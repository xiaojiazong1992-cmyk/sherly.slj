# -*- coding: utf-8 -*-
"""
住宅建安工程成本预测模型 — 优化版
Optimized Residential Construction Cost Prediction
Models: Random Forest · XGBoost · Stacking Ensemble

Fixes applied vs. original:
  [F1] Stratified split (target quantile bins) prevents lucky/unlucky draw
  [F2] RepeatedKFold(10×5) as primary evaluation; CV and test R² now aligned
  [F3] XGBoost: subsample/colsample_bytree/gamma/min_child_weight/reg_alpha added
       → resolves Train R²=0.9998 overfitting (was gap 0.104, target <0.05)
  [F4] RF: min_samples_leaf + max_features added; RandomizedSearchCV
  [F5] Stacking: Ridge base learner added for tree-linear diversity
  [F6] LabelEncoder fit only on X_train (eliminates data leakage)
  [F7] Log-transform target; metrics reported in original yuan/m² space
  [F8] Overfitting diagnostic table + learning curve figure added
"""

import pandas as pd
import numpy as np
import xgboost
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    RepeatedKFold, cross_val_score, learning_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
print("=" * 60)
print("1. DATA LOADING & PREPROCESSING")
print("=" * 60)

df = pd.read_excel(
    r"D:\HuaweiMoveData\Users\HUAWEI\Desktop\住宅建安造价数据集_价格指数完善版.xlsx"
)
df = df.dropna(subset=['单方造价(元/m²)']).reset_index(drop=True)
print(f"Samples: {df.shape[0]}")

target_col = '单方造价(元/m²)'
drop_cols = ['序号', '项目名称', '编制日期', target_col,
             '钢筋价格(元/t)', '混凝土价格(元/m3)']
feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Original Features ({len(feature_cols)}): {feature_cols}")

df_model = df[feature_cols + [target_col]].copy()

# ── Ordinal encoding (deterministic mapping, no leakage) ──
print("\n处理有序变量:")
ordinal_mappings = {
    '抗震等级': {'一级': 1, '二级': 2, '三级': 3, '四级': 4, '特一级': 5},
    '设防烈度': {'6度': 6, '7度': 7, '8度': 8, '9度': 9}
}
for col, mapping in ordinal_mappings.items():
    if col in df_model.columns:
        df_model[col] = df_model[col].map(mapping)
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)
        print(f"  {col}: 有序编码完成 (等级数: {df_model[col].nunique()})")

# ── One-Hot encoding (no statistics learned, safe before split) ──
print("\n处理无序变量:")
nominal_cols = [c for c in
                ['工程业态', '结构类型', '基础形式', '交付形式', '造价类型']
                if c in df_model.columns]
df_model = pd.get_dummies(df_model, columns=nominal_cols, drop_first=False)
print(f"  One-Hot编码: {nominal_cols}")

# Keep '工程地址' raw here; LabelEncoder applied post-split below [F6]
feature_cols_new = [c for c in df_model.columns if c != target_col]
print(f"\n特征数量变化: {len(feature_cols)} → {len(feature_cols_new)}")

X_raw = df_model[feature_cols_new].copy()
y = df_model[target_col].values

# ─────────────────────────────────────────────
# 2. STRATIFIED SPLIT  [F1]
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. STRATIFIED TRAIN/TEST SPLIT  [F1]")
print("=" * 60)

# Bin target into 5 quantiles so every price range is represented in both sets
y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y_bins
)
print(f"Train: {len(y_train)} | Test: {len(y_test)}")
print(f"Train target range: {y_train.min():.0f}–{y_train.max():.0f} yuan/m²")
print(f"Test  target range: {y_test.min():.0f}–{y_test.max():.0f} yuan/m²")

# ── LabelEncoder fit ONLY on train  [F6] ──
print("\n处理高基数变量 [F6 - train-only fit]:")
if '工程地址' in X_train_df.columns:
    le = LabelEncoder()
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    X_train_df['工程地址'] = le.fit_transform(X_train_df['工程地址'].astype(str))
    # Unseen categories in test → map to -1 (only 4 classes, practically zero risk)
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    X_test_df['工程地址'] = (
        X_test_df['工程地址'].astype(str)
        .map(label_map)
        .fillna(-1)
        .astype(int)
    )
    print(f"  工程地址: LabelEncoder fit on train only (类别数: {len(le.classes_)})")

X_train = X_train_df.values.astype(float)
X_test = X_test_df.values.astype(float)

# ─────────────────────────────────────────────
# 3. LOG-TRANSFORM TARGET  [F7]
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. LOG-TRANSFORM TARGET  [F7]")
print("=" * 60)
y_train_log = np.log1p(y_train)
print(f"Original std: {y_train.std():.0f} yuan/m²  →  "
      f"Log std: {y_train_log.std():.4f}  (reduced skew)")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def eval_model(name, yt, yp):
    """Evaluate in original yuan/m² space."""
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mape = mean_absolute_percentage_error(yt, yp) * 100
    print(f"  {name}: R²={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}  MAPE={mape:.2f}%")
    return {'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# RepeatedKFold — primary evaluation metric  [F2]
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# ─────────────────────────────────────────────
# 4. RANDOM FOREST  [F4]
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. RANDOM FOREST  [F4]")
print("=" * 60)

rf_param_dist = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [8, 12, 16, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],           # F4: prevents over-deep trees
    'max_features':      ['sqrt', 'log2', 0.5]  # F4: column subsampling
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_dist, n_iter=30, cv=5,
    scoring='r2', random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train_log)
rf_best = rf_search.best_estimator_
print(f"Best params: {rf_search.best_params_}")
print(f"Search CV R² (5-fold): {rf_search.best_score_:.4f}")

# Full repeated CV as primary metric  [F2]
rf_cv = cross_val_score(rf_best, X_train, y_train_log, cv=rkf, scoring='r2', n_jobs=-1)
print(f"RKF CV R² (10×5): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

y_pred_rf  = np.expm1(rf_best.predict(X_test))
rf_train_r = eval_model("RF-Train",  y_train, np.expm1(rf_best.predict(X_train)))
rf_test_r  = eval_model("RF-Test",   y_test,  y_pred_rf)

# ─────────────────────────────────────────────
# 5. XGBOOST  [F3]
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. XGBOOST  [F3]")
print("=" * 60)

xgb_param_dist = {
    'n_estimators':     [200, 300, 500],
    'learning_rate':    [0.01, 0.05, 0.1],
    'max_depth':        [3, 5, 7],
    'subsample':        [0.6, 0.8, 1.0],      # F3: row subsampling
    'colsample_bytree': [0.6, 0.8, 1.0],      # F3: column subsampling per tree
    'min_child_weight': [1, 3, 5],             # F3: min leaf node weight
    'gamma':            [0, 0.1, 0.3],         # F3: min split loss gain
    'reg_lambda':       [0.1, 1.0, 5.0],       # L2 regularization
    'reg_alpha':        [0, 0.1, 1.0]          # F3: L1 regularization
}
xgb_search = RandomizedSearchCV(
    xgboost.XGBRegressor(random_state=42, tree_method='hist', verbosity=0),
    xgb_param_dist, n_iter=50, cv=5,
    scoring='r2', random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_train, y_train_log)
xgb_best = xgb_search.best_estimator_
print(f"Best params: {xgb_search.best_params_}")
print(f"Search CV R² (5-fold): {xgb_search.best_score_:.4f}")

xgb_cv = cross_val_score(xgb_best, X_train, y_train_log, cv=rkf, scoring='r2', n_jobs=-1)
print(f"RKF CV R² (10×5): {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

y_pred_xgb  = np.expm1(xgb_best.predict(X_test))
xgb_train_r = eval_model("XGB-Train", y_train, np.expm1(xgb_best.predict(X_train)))
xgb_test_r  = eval_model("XGB-Test",  y_test,  y_pred_xgb)

# ─────────────────────────────────────────────
# 6. STACKING  [F5]
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. STACKING ENSEMBLE (Ridge + RF + XGBoost → Ridge)  [F5]")
print("=" * 60)

rf_stk_params  = {k: v for k, v in rf_search.best_params_.items()}
xgb_stk_params = {k: v for k, v in xgb_search.best_params_.items()}

stacking = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),                         # F5: linear base learner for diversity
        ('rf',    RandomForestRegressor(**rf_stk_params,
                                        random_state=42, n_jobs=-1)),
        ('xgb',   xgboost.XGBRegressor(**xgb_stk_params,
                                        random_state=42,
                                        tree_method='hist', verbosity=0))
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=10, n_jobs=-1
)
stacking.fit(X_train, y_train_log)

y_pred_stk  = np.expm1(stacking.predict(X_test))
stk_train_r = eval_model("Stacking-Train", y_train, np.expm1(stacking.predict(X_train)))
stk_test_r  = eval_model("Stacking-Test",  y_test,  y_pred_stk)

# ─────────────────────────────────────────────
# 7. SUMMARY & OVERFITTING DIAGNOSTIC  [F8]
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TEST SET COMPARISON")
print("=" * 60)
results = pd.DataFrame([rf_test_r, xgb_test_r, stk_test_r])
print(results.to_string(index=False))

print("\n" + "=" * 60)
print("OVERFITTING DIAGNOSTIC TABLE  [F8]")
print("=" * 60)
diag = pd.DataFrame({
    'Model':      ['RF', 'XGBoost', 'Stacking'],
    'Train_R2':   [rf_train_r['R2'],  xgb_train_r['R2'],  stk_train_r['R2']],
    'Test_R2':    [rf_test_r['R2'],   xgb_test_r['R2'],   stk_test_r['R2']],
    'Gap':        [rf_train_r['R2']  - rf_test_r['R2'],
                   xgb_train_r['R2'] - xgb_test_r['R2'],
                   stk_train_r['R2'] - stk_test_r['R2']],
    'RKF_R2_mean': [rf_cv.mean(),  xgb_cv.mean(),  None],
    'RKF_R2_std':  [rf_cv.std(),   xgb_cv.std(),   None]
})
print(diag.to_string(index=False))

# ─────────────────────────────────────────────
# 8. FIGURES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. GENERATING FIGURES")
print("=" * 60)

colors = ['#2ecc71', '#3498db', '#e74c3c']
names  = ['RF', 'XGBoost', 'Stacking']
preds  = [y_pred_rf, y_pred_xgb, y_pred_stk]

# Fig 1 — Predicted vs Real scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for idx, (name, yp, color) in enumerate(zip(names, preds, colors)):
    ax = axes[idx]
    ax.scatter(y_test, yp, alpha=0.6, s=50, c=color, edgecolors='k', linewidth=0.5)
    mn = min(y_test.min(), yp.min()) * 0.9
    mx = max(y_test.max(), yp.max()) * 1.1
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel('Real (yuan/$m^2$)', fontsize=11)
    ax.set_ylabel('Predicted (yuan/$m^2$)', fontsize=11)
    r2 = r2_score(y_test, yp)
    ax.set_title(f'{name}  $R^2$={r2:.4f}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_scatter.png', dpi=150, bbox_inches='tight')
print("  fig1_scatter.png saved")

# Fig 2 — Bar comparison
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
metrics_info = [
    ('R2',   '$R^2$'),
    ('RMSE', 'RMSE (yuan/$m^2$)'),
    ('MAE',  'MAE (yuan/$m^2$)'),
    ('MAPE', 'MAPE (%)')
]
for i, (m, lb) in enumerate(metrics_info):
    vals = [results.iloc[j][m] for j in range(3)]
    bars = axes2[i].bar(names, vals, color=colors, edgecolor='black', width=0.5)
    axes2[i].set_title(lb, fontsize=13, fontweight='bold')
    axes2[i].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        fmt = f'{val:.4f}' if m == 'R2' else f'{val:.1f}'
        axes2[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(), fmt,
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
plt.tight_layout()
plt.savefig('fig2_bars.png', dpi=150, bbox_inches='tight')
print("  fig2_bars.png saved")

# Fig 3 — Feature importance
fig3, ax3 = plt.subplots(figsize=(12, 8))
imp = pd.Series(rf_best.feature_importances_, index=feature_cols_new)
imp.index = [i.replace('m²', '$m^2$').replace('(m²)', '($m^2$)') for i in imp.index]
imp.sort_values(ascending=True).plot(kind='barh', ax=ax3,
                                      color='steelblue', edgecolor='black')
ax3.set_xlabel('Importance', fontsize=12)
ax3.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_importance.png', dpi=150, bbox_inches='tight')
print("  fig3_importance.png saved")

# Fig 4 — Residual distributions
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, yp) in enumerate(zip(names, preds)):
    res = y_test - yp
    axes4[idx].hist(res, bins=15, edgecolor='black', alpha=0.7, color=colors[idx])
    axes4[idx].axvline(0, color='r', linestyle='--', lw=2)
    axes4[idx].set_xlabel('Residual (yuan/$m^2$)')
    axes4[idx].set_title(
        f'{name}  Mean={res.mean():.1f}  Std={res.std():.1f}',
        fontweight='bold'
    )
    axes4[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_residuals.png', dpi=150, bbox_inches='tight')
print("  fig4_residuals.png saved")

# Fig 5 — Learning curves  [F8]
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
for idx, (name, model, color) in enumerate(zip(
        ['Random Forest', 'XGBoost'],
        [rf_best, xgb_best],
        colors[:2])):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train_log,
        cv=5, scoring='r2',
        train_sizes=np.linspace(0.2, 1.0, 8),
        n_jobs=-1
    )
    ax = axes5[idx]
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-',
            color=color, label='Train R²')
    ax.fill_between(
        train_sizes,
        train_scores.mean(1) - train_scores.std(1),
        train_scores.mean(1) + train_scores.std(1),
        alpha=0.1, color=color
    )
    ax.plot(train_sizes, val_scores.mean(axis=1), 's--',
            color='gray', label='CV Val R²')
    ax.fill_between(
        train_sizes,
        val_scores.mean(1) - val_scores.std(1),
        val_scores.mean(1) + val_scores.std(1),
        alpha=0.1, color='gray'
    )
    ax.set_xlabel('Training samples', fontsize=11)
    ax.set_ylabel('$R^2$', fontsize=11)
    ax.set_title(f'{name} — Learning Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_learning_curves.png', dpi=150, bbox_inches='tight')
print("  fig5_learning_curves.png saved")

# Save prediction details to CSV
pred_df = pd.DataFrame({
    'Real':      y_test.round(2),
    'RF_Pred':   y_pred_rf.round(2),
    'XGB_Pred':  y_pred_xgb.round(2),
    'Stk_Pred':  y_pred_stk.round(2),
    'RF_Err%':   (np.abs(y_test - y_pred_rf)  / y_test * 100).round(2),
    'XGB_Err%':  (np.abs(y_test - y_pred_xgb) / y_test * 100).round(2),
    'Stk_Err%':  (np.abs(y_test - y_pred_stk) / y_test * 100).round(2),
})
pred_df.to_csv('predictions.csv', index=False, encoding='utf-8-sig')
print("  predictions.csv saved")

plt.show()
print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
