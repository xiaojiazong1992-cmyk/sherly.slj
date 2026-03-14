# -*- coding: utf-8 -*-
"""
run_all.py
End-to-end pipeline: data -> EDA -> features -> models -> SHAP -> Excel + figures
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ── Set project root so src imports work ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend

# ── Font setup ────────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'SimHei' in available_fonts:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    print("[Font] Using SimHei")
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("[Font] SimHei not found, using DejaVu Sans")
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR  = os.path.join(OUTPUTS_DIR, 'figures')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300

def fig_path(name):
    return os.path.join(FIGURES_DIR, name)

def out_path(name):
    return os.path.join(OUTPUTS_DIR, name)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Data collection (fallback)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Data Collection")
print("="*60)

try:
    from src.data_scraper import generate_fallback_data, get_material_price_df
    df_raw = generate_fallback_data(n=350, random_state=42)
    print(f"[OK] Raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} cols")

    # Save raw data
    df_raw.to_excel(out_path('01_raw_data.xlsx'), index=False)
    print(f"[OK] Saved: outputs/01_raw_data.xlsx")

    step1_ok = True
except Exception as e:
    print(f"[FAIL] Step 1 error: {e}")
    import traceback; traceback.print_exc()
    step1_ok = False
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Preprocessing / Cleaning
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Preprocessing & Cleaning")
print("="*60)

try:
    from src.feature_engineering import remove_outliers_iqr, dedup_by_key_fields, report_missing
    from src.preprocessing import add_derived_features, log_transform_target, log_transform_area

    df_clean = df_raw.copy()
    df_clean = dedup_by_key_fields(df_clean)
    df_clean = remove_outliers_iqr(df_clean, 'unit_cost', factor=1.5)
    report_missing(df_clean)

    # Add derived features
    df_clean = add_derived_features(df_clean)
    # Log-transform area and target
    df_clean = log_transform_area(df_clean, col='total_area')
    df_clean = log_transform_target(df_clean, target_col='unit_cost')

    print(f"[OK] Clean data: {df_clean.shape[0]} rows, {df_clean.shape[1]} cols")

    # Build stats summary
    numeric_summary = df_clean.select_dtypes(include=[np.number]).describe().T
    numeric_summary.columns = ['count','mean','std','min','25%','50%','75%','max']

    with pd.ExcelWriter(out_path('02_clean_data.xlsx')) as writer:
        df_clean.to_excel(writer, sheet_name='clean_data', index=False)
        numeric_summary.to_excel(writer, sheet_name='stats')

    print(f"[OK] Saved: outputs/02_clean_data.xlsx")
    step2_ok = True
except Exception as e:
    print(f"[FAIL] Step 2 error: {e}")
    import traceback; traceback.print_exc()
    step2_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: EDA – Correlation heatmap (fig3-1) + VIF (fig3-2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: EDA Figures (fig3-1, fig3-2)")
print("="*60)

try:
    from src.feature_engineering import (
        plot_correlation_heatmap, check_vif, plot_vif_barh
    )
    from src.preprocessing import NUMERIC_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES

    # Pearson correlation heatmap (fig3-1)
    numeric_cols_for_corr = [c for c in
                              NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
                              if c in df_clean.columns]
    corr_matrix = plot_correlation_heatmap(
        df_clean, numeric_cols_for_corr,
        target_col='unit_cost_log',
        save_path=fig_path('fig3-1_correlation_heatmap.png')
    )
    plt.close('all')

    # VIF analysis (fig3-2)
    vif_cols = [c for c in NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
                if c in df_clean.columns]
    X_vif = df_clean[vif_cols].dropna()
    vif_data = check_vif(X_vif, threshold=10)
    plot_vif_barh(vif_data, threshold=10,
                  save_path=fig_path('fig3-2_vif.png'))
    plt.close('all')

    # Save feature stats to Excel
    corr_target = df_clean[numeric_cols_for_corr + ['unit_cost_log']].corr(method='pearson')
    with pd.ExcelWriter(out_path('03_feature_stats.xlsx')) as writer:
        vif_data.to_excel(writer, sheet_name='VIF', index=False)
        corr_target.to_excel(writer, sheet_name='Correlation')

    print(f"[OK] Saved: outputs/03_feature_stats.xlsx")
    step3_ok = True
except Exception as e:
    print(f"[FAIL] Step 3 error: {e}")
    import traceback; traceback.print_exc()
    step3_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Distribution figures (fig4-1) + time-split
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Distribution Figures + Train/Test Split")
print("="*60)

try:
    from src.feature_engineering import compare_log_transform
    from src.preprocessing import time_based_split, year_expanding_cv

    # fig4-1: distribution comparison
    compare_log_transform(
        df_raw['unit_cost'],
        col_name='unit_cost',
        save_path=fig_path('fig4-1_distribution.png')
    )
    plt.close('all')

    # Time-based split
    train_df, test_df = time_based_split(df_clean, test_year=2025)
    print(f"[OK] Train: {len(train_df)}, Test: {len(test_df)}")

    from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES, TARGET

    feature_cols = (NUMERIC_FEATURES + CATEGORICAL_FEATURES +
                    ORDINAL_FEATURES + BINARY_FEATURES)
    # Keep only columns that actually exist
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET]

    # Build CV splits (list form for sklearn) - convert to positional indices
    # year_expanding_cv returns DataFrame index labels; sklearn needs positional (iloc) indices
    train_df_reset = train_df.reset_index(drop=True)
    X_train_reset  = train_df_reset[feature_cols]
    y_train_reset  = train_df_reset[TARGET]

    # Rebuild positional CV splits on the reset DataFrame
    cv_splits_raw = list(year_expanding_cv(train_df_reset, year_col='completion_year'))
    # These are already positional since we reset_index above
    cv_splits = cv_splits_raw
    print(f"[OK] CV folds: {len(cv_splits)}")

    step4_ok = True
except Exception as e:
    print(f"[FAIL] Step 4 error: {e}")
    import traceback; traceback.print_exc()
    step4_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Model training
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Model Training (Ridge, RF, XGBoost, Blending, Stacking)")
print("="*60)

try:
    from src.models import (train_ridge, train_random_forest, train_xgboost,
                             train_blending, predict_blending, train_stacking)
    from src.preprocessing import build_preprocessor, get_feature_names

    preprocessor = build_preprocessor()

    # Use positional-indexed training data and CV splits
    # (cv_splits built on train_df_reset which has positional integer index)

    # --- Ridge ---
    print("\n[Ridge]")
    best_ridge = train_ridge(X_train_reset, y_train_reset, cv_splits,
                             preprocessor=build_preprocessor())

    # --- Random Forest ---
    print("\n[Random Forest]")
    best_rf = train_random_forest(X_train_reset, y_train_reset, cv_splits,
                                  preprocessor=build_preprocessor())

    # --- XGBoost ---
    print("\n[XGBoost]")
    best_xgb = train_xgboost(X_train_reset, y_train_reset, cv_splits,
                              preprocessor=build_preprocessor())

    # --- Blending ---
    # cross_val_predict requires non-overlapping partitions; expanding-window
    # folds are not partitions (train sets overlap). Compute OOF manually:
    print("\n[Blending] Computing OOF predictions manually (expanding window)...")
    oof_rf_arr  = np.zeros(len(y_train_reset))
    oof_xgb_arr = np.zeros(len(y_train_reset))

    import copy
    from sklearn.pipeline import clone as sk_clone

    for tr_idx, val_idx in cv_splits:
        Xf_tr = X_train_reset.iloc[tr_idx]
        yf_tr = y_train_reset.iloc[tr_idx]
        Xf_val = X_train_reset.iloc[val_idx]

        rf_fold  = sk_clone(best_rf);  rf_fold.fit(Xf_tr, yf_tr)
        xgb_fold = sk_clone(best_xgb); xgb_fold.fit(Xf_tr, yf_tr)

        oof_rf_arr[val_idx]  = rf_fold.predict(Xf_val)
        oof_xgb_arr[val_idx] = xgb_fold.predict(Xf_val)

    from scipy.optimize import minimize

    def mape_obj(weights):
        w1, w2 = weights
        pred_blend = w1 * oof_rf_arr + w2 * oof_xgb_arr
        y_real  = np.expm1(y_train_reset.values)
        y_blend = np.expm1(pred_blend)
        mask = y_real > 0
        return np.mean(np.abs((y_real[mask] - y_blend[mask]) / y_real[mask])) * 100

    res_opt = minimize(mape_obj, x0=[0.5, 0.5],
                       bounds=[(0,1),(0,1)],
                       constraints={'type':'eq','fun': lambda w: 1-sum(w)},
                       method='SLSQP')
    w_rf, w_xgb = res_opt.x
    print(f"[Blending] Optimal weights: RF={w_rf:.3f}, XGB={w_xgb:.3f}")

    # --- Stacking ---
    print("\n[Stacking]")
    # Get best RF and XGB params (strip 'regressor__' prefix later in train_stacking)
    # Exclude keys that train_stacking already passes explicitly (random_state, n_jobs, etc.)
    _EXCLUDE_RF  = {'random_state', 'n_jobs', 'warm_start', 'oob_score'}
    _EXCLUDE_XGB = {'random_state', 'n_jobs', 'verbosity', 'objective',
                    'eval_metric', 'use_label_encoder'}

    rf_best_params  = {k: v for k, v in
                       best_rf.named_steps['regressor'].get_params().items()
                       if k not in _EXCLUDE_RF}
    xgb_best_params = {k: v for k, v in
                       best_xgb.named_steps['regressor'].get_params().items()
                       if k not in _EXCLUDE_XGB}

    # For stacking we pass preprocessed data
    preprocessor_for_stack = build_preprocessor()
    preprocessor_for_stack.fit(X_train_reset, y_train_reset)
    X_train_proc = preprocessor_for_stack.transform(X_train_reset)
    X_test_proc  = preprocessor_for_stack.transform(X_test)

    # Re-package params with 'regressor__' prefix as train_stacking uses strip_prefix
    rf_params_prefixed  = {f'regressor__{k}': v for k, v in rf_best_params.items()}
    xgb_params_prefixed = {f'regressor__{k}': v for k, v in xgb_best_params.items()}

    # StackingRegressor also needs non-overlapping CV; use simple k-fold here
    from sklearn.model_selection import KFold
    stack_cv = KFold(n_splits=4, shuffle=True, random_state=42)

    best_stacking = train_stacking(
        X_train_proc, y_train_reset, stack_cv,
        rf_params_prefixed, xgb_params_prefixed
    )

    step5_ok = True
    print("[OK] All models trained")
except Exception as e:
    print(f"[FAIL] Step 5 error: {e}")
    import traceback; traceback.print_exc()
    step5_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Evaluation + figures (fig5-1, fig5-2, fig5-3, fig5-7)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Evaluation + Figures")
print("="*60)

try:
    from src.evaluation import (evaluate_model, build_results_table,
                                  plot_model_comparison, plot_pred_vs_actual,
                                  plot_residuals, plot_learning_curve)

    # Generate predictions (log space)
    pred_ridge    = best_ridge.predict(X_test)
    pred_rf       = best_rf.predict(X_test)
    pred_xgb      = best_xgb.predict(X_test)
    pred_blending = predict_blending(best_rf, best_xgb, w_rf, w_xgb, X_test)
    pred_stacking = best_stacking.predict(X_test_proc)

    y_test_arr = y_test.values

    # Evaluate each model
    results = []
    preds_dict = {}
    for name, pred in [
        ('Ridge',    pred_ridge),
        ('RandomForest', pred_rf),
        ('XGBoost',  pred_xgb),
        ('Blending', pred_blending),
        ('Stacking', pred_stacking),
    ]:
        res = evaluate_model(y_test_arr, pred, model_name=name)
        results.append(res)
        preds_dict[name] = pred

    # Table with Bootstrap CI
    df_results = build_results_table(results, y_test_arr, preds_dict, n_bootstrap=1000)
    df_results.to_excel(out_path('04_model_results.xlsx'), index=False)
    print(f"[OK] Saved: outputs/04_model_results.xlsx")

    # fig5-1: model comparison bar chart
    plot_model_comparison(results, save_path=fig_path('fig5-1_model_comparison.png'))
    plt.close('all')

    # Best model = Stacking (usually best; pick by lowest MAPE)
    best_name = min(results, key=lambda r: r['MAPE'])['model']
    best_pred = preds_dict[best_name]
    print(f"[INFO] Best model: {best_name}")

    # fig5-2: pred vs actual
    plot_pred_vs_actual(y_test_arr, best_pred, model_name=best_name,
                        save_path=fig_path('fig5-2_pred_vs_actual.png'))
    plt.close('all')

    # fig5-3: residuals
    plot_residuals(y_test_arr, best_pred, model_name=best_name,
                   save_path=fig_path('fig5-3_residuals.png'))
    plt.close('all')

    # fig5-7: learning curve (use best_xgb as it supports learning curve via pipeline)
    plot_learning_curve(
        best_xgb, X_train_reset, y_train_reset, cv_splits,
        model_name='XGBoost',
        save_path=fig_path('fig5-7_learning_curve.png')
    )
    plt.close('all')

    step6_ok = True
except Exception as e:
    print(f"[FAIL] Step 6 error: {e}")
    import traceback; traceback.print_exc()
    step6_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: SHAP Analysis + figures (fig5-4, fig5-5, fig5-6)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: SHAP Analysis")
print("="*60)

try:
    import shap

    # Use the XGBoost model's preprocessor to get transformed features
    xgb_preprocessor = best_xgb.named_steps['preprocessor']
    xgb_regressor    = best_xgb.named_steps['regressor']

    X_test_shap  = xgb_preprocessor.transform(X_test)
    X_train_shap = xgb_preprocessor.transform(X_train)

    # Get feature names after preprocessing
    try:
        feature_names_out = xgb_preprocessor.get_feature_names_out()
    except Exception:
        feature_names_out = [f'f{i}' for i in range(X_test_shap.shape[1])]

    # Clean feature names (remove transformer prefix like 'num__', 'cat__', etc.)
    clean_names = []
    for n in feature_names_out:
        parts = n.split('__', 1)
        clean_names.append(parts[-1] if len(parts) > 1 else n)

    # TreeExplainer for XGBoost
    explainer   = shap.TreeExplainer(xgb_regressor)
    shap_values = explainer.shap_values(X_test_shap)

    # ── fig5-4: SHAP summary bar plot ────────────────────────────────────────
    fig54, ax54 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_shap,
                      feature_names=clean_names,
                      plot_type='bar',
                      show=False,
                      max_display=15)
    plt.title('图5-4 SHAP特征重要性（全局）', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path('fig5-4_shap_bar.png'), dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] fig5-4 saved")

    # ── fig5-5: SHAP beeswarm summary plot ───────────────────────────────────
    fig55 = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_shap,
                      feature_names=clean_names,
                      plot_type='dot',
                      show=False,
                      max_display=15)
    plt.title('图5-5 SHAP蜂群图（特征影响方向）', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path('fig5-5_shap_beeswarm.png'), dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] fig5-5 saved")

    # ── fig5-6: SHAP dependence plot for top feature ──────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs_shap))
    top_feature = clean_names[top_idx]
    print(f"[INFO] Top SHAP feature: {top_feature} (idx={top_idx})")

    fig56 = plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        top_idx, shap_values, X_test_shap,
        feature_names=clean_names,
        show=False,
        interaction_index=None,
    )
    plt.title(f'图5-6 SHAP依赖图 ({top_feature})', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path('fig5-6_shap_dependence.png'), dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] fig5-6 saved")

    # ── Build SHAP importance DataFrame ──────────────────────────────────────
    shap_importance = pd.DataFrame({
        'feature': clean_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_importance['rank'] = range(1, len(shap_importance) + 1)

    shap_importance.to_excel(out_path('05_shap_importance.xlsx'), index=False)
    print(f"[OK] Saved: outputs/05_shap_importance.xlsx")

    step7_ok = True
except Exception as e:
    print(f"[FAIL] Step 7 error: {e}")
    import traceback; traceback.print_exc()
    step7_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Additional EDA figures (fig4-2, fig4-3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8: Additional EDA Figures (fig4-2, fig4-3)")
print("="*60)

try:
    import seaborn as sns

    # ── fig4-2: Unit cost by province (box plot) ─────────────────────────────
    fig42, ax42 = plt.subplots(figsize=(14, 6))
    province_order = (df_raw.groupby('province')['unit_cost']
                      .median().sort_values(ascending=False).index.tolist())
    sns.boxplot(data=df_raw, x='province', y='unit_cost',
                order=province_order, palette='Set3', ax=ax42)
    ax42.set_title('图4-2 各省份单方造价分布（箱线图）', fontsize=13)
    ax42.set_xlabel('省份', fontsize=11)
    ax42.set_ylabel('单方造价（元/㎡）', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path('fig4-2_unit_cost_by_province.png'), dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] fig4-2 saved")

    # ── fig4-3: Unit cost by structure type and year ──────────────────────────
    fig43, axes43 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: unit cost by structure type
    sns.boxplot(data=df_raw, x='structure_type', y='unit_cost',
                palette='Set2', ax=axes43[0])
    axes43[0].set_title('按结构类型', fontsize=12)
    axes43[0].set_xlabel('结构类型', fontsize=11)
    axes43[0].set_ylabel('单方造价（元/㎡）', fontsize=11)
    axes43[0].tick_params(axis='x', rotation=15)

    # Right: unit cost by completion year
    year_means = df_raw.groupby('completion_year')['unit_cost'].mean().reset_index()
    axes43[1].bar(year_means['completion_year'].astype(str),
                  year_means['unit_cost'],
                  color='steelblue', alpha=0.8, edgecolor='white')
    axes43[1].set_title('按竣工年份（均值）', fontsize=12)
    axes43[1].set_xlabel('竣工年份', fontsize=11)
    axes43[1].set_ylabel('平均单方造价（元/㎡）', fontsize=11)
    axes43[1].tick_params(axis='x', rotation=15)

    plt.suptitle('图4-3 造价影响因素分析', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path('fig4-3_cost_analysis.png'), dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] fig4-3 saved")

    step8_ok = True
except Exception as e:
    print(f"[FAIL] Step 8 error: {e}")
    import traceback; traceback.print_exc()
    step8_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

steps = [
    ("Step 1 - Data Collection",   step1_ok),
    ("Step 2 - Preprocessing",     step2_ok),
    ("Step 3 - EDA (corr + VIF)",  step3_ok),
    ("Step 4 - Distribution + Split", step4_ok),
    ("Step 5 - Model Training",    step5_ok),
    ("Step 6 - Evaluation",        step6_ok),
    ("Step 7 - SHAP Analysis",     step7_ok),
    ("Step 8 - Extra EDA figs",    step8_ok),
]
for name, ok in steps:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}")

print("\n── Excel Outputs ──")
for f in sorted(os.listdir(OUTPUTS_DIR)):
    if f.endswith('.xlsx'):
        full = os.path.join(OUTPUTS_DIR, f)
        print(f"  {full}")

print("\n── Figures ──")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith('.png'):
        full = os.path.join(FIGURES_DIR, f)
        print(f"  {full}")

print("\nDone.")
