# -*- coding: utf-8 -*-
"""
models.py
模型定义与训练：Ridge → 随机森林 → XGBoost → Blending → Stacking
严格使用Pipeline封装，防止数据泄露
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict
from scipy.stats import uniform, randint
from scipy.optimize import minimize
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor

# 模型保存目录
MODEL_DIR = Path('outputs/saved_models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# M0: Ridge 线性基线
# ─────────────────────────────────────────────

def train_ridge(X_train, y_train, cv_splits, preprocessor=None):
    """
    M0: Ridge回归线性基线
    通过GridSearchCV搜索正则化系数alpha
    参数:
        X_train: 训练特征DataFrame
        y_train: 目标变量（log空间）
        cv_splits: Expanding Window CV分割列表
        preprocessor: 预处理器（None则使用默认）
    返回:
        best_ridge: 最优Ridge Pipeline
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    ridge_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge()),
    ])

    param_grid = {'regressor__alpha': [0.01, 0.1, 1.0, 10, 100, 1000]}

    cv = GridSearchCV(
        ridge_pipe, param_grid,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0,
    )
    cv.fit(X_train, y_train)

    print(f"[Ridge] 最优alpha: {cv.best_params_['regressor__alpha']}")
    print(f"[Ridge] CV MAE: {-cv.best_score_:.4f}")

    joblib.dump(cv.best_estimator_, MODEL_DIR / 'ridge_best.pkl')
    return cv.best_estimator_


# ─────────────────────────────────────────────
# M1: 随机森林
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train, cv_splits, preprocessor=None):
    """
    M1: 随机森林回归
    GridSearchCV网格搜索最优超参数
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    param_grid = {
        'regressor__n_estimators':    [100, 200, 300],
        'regressor__max_depth':       [8, 12, 16, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf':  [1, 2, 4],
    }

    cv = GridSearchCV(
        rf_pipe, param_grid,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
    )
    cv.fit(X_train, y_train)

    print(f"\n[RF] 最优参数: {cv.best_params_}")
    print(f"[RF] CV MAE: {-cv.best_score_:.4f}")

    joblib.dump(cv.best_estimator_, MODEL_DIR / 'rf_best.pkl')
    return cv.best_estimator_


# ─────────────────────────────────────────────
# M2: XGBoost（两步搜索）
# ─────────────────────────────────────────────

def train_xgboost(X_train, y_train, cv_splits, preprocessor=None):
    """
    M2: XGBoost两步超参数搜索
    第一步：RandomizedSearchCV粗搜索（50次迭代）
    第二步：GridSearchCV在最优附近细搜索
    两步搜索兼顾搜索效率与精度
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    xgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            eval_metric='mae',
            verbosity=0,
        )),
    ])

    # ── 第一步：粗搜索（RandomizedSearchCV, 50次）──
    coarse_params = {
        'regressor__n_estimators':    randint(100, 400),
        'regressor__max_depth':       randint(3, 10),
        'regressor__learning_rate':   uniform(0.01, 0.19),
        'regressor__subsample':       uniform(0.6, 0.4),
        'regressor__colsample_bytree': uniform(0.6, 0.4),
        'regressor__min_child_weight': randint(1, 8),
    }

    coarse_search = RandomizedSearchCV(
        xgb_pipe, coarse_params,
        n_iter=50,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    coarse_search.fit(X_train, y_train)

    bp = coarse_search.best_params_
    print(f"\n[XGB粗搜索] 最优参数: {bp}")
    print(f"[XGB粗搜索] CV MAE: {-coarse_search.best_score_:.4f}")

    # ── 第二步：细搜索（GridSearchCV，在最优附近精调）──
    def clip_int(val, low, high):
        return max(low, min(high, int(val)))

    fine_params = {
        'regressor__max_depth': [
            clip_int(bp['regressor__max_depth'] - 1, 2, 12),
            clip_int(bp['regressor__max_depth'],     2, 12),
            clip_int(bp['regressor__max_depth'] + 1, 2, 12),
        ],
        'regressor__learning_rate': [
            round(bp['regressor__learning_rate'] * 0.7, 4),
            round(bp['regressor__learning_rate'],       4),
            round(bp['regressor__learning_rate'] * 1.3, 4),
        ],
        'regressor__n_estimators':     [bp['regressor__n_estimators']],
        'regressor__subsample':        [round(bp['regressor__subsample'], 3)],
        'regressor__colsample_bytree': [round(bp['regressor__colsample_bytree'], 3)],
        'regressor__min_child_weight': [bp['regressor__min_child_weight']],
    }
    # 去重
    fine_params['regressor__max_depth']    = list(set(fine_params['regressor__max_depth']))
    fine_params['regressor__learning_rate'] = list(set(fine_params['regressor__learning_rate']))

    fine_search = GridSearchCV(
        xgb_pipe, fine_params,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
    )
    fine_search.fit(X_train, y_train)

    print(f"\n[XGB细搜索] 最优参数: {fine_search.best_params_}")
    print(f"[XGB细搜索] CV MAE: {-fine_search.best_score_:.4f}")

    joblib.dump(fine_search.best_estimator_, MODEL_DIR / 'xgb_best.pkl')
    return fine_search.best_estimator_


# ─────────────────────────────────────────────
# M3a: Optimized Blending（OOF权重优化）
# ─────────────────────────────────────────────

def train_blending(best_rf, best_xgb, X_train, y_train, cv_splits):
    """
    M3a: Optimized Blending
    使用OOF（Out-of-Fold）预测在训练集内部优化融合权重
    严格防止测试集信息泄露：权重求解过程完全不接触测试集
    参数:
        best_rf:  已训练的RF Pipeline
        best_xgb: 已训练的XGB Pipeline
        X_train, y_train: 训练数据
        cv_splits: Expanding Window CV分割
    返回:
        w_rf, w_xgb: 最优融合权重（两者之和=1）
    """
    print("[Blending] 计算OOF预测...")

    # 用Expanding Window CV在训练集内部做OOF预测
    oof_rf  = cross_val_predict(best_rf,  X_train, y_train, cv=cv_splits)
    oof_xgb = cross_val_predict(best_xgb, X_train, y_train, cv=cv_splits)

    def mape_objective(weights):
        """以MAPE作为目标函数（在原始空间计算）"""
        w1, w2 = weights
        pred_blend = w1 * oof_rf + w2 * oof_xgb
        y_real  = np.expm1(y_train.values)
        y_blend = np.expm1(pred_blend)
        # 避免除零
        mask = y_real > 0
        return np.mean(np.abs((y_real[mask] - y_blend[mask]) / y_real[mask])) * 100

    # SLSQP约束优化：权重在[0,1]，且总和=1
    result = minimize(
        mape_objective,
        x0=[0.5, 0.5],
        bounds=[(0, 1), (0, 1)],
        constraints={'type': 'eq', 'fun': lambda w: 1 - sum(w)},
        method='SLSQP',
    )

    w_rf, w_xgb = result.x
    oof_mape = mape_objective([w_rf, w_xgb])

    print(f"[Blending] 最优权重: RF={w_rf:.3f}, XGBoost={w_xgb:.3f}")
    print(f"[Blending] OOF MAPE: {oof_mape:.2f}%")

    # 保存权重
    joblib.dump({'w_rf': w_rf, 'w_xgb': w_xgb},
                MODEL_DIR / 'blending_weights.pkl')
    return w_rf, w_xgb


def predict_blending(best_rf, best_xgb, w_rf, w_xgb, X):
    """Blending推理：加权平均两个模型的预测（log空间）"""
    pred_rf  = best_rf.predict(X)
    pred_xgb = best_xgb.predict(X)
    return w_rf * pred_rf + w_xgb * pred_xgb


# ─────────────────────────────────────────────
# M3b: Stacking
# ─────────────────────────────────────────────

def train_stacking(X_train_proc, y_train, cv_splits,
                   rf_params, xgb_params):
    """
    M3b: Stacking（先对数据预处理，再在处理后数据上做Stacking）
    基学习器：RF + XGBoost
    元学习器：Ridge
    注意：输入X_train_proc为已经过preprocessor.fit_transform的数组
    参数:
        X_train_proc: 预处理后的训练特征数组
        y_train:      目标变量（log空间）
        cv_splits:    Expanding Window CV分割（list of (train_idx, val_idx)）
        rf_params:    RF裸模型最优参数（不含'regressor__'前缀）
        xgb_params:   XGB裸模型最优参数（不含'regressor__'前缀）
    返回:
        stacking: 已训练的StackingRegressor
    """
    # 从Pipeline参数提取裸模型参数（去掉'regressor__'前缀）
    def strip_prefix(params, prefix='regressor__'):
        return {k.replace(prefix, ''): v
                for k, v in params.items() if k.startswith(prefix)}

    rf_bare_params  = strip_prefix(rf_params)
    xgb_bare_params = strip_prefix(xgb_params)

    # 确保XGB参数合法
    xgb_bare_params.setdefault('objective', 'reg:squarederror')
    xgb_bare_params.setdefault('verbosity', 0)

    stacking = StackingRegressor(
        estimators=[
            ('rf',  RandomForestRegressor(**rf_bare_params,  random_state=42, n_jobs=-1)),
            ('xgb', XGBRegressor(**xgb_bare_params, random_state=42)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=cv_splits,        # Expanding Window CV
        passthrough=False,   # 只传递元特征
        n_jobs=-1,
    )
    stacking.fit(X_train_proc, y_train)

    print("[Stacking] 元学习器系数（Ridge）：",
          stacking.final_estimator_.coef_)

    joblib.dump(stacking, MODEL_DIR / 'stacking_best.pkl')
    return stacking


# ─────────────────────────────────────────────
# 模型加载辅助
# ─────────────────────────────────────────────

def load_model(model_name):
    """
    从outputs/saved_models/加载已保存的模型
    model_name: 'ridge' / 'rf' / 'xgb' / 'stacking' / 'blending_weights'
    """
    path_map = {
        'ridge':            MODEL_DIR / 'ridge_best.pkl',
        'rf':               MODEL_DIR / 'rf_best.pkl',
        'xgb':              MODEL_DIR / 'xgb_best.pkl',
        'stacking':         MODEL_DIR / 'stacking_best.pkl',
        'blending_weights': MODEL_DIR / 'blending_weights.pkl',
    }
    path = path_map.get(model_name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"未找到模型文件: {path}")
    return joblib.load(path)
