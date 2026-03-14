# -*- coding: utf-8 -*-
"""
evaluation.py
评估模块：基础指标 + Bootstrap 95%置信区间 + 论文图表（5-1 ~ 5-3, 5-7）
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

def _setup_chinese_font():
    candidates = ['SimHei', 'SimSun', 'Microsoft YaHei',
                  'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
                  'Noto Sans CJK SC', 'PingFang SC']
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), None)
    if chosen:
        matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        import os
        wqy = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        if os.path.exists(wqy):
            prop = matplotlib.font_manager.FontProperties(fname=wqy)
            matplotlib.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()
DPI = 300


# ─────────────────────────────────────────────
# 基础评估函数
# ─────────────────────────────────────────────

def evaluate_model(y_true_log, y_pred_log, model_name='Model'):
    """
    评估模型性能（在原始造价空间，不是log空间）
    先将预测值和真实值从log空间还原，再计算指标
    指标：MAPE(%), RMSE(元), MAE(元), R²
    """
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.expm1(np.asarray(y_pred_log))

    # 防止除零
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"[{model_name}]  MAPE={mape:.2f}%  |  "
          f"RMSE={rmse:.0f}元  |  MAE={mae:.0f}元  |  R²={r2:.4f}")

    return {'model': model_name, 'MAPE': mape, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


# ─────────────────────────────────────────────
# Bootstrap 95%置信区间
# ─────────────────────────────────────────────

def bootstrap_mape(y_true_log, y_pred_log, n_bootstrap=1000, random_state=42):
    """
    Bootstrap重采样计算MAPE的95%置信区间
    方法：有放回抽样 n_bootstrap 次，每次计算MAPE，取2.5%和97.5%分位数
    返回: (均值, 95%CI下界, 95%CI上界)
    """
    rng = np.random.RandomState(random_state)
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.expm1(np.asarray(y_pred_log))
    n = len(y_true)

    mapes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        mask = yt > 0
        if mask.sum() == 0:
            continue
        mapes.append(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)

    mean_mape = np.mean(mapes)
    ci_low, ci_high = np.percentile(mapes, [2.5, 97.5])
    return mean_mape, ci_low, ci_high


def build_results_table(results_list, y_true_log, preds_dict, n_bootstrap=1000):
    """
    构建性能对比表（表5-1），含Bootstrap 95%置信区间
    参数:
        results_list: [evaluate_model的返回字典, ...]
        y_true_log:   真实值（log空间）
        preds_dict:   {model_name: y_pred_log}
        n_bootstrap:  Bootstrap次数
    返回:
        DataFrame（可直接输出为论文表格）
    """
    rows = []
    for res in results_list:
        name = res['model']
        if name in preds_dict:
            mean_m, ci_l, ci_h = bootstrap_mape(
                y_true_log, preds_dict[name], n_bootstrap)
            ci_str = f"{mean_m:.2f}% [{ci_l:.2f}%, {ci_h:.2f}%]"
        else:
            ci_str = f"{res['MAPE']:.2f}% [N/A]"

        rows.append({
            '模型': name,
            'MAPE (%)': f"{res['MAPE']:.2f}",
            'MAPE 95%CI': ci_str,
            'RMSE (元/㎡)': f"{res['RMSE']:.0f}",
            'MAE (元/㎡)':  f"{res['MAE']:.0f}",
            'R²': f"{res['R2']:.4f}",
        })

    df_table = pd.DataFrame(rows)
    print("\n表5-1 各模型性能对比（测试集）")
    print(df_table.to_string(index=False))
    return df_table


# ─────────────────────────────────────────────
# 图5-1: 模型性能对比柱状图
# ─────────────────────────────────────────────

def plot_model_comparison(results_list, metric='MAPE',
                           save_path='outputs/figures/fig5_1_model_comparison.png'):
    """
    图5-1: 各模型MAPE/R²对比分组柱状图
    同时展示MAPE（越小越好）和R²（越大越好）
    """
    models = [r['model'] for r in results_list]
    mapes  = [r['MAPE'] for r in results_list]
    r2s    = [r['R2']   for r in results_list]

    x = np.arange(len(models))
    width = 0.35
    colors_mape = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
    colors_r2   = ['#b07aa1', '#ff9da7', '#9c755f', '#bab0ac', '#edc948']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MAPE对比（越低越好）
    bars1 = ax1.bar(x, mapes, color=colors_mape[:len(models)],
                    alpha=0.85, edgecolor='white', width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax1.set_ylabel('MAPE (%)', fontsize=11)
    ax1.set_title('各模型MAPE对比（越低越好）', fontsize=12)
    for bar, val in zip(bars1, mapes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    # R²对比（越高越好）
    bars2 = ax2.bar(x, r2s, color=colors_r2[:len(models)],
                    alpha=0.85, edgecolor='white', width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_title('各模型R²对比（越高越好）', fontsize=12)
    ax2.set_ylim(max(0, min(r2s) - 0.1), 1.05)
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('图5-1 模型性能对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"[图5-1] 已保存至 {save_path}")
    return fig


# ─────────────────────────────────────────────
# 图5-2: 预测值 vs 实际值散点图
# ─────────────────────────────────────────────

def plot_pred_vs_actual(y_true_log, y_pred_log, model_name='最优模型',
                         save_path='outputs/figures/fig5_2_pred_vs_actual.png'):
    """
    图5-2: 预测值 vs 实际值散点图（原始空间）
    包含45°参考线和R²标注
    """
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.expm1(np.asarray(y_pred_log))
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.6, color='#4e79a7',
               edgecolors='white', linewidths=0.5, s=60, label='样本点')

    # 45°完美预测线
    lims = [min(y_true.min(), y_pred.min()) * 0.95,
            max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='完美预测线(45°)')

    ax.set_xlabel('实际单方造价（元/㎡）', fontsize=12)
    ax.set_ylabel('预测单方造价（元/㎡）', fontsize=12)
    ax.set_title(f'图5-2 {model_name} 预测值 vs 实际值\n(R²={r2:.4f})',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"[图5-2] 已保存至 {save_path}")
    return fig


# ─────────────────────────────────────────────
# 图5-3: 残差分布图
# ─────────────────────────────────────────────

def plot_residuals(y_true_log, y_pred_log, model_name='最优模型',
                   save_path='outputs/figures/fig5_3_residuals.png'):
    """
    图5-3: 残差分布诊断图
    左：残差直方图（检验残差是否近似正态）
    右：残差 vs 预测值散点图（检验方差齐性）
    """
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.expm1(np.asarray(y_pred_log))
    residuals = y_true - y_pred
    rel_residuals = residuals / y_true * 100  # 相对残差(%)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左：残差直方图
    ax1.hist(residuals, bins=25, color='#59a14f', alpha=0.75, edgecolor='white')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='零线')
    ax1.axvline(x=residuals.mean(), color='orange', linestyle='-',
                linewidth=1.5, label=f'均值={residuals.mean():.1f}')
    ax1.set_xlabel('残差（元/㎡）', fontsize=11)
    ax1.set_ylabel('频次', fontsize=11)
    ax1.set_title('残差分布直方图', fontsize=12)
    ax1.legend(fontsize=9)

    # 右：残差 vs 预测值（检验方差齐性）
    ax2.scatter(y_pred, rel_residuals, alpha=0.5, color='#e15759',
                edgecolors='white', linewidths=0.3, s=50)
    ax2.axhline(y=0,   color='red',   linestyle='--', linewidth=1.5)
    ax2.axhline(y=10,  color='gray',  linestyle=':', linewidth=1)
    ax2.axhline(y=-10, color='gray',  linestyle=':', linewidth=1)
    ax2.set_xlabel('预测值（元/㎡）', fontsize=11)
    ax2.set_ylabel('相对残差 (%)', fontsize=11)
    ax2.set_title('残差 vs 预测值', fontsize=12)

    plt.suptitle(f'图5-3 {model_name} 残差诊断图', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"[图5-3] 已保存至 {save_path}")
    return fig


# ─────────────────────────────────────────────
# 图5-7: 学习曲线
# ─────────────────────────────────────────────

def plot_learning_curve(model, X_train_proc, y_train, cv_splits,
                         model_name='模型',
                         save_path='outputs/figures/fig5_7_learning_curve.png'):
    """
    图5-7: 学习曲线（过拟合诊断）
    展示训练集和验证集MAE随训练样本量变化的趋势
    训练集-验证集差距大 → 过拟合；验证集持续下降 → 欠拟合
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_proc, y_train,
        cv=cv_splits,
        train_sizes=np.linspace(0.2, 1.0, 8),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
    )

    # neg_MAE → MAE
    train_mae = -train_scores.mean(axis=1)
    val_mae   = -val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std   = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mae, 'o-', color='#4e79a7',
            label='训练集 MAE', linewidth=2, markersize=6)
    ax.plot(train_sizes, val_mae, 's--', color='#e15759',
            label='验证集 MAE', linewidth=2, markersize=6)

    # 置信区间阴影
    ax.fill_between(train_sizes,
                    train_mae - train_std, train_mae + train_std,
                    alpha=0.15, color='#4e79a7')
    ax.fill_between(train_sizes,
                    val_mae - val_std, val_mae + val_std,
                    alpha=0.15, color='#e15759')

    ax.set_xlabel('训练集样本量', fontsize=12)
    ax.set_ylabel('MAE（log空间）', fontsize=12)
    ax.set_title(f'图5-7 {model_name} 学习曲线', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"[图5-7] 已保存至 {save_path}")
    return fig
