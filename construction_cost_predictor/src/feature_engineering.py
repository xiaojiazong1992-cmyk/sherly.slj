# -*- coding: utf-8 -*-
"""
feature_engineering.py
特征工程与指标筛选：Pearson相关性分析 + VIF多重共线性检验
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 全局图表设置（自动检测可用中文字体）
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
# 1. K-S正态性检验（验证log变换必要性）
# ─────────────────────────────────────────────

def ks_normality_test(data, name):
    """
    Kolmogorov-Smirnov正态性检验
    H0: 数据服从正态分布
    p > 0.05 → 不拒绝正态性假设
    p ≤ 0.05 → 拒绝正态性假设，需要变换
    """
    # 标准化后再做K-S检验
    stat, p = kstest(data, 'norm', args=(data.mean(), data.std()))
    conclusion = '不拒绝' if p > 0.05 else '拒绝'
    print(f"[K-S检验] {name}")
    print(f"  统计量 D = {stat:.4f}, p值 = {p:.6f}")
    print(f"  → {conclusion}正态性假设（α=0.05）\n")
    return stat, p


def compare_log_transform(series, col_name='unit_cost',
                           save_path='outputs/figures/fig4_1_distribution.png'):
    """
    对比log变换前后的分布，绘制图4-1
    包含：直方图 + KDE曲线 + K-S检验结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：原始分布
    axes[0].hist(series, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].set_title(f'原始{col_name}分布', fontsize=13)
    axes[0].set_xlabel('造价（元/㎡）', fontsize=11)
    axes[0].set_ylabel('频次', fontsize=11)
    stat0, p0 = ks_normality_test(series.dropna(), '原始造价')
    axes[0].text(0.97, 0.95, f'K-S: p={p0:.4f}',
                 transform=axes[0].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # 右图：log1p变换后分布
    log_series = np.log1p(series)
    axes[1].hist(log_series, bins=30, color='tomato', alpha=0.7, edgecolor='white')
    axes[1].set_title(f'log1p变换后{col_name}分布', fontsize=13)
    axes[1].set_xlabel('log1p(造价)', fontsize=11)
    axes[1].set_ylabel('频次', fontsize=11)
    stat1, p1 = ks_normality_test(log_series.dropna(), 'log1p变换后')
    axes[1].text(0.97, 0.95, f'K-S: p={p1:.4f}',
                 transform=axes[1].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.suptitle('图4-1 单方造价分布对比（log变换前后）', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"[图4-1] 已保存至 {save_path}")
    return fig


# ─────────────────────────────────────────────
# 2. Pearson相关性热力图（图3-1）
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df, numeric_cols, target_col='unit_cost_log',
                              save_path='outputs/figures/fig3_1_correlation.png'):
    """
    绘制Pearson相关性热力图（图3-1）
    展示数值特征之间及与目标变量的相关系数
    """
    cols = [c for c in numeric_cols + [target_col] if c in df.columns]
    corr = df[cols].corr(method='pearson')

    # 中文列名映射（用于图表展示）
    col_name_map = {
        'total_area':          '总建筑面积',
        'above_floors':        '地上层数',
        'under_floors':        '地下层数',
        'seismic_intensity':   '抗震烈度',
        'steel_price_index':   '钢材价格指数',
        'cement_price_index':  '水泥价格指数',
        'floor_density':       '单层平均面积',
        'decoration_level':    '装修标准',
        'is_prefab':           '装配式',
        'is_high_rise':        '是否高层',
        'unit_cost_log':       'log造价',
    }

    display_cols = [col_name_map.get(c, c) for c in cols]
    corr.index   = display_cols
    corr.columns = display_cols

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # 只显示下三角

    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={'size': 8},
    )
    ax.set_title('图3-1 特征Pearson相关性热力图', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"[图3-1] 已保存至 {save_path}")

    # 输出与目标变量相关性排序
    if target_col in df.columns:
        target_corr = df[cols].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        print("\n与目标变量（log造价）的Pearson相关系数排序：")
        for feat, val in target_corr.items():
            print(f"  {col_name_map.get(feat, feat):15s}: {val:.4f}")

    return corr


# ─────────────────────────────────────────────
# 3. VIF多重共线性检验（图3-2）
# ─────────────────────────────────────────────

def check_vif(X_df, threshold=10):
    """
    计算方差膨胀因子（VIF）
    VIF > 10：存在严重多重共线性，需要处理
    VIF 5-10：中等多重共线性，需要关注
    VIF < 5：多重共线性较弱，可接受
    """
    # 去除含NaN的行，确保VIF计算正常
    X_clean = X_df.dropna()

    vif_data = pd.DataFrame({
        'feature': X_clean.columns,
        'VIF': [
            variance_inflation_factor(X_clean.values.astype(float), i)
            for i in range(X_clean.shape[1])
        ],
    }).sort_values('VIF', ascending=False).reset_index(drop=True)

    # 标注严重程度
    vif_data['状态'] = vif_data['VIF'].apply(
        lambda v: '⚠ 严重' if v > 10 else ('△ 中等' if v > 5 else '✓ 正常')
    )

    print("\nVIF多重共线性检验结果：")
    print(vif_data.to_string(index=False))

    severe = vif_data[vif_data['VIF'] > threshold]
    if not severe.empty:
        print(f"\n⚠ VIF>{threshold}的特征（建议处理）：{severe['feature'].tolist()}")
    else:
        print(f"\n✓ 所有特征VIF均低于阈值{threshold}，无严重多重共线性")

    return vif_data


def plot_vif_barh(vif_data, threshold=10,
                  save_path='outputs/figures/fig3_2_vif.png'):
    """
    绘制VIF多重共线性检验水平柱状图（图3-2）
    红色虚线标注阈值=10
    """
    # 中文特征名映射
    col_name_map = {
        'total_area':          '总建筑面积(log)',
        'above_floors':        '地上层数',
        'under_floors':        '地下层数',
        'seismic_intensity':   '抗震烈度',
        'steel_price_index':   '钢材价格指数',
        'cement_price_index':  '水泥价格指数',
        'floor_density':       '单层平均面积',
        'decoration_level':    '装修标准',
        'is_prefab':           '装配式(0/1)',
        'is_high_rise':        '是否高层(0/1)',
    }

    vif_plot = vif_data.copy()
    vif_plot['feature'] = vif_plot['feature'].map(
        lambda x: col_name_map.get(x, x))
    vif_plot = vif_plot.sort_values('VIF', ascending=True)

    colors = ['#d62728' if v > threshold else '#1f77b4'
              for v in vif_plot['VIF']]

    fig, ax = plt.subplots(figsize=(9, max(5, len(vif_plot) * 0.45)))
    bars = ax.barh(vif_plot['feature'], vif_plot['VIF'],
                   color=colors, alpha=0.8, edgecolor='white')

    # 阈值线
    ax.axvline(x=threshold, color='red', linestyle='--',
               linewidth=1.5, label=f'警戒阈值 VIF={threshold}')

    # 标注数值
    for bar, val in zip(bars, vif_plot['VIF']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', ha='left', fontsize=9)

    ax.set_xlabel('方差膨胀因子（VIF）', fontsize=11)
    ax.set_title('图3-2 VIF多重共线性检验', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, max(vif_plot['VIF'].max() * 1.15, threshold * 1.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"[图3-2] 已保存至 {save_path}")
    return fig


# ─────────────────────────────────────────────
# 4. 数据清洗辅助函数
# ─────────────────────────────────────────────

def remove_outliers_iqr(df, col, factor=1.5):
    """
    IQR方法去除异常值
    保留 [Q1 - factor*IQR, Q3 + factor*IQR] 范围内的数据
    """
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    mask = df[col].between(lower, upper)
    n_removed = (~mask).sum()
    print(f"[IQR清洗] {col}: 移除{n_removed}条异常值 "
          f"（范围：[{lower:.1f}, {upper:.1f}]）")
    return df[mask].copy()


def dedup_by_key_fields(df, key_cols=None):
    """
    按关键字段组合去重（四字段联合去重）
    默认key_cols：省份 + 年份 + 面积 + 层数
    """
    if key_cols is None:
        key_cols = ['province', 'completion_year',
                    'total_area', 'above_floors']
    available = [c for c in key_cols if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=available).copy()
    print(f"[去重] 使用字段{available}，去除{before - len(df)}条重复记录")
    return df


def report_missing(df):
    """输出缺失值报告"""
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    report = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例(%)': missing_pct.round(2),
    }).query('缺失数量 > 0').sort_values('缺失比例(%)', ascending=False)

    if report.empty:
        print("✓ 无缺失值")
    else:
        print("缺失值报告：")
        print(report.to_string())
    return report
