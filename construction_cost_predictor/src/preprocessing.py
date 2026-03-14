# -*- coding: utf-8 -*-
"""
preprocessing.py
数据预处理模块：Pipeline封装，严格防止数据泄露
核心原则：所有fit操作只在训练集上执行，测试集只做transform
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# ─────────────────────────────────────────────
# 特征分组定义（全局，供其他模块导入）
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    'total_area',          # A1: 总建筑面积（已做log1p变换）
    'above_floors',        # A2: 地上层数
    'under_floors',        # A3: 地下层数
    'seismic_intensity',   # B3: 抗震设防烈度（6/7/8）
    'steel_price_index',   # E2: 钢材价格指数
    'cement_price_index',  # E3: 水泥价格指数
    'floor_density',       # F1: 单层平均面积（衍生）
]

CATEGORICAL_FEATURES = [
    'structure_type',    # B1: 结构类型（框架/剪力墙/框剪/砌体）
    'foundation_type',   # B2: 基础形式
    'housing_type',      # C2: 住宅类型
    'province',          # D1: 省份/城市
]

ORDINAL_FEATURES = [
    'decoration_level',  # C1: 装修标准（0毛坯/1简装/2精装/3豪装）
]

BINARY_FEATURES = [
    'is_prefab',       # B4: 是否装配式
    'is_high_rise',    # F2: 是否高层（above_floors>=12）
]

TARGET = 'unit_cost_log'


# ─────────────────────────────────────────────
# Pipeline构建函数
# ─────────────────────────────────────────────

def build_preprocessor(numeric_features=None,
                        categorical_features=None,
                        ordinal_features=None,
                        binary_features=None):
    """
    构建ColumnTransformer预处理Pipeline
    - 数值特征：KNN填补缺失值 → 标准化
    - 分类特征：众数填补 → 独热编码（drop='first'减少共线性）
    - 有序特征：直接OrdinalEncoder编码
    - 二元特征：直接传递（passthrough）

    注意：此预处理器只能在训练集上fit，测试集只能transform！
    """
    if numeric_features     is None: numeric_features     = NUMERIC_FEATURES
    if categorical_features is None: categorical_features = CATEGORICAL_FEATURES
    if ordinal_features     is None: ordinal_features     = ORDINAL_FEATURES
    if binary_features      is None: binary_features      = BINARY_FEATURES

    # 数值特征：KNN填补（比均值填补更合理）+ 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler',  StandardScaler()),
    ])

    # 分类特征：众数填补 + 独热编码（drop='first'避免虚拟变量陷阱）
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first',
                                  handle_unknown='ignore',
                                  sparse_output=False)),
    ])

    # 有序特征：直接编码（decoration_level已为0/1/2/3数值）
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer,     numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer,     ordinal_features),
            ('bin', 'passthrough',           binary_features),
        ],
        remainder='drop',   # 丢弃未指定的列
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names(preprocessor, categorical_features=None):
    """
    提取ColumnTransformer处理后的特征名列表
    用于SHAP分析时标注特征名称
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == 'remainder' or transformer == 'drop':
            continue
        if transformer == 'passthrough':
            feature_names.extend(cols)
        elif hasattr(transformer, 'steps'):
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, 'get_feature_names_out'):
                try:
                    names = last_step.get_feature_names_out(cols)
                    feature_names.extend(names)
                except Exception:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)

    return feature_names


# ─────────────────────────────────────────────
# 特征工程辅助函数（在建模前调用）
# ─────────────────────────────────────────────

def add_derived_features(df):
    """
    添加衍生特征（F1、F2），在Pipeline外部、划分前调用
    F1: floor_density = total_area / above_floors（单层平均面积）
    F2: is_high_rise  = 1 if above_floors >= 12 else 0
    """
    df = df.copy()
    # 防止除零
    df['floor_density'] = np.where(
        df['above_floors'] > 0,
        df['total_area'] / df['above_floors'],
        df['total_area'],
    )
    df['is_high_rise'] = (df['above_floors'] >= 12).astype(int)
    return df


def log_transform_target(df, target_col='unit_cost'):
    """
    对目标变量做log1p变换，并保留原始列供还原
    返回带 unit_cost_log 列的 DataFrame
    """
    df = df.copy()
    df['unit_cost_log'] = np.log1p(df[target_col])
    return df


def log_transform_area(df, col='total_area'):
    """
    对总建筑面积做log1p变换（右偏分布处理）
    """
    df = df.copy()
    df[col] = np.log1p(df[col])
    return df


# ─────────────────────────────────────────────
# 时序划分函数
# ─────────────────────────────────────────────

def time_based_split(df, test_year=2025, year_col='completion_year'):
    """
    严格按时间划分训练集/测试集
    训练集：test_year之前的所有年份
    测试集：test_year当年数据
    防止"用未来预测过去"的数据泄露
    """
    train_df = df[df[year_col] <  test_year].copy()
    test_df  = df[df[year_col] == test_year].copy()

    print(f"训练集: {len(train_df)} 条 "
          f"（{df[year_col].min()}-{test_year-1}年）")
    print(f"测试集: {len(test_df)} 条 （{test_year}年）")

    if len(test_df) == 0:
        print(f"警告：{test_year}年无数据，将使用最后一年作为测试集")
        last_year = df[year_col].max()
        train_df  = df[df[year_col] <  last_year].copy()
        test_df   = df[df[year_col] == last_year].copy()
        print(f"调整后 - 训练集:{len(train_df)}, 测试集:{len(test_df)}")

    return train_df, test_df


def year_expanding_cv(df, year_col='completion_year'):
    """
    Expanding Window交叉验证生成器（尊重时间顺序）
    Fold 1: train=[year0],         val=[year1]
    Fold 2: train=[year0, year1],  val=[year2]
    ...
    适用于时间序列数据，防止未来信息泄露
    """
    years = sorted(df[year_col].unique())
    if len(years) < 2:
        raise ValueError(f"至少需要2个不同年份，当前只有: {years}")

    for i in range(1, len(years)):
        train_idx = df[df[year_col].isin(years[:i])].index
        val_idx   = df[df[year_col] == years[i]].index
        print(f"  CV Fold {i}: 训练={list(years[:i])}, 验证=[{years[i]}] "
              f"| 训练{len(train_idx)}条, 验证{len(val_idx)}条")
        yield list(train_idx), list(val_idx)
