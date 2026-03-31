# 住宅建安工程成本预测模型 — 需求与优化说明文档

## 1. 项目背景

本项目基于 206 条住宅建安工程历史数据，采用集成学习方法（随机森林、XGBoost、Stacking）
预测单方建安造价（元/m²），为工程前期成本估算提供数据支持。

**目标变量**：单方造价（元/m²），范围约 1000–5000 元
**特征数量**：18 个原始特征 → One-Hot 展开后 36 个
**数据量**：206 条样本（训练集 164 条，测试集 42 条）

---

## 2. 原始方案问题分析

### 2.1 XGBoost 严重过拟合（最关键问题）

| 指标 | 训练集 | 测试集 | 差值 |
|------|--------|--------|------|
| R²   | 0.9998 | 0.8955 | **0.104** |
| RMSE | 12.5   | 258.4  | —    |

**根因**：`GridSearchCV` 仅调节 4 个参数（`n_estimators`、`learning_rate`、
`max_depth`、`reg_lambda`），缺少以下关键正则化参数：

| 缺失参数 | 作用 |
|---------|------|
| `subsample` | 每棵树随机采样行比例，防止记忆训练集 |
| `colsample_bytree` | 每棵树随机采样列比例 |
| `min_child_weight` | 叶节点最小样本权重，防止过深分裂 |
| `gamma` | 节点分裂所需最小增益 |
| `reg_alpha` | L1 正则化系数 |

### 2.2 评估结果不可靠（数据集小 + 无分层）

| 模型 | CV R²（5 折） | 测试集 R² | 差值 |
|------|--------------|-----------|------|
| RF      | 0.7724 | 0.9164 | **0.144** |
| XGBoost | 0.8123 | 0.8955 | **0.073** |

CV R² 与测试集 R² 差距过大，说明 `random_state=42` 碰巧产生了"容易"的测试集，
结果虚高，不具代表性。

**根本原因**：
- 仅 42 条测试样本，单个 ±500 元/m² 异常值可影响 R² 约 ±0.02
- `train_test_split` 无分层（`stratify=None`），各价格区间分布不保证均衡
- 5 折交叉验证重复次数不足，方差过大

### 2.3 Stacking 劣于单模型

```
Stacking R²=0.8767  <  XGBoost R²=0.8955  <  RF R²=0.9164
```

集成反而降低精度，原因：
- RF 与 XGBoost 均为树模型，OOF 预测高度相关，多样性不足
- 缺少线性基学习器作为互补
- Ridge 元学习器无法从两个相关信号中提取额外增益

### 2.4 数据泄漏风险

`LabelEncoder` 在 `train_test_split` 前对全量数据执行 `fit`，导致测试集的类别
信息泄露到编码过程。此处工程地址仅 4 类，实际影响微小，但不符合规范，应修正。

---

## 3. 优化需求

### [F1] 分层随机划分

**需求**：按目标变量的 5 分位数分箱，传入 `stratify=y_bins` 参数，保证训练集与
测试集价格区间分布一致。

```python
y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)
```

**预期效果**：消除因随机种子导致的测试集"容易/困难"偏差，测试集各价格段样本均衡。

---

### [F2] 以 RepeatedKFold 为主评估指标

**需求**：使用 `RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)` 进行
50 次交叉验证，报告 `均值 ± 标准差`，作为模型泛化能力的主要度量。

```python
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train_log, cv=rkf,
                             scoring='r2', n_jobs=-1)
print(f"RKF CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

**验收标准**：CV R² 与测试集 R² 差值 < 0.05。

---

### [F3] 修复 XGBoost 过拟合

**需求**：扩展超参数搜索空间，改用 `RandomizedSearchCV(n_iter=50)`：

| 参数 | 候选值 | 说明 |
|------|--------|------|
| `n_estimators` | 200, 300, 500 | 树的数量 |
| `learning_rate` | 0.01, 0.05, 0.1 | 学习率 |
| `max_depth` | 3, 5, 7 | 最大深度（降低上限） |
| `subsample` | 0.6, 0.8, 1.0 | **新增**：行采样 |
| `colsample_bytree` | 0.6, 0.8, 1.0 | **新增**：列采样 |
| `min_child_weight` | 1, 3, 5 | **新增**：叶节点最小权重 |
| `gamma` | 0, 0.1, 0.3 | **新增**：分裂最小增益 |
| `reg_lambda` | 0.1, 1.0, 5.0 | L2 正则（扩展范围） |
| `reg_alpha` | 0, 0.1, 1.0 | **新增**：L1 正则 |

**验收标准**：XGBoost 训练集与测试集 R² 差值 < 0.05（原始为 0.104）。

---

### [F4] 优化随机森林超参数搜索

**需求**：在原有参数基础上增加：

| 新增参数 | 候选值 | 作用 |
|---------|--------|------|
| `min_samples_leaf` | 1, 2, 4 | 防止树过深，增强泛化 |
| `max_features` | 'sqrt', 'log2', 0.5 | 特征子采样，降低相关性 |

改用 `RandomizedSearchCV(n_iter=30)` 提升搜索效率。

---

### [F5] 改进 Stacking 多样性

**需求**：增加 `Ridge(alpha=1.0)` 作为第三基学习器，使集成中同时存在线性模型与
两个树模型，形成互补。

```python
stacking = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),   # 线性模型，与树模型互补
        ('rf',    rf_best),
        ('xgb',   xgb_best)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=10
)
```

**验收标准**：Stacking 测试集 R² ≥ 最优单模型 R²。

---

### [F6] 消除 LabelEncoder 数据泄漏

**需求**：`LabelEncoder` 仅在 `X_train` 上执行 `fit`，在 `X_test` 上仅执行
`transform`。测试集中出现的未见类别映射为 -1。

```python
le = LabelEncoder()
X_train_df['工程地址'] = le.fit_transform(X_train_df['工程地址'].astype(str))
X_test_df['工程地址']  = X_test_df['工程地址'].astype(str).map(
    dict(zip(le.classes_, le.transform(le.classes_)))
).fillna(-1).astype(int)
```

---

### [F7] 目标变量对数变换

**需求**：对目标 `y` 执行 `log1p` 变换后训练，预测时执行 `expm1` 逆变换。
所有评估指标均在原始元/m² 空间计算，保持与原始结果可比性。

```python
y_train_log = np.log1p(y_train)          # 训练
y_pred = np.expm1(model.predict(X_test)) # 预测逆变换
```

**目的**：降低高造价异常值对损失函数的影响，改善残差正态性，有助于树模型的分裂决策。

---

### [F8] 过拟合诊断图表

**需求**：新增以下诊断输出：

1. **过拟合诊断表**：包含 Train R²、Test R²、Gap、RKF CV 均值 ± 标准差
2. **学习曲线图**（`fig5_learning_curves.png`）：展示 RF 和 XGBoost 训练样本数
   与 Train/Val R² 的关系，直观判断偏差/方差问题

---

## 4. 输出物清单

| 文件 | 说明 |
|------|------|
| `cost_prediction_optimized.py` | 完整优化脚本（含 [F1]–[F8] 全部修复） |
| `fig1_scatter.png` | 预测值 vs 真实值散点图（3 模型） |
| `fig2_bars.png` | R²/RMSE/MAE/MAPE 对比柱状图 |
| `fig3_importance.png` | RF 特征重要性横向柱状图 |
| `fig4_residuals.png` | 残差分布直方图（3 模型） |
| `fig5_learning_curves.png` | RF & XGBoost 学习曲线（新增） |
| `predictions.csv` | 测试集逐条预测详情及误差率 |

---

## 5. 验收标准

| 验收项 | 指标 | 目标 | 原始值 |
|-------|------|------|--------|
| XGBoost 过拟合修复 | Train R² − Test R² | **< 0.05** | 0.104 |
| CV 评估可靠性 | RKF CV R² Std | **< 0.05** | — |
| CV 与测试对齐 | \|CV R² − Test R²\| | **< 0.05** | 0.144 (RF) |
| Stacking 有效性 | Stacking R² vs 最优单模型 | **≥ 0 提升** | −0.039 (退步) |
| 代码可运行 | 无异常退出 | **Pass** | Pass |
| 输出完整 | 5 张图 + 1 个 CSV | **全部生成** | 4 张图 |

---

## 6. 依赖环境

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.3
xgboost>=1.7
matplotlib>=3.6
openpyxl>=3.0
```

---

## 7. 修复对照表

| 标签 | 问题 | 解决方案 | 关键代码位置 |
|------|------|----------|-------------|
| F1 | 无分层随机划分 | `stratify=y_bins`（5 分位数箱） | `train_test_split` |
| F2 | 评估不可靠 | `RepeatedKFold(10, 5)` 作主评估 | `cross_val_score` |
| F3 | XGBoost 过拟合 | 扩展正则化参数搜索 + `RandomizedSearchCV(50)` | `xgb_param_dist` |
| F4 | RF 泛化弱 | `min_samples_leaf` + `max_features` + `RandomizedSearchCV(30)` | `rf_param_dist` |
| F5 | Stacking 退步 | 增加 Ridge 基学习器 | `StackingRegressor` |
| F6 | 数据泄漏 | LabelEncoder 仅 fit 训练集 | `le.fit_transform(X_train)` |
| F7 | 目标分布偏斜 | `log1p` 变换训练，`expm1` 逆变换评估 | `y_train_log` |
| F8 | 缺少诊断工具 | 过拟合诊断表 + 学习曲线图 | `diag` DataFrame + `fig5` |
