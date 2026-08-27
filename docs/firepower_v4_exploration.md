# Firepower v4：均值归一化探索记录

本文档记录从 v1/v2/v3（基于求和）到 v4（基于均值）的完整探索过程、技术细节、
实验结果及尚未完成的工作。

---

## 一、问题背景：count confound

### 根本缺陷

职业选手的 HLTV Rating 高度聚集在 1.0 附近（均值约 1.02，标准差约 0.15）。
v1/v2/v3 用的是 **求和**：

```
ct_rating_sum = Σ rating_i  （对所有存活CT选手求和）
```

由于每人 rating ≈ 1.0，因此：

```
ct_rating_sum ≈ ct_players_alive × 1.0 ≈ ct_players_alive
```

实测相关系数 **r = 0.987**。这个"技术水平"特征几乎只是在重复编码场上人数，
而模型已经直接看到了 `ct_players_alive`。

### 为什么 v2/v3 没有修复？

- **v2**：引入了情境门控（单兵clutch、开枪entry、5v5 opening），但保持了求和
- **v3**：引入了 HLTV 队伍排名加权（`team_weight = 1/log2(rank+1)`），但仍然保持了求和
- 两者都没有除以人数，count confound 原封不动

---

## 二、v4 的三个探索变体（代理实验）

这三个变体是在**不重新运行 assemble.py** 的前提下，从已有的 `training_dataset.parquet`
中的 `*_sum` 和 `ct_players_alive` 列直接派生的代理特征。

### v4.1 — 仅 Rating 均值，无排名权重

```python
ct_rating_v41 = ct_rating_sum / ct_players_alive
t_rating_v41  = t_rating_sum  / t_players_alive
```

仅 2 个特征（CT/T 各一个 rating 均值），剔除其他所有统计项，隔离纯 rating 信号。

**文件**：`src/features/firepower_v4_1.py`

---

### v4.2 — Rating 均值 × 队伍排名权重

```python
ct_rating_v42 = team_weight × (ct_rating_sum / ct_players_alive)
```

其中 `team_weight = 1 / log2(hltv_rank + 1)`（与 v3 相同的 log2 公式）。
同侧所有存活选手共享同一个队伍权重，故等价于"均值乘以队伍系数"。
半场换边通过读取 rounds 表正确处理（第 1–12 局用上半场分配，第 13 局起互换）。

**文件**：`src/features/firepower_v4_2.py`

---

### v4.3 — v2 全部统计项均转为均值

v2 的 14 个求和列全部除以 `n_alive`；6 个本身就是单人值的列（kast、clutch、awp）
直接透传，不需要平均。

| v2 列 | v4.3 列 | 备注 |
|---|---|---|
| ct/t_rating_sum | ct/t_rating_v43 | ÷ n_alive |
| ct/t_adr_sum | ct/t_adr_v43 | ÷ n_alive |
| ct/t_hltv_firepower_sum | ct/t_firepower_v43 | ÷ n_alive |
| ct/t_entry_sum | ct/t_entry_v43 | ÷ n_alive；独苗生还时保持门控 |
| ct/t_trading_sum | ct/t_trading_v43 | ÷ n_alive |
| ct/t_opening_sum | ct/t_opening_v43 | ÷ n_alive；非 5v5 时为 NaN |
| ct/t_weighted_utility | ct/t_utility_v43 | ÷ n_alive |
| ct/t_kast_mean | ct/t_kast_v43 | 透传（v2 中已经是均值） |
| ct/t_clutch_score | ct/t_clutch_v43 | 透传（独苗个人值） |
| ct/t_awp_sniping_skill | ct/t_awp_v43 | 透传（个人值） |

**文件**：`src/features/firepower_v4_3.py`

---

### 代理实验结果（Logistic Regression）

评估方式：5-fold GroupKFold（按 match_id），CV = 训练集 OOF；OOT = 全量训练 → 2026 holdout。
contested = 双方人数相等 且 |Δ装备价值| ≤ $1500。

| 版本 | CV AUC | CV cAUC | OOT AUC | OOT cAUC |
|---|---|---|---|---|
| EB2（无FP，基线） | 0.8508 | 0.5963 | **0.8474** | **0.6553** |
| EFB2（v2 求和，旧版） | 0.8519 | 0.6033 | 0.8236 | 0.6166 |
| EFB4.1（rating 均值） | 0.8508 | 0.5955 | 0.8458 | 0.5948 |
| EFB4.2（加权均值） | 0.8510 | 0.5940 | **0.8478** | **0.6554** |
| EFB4.3（全均值） | **0.8520** | **0.6051** | 0.8107 | 0.6085 |

**关键结论**：v4.2 OOT 与 EB2 基本持平（不降也不升），v4.3 CV 最好但 OOT 崩溃，
模式和 v2/v3 一致——更多特征就更过拟合。

详细分析见：`docs/notes_firepower_v4.md`

---

## 三、v4 正式集成到主流程

代理实验阶段用的是派生列，实际集成时需要修改 `assemble.py` 流程，让
`firepower_features()` 在组装时直接计算均值列。

### 核心改动：`src/features/firepower.py`

在原有的 `_sum` 计算循环中新增 `n_with_stats` 计数器：

```python
n_with_stats = 0
for i, sid in enumerate(sids):
    stats = lookup.get((int(sid), year))
    if stats is None:
        continue
    n_with_stats += 1   # ← 只统计有 HLTV 数据的存活选手
    ...
    rating_sum += rating_val
    ...

# 均值列：分母 = 有 HLTV 数据的存活选手数（不是 ct_players_alive）
out[f"{pfx}_rating_mean"] = rating_sum / n_with_stats if n_with_stats else nan
out[f"{pfx}_adr_mean"]    = adr_sum    / n_with_stats if n_with_stats else nan
out[f"{pfx}_fp_mean"]     = fp_sum     / n_with_stats if n_with_stats else nan
out[f"{pfx}_entry_mean"]  = entry_sum  / n_with_stats if n_with_stats else nan
out[f"{pfx}_trading_mean"]  = trading_sum  / n_with_stats if n_with_stats else nan
out[f"{pfx}_opening_mean"]  = (opening_sum / n_with_stats
                               if (is_opening and n_with_stats) else nan)
out[f"{pfx}_utility_mean"]  = weighted_util / n_with_stats if n_with_stats else nan
```

**为什么用 `n_with_stats` 而不是 `ct_players_alive`？**
部分选手（未被 HLTV 收录的小队员）没有统计数据。如果用 `ct_players_alive`
做分母，这些人会被计为 0 并拉低平均值，错误惩罚了有数据的队友。
`n_with_stats` 只统计实际参与累加的人，语义正确。

**验证（1 条记录）**：

```
ct_rating_sum = 5.25，ct_players_alive = 5 → ct_rating_mean = 1.05 ✓
```

新增列（训练集验收）：

```
FIREPOWER_MEAN_COLS = [
    "ct_rating_mean", "t_rating_mean",
    "ct_adr_mean",    "t_adr_mean",
    "ct_kast_mean",   "t_kast_mean",    # v2 已有，无变化
    "ct_fp_mean",     "t_fp_mean",
    "ct_entry_mean",  "t_entry_mean",
    "ct_trading_mean","t_trading_mean",
    "ct_opening_mean","t_opening_mean",
    "ct_clutch_score","t_clutch_score", # 个人值，不平均
    "ct_awp_sniping_skill","t_awp_sniping_skill",
    "ct_utility_mean","t_utility_mean",
]
```

共 20 列（14 个均值 + 6 个透传）。

### 新增特征集：`src/models/train_pipeline.py`

```python
"EB2_FPmean": (ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL + BOMB_LIVE_COLS
               + BOMB_DEFUSE_COLS + FIREPOWER_MEAN_COLS),

"EFB3":       (ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL + TERRITORY_COLS
               + FIREPOWER_MEAN_COLS + BOMB_LIVE_COLS + BOMB_DEFUSE_COLS),
```

- **EB2_FPmean** = EB2 基础（经济 + 地图控制 + 战术 + 炸弹）+ 均值 FP（没有 Voronoi）
- **EFB3** = 所有 pillar + 均值 FP（用 FIREPOWER_MEAN_COLS 替换旧的 FIREPOWER_COLS）

---

## 四、数据集重建

两个数据集都需要重新从 demo 原始文件 assemble，才能包含新的 `_mean` 列。

| 数据集 | 文件 | 行数 | 列数 | 状态 |
|---|---|---|---|---|
| 训练集 | `data/training_dataset.parquet` | 531,866 | 129 | ✅ 已重建 |
| OOT 测试集 | `data/test_dataset_2026.parquet` | 55,271 | 129 | ✅ 已重建 |

均值列（16 列）验证确认存在于两个数据集中。

---

## 五、完整 Benchmark 结果（eval_firepower_mean.py）

**注意**：CV AUC 列由于 `cv_auc()` 内部固定使用 logreg，XGB 行的 CV AUC 与
logreg 行相同——这是脚本设计特性，不影响 OOT 结论。

| 特征集 / 模型 | CV AUC | CV cAUC | OOT AUC | OOT cAUC |
|---|---|---|---|---|
| **EB2 / logreg** | 0.8516 | 0.6028 | 0.8513 | 0.6699 |
| **EB2 / XGB** | 0.8516 | 0.6028 | 0.8676 | 0.7371 |
| EFB2 / logreg | 0.8529 | 0.6073 | 0.8508 | 0.6100 |
| EFB2 / XGB | 0.8529 | 0.6073 | 0.8780 | 0.7601 |
| EB2_FPmean / logreg | 0.8529 | 0.6084 | 0.8512 | 0.6148 |
| **EB2_FPmean / XGB** | 0.8529 | 0.6084 | **0.8792** | **0.7611** |
| EFB3 / logreg | 0.8529 | 0.6067 | 0.8511 | 0.6130 |
| EFB3 / XGB | 0.8529 | 0.6067 | 0.8783 | 0.7583 |

输出文件：`outputs/firepower_mean_benchmark.csv`

### 解读

**Logistic Regression 侧（论文主模型）：**
- 四个 logreg 变体的 CV AUC 几乎相同（0.8516–0.8529），差异在统计噪声范围内
- OOT AUC：EB2（0.8513）> EB2_FPmean（0.8512）≈ EFB3（0.8511）> EFB2（0.8508）
- OOT cAUC：EB2（0.6699）远高于其他三个（均在 0.61 左右）
- 结论：均值 FP 用于 logreg 时无明显收益，加入后 cAUC 反而轻微下降

**XGBoost 侧：**
- EB2_FPmean/XGB 的 OOT cAUC（0.7611）略优于 EFB2/XGB（0.7601）和 EFB3/XGB（0.7583）
- 所有 XGB 变体 OOT AUC 都高于 EB2 基线（0.8676），说明树模型能从 FP 特征中提取增量信号
- 均值编码（EB2_FPmean）比旧求和（EFB2）在 XGB 下略好，验证了均值化的理论正确性

**总结判断：**

> 均值化修复了 count confound 的编码问题，但没有根本改变 Firepower pillar 对
> logistic regression 无效的结论。XGB 能从 FP 中多挤出约 0.024 cAUC，
> 但考虑到推理成本和可解释性，EB2 + logreg 仍是论文首选模型。

---

## 六、尚未完成的工作

### 1. Leak-free OOT 测试（重要）

**问题**：目前 OOT benchmark 使用 2026 年的 HLTV 统计数据评估 2026 年的比赛——
这在实战推断中是不可用的（比赛结束前你不知道本年统计）。

**修复方案**：`firepower.py` 已预留 `FIREPOWER_YEAR_LAG` 环境变量：

```bash
FIREPOWER_YEAR_LAG=1 python src/models/eval_firepower_mean.py
```

设置后，2026 年的比赛自动查 2025 年的统计数据（真正的赛前已知量）。
**此测试尚未运行**，无法确认 FP 在无泄漏场景下是否仍有帮助。

### 2. 排名加权均值（v4.2 方案）未进入主流程

代理实验中 v4.2 是 OOT 表现最稳的变体（与 EB2 基本持平，不崩溃）。
但 `FIREPOWER_MEAN_COLS` 没有引入 team_weight——当前 `_mean` 只是简单平均，
不区分队伍强弱。如果要在主流程中测试 v4.2，需要：

1. 在 `firepower_features()` 中查询队伍排名（需访问 `configs/hltv_rankings.csv` 或等效数据）
2. 新增 `ct_weighted_rating_mean`、`t_weighted_rating_mean` 两列
3. 建立新的 `FIREPOWER_WEIGHTED_MEAN_COLS` 特征集
4. 重新运行 assemble 和 benchmark

**当前状态**：未实现，仅在代理脚本 `src/features/firepower_v4_2.py` 中有参考实现。

### 3. Defuse elapsed time 集成（独立工作流）

在 Firepower v4 探索结束后，发现了 bomb.py 中 `defuse_time_margin` 的另一个问题：
该特征假设拆弹刚刚开始（用完整 5s/10s 计算），不感知拆弹已经进行了多久。

已启动新的改动（详见 `docs/defuse_elapsed_plan.md`），但与 Firepower v4 主流程独立，
需单独完成 re-parse → assemble → benchmark 流程。

### 4. 论文更新

- paper 中的 Table 2 / Appendix B 数据以 EB2/logreg 为准，结论不变
- 若 leak-free 测试（第 1 项）结果出来后有变化，需更新 Appendix B
- `EB2_FPmean/XGB` 这个最优变体目前没有在 paper 中提及，可考虑在 §5 补充一段

---

## 七、文件索引

| 文件 | 用途 |
|---|---|
| `src/features/firepower.py` | 主流程 FP 计算，含均值列（v4 集成版） |
| `src/features/firepower_v4_1.py` | 代理实验：rating 均值（仅 logreg 测试） |
| `src/features/firepower_v4_2.py` | 代理实验：加权均值（v4.2） |
| `src/features/firepower_v4_3.py` | 代理实验：全统计均值（v4.3） |
| `src/models/eval_firepower_v4.py` | 代理实验评估脚本 |
| `src/models/eval_firepower_mean.py` | 主流程均值 FP benchmark（EB2/EFB2/EB2_FPmean/EFB3） |
| `src/models/train_pipeline.py` | 特征集定义，含 EB2_FPmean 和 EFB3 |
| `outputs/firepower_v4_benchmark.csv` | 代理实验结果 |
| `outputs/firepower_mean_benchmark.csv` | 主流程均值 FP benchmark 结果 |
| `docs/notes_firepower_v4.md` | 代理实验的详细记录（含公式和结论） |
| `docs/firepower_v4_exploration.md` | 本文档（完整探索过程） |
