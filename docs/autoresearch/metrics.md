# Keep/Discard 指标定义 — Metric Decision Logic

HGT-autoresearch 的实验评判标准。

---

## Primary Metric: `val_f1_calibrated`

**定义:** Validation set 上的 F1 score，使用 calibrated threshold（在 101 个等间距阈值上扫描得到的最优 F1 对应的阈值）。

**来源:** `outputs/results/validation_metrics.json` → `f1_calibrated`

**为什么选这个:**
- F1 平衡了 precision 和 recall，适合类别不平衡的告警预测任务。
- Calibrated threshold 避免了硬编码 0.5 阈值的偏差。
- 这是一个 edge-level metric，直接衡量每个 alarm entity 的预测准确性。

---

## Secondary Metrics (tie-breaking & monitoring)

| 指标 | Key in JSON | 方向 | 用途 |
|------|------------|------|------|
| `val_graph_accuracy_calibrated` | `graph_accuracy_calibrated` | higher = better | 整图准确率，运维场景最关心的 |
| `val_auc` | `auc` | higher = better | 排序质量，threshold-independent |
| `val_ap` | `ap` | higher = better | 类不平衡下的排序质量 |
| `val_loss` | eval_loss (from trainer) | lower = better | 训练信号质量 |
| `val_graph_perfect_or_one_fp_calibrated` | `graph_perfect_or_one_fp_calibrated` | higher = better | 宽松整图准确率 (允许≤1 FP) |

---

## Decision Matrix

### Step 1: Compare `val_f1_calibrated` with current best

| Condition | Decision |
|-----------|----------|
| f1_new > f1_best + 0.005 | **KEEP** — 显著改善 |
| f1_new > f1_best | **KEEP** — 改善（检查 Step 2） |
| f1_new == f1_best (±0.001) | Go to **Step 2** (tie-break) |
| f1_new < f1_best | **DISCARD** |

### Step 2: Tie-breaking (when F1 delta < 0.001)

| Condition | Decision |
|-----------|----------|
| graph_accuracy_calibrated improved | **KEEP** |
| auc improved AND simpler code | **KEEP** |
| Code is strictly simpler (fewer lines, removed hacks) | **KEEP** |
| graph_accuracy_calibrated equal or worse AND code not simpler | **DISCARD** |

### Step 3: Complexity override

即使 F1 改善了，如果：
- 改善 < 0.005 **AND** 增加了 > 30 行 non-trivial 代码 → 倾向 **DISCARD**
- 改善 < 0.005 **AND** 删除了代码 → 倾向 **KEEP**

这是 autoresearch 的 simplicity criterion 的 HGT 版本。

---

## Metric Extraction

### 从 validation_metrics.json 提取

```python
import json

with open("outputs/results/validation_metrics.json") as f:
    m = json.load(f)

# Primary
f1_calibrated = m["f1_calibrated"]

# Secondary
graph_accuracy = m["graph_accuracy_calibrated"]
auc = m["auc"]
ap = m["ap"]
graph_perfect_1fp = m["graph_perfect_or_one_fp_calibrated"]

# Edge-level detail
precision = m["precision_calibrated"]
recall = m["recall_calibrated"]
best_threshold = m["best_threshold"]
```

### 从 run.log 提取（如果 JSON 不存在）

```bash
# 如果 research mode 正常完成，会写 validation_metrics.json
# 如果只有 log，检查是否有 crash
tail -n 50 run.log
```

---

## Crash Handling

| Situation | Action |
|-----------|--------|
| OOM | 减小 model size 或 batch size，revert |
| 维度不匹配 | 修复 in_dim/n_hid/n_heads 关系 |
| NaN loss | 降低 learning rate，revert |
| Config 校验失败 | 修复 YAML 格式/类型 |
| 超时 (>1 hour) | kill, 减小 epochs 或 model，检查是否有死循环 |

所有 crash 记录 status=`crash`，val_f1_cal=0.0000。

---

## Baseline Expectations

基于当前默认配置（n_hid=64, num_layers=4, n_heads=4, lr=0.001, epochs=8, 8000 train / 500 val graphs），预期的 baseline 范围：

| Metric | Reasonable Range |
|--------|-----------------|
| val_f1_calibrated | 0.70 – 0.90 |
| val_graph_accuracy_calibrated | 0.50 – 0.80 |
| val_auc | 0.85 – 0.95 |
| val_loss | 0.10 – 0.50 |

第一次运行建立实际 baseline 后，这些范围会被具体数值替代。

---

## Long-term Progress Tracking

随着实验推进，Agent 应该关注：

1. **收益递减曲线:** 如果连续 5 次实验都 discard，考虑更大的架构变化而不是微调。
2. **Metric 相关性:** 如果 F1 涨但 graph_accuracy 跌，说明模型在 easy edges 上变好了但 hard graphs 没改善。需要关注。
3. **Overfitting 信号:** 如果 train loss 持续下降但 val_f1 停滞或下降，加 dropout / weight decay / 减小模型。
4. **Data distribution 影响:** 如果改了 synthetic config 参数，baseline 失效。需要重新建立。
