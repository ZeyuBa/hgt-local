# 实验约束清单 — Experiment Constraints

HGT-autoresearch 的硬性约束和软性约束。Agent 必须在这些边界内实验。

---

## Hard Constraints (硬约束 — 违反即 crash/discard)

### HC-1: 评估协议不可变
- `compute_link_prediction_metrics()` in `src/training/trainer.py` 是 ground truth。
- 不得修改指标计算逻辑、阈值扫描方式、或 mask 规则。
- **理由:** 指标可比性是 keep/discard 决策的基石。

### HC-2: 数据管线不可变
- `src/dataset/` 下的数据加载、collate、bucket sampler 不得修改。
- `training_data/` 下的拓扑生成和标注代码不得修改。
- 可以通过 `configs/config.yaml` 的 `synthetic` 节调参（num_sites, noise_probability 等）。
- **理由:** 数据一致性。模型变化和数据变化不能混在同一个实验里。

### HC-3: 固定入口和配置加载
- `main.py` 和 `src/training/config.py` 不得修改。
- **理由:** 实验复现性。任何实验都能用同一个 entrypoint 重跑。

### HC-4: 测试集隔离
- Research mode 只看 validation metrics。
- 绝对不得读取、引用、或优化 test set 结果。
- **理由:** 防止过拟合评估集。Test set 留给人类做最终判断。

### HC-5: 不新增依赖
- 不得 `pip install` 或添加新包到 requirements。
- 只能使用项目已有的依赖。
- **理由:** 环境隔离和可复现性。

### HC-6: 图结构约束不可变
- 3 node types (ne, alarm_entity, alarm)、9 relation types 是固定的。
- `model.num_types=3` 和 `model.num_relations=9` 不得更改。
- **理由:** 这是问题定义的一部分，不是超参。

### HC-7: 特征维度一致性
- 如果修改 `feature_extraction.py` 的特征布局，必须同步修改 `model.in_dim`。
- 特征维度不匹配 = 必定 crash。
- **理由:** 模型输入和特征提取必须对齐。

---

## Soft Constraints (软约束 — 需要权衡)

### SC-1: 训练时间预算
- 没有固定时间预算。更长训练（更多 epochs、更大模型）是被鼓励的，只要 metrics 有改善。
- **1 小时硬性安全上限:** 使用 `timeout 3600` 包裹运行命令。超过 1 小时 → kill 并视为 crash。
- 超时通常意味着实验出了问题（死循环、配置错误、模型过大），不是正常训练该花的时间。
- **权衡:** 30 分钟训练但 F1 显著提升？完全可以。45 分钟但只涨了 0.002？考虑减少 epochs 来加速迭代。

### SC-2: 模型复杂度
- 当前模型 ~284K 参数。
- 可以增大到 ~1M，但要确保 val_f1 有对应的显著提升。
- 大模型 + 微小改善 = 不值得。
- **权衡:** 参数量翻倍但 F1 只涨 0.01？discard。参数量翻倍但 F1 涨 0.05？keep。

### SC-3: 简洁性原则 (Simplicity Criterion)
- 借鉴 autoresearch 原则：同等效果下，更简单的代码更好。
- 小改善 + 大量 hacky 代码 = 不值得 keep。
- 删除代码且效果不变 = 绝对 keep。
- 同等 F1 但更简单的实现 = keep。
- **权衡:** 0.005 F1 improvement + 50 行新代码？Questionable。0.005 F1 improvement + 删掉 20 行？Definitely keep。

### SC-4: 数据规模
- 基准数据规模：8000 train / 500 val / 1000 test graphs。这是锁定的基准规模。
- Research mode 下应设 `reuse_existing_splits: true`，确保数据不重新生成。
- 不建议在实验中调整 split_sizes —— 数据规模变化会让 metrics 不可比。
- 如果需要测试数据分布变化（如 num_sites, noise_probability），需要先重新生成数据并重建 baseline。

### SC-5: 单变量原则
- 每次实验尽量只改一个东西。
- 如果改了两个变量且 F1 提升，很难知道是哪个起作用的。
- **权衡:** 如果两个变化逻辑上紧密关联（例如同时增大 n_hid 和减少 n_heads 来保持 head_dim），可以一起改。

### SC-6: reuse_existing_splits
- Research mode 下应设置 `synthetic.reuse_existing_splits: true`。
- 这确保每次实验用同一份数据，变化只来自模型/训练。
- 如果需要测试数据分布变化的影响，需先 discard 旧 baseline 并重建。

---

## 约束优先级

```
HC (Hard)  >  Primary Metric (val_f1_cal)  >  SC (Soft)
```

Hard constraint 违反 → 无条件 crash/revert。
Soft constraint 违反 → 需要足够的 metric 改善来证明合理性。

---

## Checklist for each experiment

在每次实验前，Agent 应自检：

- [ ] 我修改的文件在允许范围内？
- [ ] 我没有动评估代码？
- [ ] 我没有引入新依赖？
- [ ] 特征维度和 model.in_dim 一致？
- [ ] num_types=3, num_relations=9 没变？
- [ ] 我用的是 research run mode？
- [ ] 我只看 validation metrics，不看 test？
- [ ] 这次只改了一个变量（或逻辑相关的一组）？
- [ ] 预计训练时间在 1 小时以内（使用 timeout 3600）？
