---
name: autoresearch-hgt
description: Run autonomous HGT alarm-prediction research loops with strict keep/discard decisions. Use when users ask to run, continue, or optimize iterative experiments on this repo (e.g. "启动 autoresearch", "跑一轮实验", "做超参搜索", "按验证集指标保留/回滚"). Not for generic model training outside this repository.
---

# Purpose
Execute reliable, repeatable research loops for this repository's HGT link-prediction pipeline.

# When to use
Use this skill when the user wants:
- autonomous experiment iteration in this repo;
- metric-driven keep/discard decisions;
- disciplined hyperparameter/model exploration under fixed evaluation rules.

Do **not** use for unrelated repos or ad-hoc one-off analysis without experiment loop control.

# Fast start workflow
1. Read `references/workflow.md`.
2. Enforce boundaries from `references/constraints.md`.
3. Evaluate experiments by `references/decision-policy.md`.
4. Use `scripts/extract_val_metrics.py run.log` after each run.
5. Write mutable session artifacts to `outputs/research/<run-tag>/` (not `docs/`).

# Required operating rules
- Always run Python via `conda run -n miso`.
- Default run command:
  - `timeout 3600 conda run -n miso python main.py --config configs/config.yaml --mode train > run.log 2>&1`
- Only use validation metrics for keep/discard.
- Never optimize using test-set results during research loop.
- Commit each experiment **before** execution with `research: <short description>`.

# Session artifact layout (规范化存储)
Create one folder per run tag:

```text
outputs/research/<run-tag>/
├── progress.md
├── results.tsv
└── session.log
```

Use templates from `assets/`:
- `assets/progress-template.md`
- `assets/results-template.tsv`

# References
- Workflow details: `references/workflow.md`
- Hard/soft constraints: `references/constraints.md`
- Keep/discard decision logic: `references/decision-policy.md`
- Experiment idea bank: `references/experiment-strategy.md`
- Principles: `references/core-principles-hgt.md`

# Scripts
- Run `scripts/extract_val_metrics.py` to parse final validation metrics from `run.log`.
