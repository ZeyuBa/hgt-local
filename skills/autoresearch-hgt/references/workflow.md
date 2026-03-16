# HGT Autoresearch Workflow

## 1) Setup
1. Agree run tag (`<date-or-topic>`), create branch `hgt-research/<tag>`.
2. Read this skill's references (workflow/constraints/decision policy).
3. Verify synthetic splits exist; if missing, run once with default config.
4. Set `synthetic.reuse_existing_splits: true` for comparable experiments.
5. Create session folder `outputs/research/<tag>/` and initialize:
   - `progress.md` from `assets/progress-template.md`
   - `results.tsv` from `assets/results-template.tsv`
   - `session.log` (append-only)

## 2) Allowed changes
- Primary lever: `configs/config.yaml`
- Optional model-side changes: `src/models/` and `src/graph/feature_extraction.py`
- Optional new support files under `src/`

## 3) Forbidden changes
- `src/training/trainer.py` metric computation logic
- `src/training/config.py`
- `main.py`
- `tests/`, `src/dataset/`, `training_data/`
- Installing new dependencies

## 4) Experiment loop
1. Propose one hypothesis (prefer single-variable change).
2. Update `progress.md` current experiment section.
3. Apply change and commit (`research: ...`).
4. Run training with 1-hour timeout.
5. Parse metrics via `scripts/extract_val_metrics.py run.log`.
6. Record metrics in `results.tsv` and details in `session.log`.
7. Make keep/discard decision per `decision-policy.md`.
8. If discard, `git reset --hard HEAD~1`; if keep, advance baseline.

## 5) Failure handling
- Missing eval metrics => crash; inspect `tail -n 50 run.log`.
- OOM/shape mismatch/NaN/timeout => record crash, revert, adjust, retry.

