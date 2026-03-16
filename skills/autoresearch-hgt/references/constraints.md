# Experiment Constraints

## Hard constraints (must obey)
1. **Evaluation protocol fixed**: do not alter metrics logic in `src/training/trainer.py`.
2. **Data pipeline fixed**: do not edit `src/dataset/` or `training_data/`.
3. **Entry/config loader fixed**: do not edit `main.py` or `src/training/config.py`.
4. **Test-set isolation**: keep/discard uses validation metrics only.
5. **No new dependencies**.
6. **Problem-definition constants fixed**: keep `num_types=3`, `num_relations=9`.
7. **Feature-model dimension consistency**: if feature layout changes, align `model.in_dim`.

## Soft constraints (optimize with judgment)
- Use `timeout 3600` as safety cap.
- Favor simpler changes when gains are tied.
- Prefer one-variable experiments for interpretability.
- Keep `reuse_existing_splits: true` unless intentionally rebuilding baseline.

## Priority
Hard constraints > primary metric > soft constraints.
