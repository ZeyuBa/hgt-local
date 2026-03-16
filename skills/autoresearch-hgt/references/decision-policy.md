# Keep/Discard Decision Policy

## Primary metric
- `val_f1_cal` (from HF key `eval_edge_best_f1`) is primary.

## Secondary tie-breakers
- `val_graph_acc` (`eval_graph_accuracy`)
- `val_auc` (`eval_edge_auc`)
- code simplicity/maintainability

## Decision matrix
1. `f1_new > f1_best + 0.005` => **KEEP**.
2. `f1_new > f1_best` => **KEEP**.
3. `|f1_new - f1_best| <= 0.001` => tie-break:
   - graph accuracy better => KEEP
   - or AUC better + code simpler => KEEP
   - else DISCARD
4. `f1_new < f1_best` => **DISCARD**.

## Complexity override
If gain is tiny (<0.005) but change adds heavy complexity, prefer DISCARD.
If tied/slightly better with simpler code, prefer KEEP.

## Crash rule
If run crashes or metrics cannot be extracted:
- status = `crash`
- `val_f1_cal = 0.0000`, `val_graph_acc = 0.0000`, `val_auc = 0.0000`
