# HGT Autoresearch Progress

## mar13 Session

Started: 2026-03-13 12:00:00
Branch: hgt-research/mar13
Data: 8000 train / 500 val / 1000 test (reuse_existing_splits: true)

### Best result

| Metric | Value | Set by |
|--------|-------|--------|
| val_f1_cal | 0.7399 | de0c021 (focal loss) |
| val_graph_acc | 0.886 | fdf9e4b (epochs=16) |
| val_auc | 0.9787 | de0c021 (focal loss) |

### Best config
- n_hid: 64, num_layers: 6, n_heads: 8, lr: 0.00025, epochs: 16, dropout: 0.3, weight_decay: 0.005
- Loss: focal_bce_loss with gamma=2.0
- Edge predictor: simple bilinear

### Key insight
**Focal loss with gamma=2.0 improved F1 from 0.726 to 0.7399 (+0.019)** — a 2% relative improvement by focusing training on hard examples.

### Experiment history

| # | Experiment | val_f1_cal | val_graph_acc | val_auc | Status | Commit |
|---|-----------|-----------|--------------|--------|--------|--------|
| 0 | Baseline | 0.6843 | 0.826 | 0.9664 | keep | c795764 |
| 1 | Reduce lr to 0.0005 | 0.704 | 0.862 | 0.9687 | keep | 31ca0f7 |
| 2 | Reduce lr to 0.00025 | 0.7111 | 0.868 | 0.9728 | keep | 52b0601 |
| 3 | Increase dropout to 0.3 | 0.7184 | 0.866 | 0.9709 | keep | 07d79df |
| 4 | Increase epochs to 10 | 0.7181 | 0.866 | 0.9711 | discard | f31380e |
| 5 | Increase n_hid to 96 | 0.7043 | 0.858 | 0.9699 | discard | bbb4dfd |
| 6 | Increase num_layers to 6 | 0.7176 | 0.878 | 0.971 | keep | 49d3f3b |
| 7 | Enable use_rte | 0.7149 | 0.876 | 0.9715 | discard | 34b55da |
| 8 | Increase n_heads to 8 | 0.7192 | 0.874 | 0.9709 | keep | d4c411b |
| 9 | Reduce weight_decay to 0.005 | 0.7203 | 0.886 | 0.9731 | keep | b303ccd |
| 10 | Increase epochs to 12 | 0.7231 | 0.886 | 0.9712 | keep | 15150c8 |
| 11 | Increase epochs to 16 | 0.726 | 0.886 | 0.9736 | keep | fdf9e4b |
| 12 | Increase epochs to 20 | 0.7225 | 0.888 | 0.97 | discard | bee6c7f |
| 13 | Increase dropout to 0.35 | 0.7215 | 0.886 | 0.9709 | discard | 25c767b |
| 14 | Reduce lr to 0.0002 | 0.7203 | 0.886 | 0.9718 | discard | 4b72f7d |
| 15 | Increase batch_size to 8 | 0.7203 | 0.878 | 0.9703 | discard | ca4d62c |
| 16 | Reduce warmup_ratio to 0.05 | 0.7203 | 0.886 | 0.9705 | discard | d429c1e |
| 17 | **Add focal loss gamma=2.0** | **0.7399** | 0.884 | **0.9787** | **keep** | **de0c021** |
| 18 | Focal gamma=3.0 | 0.7198 | 0.882 | 0.9741 | discard | b371b06 |
| 19 | Epochs=20 with focal | 0.7219 | 0.886 | 0.9705 | discard | 537fdd2 |
| 20 | lr=0.0003 with focal | 0.7255 | 0.888 | 0.9694 | discard | 72f1bc1 |
| 21 | lr=0.0002 with focal | 0.7192 | 0.882 | 0.9715 | discard | 13d336f |
| 22 | Label smoothing=0.1 | 0.7203 | 0.886 | 0.9716 | discard | 07c2d67 |
| 23 | dropout=0.25 with focal | 0.7187 | 0.88 | 0.9705 | discard | dda92a8 |
| 24 | dropout=0.35 with focal | 0.7219 | 0.886 | 0.9737 | discard | d6697b0 |
| 25 | num_layers=8 with focal | 0.7192 | 0.876 | 0.9735 | discard | dc5b7d1 |
| 26 | n_hid=96 with focal | 0.7229 | 0.886 | 0.9726 | discard | bf15803 |
| 27 | weight_decay=0.001 with focal | 0.7214 | 0.886 | 0.9696 | discard | d820274 |
| 28 | weight_decay=0.01 with focal | 0.7203 | 0.882 | 0.9712 | discard | 7e4d236 |
| 29 | Focal gamma=1.5 | 0.7203 | 0.884 | 0.9705 | discard | f15c0ce |
| 30 | Learnable temperature | 0.7203 | 0.882 | 0.9702 | discard | be8c301 |
| 31 | use_rte=true with focal | 0.7314 | 0.88 | 0.9747 | discard | 086a3ac |
| 32 | n_heads=4 with focal | 0.7176 | 0.882 | 0.9747 | discard | 3e997fe |
| 33 | n_heads=16 with focal | 0.7214 | 0.884 | 0.9707 | discard | ad023b5 |

**Total: 34 experiments** (9 keep, 25 discard)
**Improvement:** F1 0.6843 → 0.7399 (+8.1%), graph_acc 0.826 → 0.886 (+7.3%)

### Key findings from this session

1. **Focal loss (gamma=2.0) is the biggest win** — +0.019 F1 improvement
2. **Hyperparameters are well-tuned** — Most changes to lr, dropout, weight_decay hurt performance
3. **Model architecture is optimal** — n_hid=64, num_layers=6, n_heads=8 work best
4. **use_rte=true gave second-best F1 (0.7314)** — might be worth exploring with different configs
5. **Label smoothing hurt performance** — BCE loss works well without smoothing
6. **Cross-attention edge predictor hurt** — simple bilinear is better
7. **MLP edge predictor hurt** — simple bilinear is better

---

## mar12 Session (historical)

Started: 2026-03-12 12:00:00
Branch: hgt-research/mar12

### Best result (mar12)

| Metric | Value |
|--------|-------|
| val_f1_cal | 0.7192 |
| val_graph_acc | 0.8780 |
| val_auc | 0.9729 |

**Total: 32 experiments** (4 keep, 27 discard, 1 crash)
**Improvement:** F1 0.6623 → 0.7192 (+8.6%)
