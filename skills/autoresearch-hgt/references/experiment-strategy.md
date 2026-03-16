# Experiment Strategy

## Quick wins
- learning-rate sweep: 0.0005 / 0.001 / 0.002 / 0.005
- hidden size: 64 / 128 / 256
- layers: 2 / 4 / 6 / 8
- dropout: 0.1 / 0.2 / 0.3
- epochs: 8 / 16 / 32 / 64

## Architecture exploration
- n_heads vs n_hid ratio
- edge predictor variants
- residual/skip behavior
- `use_rte` toggle

## Regularization/data knobs
- weight decay
- warmup ratio
- batch size
- synthetic noise and topology complexity

## If plateau
When 5+ consecutive discards occur, move from micro-tuning to architecture-level hypotheses.
