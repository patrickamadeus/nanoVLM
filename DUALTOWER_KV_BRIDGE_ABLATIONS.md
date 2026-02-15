# DualTower KV Bridge (All Layers): Ablations and Knobs

## Goal
Train only a learnable KV bridge that maps left-tower KV cache into right-tower KV space across **all LM layers**, while freezing both towers.

## Bridge-Only Baseline Command

```bash
source .venv/bin/activate && python train.py \
  --dualtower \
  --enable_kv_bridge \
  --dualtower_bridge_only \
  --lr_kv_bridge 1e-4
```

## Config Knobs

### VLMConfig (`models/config.py`)
- `kv_bridge_enabled` (`bool`, default `False`): Enables KV bridge in dualtower mode.
- `kv_bridge_type` (`str`, default `"linear"`): `linear` or `mlp`.
- `kv_bridge_mlp_ratio` (`float`, default `2.0`): Hidden expansion ratio for MLP bridge.
- `kv_bridge_use_rmsnorm` (`bool`, default `True`): Apply RMSNorm on K/V before bridge projections.
- `kv_bridge_residual` (`bool`, default `True`): Residual bridge output (`K + ΔK`, `V + ΔV`).

### TrainConfig (`models/config.py`)
- `lr_kv_bridge` (`float | None`, default `1e-4`): Bridge learning rate.
- `dualtower_bridge_only` (`bool`, default `False`): Freeze left and right towers; train only bridge.

### CLI Knobs (`train.py`)
- `--enable_kv_bridge`
- `--kv_bridge_type {linear,mlp}`
- `--kv_bridge_mlp_ratio <float>`
- `--kv_bridge_no_rmsnorm`
- `--kv_bridge_no_residual`
- `--lr_kv_bridge <float>`
- `--dualtower_bridge_only`
- `--left_tower_prefill_no_grad`

## Recommended Ablation Runs

### A1: Linear bridge (residual)
```bash
source .venv/bin/activate && python train.py \
  --dualtower --enable_kv_bridge --dualtower_bridge_only \
  --kv_bridge_type linear --lr_kv_bridge 1e-4
```

### A2: Linear bridge (no residual)
```bash
source .venv/bin/activate && python train.py \
  --dualtower --enable_kv_bridge --dualtower_bridge_only \
  --kv_bridge_type linear --kv_bridge_no_residual --lr_kv_bridge 1e-4
```

### A3: MLP bridge (residual, ratio 2.0)
```bash
source .venv/bin/activate && python train.py \
  --dualtower --enable_kv_bridge --dualtower_bridge_only \
  --kv_bridge_type mlp --kv_bridge_mlp_ratio 2.0 --lr_kv_bridge 1e-4
```

### A4: MLP bridge (residual, ratio 4.0)
```bash
source .venv/bin/activate && python train.py \
  --dualtower --enable_kv_bridge --dualtower_bridge_only \
  --kv_bridge_type mlp --kv_bridge_mlp_ratio 4.0 --lr_kv_bridge 7e-5
```

### A5: MLP bridge (residual + no RMSNorm)
```bash
source .venv/bin/activate && python train.py \
  --dualtower --enable_kv_bridge --dualtower_bridge_only \
  --kv_bridge_type mlp --kv_bridge_no_rmsnorm --lr_kv_bridge 1e-4
```

## Notes
- `--dualtower_bridge_only` requires `--dualtower` and `--enable_kv_bridge`.
- In bridge-only mode, tower parameters are force-frozen regardless of tower LR settings.
- Bridge-only mode auto-enables `left_tower_prefill_no_grad` for memory savings.
- KV bridge is applied to all layers (`lm_n_blocks`) by construction.
