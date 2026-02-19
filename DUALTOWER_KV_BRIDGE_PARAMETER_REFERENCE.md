# DualTower KV Bridge Parameter Reference

This document explains every KV-bridge parameter and how it changes behavior.

## Where KV bridge is applied
- The bridge transforms **donor K/V cache** from the left tower before the right tower uses it.
- It runs per decoder layer, on tensors shaped like `[B, H_kv, T, D_head]`.

## Bridge modes

### `kv_bridge_type: scaled_linear`
Formula (per key/value path):
- `y = W_n(...W_2(W_1(x))...) * s`
- `W_i` are linear layers without nonlinearity.
- `s` is a learnable per-head-dim scale vector.

Interpretation:
- Pure affine rotation/reshaping in head space.
- Good first stage when you want a conservative mapping.

Uses these knobs:
- `kv_bridge_linear_depth`
- `kv_bridge_use_rmsnorm`

Ignores these knobs:
- `kv_bridge_adapter_depth`
- `kv_bridge_adapter_expansion`
- `kv_bridge_residual`
- `kv_bridge_init_mode`
- `kv_bridge_mlp_ratio`

### `kv_bridge_type: residual_nonlinear`
Formula (per key/value path):
- `f(x) = out( SiLU(hidden(...SiLU(in(x))...)) )`
- `y = (x + f(x)) * s`
- output projection is zero-initialized, so initially `f(x)=0` and `y≈x` (identity).

Interpretation:
- Learns residual correction instead of full overwrite.
- Nonlinearity makes it more expressive than scaled linear.

Uses these knobs:
- `kv_bridge_adapter_depth`
- `kv_bridge_adapter_expansion`
- `kv_bridge_use_rmsnorm`

Ignores these knobs:
- `kv_bridge_linear_depth`
- `kv_bridge_residual`
- `kv_bridge_init_mode`
- `kv_bridge_mlp_ratio`

## Shared/legacy knobs

### `kv_bridge_use_rmsnorm` (bool)
- Applies head-dim RMSNorm to K and V before bridge transform.
- `true`: stabilizes magnitude and can improve training stability.
- `false`: raw K/V values are transformed directly.

### `kv_bridge_linear_depth` (int, new mode)
- Used only by `scaled_linear`.
- Number of stacked linear layers in the transform path.
- `1`: single linear map (most conservative).
- `>1`: deeper affine transform, still no nonlinearity.

### `kv_bridge_adapter_depth` (int, new mode)
- Used only by `residual_nonlinear`.
- Number of linear layers in residual branch `f(x)`.
- Minimum is clamped to `2`:
  - `2` means `dim -> hidden -> dim`.
  - `>2` inserts extra hidden layers.

### `kv_bridge_adapter_expansion` (float, new mode)
- Used only by `residual_nonlinear`.
- Hidden width factor in residual branch:
  - `hidden = round(dim * expansion)`.
- `1.0`: no width expansion.
- `>1.0`: wider residual branch, higher capacity.

### `kv_bridge_residual` (legacy)
- Only used by legacy `linear` / `mlp` modes.
- Not used by `scaled_linear` / `residual_nonlinear`.

### `kv_bridge_init_mode` (legacy)
- Only used by legacy `linear` / `mlp` modes.
- Not used by `scaled_linear` / `residual_nonlinear`.

### `kv_bridge_mlp_ratio` (legacy)
- Only used by legacy `mlp` mode.
- Not used by `scaled_linear` / `residual_nonlinear`.

## Practical training sequence (recommended)
1. Start with `scaled_linear`:
- `kv_bridge_type: scaled_linear`
- `kv_bridge_linear_depth: 1`
- keep `kv_bridge_use_rmsnorm: true`

2. If underfitting donor transport, move to `residual_nonlinear`:
- `kv_bridge_type: residual_nonlinear`
- `kv_bridge_adapter_depth: 2`
- `kv_bridge_adapter_expansion: 1.0` (then try 2.0)

## Notes on "residual"
There are two different meanings in the codebase:
- Legacy residual (`kv_bridge_residual`): for old `linear/mlp` bridge.
- Residual nonlinear bridge (`residual_nonlinear`): always uses `x + f(x)` inside the mode.

For your new modes, focus on `kv_bridge_type`, `kv_bridge_linear_depth`, `kv_bridge_adapter_depth`, `kv_bridge_adapter_expansion`, and `kv_bridge_use_rmsnorm`.
