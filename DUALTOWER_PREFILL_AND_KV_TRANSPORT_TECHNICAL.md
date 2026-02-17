# DualTower Prefill and KV Transport Technical Notes

## 1) Goal and Current Design Contract

This implementation now supports your intended separation:

- Left tower is the donor of prompt K/V (scope-controlled by `left_mask_scope`).
- Right tower is the query side and decoding side.
- Decode cache grows on the right tower in standard autoregressive append mode.

Key rule:

- Q is always computed by the right tower.
- K/V can be donated from left tower according to `left_mask_scope`.

## 2) Disparity vs Previous Design (Why Outputs Diverged)

### Legacy (`DualTowerVLM` old folder)

- Donor region was contiguous prefix (`[:last_img_idx+1]`).
- Right dual-prefill used a custom mask:
  - donor block bidirectional internally,
  - text block causal,
  - text queries could attend donor block.
- This behavior lived in `dual_language_model.py` custom mask path.

### New masking implementation (before fix)

- Donor region came from boolean `replace_mask` (`visual_only` / `visual_sys` / `full`).
- K/V donor replacement was correct, but right prefill attention used plain causal semantics.
- That mismatch (replacement semantics vs interaction semantics) caused degradation, especially in `visual_only` / `visual_sys`.

## 3) Fixes Implemented

### A) Restored legacy-style interaction for partial donor scopes

- During dual-prefill with partial donor (`visual_only` / `visual_sys`), a dual-prefill mask is now applied.
- For `full`, this custom bidirectional donor mask is intentionally skipped to avoid non-causal full-prompt leakage.

Files:

- `models/language_model.py`

### B) Decode continuity preserved

- Decode position uses cache-length continuation.
- No RoPE restart in decode.

Files:

- `models/dual_tower/dual_tower.py`
- `models/language_model.py`

### C) Optional pad-aware RoPE

- Added `vlm.lm_pad_aware_rope`:
  - `false` (default): absolute positions.
  - `true`: prefill positions compact by non-pad tokens (legacy-style).

Files:

- `models/config.py`
- `models/language_model.py`

### D) Right prefill strategy knob (generation path)

- Added `vlm.right_prefill_mode`:
  - `full`: full right prefill (original behavior).
  - `non_donor_only`: minimal right prefill behavior:
    - `left_mask_scope=full`: single last-token query bootstrap.
    - `left_mask_scope=visual_only|visual_sys`: query only on non-donor (unprocessed) span.

Files:

- `models/config.py`
- `models/dual_tower/dual_tower.py`
- `generate.py`

## 4) Important Behavioral Notes

1. `right_prefill_mode` is used in generation path (`DualTowerVLM.generate`), not training forward.
2. In `non_donor_only` with visual scopes, current implementation is constrained to `batch_size=1` for generation.
3. If `use_kv_bridge=true`, donor K/V is bridged-left K/V, not raw left K/V.
4. Keep `lm_pad_aware_rope` consistent between training and inference for a checkpoint family.

## 5) Knobs Summary

- `vlm.left_mask_scope`: `visual_only | visual_sys | full`
- `vlm.right_prefill_mode`: `full | non_donor_only`
- `vlm.use_kv_bridge`: `true | false`
- `vlm.kv_bridge_type`: `linear | mlp`
- `vlm.kv_bridge_mlp_ratio`: float
- `vlm.lm_pad_aware_rope`: `true | false`

## 6) Configs Added (Full YAMLs)

- `configs/train.full-kv.bootstrap.kvbridge-mlp4.yaml`
- `configs/train.full-kv.bootstrap.dualtower-nobridge-lefttrain-lm1e5-mp5e5.yaml`
- `configs/train.visual-sys.bootstrap.dualtower-nobridge-lefttrain.yaml`
- `configs/train.full-kv.bootstrap.dualtower-nobridge-lefttrain.yaml`
- `configs/train.full-kv.fullquery.dualtower-nobridge-lefttrain.yaml`

## 7) Practical Mapping to Your Requested Modes

1. Full KV + bootstrap + KV bridge MLP 4.0:
   - `left_mask_scope=full`
   - `right_prefill_mode=non_donor_only`
   - `use_kv_bridge=true`
   - `kv_bridge_type=mlp`
   - `kv_bridge_mlp_ratio=4.0`

2. Full KV + bootstrap + no bridge, left trained (explicit lr):
   - `left_mask_scope=full`
   - `right_prefill_mode=non_donor_only`
   - `use_kv_bridge=false`
   - `lr_mp=5e-5`
   - `lr_left_tower=1e-5`
   - `lr_right_tower=0.0`

3. Visual+sys + no bridge, left trained:
   - `left_mask_scope=visual_sys`
   - `right_prefill_mode=non_donor_only` (non-donor query span)
   - `use_kv_bridge=false`
   - `lr_mp=5e-5`
   - `lr_left_tower=1e-5`
   - `lr_right_tower=0.0`

4. Full KV + bootstrap + no bridge, left trained (fallback-lr variant):
   - same behavior as #2 but left LR via fallback (`lr_left_tower=null`, `lr_language_backbone=1e-5`)

5. Full KV + full right query recomputation + no bridge, left trained:
   - `left_mask_scope=full`
   - `right_prefill_mode=full`
   - `use_kv_bridge=false`
   - `lr_mp=5e-5`
   - `lr_left_tower=1e-5`
   - `lr_right_tower=0.0`

