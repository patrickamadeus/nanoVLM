# DualTower Stabilization: Consolidated Technical Report

## 1. Problem Statement and Skepticism

Observed behavior:
- Same dataset and setup as baseline nanoVLM.
- ~1000 training steps, batch size 128.
- Loss decreases below 1.0.
- Dual-tower generation remains incoherent/gibberish.

Core skepticism:
- This pattern strongly suggests a semantic mismatch between training and decoding, or a structural bug in attention/KV/masking, rather than simple undertraining.

Initial high-risk suspects:
- Divergent forward paths (`forward` vs special dual forward path).
- KV cache replacement order/timing errors.
- Positional index / decode `start_pos` misalignment.
- Center-padding and split-index coupling.
- 4D mask broadcasting mistakes and shape mismatches.
- Packing boundary leakage.

---

## 2. Architectural Direction We Converged On

Target architecture:
- Left tower: visual processing branch.
- Right tower: language modeling branch.
- Right tower uses processed visual representations through KV replacement (not raw visual token semantics).
- Keep packing configurable; do not maintain unnecessary special-case machinery.

Simplification decisions:
1. Remove dedicated dual language model complexity.
2. Reuse shared/stable `models/language_model.py`.
3. Keep only a minimal dual-prefill KV replacement hook.
4. Remove center-padding and split-index alignment mechanisms from core dual path.
5. Keep right tower frozen by default and support bridge-only mode (freeze both towers, train only KV bridge) for low-cost alignment.

---

## 3. High-Critical Actionables (Prioritized)

### P0 (already implemented)
1. **Unify right tower on shared LM implementation**
   - Avoid duplicated attention pipelines and forward divergence.
   - Done by removing dedicated `dual_language_model.py` usage and using `language_model.py` with dual-prefill hook.

2. **Guarantee KV replacement before attention**
   - Replacement must occur before SDPA/manual attention in each layer.
   - Implemented in shared LM attention path.

3. **Fix decode positional continuity**
   - Decode `start_pos` must derive from KV cache length, not prompt-length shortcuts.
   - Implemented in dual generate path.

4. **Remove center-padding dependency**
   - Full-sequence + masking + KV semantics are now the primary mechanism.
   - Center-padding no longer required in core dual train/generate flow.

5. **Freeze right tower by default**
   - `TrainConfig.lr_right_tower = 0.0` default means no right-tower updates unless overridden.

### P1 (implemented)
6. **Left-tower masking as explicit config knob**
   - `left_tower_mask_mode`:
     - `visual_only` (default)
     - `visual_plus_prefix`
     - `full`
   - Enables controlled ablation without changing architecture code.

7. **Packing mode as explicit train-time knob**
   - `--packing` / `--no_packing` supported.

### P2 (still a design choice, not yet fixed)
8. **Packed cross-boundary isolation**
   - Current vanilla packed path does not enforce strict boundary isolation; later tokens can still attend earlier packed samples (causal + key padding only).
   - If strict isolation is desired, add segment-aware causal mask.
   - If accepted as-is, keep for throughput and simplicity.

---

## 4. Technical Findings from Verification

## 4.1 KV Replacement Semantics (Critical)

Validated behavior:
- Dual-prefill replacement is layer-local and pre-attention.
- For positions in `img_mask`, right-tower K/V are overwritten by cached left-tower K/V.
- For non-image positions, right-tower current K/V are used.
- Replacement flag is cleared after prefill replacement.

Direct test result:
- `dual_prefill_key_max_err = 0.0`
- `dual_prefill_value_max_err = 0.0`
- `dual_prefill_flag_cleared = True`

Interpretation:
- The feared failure mode “layer-1 attends raw image K/V before replacement” is not supported by current implementation under tested path.

## 4.2 Packing Boundary Behavior

Validated behavior:
- Packed samples are concatenated with a separator token where `attention_mask=0`.
- Attention implementation uses:
  - causal masking
  - key padding masking
- This does **not** create strict segment isolation.

Direct test result:
- `packed_boundary_leakage_max_logit_delta` is non-zero (e.g., `0.376...` / `0.532...` in repeated checks).

Interpretation:
- Changing tokens in one packed segment affects logits in subsequent packed segments.
- This is expected under current masking semantics and is not a bug unless strict isolation is a requirement.

## 4.3 Training/Decode Smoke Stability

Dual-tower one-step smoke with right tower frozen:

- `no_packing`
  - `loss_per_token=10.693313`
  - `grad_norm_left_decoder=1.317911`
  - `grad_norm_right_decoder=0.000000` (correctly frozen)
  - `grad_norm_mp=0.654854`

- `packing`
  - `loss_per_token=10.687840`
  - `grad_norm_left_decoder=1.084213`
  - `grad_norm_right_decoder=0.000000` (correctly frozen)
  - `grad_norm_mp=0.591054`

Interpretation:
- Both modes are numerically stable in smoke conditions.
- Freezing policy is respected.
- Smoke generation from tiny random-init model is not quality-representative.

---

## 5. Changes Completed

Implemented code/documentation updates include:
- Reuse of shared `language_model.py` for right tower with minimal dual-prefill hook.
- Removal of dedicated dual language model path.
- Removal of center-padding dependency from core dual logic.
- Decode `start_pos` fix via KV length in dual generate.
- New `left_tower_mask_mode` config and CLI.
- Existing packing toggle (`--packing` / `--no_packing`) retained.
- Right tower non-trainable by default via LR config.
- Optional all-layer KV bridge (`linear`/`mlp`) inserted between left KV cache and right dual-prefill replacement.
- Optional bridge-only mode: freezes left and right towers and optimizes only KV bridge params.

Additional test hygiene:
- Updated stale test in `tests/test_vision_language_model.py` to match current generation API.

---

## 6. Expected Outcomes (Generation and Scores)

Expected immediate effects from current fixes:
1. **Lower risk of train/decode semantic drift** due to single shared LM path.
2. **Lower risk of KV/position bugs** due to pre-attention replacement + KV-length decode indexing.
3. **Higher debugging clarity** due to left-mask and packing knobs.

What this should improve:
- Coherence should improve once data/mask policy is aligned with objective and enough optimization is done.
- If gibberish persists despite these structural fixes, the next likely causes are:
  - mask policy mismatch (`visual_only` vs `visual_plus_prefix` vs `full`)
  - right tower frozen too aggressively for current dataset/domain
  - packing cross-boundary leakage effects on learned conditioning
  - data-label alignment quality

Expected score trajectory:
- Do not infer quality from loss alone.
- Use task metrics + decode samples jointly.
- Evaluate with:
  1. `--no_packing` baseline for correctness debugging.
  2. `--packing` once decode is stable.
  3. identical prompts and seeds for A/B comparisons.

---

## 7. Recommended Next Execution Plan

1. Run KV-bridge ablations in bridge-only mode (`--dualtower --enable_kv_bridge --dualtower_bridge_only`).
2. Compare linear vs MLP bridge and RMSNorm/residual toggles while keeping towers frozen.
3. If coherence remains poor, then unfreeze right tower partially with a small LR and keep bridge active.

Detailed commands and knob matrix: `DUALTOWER_KV_BRIDGE_ABLATIONS.md`.

---

## 8. Bottom Line

The major complexity risks (dedicated dual LM, center-padding coupling, decode position drift) have been removed or mitigated.  
The remaining major semantic decision is packed cross-boundary behavior: accepted throughput tradeoff vs strict segment isolation requirement.
