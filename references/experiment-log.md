## 2026-02-13 — MoMH with Packed Samples Segment Isolation Gap

**Type:** Retrospective
**General description:** Reviewed MoMH behavior under document packing to verify whether packed samples remain isolated.

### Details

We validated the current MoMH implementation path in `nanoVLM_main` and compared it to the sibling `flex_attention` project.

Current behavior:
- MoMH uses an explicit per-token modality mask (`is_vision`) derived from `input_ids == image_token_id`.
- This correctly supports multiple image-token regions in one packed sequence, including interleaving with text.
- Packing currently concatenates multiple samples and inserts separators with `attention_mask=0` and `labels=-100`.
- Attention/MoMH masking uses `attention_mask` for content vs padding, but does not enforce sample-segment isolation.

Implication:
- Later packed samples can still causally attend to earlier packed samples (cross-sample leakage), including MoMH head patterns across segment boundaries.

### Key Points

- The MoMH "single contiguous vision span" assumption has already been removed in the current implementation.
- The remaining packing risk is segment-boundary leakage, not modality-identification failure.
- If strict per-sample isolation is required, add segment-aware masking (same-segment constraint) in attention/MoMH.

### Links

- Decoder MoMH modality path: `models/language_model.py`
- VLM prefill block-mask creation: `models/vision_language_model.py`
- Packing/separator behavior: `data/advanced_datasets.py`

## 2026-02-13 — Torch Compile Regionalization for DualTower Debug Config

**Type:** Observation
**General description:** Reworked compile scope to avoid wrapper-level recompiles and measured before/after impact on the same 40-step debug setup.

### Details

We replaced full-model `torch.compile` in `train.py` with a regional compile strategy:
- Compile only decoder blocks (`left_tower.decoder.blocks`, `right_tower.blocks`) for dualtower.
- Keep variable image-count modules (`vision_encoder` blocks and `MP`) eager.
- Add explicit startup logging for compile strategy and compiled module counts.

Validation runs on `configs/train.small_debug.dualtower.yaml`:
- Historical compiled baseline (`qqiatehi`): epoch time `109.31s`.
- New patch v1 (`hb2w5snt`, decoder+vision+MP compiled): epoch time `54.71s`.
- New patch v2 (`ujacktuh`, decoder-only compile): epoch time `29.88s`.

Loss stayed aligned across runs (`train/epoch_loss` around `9.692` at step 40).

### Key Points

- Regional compile removed the prior top-level recompile failure on `len(images[0]) == 1`.
- Decoder-only compile outperformed broader regional compile for this short 40-step run.
- Residual recompiles are limited to early warmup in decoder blocks and no longer trigger `recompile_limit`.

### Links

- Report: `training_reports/torch-compile-regionalization-dualtower-2026-02-13.md`
- Code: `train.py`

## 2026-02-13 — Torch Compile Fullgraph + Mark Dynamic Debug Pass

**Type:** Observation
**General description:** Added `fullgraph=True` on compiled decoder blocks and marked batch `(B, T)` dimensions dynamic, then validated with `TORCH_LOGS="recompiles"`.

### Details

Changes in `train.py`:
- Regional compile now sets `fullgraph=True` for compiled decoder blocks.
- Added `_maybe_mark_batch_dynamic(...)` and applied it on `input_ids`, `labels`, and `attention_mask` when compile is enabled.

Diagnostic run:
- Command used: `TORCH_LOGS="recompiles" python train.py --config /tmp/train.small_debug.dualtower.compile_true.yaml`
- Run ID: `b80q6nwv`
- Remaining recompiles were limited to early warmup:
  - `tensor 'x' requires_grad mismatch`
  - `block_kv_cache is None`
- Previous shape-driven recompile signals were not observed in this run.

### Key Points

- Dynamic marking helped steer recompiles away from shape mismatch causes.
- Remaining recompiles are semantic path specialization, not dynamic `(B,T)` shape guards.
- `fullgraph=True` did not improve short-run wall-clock time for this 40-step debug workload.

### Links

- Report: `training_reports/torch-compile-regionalization-dualtower-2026-02-13.md`
- Code: `train.py`
