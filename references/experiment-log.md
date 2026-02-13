## 2026-02-13 — Activation Checkpointing Integration (regular + selective)

**Type:** Observation
**General description:** Added activation checkpointing modes with compile-aware validation and ran smoke tests for compile/selective, non-compile/regular, and invalid non-compile/selective settings.

### Details

Code integration:
- Added `vlm.activation_checkpointing` and `vlm.activation_checkpointing_mode` config fields.
- Added shared AC helper (`models/activation_checkpointing.py`) with:
  - `regular`: `checkpoint(..., use_reentrant=False)`
  - `selective`: `create_selective_checkpoint_contexts(..., allow_cache_entry_mutation=True)`
- Wired activation checkpointing into LM and ViT block loops with closure-safe block binding.
- Added fail-fast validation in `train.py`: `selective` mode requires `train.compile=True`.

Smoke validation:
- `compile=true + selective`: completed 2 training steps successfully.
- `compile=false + regular`: completed 1 training step successfully.
- `compile=false + selective`: raised the expected fail-fast `ValueError` before dataloader/model setup.

### Key Points

- Integration is minimal and explicit: one helper module, two config fields, and block-loop wiring.
- Non-compile path now has an enforced single valid AC mode (`regular`).
- Selective mode remains available for compile runs without silent fallback behavior.

### Links

- Report: `training_reports/activation-checkpointing-integration-2026-02-13.md`
- Code: `models/activation_checkpointing.py`, `models/language_model.py`, `models/vision_transformer.py`, `models/config.py`, `train.py`

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

## 2026-02-13 — Prepared Current MoMH Compile+Selective Run Config

**Type:** Configuration
**General description:** Added a long-run current-style config for nanoVLM with MoMH enabled, compile enabled, and selective activation checkpointing, including explicit W&B settings.

### Details

Created `configs/train.current.momh.compile-selective.yaml` by deriving from `configs/train.current.yaml` and applying:
- `vlm.momh_enabled: true`
- `vlm.momh_head_pct_vision: 0.2`
- `vlm.momh_head_pct_text: 0.3`
- `vlm.activation_checkpointing: true`
- `vlm.activation_checkpointing_mode: "selective"`
- `train.compile: true`
- Explicit W&B fields:
  - `train.wandb_entity: patrickirawan-mbzuai`
  - `train.wandb_project: momh`
  - `train.wandb_run_name_prefix: momh-compile-selective`
  - `train.log_wandb: true`

### Key Points

- This is a `train.current`-style configuration intended for full runs, not a short preflight config.
- W&B routing is now explicit in the YAML and does not depend on dataclass defaults.

### Links

- Config: `configs/train.current.momh.compile-selective.yaml`

## 2026-02-13 — 4096 Context Config + Backbone Context Preservation Fix

**Type:** Configuration
**General description:** Updated the current MoMH compile/selective config to 4096 context and fixed LM backbone loading so user-configured context length is preserved instead of being overwritten by HF defaults.

### Details

Config updates in `configs/train.current.momh.compile-selective.yaml`:
- `vlm.lm_max_position_embeddings: 4096`
- `vlm.lm_max_length: 4096`
- `train.max_sample_length: 4096`
- `train.resume_from_vlm_checkpoint: false` (already set previously so MoMH/selective settings remain active)

Code updates in `models/language_model.py` (`LanguageModel.from_pretrained`):
- Preserve requested `lm_max_position_embeddings` and `lm_max_length` from config when loading HF backbone weights.
- Add robust `rope_theta` resolution:
  - use `hf_config.rope_theta` when available
  - fallback to `hf_config.rope_parameters["rope_theta"]`
  - fail fast with explicit `ValueError` if neither exists

Validation:
- Instantiating `VisionLanguageModel(..., load_backbone=True)` with the 4096 config now yields:
  - `cfg_lm_max_position_embeddings = 4096`
  - `cfg_lm_max_length = 4096`
  - `decoder.rotary_embd.original_max_seq_len = 4096`
- Fast test suite check: `pytest -q tests/test_activation_checkpointing.py` passed (`4 passed`).

### Key Points

- The earlier "config says 4096 but model still uses 8192" issue was real in the backbone load path and is now fixed.
- This change affects backbone initialization only; it does not silently change any training hyperparameters.

### Links

- Config: `configs/train.current.momh.compile-selective.yaml`
- Code: `models/language_model.py`

## 2026-02-13 — Batch Size Probe Blocked by GPU/PyTorch Capability Mismatch

**Type:** Observation
**General description:** Attempted quick OOM-based max-batch probe for the 4096-context MoMH compile/selective setup, but execution fails before memory pressure due to unsupported GPU architecture in current PyTorch build.

### Details

Probe outcome:
- First probe at `batch_size=1` failed with:
  - `CUDA error: no kernel image is available for execution on the device`
- No valid OOM boundary could be measured because kernels do not launch successfully.

Environment check:
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`sm_120`)
- Installed torch: `2.10.0+cu126`
- PyTorch warning indicates current build supports up to `sm_90` and recommends CUDA 12.8/13.0 builds for this GPU.

### Key Points

- Max non-OOM batch size is currently **unknown** on this machine until PyTorch/CUDA wheel supports `sm_120`.
- This is an environment compatibility issue, not a model/config OOM result.

### Links

- Config under test: `configs/train.current.momh.compile-selective.yaml`

## 2026-02-13 — Max Microbatch Probe on Blackwell After Torch Reinstall

**Type:** Observation
**General description:** Re-ran OOM boundary probing on the `4096` MoMH compile/selective config after reinstalling torch with Blackwell support.

### Details

Environment:
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`sm_120`)
- torch: `2.10.0+cu128`

Config under test:
- `configs/train.current.momh.compile-selective.yaml`
- `vlm.lm_max_position_embeddings=4096`, `vlm.lm_max_length=4096`, `train.max_sample_length=4096`
- `vlm.momh_enabled=true`
- `vlm.activation_checkpointing=true`, `vlm.activation_checkpointing_mode=selective`
- `train.compile=true`
- `train.resume_from_vlm_checkpoint=false`, `vlm.vlm_load_backbone_weights=true`

Method:
- Synthetic one-step train probe (forward + backward + optimizer step) with real model init/backbone load.
- Regional decoder compile path (`_apply_regional_compile`) enabled.
- Exponential search + binary search over microbatch size, then fresh-process confirmation.

Measured boundary:
- Success up to `microbatch=32`
- First OOM at `microbatch=33`

Fresh-process confirmation:
- `bs=32` -> success (`peak_mb=91425.61`)
- `bs=33` -> OOM (`peak_mb=87547.14`, early-fail peak)

### Key Points

- Current best known per-step microbatch ceiling for this setup is `32`.
- Effective global batch still depends on `gradient_accumulation_steps` and number of GPUs.
- Config was updated to use `train.batch_size: 32` in `configs/train.current.momh.compile-selective.yaml`.

### Links

- Config: `configs/train.current.momh.compile-selective.yaml`
- Code path: `train.py` (`_apply_regional_compile`), `models/vision_language_model.py`

## 2026-02-13 — Preflight Config Alignment + W&B/HF_HOME Validation

**Type:** Configuration
**General description:** Aligned MoMH preflight configs to the same 4096-context current baseline, enabled explicit W&B logging for preflight, and validated online W&B logging with `HF_HOME` exported first.

### Details

Preflight configs were rebuilt from the same base as `configs/train.current.momh.compile-selective.yaml` and then specialized per preflight intent:
- `configs/train.preflight.momh.checkpoint-load.yaml`
- `configs/train.preflight.momh.stability.yaml`
- `configs/train.preflight.momh.compile-selective.yaml`

Shared aligned settings:
- `mode: nanovlm`
- `lm_max_position_embeddings=4096`, `lm_max_length=4096`, `max_sample_length=4096`
- `batch_size=32`, `gradient_accumulation_steps=16`
- `momh_enabled=true`
- `wandb_entity=patrickirawan-mbzuai`
- `log_wandb=true`

Step-specific toggles:
- checkpoint-load: `max_training_steps=2`, `compile=false`, `resume_from_vlm_checkpoint=true`, `activation_checkpointing_mode=regular`
- stability: `max_training_steps=40`, `compile=false`, `resume_from_vlm_checkpoint=false`, `activation_checkpointing_mode=regular`
- compile-selective: `max_training_steps=40`, `compile=true`, `resume_from_vlm_checkpoint=false`, `activation_checkpointing_mode=selective`

W&B reliability updates:
- `train.py` now passes `entity=train_cfg.wandb_entity` in `wandb.init(...)` so config-driven entity selection is honored.
- `train.py` `_apply_checkpoint_cfg_overrides(...)` now preserves requested runtime behavior fields (`momh_*`, `activation_checkpointing*`, and max-length overrides) when `resume_from_vlm_checkpoint=true`.
- `runpod_train_and_stop.sh` now exports `HF_HOME=${HF_HOME:-/workspace/huggingface}` after venv activation.
- `AGENTS.md` preflight command examples now include `export HF_HOME=/workspace/huggingface && ...` prefixes.
- Ran smoke logging with required command prefix:
  - `export HF_HOME=/workspace/huggingface && source .venv/bin/activate && python ...`
  - Successful run URL: `https://wandb.ai/patrickirawan-mbzuai/momh-preflight/runs/xx3ec62n`

Validation:
- Config parsing checks passed for all aligned configs.
- Added `tests/test_train_checkpoint_cfg.py` to lock checkpoint-config override behavior.
- `pytest -q tests/test_train_checkpoint_cfg.py tests/test_activation_checkpointing.py` passed (`5 passed`).

### Key Points

- Preflight is now configuration-consistent with the current MoMH compile/selective run setup.
- Script invocations should always export `HF_HOME=/workspace/huggingface` first.
- W&B online logging is confirmed functional in this environment.

### Links

- Configs: `configs/train.current.momh.compile-selective.yaml`, `configs/train.preflight.momh.checkpoint-load.yaml`, `configs/train.preflight.momh.stability.yaml`, `configs/train.preflight.momh.compile-selective.yaml`
- Code: `train.py`
