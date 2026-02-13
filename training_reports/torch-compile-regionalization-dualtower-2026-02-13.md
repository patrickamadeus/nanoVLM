# Experiment: torch-compile-regionalization-dualtower

- **Date**: 2026-02-13
- **Author**: Codex-assisted
- **Goal**: Reduce torch.compile overhead/recompilation for DualTower training while keeping loss behavior consistent on the existing 40-step debug configuration.
- **General description**: We replaced full-model compile with regional compile and measured impact on the same config/run length.
- **Models**: DualTowerVLM (50,791,552 params)
- **Datasets**: `patrickamadeus/the_cauldron` with config `sample_1pct`

---

## 1. Setup

### 1.1 Model & task

- Model: DualTowerVLM (`mode: dualtower`)
- Task: short training debug run (`max_training_steps=40`)
- Config: `configs/train.small_debug.dualtower.yaml`

### 1.2 Data

- Train split: `train`
- Validation split: `validation` (not used in these 40-step runs due eval interval)
- Packing enabled (`use_packing=true`)

### 1.3 Base hyperparameters

- `batch_size=4`, `gradient_accumulation_steps=4` (effective batch size 16)
- `max_sample_length=512`, `lm_max_length=512`
- `lr_mp=5e-4`, `lr_left_tower=1e-4` (via fallback), `lr_right_tower=0`
- Hardware: 1x NVIDIA GeForce RTX 4090

---

## 2. Runs

### 2.1 Run table

| run_id | config_name / label | key_param_changed | job_id / link | logs_file |
|--------|---------------------|-------------------|---------------|-----------|
| r0 | baseline-uncompiled | `compile=false` | `n1bvvbmm` / https://wandb.ai/patrickirawan-mbzuai/dualtower-debug/runs/n1bvvbmm | `wandb/run-20260213_142935-n1bvvbmm/files/output.log` |
| r1 | baseline-compiled-full-model | top-level `torch.compile(model)` | `qqiatehi` / https://wandb.ai/patrickirawan-mbzuai/dualtower-debug/runs/qqiatehi | `wandb/run-20260213_143104-qqiatehi/files/output.log` |
| r2 | patch-v1-regional-wide | compiled decoder+vision+MP submodules | `hb2w5snt` / https://wandb.ai/patrickirawan-mbzuai/dualtower-debug/runs/hb2w5snt | `wandb/run-20260213_184015-hb2w5snt/files/output.log` |
| r3 | patch-v2-regional-decoder-only | compiled decoder blocks only | `ujacktuh` / https://wandb.ai/patrickirawan-mbzuai/dualtower-debug/runs/ujacktuh | `wandb/run-20260213_184315-ujacktuh/files/output.log` |
| r4 | patch-v3-fullgraph+mark_dynamic | decoder-only compile with `fullgraph=True` + `(B,T)` dynamic marking on batch tensors | `b80q6nwv` / https://wandb.ai/patrickirawan-mbzuai/dualtower-debug/runs/b80q6nwv | `wandb/run-20260213_192216-b80q6nwv/files/output.log` |

### 2.2 Notes per run

- r1 emitted `torch._dynamo hit config.recompile_limit (8)` with guard `len(images[0]) == 1` in `models/dual_tower/dual_tower.py`.
- r2 removed wrapper-level guard failures but still recompiled on variable image-batch dimension in compiled ViT/MP.
- r3 removed those ViT/MP shape recompiles by keeping variable image-count modules eager.
- r4 (`TORCH_LOGS="recompiles"`) showed only two warmup recompiles:
  - `tensor 'x' requires_grad mismatch`
  - `block_kv_cache is None` branch specialization
  No size-mismatch recompiles were observed.

---

## 3. Results

### 3.1 Metrics

| run_id | main_metric_name | main_metric_value | other_metrics | notes |
|--------|------------------|-------------------|---------------|-------|
| r0 | `train/epoch_duration` | `9.45s` | `train/epoch_loss=9.6921`, `train/epoch_tokens_per_second=25694.22` | Fastest end-to-end for 40 steps |
| r1 | `train/epoch_duration` | `109.31s` | `train/epoch_loss=9.6921`, `train/epoch_tokens_per_second=2220.40` | Slow due full-model compile recompiles |
| r2 | `train/epoch_duration` | `54.71s` | `train/epoch_loss=9.6920`, `train/epoch_tokens_per_second=4436.68` | 2.0x faster than r1 |
| r3 | `train/epoch_duration` | `29.88s` | `train/epoch_loss=9.6921`, `train/epoch_tokens_per_second=8122.39` | 3.66x faster than r1, 1.83x faster than r2 |
| r4 | `train/epoch_duration` | `33.14s` | `train/epoch_loss=9.6921`, `train/epoch_tokens_per_second=7323.92` | Fewer recompile causes visible; slightly slower than r3 on this short run |

### 3.2 Plots / qualitative observations

- Loss curves at step 20 and 40 remained aligned across all runs.
- For this short run length, compile startup cost still dominates total epoch time, so eager remains faster than compiled.

---

## 4. Analysis

- Full-model compile is a poor fit for this packed DualTower path because wrapper-level Python list guards trigger heavy recompilation.
- Regional compile is materially better than full-model compile.
- Decoder-only regional compile is the best compile strategy tested for this config and run length.
- With `fullgraph=True` + `mark_dynamic`, recompiles shifted away from shape guards; remaining warmup recompiles are path-specialization (`requires_grad`, `block_kv_cache`) and no longer hit `recompile_limit`.
- For this short 40-step debug run, `fullgraph=True` did not improve wall-clock time versus non-fullgraph decoder-only compile.

---

## 5. Lessons learned -> candidate skills

- Candidate skill update: torch compile on packed multimodal pipelines should default to region-level compile boundaries around tensor-stable decoder blocks.
- Candidate troubleshooting entry: avoid compiling wrapper forwards that consume Python list/image pack structures.

Relevant files:
- This report: `training_reports/torch-compile-regionalization-dualtower-2026-02-13.md`
- Code change: `train.py`
- Experiment log: `references/experiment-log.md`
