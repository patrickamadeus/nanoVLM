# Report: Activation Checkpointing Integration (regular + selective)

**Date:** 2026-02-13
**Author:** codex
**Status:** Completed

## Objective

Integrate activation checkpointing into nanoVLM with minimal surface area:
- Support `regular` and `selective` modes.
- Enforce project contract: `selective` requires `train.compile=true`; non-compile runs use `regular`.
- Validate both compile/selective and non-compile/regular execution paths.

## Setup

### Environment
- Hardware: CUDA device (single-process run)
- Software: PyTorch `2.10.0+cu126`
- Dataset: `patrickamadeus/the_cauldron` (`sample_1pct`)

### Configuration

Base config: `configs/train.small_debug.momh.yaml`, copied to temp configs with:
- `vlm.activation_checkpointing: true`
- `vlm.activation_checkpointing_mode: selective|regular`
- `train.compile: true|false`
- `train.log_wandb: false`
- `train.max_training_steps: 1-2` (smoke-run only)

## Experiments

### Run 1: Compile + selective activation checkpointing smoke

**Command:**
```bash
source .venv/bin/activate && TORCH_LOGS=recompiles python train.py --config /tmp/train.small_debug.momh.ac_smoke.yaml
```

**Results:**
- Compile strategy logged: `regional_submodule_compile_fullgraph` (`fullgraph=True`)
- Compiled modules logged: `decoder_blocks=4, vision_blocks=0, mp=0`
- Completed `max_training_steps=2` without runtime errors
- Final logged line: `Epoch 1 | Step 2/2 | Train Loss: 10.6214 | Time: 23.84s | T/s: 497.20`

### Run 2: Non-compile + regular activation checkpointing smoke

**Command:**
```bash
source .venv/bin/activate && python train.py --config /tmp/train.small_debug.momh.ac_regular_nocompile.yaml
```

**Results:**
- Logged: `Compile disabled.`
- Completed `max_training_steps=1` without runtime errors
- Final logged line: `Epoch 1 | Step 1/1 | Train Loss: 10.8127 | Time: 23.69s | T/s: 243.04`

### Run 3: Non-compile + selective guard validation

**Command:**
```bash
source .venv/bin/activate && python train.py --config /tmp/train.small_debug.momh.ac_smoke.compile_false.yaml
```

**Results:**
- Expected fail-fast behavior confirmed:
  `ValueError: Selective activation checkpointing requires train.compile=True. Set vlm.activation_checkpointing_mode: regular or enable compile.`

## Analysis

### What Worked
- Activation checkpointing mode wiring in LM and ViT block loops executed cleanly.
- Compile/selective path and non-compile/regular path both executed end-to-end.
- Contract guard for invalid non-compile/selective configuration triggers early and clearly.

### What Failed
- None in integration scope. Smoke runs completed as expected for valid configurations.

### Key Insights
1. A small helper abstraction is sufficient to keep AC integration simple and explicit.
2. Binding block closures (`_block=block`) avoids late-bound block capture risks during checkpoint recomputation.
3. Guarding mode/compile compatibility early in `train()` prevents expensive setup before config errors surface.

## Next Steps

- [ ] Compare memory usage (peak VRAM) between `regular` and `selective` on same step budget.
- [ ] Benchmark throughput impact after warmup on a longer fixed-shape run.
- [ ] Track final training loss parity against prior non-AC baseline on full debug run.
