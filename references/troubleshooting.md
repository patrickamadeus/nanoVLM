# Troubleshooting Guide

This file tracks recurring training/runtime issues and their validated fixes.

## Common Issues

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Packed-sample leakage with MoMH | Packed training shows cross-sample context bleed (later packed samples can use earlier sample context) and unstable/noisy behavior vs expected isolated packing | Packing concatenates samples with separator tokens (`attention_mask=0`, `labels=-100`), but attention/MoMH masks do not currently enforce same-segment isolation | Add segment-aware masking so query/key tokens must be in the same packed segment; keep this check in both standard attention and MoMH paths |
| Full-model torch.compile slowdown in DualTower | Compiled run is far slower than eager and can hit `torch._dynamo.config.recompile_limit`, with guard failures like `len(images[0]) == 1` | Compiling the top-level DualTower wrapper captures Python-list image structure from packed batches, causing frequent guard failures and recompiles | Do not compile the top-level model. Use regional compile on tensor-stable decoder blocks only, and keep variable image-count paths (`vision_encoder`, `MP`, wrapper forwards) eager |
| Residual decoder warmup recompiles with regional compile | `TORCH_LOGS="recompiles"` shows early recompiles like `tensor 'x' requires_grad mismatch` and `block_kv_cache is None` | Left/right towers execute the same decoder block code with different grad/cache semantics, so Dynamo specializes separate variants during warmup | Accept the one-time warmup recompiles, keep `mark_dynamic` for `(B,T)` tensors to avoid shape recompiles, and prioritize longer runs where warmup is amortized |
| Selective activation checkpointing fails on non-compile runs | Training exits before dataloader/model init with `ValueError: Selective activation checkpointing requires train.compile=True` | Project contract uses selective AC only for compile flows; non-compile flow is regular checkpointing only | Set `vlm.activation_checkpointing_mode: regular` when `train.compile: false`, or enable `train.compile: true` to use selective mode |
| Effective-token ratio appears unexpectedly low | Ratio near `0.1-0.2` despite packing expected to help | Metric mixed supervision density (`labels != -100`) with packing utilization, which does not match the intended effective-token definition | Use a single definition: effective tokens are non-pad tokens (`attention_mask==1`). Track `train/step_effective_token_ratio = step_effective_tokens / step_token_capacity` |
