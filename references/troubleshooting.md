# Troubleshooting Guide

This file tracks recurring training/runtime issues and their validated fixes.

## Common Issues

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Packed-sample leakage with MoMH | Packed training shows cross-sample context bleed (later packed samples can use earlier sample context) and unstable/noisy behavior vs expected isolated packing | Packing concatenates samples with separator tokens (`attention_mask=0`, `labels=-100`), but attention/MoMH masks do not currently enforce same-segment isolation | Add segment-aware masking so query/key tokens must be in the same packed segment; keep this check in both standard attention and MoMH paths |

