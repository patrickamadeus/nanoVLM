---
name: hybrid-step-token-scheduling-nanovlm
description: >
  Configure nanoVLM training cadence using explicit step- or token-based units for stop, eval, and checkpoint.
  Use when: variable effective-token throughput makes step-only cadence misleading.
metadata:
  short-description: "Hybrid steps/tokens scheduling for training cadence"
  tags:
    - scheduling
    - tokens
    - checkpoints
    - evaluation
  domain: research
  created: 2026-02-14
  author: codex
---

# Hybrid Step/Token Scheduling (nanoVLM)

## General Description

This skill captures the schedule unit system for nanoVLM training where stop,
evaluation, and checkpoint cadence can each be driven by either optimizer steps
or consumed effective tokens.

It prevents missed token-trigger events by using threshold-crossing logic
instead of modulo checks on token counts.

## When to Apply

Use this knowledge when:
- Batch token counts vary significantly across updates.
- You want eval/checkpoint cadence tied to token budget, not step count.
- You need explicit and reproducible schedule semantics in config.

Do NOT use when:
- You need the simplest legacy behavior and default step-based cadence is sufficient.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Scheduling surface | 3 independent units | `stop_unit`, `eval_unit`, `checkpoint_unit` |
| Trigger robustness | Threshold-based | Handles multiple token interval crossings in one update |
| Backward compatibility | Preserved | Defaults remain step-based |

## Recommended Practice

### Option A: Fully step-based (default-like)

```yaml
train:
  stop_unit: steps
  max_training_steps: 40000
  eval_unit: steps
  eval_interval: 500
  checkpoint_unit: steps
  checkpoint_interval: 500
```

### Option B: Token-driven eval/checkpoint

```yaml
train:
  stop_unit: steps
  max_training_steps: 40000
  eval_unit: tokens
  eval_interval_tokens: 5000000
  checkpoint_unit: tokens
  checkpoint_interval_tokens: 10000000
```

### Option C: Full token budget control

```yaml
train:
  stop_unit: tokens
  max_training_tokens: 250000000
  eval_unit: tokens
  eval_interval_tokens: 5000000
  checkpoint_unit: tokens
  checkpoint_interval_tokens: 10000000
```

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Token-based events were skipped | Modulo checks miss threshold crossings when token increments are large | Use threshold-advancement trigger logic |
| Invalid config silently ran | Missing interval/budget for selected unit | Fail-fast validation on unit-specific required fields |
| Resume changed schedule behavior | Schedule units differed from checkpoint metadata | Validate and reject continuation when units mismatch |

## Configuration

```yaml
train:
  stop_unit: steps              # steps | tokens
  max_training_steps: 40000     # required when stop_unit=steps
  max_training_tokens: null     # required when stop_unit=tokens
  eval_unit: steps              # steps | tokens
  eval_interval: 500            # required when eval_unit=steps
  eval_interval_tokens: null    # required when eval_unit=tokens
  checkpoint_unit: steps        # steps | tokens
  checkpoint_interval: 500      # required when checkpoint_unit=steps
  checkpoint_interval_tokens: null
```

## References

- Related code: `train.py`, `models/config.py`
- Related tests: `tests/test_training_schedule.py`
- Related docs: `README.md`, `references/troubleshooting.md`, `references/experiment-log.md`
