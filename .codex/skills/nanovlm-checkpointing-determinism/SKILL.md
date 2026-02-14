---
name: nanovlm-checkpointing-determinism
description: >
  Deterministic continuation for nanoVLM training using local full-state checkpoints.
  Use when: exact step-for-step replay is required after resume.
metadata:
  short-description: "Deterministic continue_from_checkpoint workflow"
  tags:
    - checkpointing
    - determinism
    - training
  domain: research
  created: 2026-02-02
  updated: 2026-02-14
  author: codex
---

# nanoVLM Checkpointing Determinism

## General Description

This skill captures the validated deterministic continuation recipe for nanoVLM.
Exact replay requires restoring model/optimizer/RNG/counters and restoring the
in-epoch dataloader cursor before continuing updates.

## When to Apply

Use this knowledge when:
- You need to resume training on preemptible hardware.
- You must compare baseline vs resumed runs step-by-step.
- You need deterministic restart for debugging or ablation comparisons.

Do NOT use when:
- Training uses streaming datasets (`stream_dataset=True`) and exact determinism is required.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Step 21/22/40 replay match | Exact | Base (`h83hz0vi`) vs Replay A (`p62u88oo`) vs Replay B (`99qaejjn`) matched `train/step_loss` and `train/batch_loss` |
| Step 40 aligned loss | `7.6042018` / `7.8357873` | `train/step_loss` / `train/batch_loss` identical across all runs |
| Stream dataset caveat | Best-effort only | `stream_dataset=true` cannot guarantee exact replay |

## Recommended Practice

### Step 1: Use deterministic data mode for replay checks

- Set `stream_dataset=False`.
- Keep schedule config identical (`max_training_steps` or token-based schedule fields).

### Step 2: Save full-state checkpoints

Use local full-state checkpoints with retention:

```yaml
train:
  checkpoint_unit: steps
  checkpoint_interval: 20
  checkpoint_dir: checkpoints
  keep_last_n_checkpoints: 3
```

### Step 3: Continue from a checkpoint with full state

```yaml
train:
  continue_from_checkpoint: /path/to/checkpoints/<run>/step-00000020-tokens-000000XXXXXX
  resume_from_vlm_checkpoint: false
```

### Step 4: Enforce cursor restore invariant

Full-state payload must include and restore:
- `step`, `consumed_tokens`, `epoch`
- optimizer state
- RNG state (`python`, `numpy`, `torch`, `torch_cuda`)
- `microbatches_seen_in_epoch` (for in-epoch dataloader cursor restore)

At resumed epoch start:
- fast-forward train iterator by `microbatches_seen_in_epoch`
- only then continue optimization steps

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Replay diverges from original after resume | In-epoch dataloader cursor not restored | Save + restore `microbatches_seen_in_epoch` and fast-forward iterator on resumed epoch |
| Continuation startup fails with mutual-exclusion error | `resume_from_vlm_checkpoint=true` with `continue_from_checkpoint` also set | Set exactly one mode: model-only init or full-state continuation |
| Non-deterministic continuation with streaming data | Streaming input order is not replayable | Use `stream_dataset=false` for strict replay checks |

## Configuration

```yaml
train:
  stream_dataset: false
  checkpoint_unit: steps
  checkpoint_interval: 20
  checkpoint_dir: checkpoints
  keep_last_n_checkpoints: 3
  continue_from_checkpoint: null
  resume_from_vlm_checkpoint: false
```

## References

- Related logs: `references/experiment-log.md`, `references/troubleshooting.md`
- Related code: `train.py`, `train_utils/checkpointing.py`
