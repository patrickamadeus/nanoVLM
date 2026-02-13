---
name: momh-packed-segment-isolation
description: >
  Prevent cross-sample attention leakage when using MoMH with packed sequences.
  Use when: multiple samples are concatenated into one sequence with separators.
metadata:
  short-description: "Segment-aware masking for MoMH under packing"
  tags:
    - momh
    - packing
    - attention-mask
    - vlm
    - stability
  domain: research
  created: 2026-02-13
  author: codex
---

# MoMH Packed Segment Isolation

## General Description

This skill captures a masking invariant for MoMH when document packing is enabled.
MoMH already handles multi-image and interleaved modality tokens via `is_vision`, but packed
sequences can still leak context across sample boundaries unless we also enforce segment isolation.

## When to Apply

Use this knowledge when:
- Training uses packed sequences where multiple samples are concatenated into one tensor row.
- Separators are represented by `attention_mask=0` gaps between content spans.
- You need each packed sample to behave like an independent causal sequence.

Do NOT use when:
- You intentionally want cross-sample context sharing inside packed rows.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Root-cause confidence | High | Reproduced by code-path inspection in packed + MoMH flow |
| Scope | Structural correctness | Applies to both standard attention and MoMH head rules |
| Expected effect | Cross-sample leakage removed | Later packed samples cannot attend earlier packed samples |

## Recommended Practice

Enforce a **same-segment** predicate in all attention rules.

### Step 1: Build per-token segment IDs from `attention_mask`

- Treat each contiguous `attention_mask==1` span as one segment.
- Assign `-1` to padding/separator tokens.

Example approach:

```python
valid = attention_mask.to(torch.bool)                    # [B, T]
prev_valid = torch.nn.functional.pad(valid[:, :-1], (1, 0), value=False)
seg_start = valid & (~prev_valid)                        # segment boundary markers
segment_id = seg_start.cumsum(dim=1) - 1                 # [B, T]
segment_id = segment_id.masked_fill(~valid, -1)
```

### Step 2: Add `same_segment` to masking conditions

- For query index `q_abs` and key index `kv_abs`, require:
  - `segment_id[b, q_abs] == segment_id[b, kv_abs]`
  - segment id is non-negative.
- Apply this gate to:
  - standard attention causal/padding path
  - MoMH V/T/VT head masks.

### Step 3: Keep modality logic unchanged

- Continue using `is_vision` from token IDs to classify modality.
- Segment isolation is an extra invariant; it should not replace modality masking.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Using only `attention_mask` (content/pad) with packing | It does not encode sample identity | Content gating is necessary but insufficient for packed isolation |
| Assuming MoMH modality mask implies sample isolation | `is_vision` labels modality, not ownership | Add explicit segment-aware constraint |
| Validating only loss curves | Leakage can be subtle and still train | Add structural mask tests across segment boundaries |

## Configuration

```yaml
# Recommended when packing + MoMH
train:
  use_packing: true

vlm:
  momh_enabled: true

# Add an implementation flag once available:
# vlm.enforce_pack_segment_isolation: true
```

## References

- Related log entry: `references/experiment-log.md`
- Related troubleshooting: `references/troubleshooting.md`
- Relevant code: `data/advanced_datasets.py`, `models/vision_language_model.py`, `models/language_model.py`
