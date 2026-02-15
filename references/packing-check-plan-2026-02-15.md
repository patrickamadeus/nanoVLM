# Packing Check Plan: Diagnose Low MoMH Performance via Response-Only Target Patterns

## Summary

This plan focuses first on a **new standalone debug script** (no training-code mutation) to measure whether packing distorts the **response-only supervision pattern** (`labels != -100`) in nanoVLM, then runs a fast set of controlled ablations to isolate root cause.

Grounded findings from current repo/run state:

- `dt6j0j2n` currently logs `effective_token_ratio ~0.91` in `wandb/run-20260214_215807-dt6j0j2n/files/output.log`, so non-pad utilization is not low by current definition.
- Current `effective_token_ratio` metric in `train.py` is based on `attention_mask` (non-pad tokens), not response targets.
- MoMH mask path in `models/momh_attention.py` has no explicit same-segment isolation for packed samples; this can allow cross-sample leakage (already consistent with `eval/check_momh_mask.py` intent and `momh-packed-segment-isolation` skill).
- MMStar scoring risk exists: it relies on first output character; your MoMH eval sample file has many non-letter starts, which can hide/warp interpretation.

## Scope and Constraints

- First implementation pass is **debugging script only** (as requested), no direct masking patch yet.
- Investigation budget: **fast sweep** (short runs, 40–200 steps).
- GPU-only execution: every runnable command should hard-fail if CUDA is unavailable.

## Important Interface/Additions (Decision-Complete)

### 1) New standalone script
Create `eval/inspect_packing_supervision.py` with CLI:

- `--config <path>` required
- `--num-batches <int>` default `32`
- `--split train|val` default `train`
- `--max-qkv <int>` default `512`
- `--seed <int>` default `0`
- `--output-json <path>` optional
- `--fail-if-no-cuda` default `true`

### 2) Script outputs (JSON + stdout summary)
For each analyzed batch and aggregated totals, report:

- `token_capacity` = `B*T`
- `content_tokens` = `(attention_mask==1).sum()`
- `target_tokens` = `(labels!=-100).sum()`
- `content_ratio` = `content_tokens/token_capacity`
- `target_ratio` = `target_tokens/token_capacity`
- `target_among_content_ratio` = `target_tokens/content_tokens`
- `segments_per_row` distribution derived from contiguous `attention_mask==1` spans
- `target_tokens_per_segment` stats
- `rows_with_zero_targets` count
- `rows_with_multi_segments_and_targets` count

MoMH leakage diagnostics on sampled positions (reusing same logic style as `eval/check_momh_mask.py`):

- `cross_segment_allowed_count_all_queries`
- `cross_segment_allowed_count_target_queries_only`
- `cross_segment_allowed_ratio_target_queries`
- `headwise_cross_segment_counts` for V/T/VT groups

### 3) No train.py modification in this cycle
Do not add W&B logging keys yet. Keep this cycle script-only to avoid touching active training pipeline.

## Implementation Plan

1. **Script scaffold and safety checks**
- Implement CUDA availability guard first.
- Reuse config loading and dataloader path from `train.py` without refactoring training code.
- Reuse segment-id derivation from `eval/check_momh_mask.py` (or duplicate minimal helper in script for independence).

2. **Supervision-pattern analysis**
- Compute target/content ratios from actual packed batches.
- Add per-row/per-segment breakdown so we can inspect “response-only pattern collapse” directly.
- Include top-K worst rows (lowest `target_ratio`) in output for quick triage.

3. **MoMH cross-segment query analysis**
- Build `is_vision` from `input_ids == image_token_id`.
- Instantiate MoMH mask mod with current head fractions from config.
- Measure leakage specifically for **target query positions** (`labels!=-100`) to test your primary hypothesis.

4. **Fast-sweep experiments (2–5 runs)**

| id | description | key_differences | notes |
|----|-------------|-----------------|-------|
| P1 | Packed supervision profile | `use_packing=true` current MoMH config; run script for `num_batches=64` | Primary diagnostic; expected ~5–10 min |
| P2 | Non-packed control profile | identical config but `use_packing=false`; same script settings | Isolates packing effect on target ratios |
| P3 | Packed + mask checker | run `eval/check_momh_mask.py --mode dataloader` on current config | Confirms segment leakage count presence/size |
| P4 | Short train ablation A | 40–100 update steps, `use_packing=true`, same MoMH | Correlate script metrics with early loss/eval |
| P5 | Short train ablation B | same as P4 but `use_packing=false` | If large gap appears quickly, packing is likely dominant |

5. **Secondary evaluation sanity check**
- Parse MMStar sample jsonl for first-char validity distribution (`A/B/C/D/other`) and report with each checkpoint eval.
- Keep existing scorer unchanged for now; this is interpretation hygiene, not evaluator rewrite.

## Test Cases and Validation Scenarios

1. **Synthetic invariant test (unit-like)**
- Build a tiny fake batch with two segments and known target positions.
- Verify segment stats and ratios are exact.

2. **Mask leak sanity test**
- On synthetic packed example, ensure script reports `cross_segment_allowed_count_target_queries_only > 0` for current mask logic (expected current behavior).

3. **Non-packed regression test**
- With `use_packing=false`, script should report one segment per row and near-zero cross-segment counts.

4. **Empty/degenerate safeguards**
- If any batch has zero target tokens, script should report and continue; if all batches zero, exit non-zero with explicit error.

## Runtime / Cost Expectations (Fast Sweep)

- `P1/P2/P3` diagnostics: ~5–15 minutes each (mostly dataloader + mask compute).
- `P4/P5` short training:
  - 40 update steps: roughly 6–10 minutes per run.
  - 100 update steps: roughly 15–25 minutes per run.
- Total first-cycle wall time: ~1.5 to 3 hours depending on chosen step counts.

## Defaults and Assumptions

- Default baseline is `configs/train.current.momh.compile-selective.yaml`.
- All execution uses:
  - `source .venv/bin/activate`
  - GPU required (`torch.cuda.is_available()` enforced by script)
- Current reference run (`dt6j0j2n`) interpreted as of **2026-02-15** local artifacts.
- We treat “low effective token ratio” concern as likely meaning **low target supervision density**, not low non-pad density.

## Success Criteria

- We can answer, with numbers:
  1. Is packing reducing `target_ratio` or `target_among_content_ratio` versus non-packed?
  2. Are MoMH target queries allowed to attend across packed segments?
  3. Do short ablations show early performance divergence consistent with these diagnostics?
- If yes, next cycle can be a targeted mask patch (segment isolation) plus minimal train-time logging additions.
