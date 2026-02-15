# AGENTS.md — Project Operating Contract

> **Domain:** research
> **Created:** 2026-01-13

This document defines the operational contract for AI agents working in this project.

---

## 1. Project Context

**Purpose:** I am training momH right now which is currently running. momH is where we differentiate the attention patters for multimodal inputs. By differentiating the heads of which modality they attend to, we hope that this will increase the modality imbalance performance and also the overall performance of the model.

The settings right now is that we load the checkpoint of the vanilla nanoVLM and then we finetune it with the momH attention pattern. We would expect that the performance will go down first and then it will go up as the model learns to use the new attention pattern. We are currently at the point where the performance is still going down but we hope that it will start going up soon.

But currently we have suspicion since the performance is too low. We are thinking that packing is the causes here. So I want you to help me investigate it. It might be incompatible with the response only style of nanoVLM

Please use MCP to see the reference run with low ratio of effective token :
1. https://wandb.ai/patrickirawan-mbzuai/momh/runs/dt6j0j2n

This is downstream performance for that run :
1. /home/yovakementchedjhieva/users/patrick/nanoVLM_root/refactor/eval_results/momh-2k1img-step-2800/20260215_172854_samples_mmstar.jsonl
2. /home/yovakementchedjhieva/users/patrick/nanoVLM_root/refactor/eval_results/momh-2k1img-step-2800/20260215_172854_results.json

Note that for vanilla model but we changed the attention to momH. The result is :
1. /home/yovakementchedjhieva/users/patrick/nanoVLM_root/refactor/eval_results/nanoVLM-230M-8k/20260215_033305_samples_mmstar.jsonl
2. /home/yovakementchedjhieva/users/patrick/nanoVLM_root/refactor/eval_results/nanoVLM-230M-8k/20260215_033305_results.json


Never use CPU in any way! Contant user if something's wrong with the GPU. But never EVER run in CPU because it keeps crashing my machine.

**Domain:** ML research and experimentation

---

## 2. Directory Structure

```
./
├── .codex/skills/          # Local skills (readable by Codex)
│   ├── registry.json       # Domain metadata and skill index
│   ├── domain-advisor/     # Planning assistance
│   └── domain-retrospective/  # Knowledge capture
├── .claude/
│   └── skills/             # Symlink to .codex/skills/ (for Claude Code)
├── .claude-plugin/
│   └── plugin.json         # Claude Code plugin manifest
├── references/
│   ├── experiment-log.md   # Chronological log
│   └── troubleshooting.md  # Error patterns and fixes
├── templates/              # Document templates
├── training_reports/          # Experiment/benchmark reports
└── AGENTS.md               # This file
```

---

## 3. Skill Commands

### `<advise>`

Invoke when planning new experiments or development tasks.

**Behavior:**
1. Reads `registry.json` to determine domain
2. Scans existing skills and reports for relevant context
3. Proposes 2-5 concrete experiments/tasks
4. Outputs markdown table with key differences

### `<retrospective>`

Invoke after completing experiments to capture learnings.

**Behavior:**
1. Reads specified reports from `training_reports/`
2. Summarizes: what worked, what failed, key insights
3. Proposes new result skills or troubleshooting entries
4. Only writes files with user approval

---

## 4. Documentation Rules

### Reports

- Store in `training_reports/`
- Use template: `templates/reports/report-template.md`
- Name format: `{description}-{YYYY-MM-DD}.md`

### Experiment Log

- Location: `references/experiment-log.md`
- Append entries chronologically
- Include: date, type, general description, details

### Troubleshooting

- Location: `references/troubleshooting.md`
- Add new error patterns as discovered
- Include: symptom, cause, solution

### Result Skills

- Location: `.codex/skills/{skill-name}/SKILL.md`
- Use template: `templates/skills/result-skill-template.md`
- Must include: description, when to apply, failure modes

---

## 5. Workflow

1. **Start session**: Type `<advise>` to get planning suggestions
2. **Execute**: Run experiments/development tasks
3. **Document**: Create reports in `training_reports/`
4. **Capture**: Type `<retrospective>` to distill learnings
5. **Iterate**: Use new skills in next `<advise>` cycle

---

## 6. Domain-Specific Notes

- Focus on reproducibility: log all hyperparameters
- Track dataset versions and preprocessing
- Document negative results - they prevent repeated mistakes

---

## 7. Conventions

- **File naming:** lowercase with hyphens (e.g., `my-experiment.md`)
- **Skill naming:** `{topic}-{finding}` (e.g., `lora-rank-optimal`)
- **Dates:** YYYY-MM-DD format
- **Configs:** YAML or JSON, copy-paste ready

## 8. Debugging
For debugging that is not affecting the loss (eg. compile), we can reduce the model size to make our iterations faster. Suggested configs that we can change on `./models/config.py`:
```
vit_hidden_dim: 256,
vit_inter_dim: 1024,
vit_patch_size: 16,
vit_img_size: 128,
vit_n_heads: 4,
vit_dropout: 0.0,
vit_n_blocks: 4,
lm_hidden_dim: 384,
lm_inter_dim: 1024,
lm_n_heads: 6,
lm_n_kv_heads: 2,
lm_n_blocks: 8,
lm_max_length: 1024,
lm_tie_weights: true,
```

## 9. MoMH Preflight Plan (Before Long Finetuning)

Run this exact order before starting long MoMH finetuning jobs.

### P1. Checkpoint load sanity (nanoVLM checkpoint)

- Config: `configs/train.preflight.momh.checkpoint-load.yaml`
- Goal: verify `resume_from_vlm_checkpoint` path and state transfer to the current nanoVLM MoMH run without runtime/load errors.
- Command:
  - `export HF_HOME=/workspace/huggingface && source .venv/bin/activate && python train.py --config configs/train.preflight.momh.checkpoint-load.yaml`
- Pass criteria:
  - Training starts and reaches step 2.
  - No checkpoint/key mismatch errors.
  - Loss is finite.

### P2. MoMH mask invariants

- Use the dataloader-mode mask checker on preflight config.
- Command:
  - `export HF_HOME=/workspace/huggingface && source .venv/bin/activate && python eval/check_momh_mask.py --mode dataloader --config configs/train.preflight.momh.stability.yaml`
- Pass criteria:
  - `padding_masking_ok`, `v_head_rules_ok`, `t_head_rules_ok`, `vt_head_rules_ok` are true.
  - Inspect `cross_segment_allowed_count`:
    - `0` means packing is segment-isolated.
    - `>0` means packed-sample leakage risk is present; either disable packing for finetuning or fix segment-aware masking first.

### P3. Non-compile MoMH stability baseline

- Config: `configs/train.preflight.momh.stability.yaml`
- Goal: establish stable loss/grad baseline before compile overhead is introduced.
- Command:
  - `export HF_HOME=/workspace/huggingface && source .venv/bin/activate && python train.py --config configs/train.preflight.momh.stability.yaml`
- Pass criteria:
  - Full 40-step run completes.
  - No NaN/Inf in loss or grad norm.
  - Throughput and loss curve look stable.

### P4. Compile + selective AC preflight (optional but recommended)

- Config: `configs/train.preflight.momh.compile-selective.yaml`
- Goal: validate compile behavior and recompilation profile before long run.
- Command:
  - `export HF_HOME=/workspace/huggingface && source .venv/bin/activate && TORCH_LOGS="recompiles" python train.py --config configs/train.preflight.momh.compile-selective.yaml`
- Pass criteria:
  - Run completes without compile/runtime failures.
  - Recompiles are warmup-only (no persistent shape thrash).
  - Loss trend is close to P3 baseline.

### Go / No-Go Gate

Start long MoMH finetuning only if P1-P3 pass.
- If `use_packing=true`, require `cross_segment_allowed_count == 0` from P2.
- If using compile for the long run, P4 must also pass.

## 10. RunPod Auto-Stop Wrapper

When running on RunPod, use the wrapper script so training stops the pod automatically after the command ends.

- Script: `./runpod_train_and_stop.sh`
- Behavior:
  - activate `.venv`
  - export `HF_HOME` (defaults to `/workspace/huggingface` in the wrapper)
  - run the training command
  - always call `runpodctl stop pod <pod_id>` at the end

### Usage

- Default training command:
  - `./runpod_train_and_stop.sh "$RUNPOD_POD_ID"`
- Explicit config:
  - `./runpod_train_and_stop.sh "$RUNPOD_POD_ID" -- python train.py --config configs/train.preflight.momh.stability.yaml`
- Compile preflight:
  - `./runpod_train_and_stop.sh "$RUNPOD_POD_ID" -- env TORCH_LOGS="recompiles" python train.py --config configs/train.preflight.momh.compile-selective.yaml`

If training fails, the script still attempts pod stop, then returns a non-zero exit code.
