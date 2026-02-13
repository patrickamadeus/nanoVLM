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
