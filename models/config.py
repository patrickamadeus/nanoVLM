from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 2048
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_pad_aware_rope: bool = False
    lm_max_length: int = 2048
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M-Instruct' #'HuggingFaceTB/SmolLM2-135M' #
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 2048
    resize_to_max_side_len: bool = False

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>", "global_image_token": "<|global_image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
      "r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
      "r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
      "r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
      "r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"})
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'lusxvr/nanoVLM-230M-8k'
    hf_repo_name: str = 'nanoVLM'
    left_mask_scope: str = "full"  # visual_only | visual_sys | full
    right_prefill_mode: str = "full"  # full | non_donor_only (generation only)
    use_right_kv_cache_gates: bool = False
    right_kv_gate_normalize: bool = False
    right_kv_gate_init_offdiag: float = 1e-3
    use_kv_bridge: bool = True
    kv_bridge_type: str = "mlp"  # linear | mlp | scaled_linear | residual_nonlinear
    kv_bridge_mlp_ratio: float = 4.0
    kv_bridge_use_rmsnorm: bool = True
    kv_bridge_residual: bool = True
    kv_bridge_init_mode: str = "default"  # default | normal | diag_eye
    kv_bridge_linear_depth: int = 1
    kv_bridge_adapter_depth: int = 2
    kv_bridge_adapter_expansion: float = 1.0


@dataclass
class TrainConfig:
    lr_mp: float = 5e-5
    lr_vision_backbone: float = 0 #0.0005 #
    lr_language_backbone: float = 1e-5 #0
    # DualTower-specific explicit LR controls. If None, falls back to lr_language_backbone.
    lr_left_tower: float | None = None
    lr_right_tower: float | None = 0.0
    lr_right_kv_gates: float | None = 1e-4
    lr_kv_bridge: float | None = 1e-4
    val_size: int = 50000  # Deprecated when using explicit train/val splits.
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = False
    eval_interval: int = 500
    checkpoint_interval: int = 200
    stats_log_interval: int = 100
    max_training_steps: int = 20_000
    warmup_ratio: float = 0.03
    max_images_per_example: int = 1
    max_images_per_knapsack: int = 18
    max_sample_length: int = 2048
    use_packing: bool = True
    compile: bool = True
    resume_from_vlm_checkpoint: bool = True # Continue training from a full VLM checkpoint.
    train_dataset_path: str = 'patrickamadeus/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("all", ) #('allava_laion', 'allava_vflan', 'cambrian(filtered)_processed', 'LLaVA_Instruct_150K', 'mmevol', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)') # 'vision_flan(filtered)', 'lvis_instruct4v',
    train_split: str = "train"
    val_split: str = "validation"
    stream_dataset: bool = False
    enable_source_filter: bool = False
    allowed_dataset_sources: tuple[str, ...] = ()
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1
    wandb_entity: str = "HuggingFace" # Indicate the entity to log to in wandb
    log_wandb: bool = True
    use_lmms_eval: bool = False # Disabled for hub-only checkpoint workflow.
    lmms_eval_tasks: str = 'mmstar' # Pass additional task as one string, seperated by commas without spaces (e.g. 'mmstar,mmmu,ocrbench')
    lmms_eval_limit: float = 100
    lmms_eval_batch_size: int = 64
    push_checkpoints_to_hub: bool = True
    save_training_state_to_hub: bool = True
    checkpoint_repo_pattern: str = "patrickamadeus/dualtower-full-step-{i}"
    hf_private: bool = False
    push_final_model_to_hub: bool = False
