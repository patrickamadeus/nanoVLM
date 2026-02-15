import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import tempfile
from dataclasses import asdict
from safetensors.torch import load_model as load_safetensors, save_model
from models.language_model import LanguageModel
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from models.utils import top_k_top_p_filtering
from train_utils.console import rank0_print
from huggingface_hub import create_repo, hf_hub_download, upload_folder



class LeftTower(VisionLanguageModel):
    def __init__(
        self,
        cfg: VLMConfig,
        *,
        load_backbone: bool = True,
        freeze_vision_encoder: bool = False,
        freeze_modality_projector: bool = False,
        freeze_language_decoder: bool = False,
    ):
        super().__init__(cfg, load_backbone=load_backbone)
        if freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        if freeze_modality_projector:
            for p in self.MP.parameters():
                p.requires_grad = False
        if freeze_language_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False


    def forward(
        self, 
        input_ids: torch.Tensor, 
        images, 
        attention_mask: torch.Tensor = None,
    ):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
        
        _, kv_cache = self.decoder(token_embd, attention_mask=attention_mask)
        
        return None, kv_cache


class RightTower(LanguageModel):
    def __init__(self, cfg: VLMConfig, *, load_backbone: bool = True, freeze_decoder: bool = False):
        if load_backbone:
            lm = LanguageModel.from_pretrained(cfg)
            super().__init__(cfg)
            self.load_state_dict(lm.state_dict())
            del lm
        else:
            super().__init__(cfg)
        
        if freeze_decoder:
            for p in self.parameters():
                p.requires_grad = False


class HeadDimRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


class KVBridgeLayer(nn.Module):
    def __init__(
        self,
        head_dim: int,
        *,
        bridge_type: str,
        mlp_ratio: float,
        use_rmsnorm: bool,
        residual: bool,
    ):
        super().__init__()
        self.bridge_type = bridge_type
        self.residual = residual
        self.norm_k = HeadDimRMSNorm(head_dim) if use_rmsnorm else nn.Identity()
        self.norm_v = HeadDimRMSNorm(head_dim) if use_rmsnorm else nn.Identity()

        if bridge_type == "linear":
            self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
            self.v_proj = nn.Linear(head_dim, head_dim, bias=False)
            if residual:
                nn.init.zeros_(self.k_proj.weight)
                nn.init.zeros_(self.v_proj.weight)
            else:
                nn.init.eye_(self.k_proj.weight)
                nn.init.eye_(self.v_proj.weight)
        elif bridge_type == "mlp":
            hidden_dim = max(1, int(round(head_dim * mlp_ratio)))
            self.k_fc1 = nn.Linear(head_dim, hidden_dim, bias=False)
            self.k_fc2 = nn.Linear(hidden_dim, head_dim, bias=False)
            self.v_fc1 = nn.Linear(head_dim, hidden_dim, bias=False)
            self.v_fc2 = nn.Linear(hidden_dim, head_dim, bias=False)
            if residual:
                # Near-identity start with non-zero gradient flow through residual MLP branch.
                nn.init.normal_(self.k_fc2.weight, mean=0.0, std=1e-4)
                nn.init.normal_(self.v_fc2.weight, mean=0.0, std=1e-4)
        else:
            raise ValueError(f"Unsupported kv_bridge_type={bridge_type!r}. Expected one of ['linear', 'mlp']")

    def _project(self, x: torch.Tensor, is_key: bool) -> torch.Tensor:
        if self.bridge_type == "linear":
            return self.k_proj(x) if is_key else self.v_proj(x)

        if is_key:
            return self.k_fc2(F.silu(self.k_fc1(x)))
        return self.v_fc2(F.silu(self.v_fc1(x)))

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_in = self.norm_k(k)
        v_in = self.norm_v(v)

        k_out = self._project(k_in, is_key=True)
        v_out = self._project(v_in, is_key=False)

        if self.residual:
            return k + k_out, v + v_out
        return k_out, v_out


class KVCacheBridge(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        bridge_type = getattr(cfg, "kv_bridge_type", "linear")
        if bridge_type not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported kv_bridge_type={bridge_type!r}. Expected one of ['linear', 'mlp']")

        head_dim = cfg.lm_hidden_dim // cfg.lm_n_heads
        self.layers = nn.ModuleList([
            KVBridgeLayer(
                head_dim,
                bridge_type=bridge_type,
                mlp_ratio=getattr(cfg, "kv_bridge_mlp_ratio", 2.0),
                use_rmsnorm=getattr(cfg, "kv_bridge_use_rmsnorm", True),
                residual=getattr(cfg, "kv_bridge_residual", True),
            )
            for _ in range(cfg.lm_n_blocks)
        ])

    def forward(self, kv_cache: list[dict]) -> list[dict]:
        if len(kv_cache) != len(self.layers):
            raise ValueError(
                f"KV bridge expected {len(self.layers)} layers, got {len(kv_cache)}."
            )

        for layer_idx, (bridge_layer, layer_cache) in enumerate(zip(self.layers, kv_cache)):
            if layer_cache is None:
                raise ValueError(f"KV bridge expected cache dict at layer {layer_idx}, got None.")

            key = layer_cache.get("key")
            value = layer_cache.get("value")
            if key is None or value is None:
                raise ValueError(f"KV bridge expected 'key' and 'value' at layer {layer_idx}.")

            bridged_key, bridged_value = bridge_layer(key, value)
            layer_cache["key"] = bridged_key
            layer_cache["value"] = bridged_value

        return kv_cache


class DualTowerVLM(nn.Module):
    def __init__(
        self,
        cfg: VLMConfig,
        *,
        load_backbone: bool = True,
        freeze_left_vision: bool = False,
        freeze_left_projector: bool = False,
        freeze_left_decoder: bool = False,
        freeze_right_decoder: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.left_tower = LeftTower(
            cfg,
            load_backbone=load_backbone,
            freeze_vision_encoder=freeze_left_vision,
            freeze_modality_projector=freeze_left_projector,
            freeze_language_decoder=freeze_left_decoder,
        )
        self.right_tower = RightTower(
            cfg,
            load_backbone=load_backbone,
            freeze_decoder=freeze_right_decoder,
        )
        self.left_tower_mask_mode = getattr(cfg, "left_tower_mask_mode", "visual_only")
        self.left_tower_prefill_no_grad = bool(getattr(cfg, "left_tower_prefill_no_grad", False))
        valid_modes = {"visual_only", "visual_plus_prefix", "full"}
        if self.left_tower_mask_mode not in valid_modes:
            raise ValueError(
                f"Unsupported left_tower_mask_mode={self.left_tower_mask_mode!r}. "
                f"Expected one of {sorted(valid_modes)}."
            )
        self.kv_bridge = KVCacheBridge(cfg) if getattr(cfg, "kv_bridge_enabled", False) else None
        self.tokenizer = self.left_tower.tokenizer
        self._kv_replace_token_ids = self._collect_kv_replace_token_ids()

    def _collect_kv_replace_token_ids(self) -> torch.Tensor:
        """
        Collect token ids whose K/V should come from the left tower during dual prefill.
        Includes all visual-structure markers from cfg.vlm_extra_tokens, e.g.:
        - image_token
        - global_image_token
        - row/col locator tokens
        """
        token_ids = set()
        for token in self.cfg.vlm_extra_tokens.values():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                continue
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.add(token_id)
        if not token_ids:
            raise ValueError("No visual token ids collected for dual-tower K/V replacement.")
        return torch.tensor(sorted(token_ids), dtype=torch.long)

    def _annotate_left_kv_cache(self, kv_cache: list[dict], input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Right tower uses this mask to keep left-tower K/V on visual marker token positions.
        # This includes <|image|>, <|global_image|>, and row/col locator tokens.
        if self.kv_bridge is not None:
            kv_cache = self.kv_bridge(kv_cache)

        img_mask = self._build_visual_token_mask(input_ids, attention_mask)
        for layer_cache in kv_cache:
            layer_cache["img_mask"] = img_mask
            layer_cache["needs_dual_prefill"] = True
        return kv_cache

    def _build_visual_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        replace_ids = self._kv_replace_token_ids.to(input_ids.device)
        visual_mask = torch.isin(input_ids, replace_ids)
        if attention_mask is not None:
            visual_mask = visual_mask & attention_mask.to(torch.bool)
        return visual_mask

    def _build_left_tower_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = self.left_tower_mask_mode

        if mode == "full":
            if attention_mask is None:
                return torch.ones_like(input_ids, dtype=torch.long)
            return attention_mask.to(dtype=torch.long)

        visual_mask = self._build_visual_token_mask(input_ids, attention_mask)
        if mode == "visual_only":
            # Left tower is constrained to visual structure tokens only.
            return visual_mask.to(dtype=torch.long)

        # mode == "visual_plus_prefix":
        # For each contiguous valid segment, include the prefix tokens before the first visual token.
        if attention_mask is None:
            valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid_mask = attention_mask.to(torch.bool)

        prefix_mask = torch.zeros_like(valid_mask)
        B, T = input_ids.shape
        for b in range(B):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue

            # Split into contiguous valid spans (packing separators are attention_mask=0 gaps).
            breaks = torch.nonzero((valid_idx[1:] - valid_idx[:-1]) > 1, as_tuple=False).flatten().tolist()
            starts = [valid_idx[0].item()] + [valid_idx[i + 1].item() for i in breaks]
            ends = [valid_idx[i].item() for i in breaks] + [valid_idx[-1].item()]

            for start, end in zip(starts, ends):
                segment_visual = visual_mask[b, start : end + 1]
                segment_visual_idx = torch.nonzero(segment_visual, as_tuple=False).flatten()
                if segment_visual_idx.numel() == 0:
                    continue
                first_visual = start + int(segment_visual_idx[0].item())
                if first_visual > start:
                    prefix_mask[b, start:first_visual] = True

        return (visual_mask | prefix_mask).to(dtype=torch.long)



    def _run_left_tower_prefill(
        self,
        input_ids: torch.Tensor,
        images,
        left_attention_mask: torch.Tensor,
    ) -> list[dict]:
        if self.left_tower_prefill_no_grad:
            if self.training and any(p.requires_grad for p in self.left_tower.parameters()):
                raise ValueError(
                    "left_tower_prefill_no_grad=True cannot be used when left tower is trainable. "
                    "Freeze left tower params or disable left_tower_prefill_no_grad."
                )
            with torch.no_grad():
                _, kv_cache = self.left_tower(
                    input_ids=input_ids,
                    images=images,
                    attention_mask=left_attention_mask,
                )
            return kv_cache

        _, kv_cache = self.left_tower(
            input_ids=input_ids,
            images=images,
            attention_mask=left_attention_mask,
        )
        return kv_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        images,
        attention_mask: torch.Tensor = None,
        targets: torch.Tensor = None,
        loss_reduction: str = "mean",
        return_loss_count: bool = False,
    ):
        left_attention_mask = self._build_left_tower_attention_mask(input_ids, attention_mask)
        # Process the full sequence through left tower, then reuse its image-token K/V in right tower.
        kv_cache = self._run_left_tower_prefill(
            input_ids=input_ids,
            images=images,
            left_attention_mask=left_attention_mask,
        )
        kv_cache = self._annotate_left_kv_cache(kv_cache, input_ids, attention_mask)

        # Process to right tower and use left's kv_cache
        # we need to embed this first (why?) because input_ids being passed is pure token ID, see @train.py / generate.py
        full_embd = self.right_tower.token_embedding(input_ids)
        logits, _ = self.right_tower(
            x=full_embd,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=0,
        )

        # Loss calculation (if any)
        loss = None
        loss_count = None
        if targets is not None:
            logits = self.right_tower.head(logits)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction=loss_reduction,
            )
            if return_loss_count:
                loss_count = (targets != -100).sum()
        
        if return_loss_count:
            if loss_count is None:
                raise ValueError("return_loss_count=True requires targets.")
            return logits, loss, loss_count

        return logits, loss
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        images,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 50,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        greedy: bool = False,
    ):
        """
        Generate tokens autoregressively using the dual tower architecture.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (B, T)
            images: Images to process (can be list or tensor)
            attention_mask (torch.Tensor, optional): Attention mask of shape (B, T)
            max_new_tokens (int): Number of new tokens to generate
            top_k (int): Top-k filtering parameter for sampling
            top_p (float): Top-p (nucleus) filtering parameter for sampling
            temperature (float): Temperature for sampling (higher = more random)
            greedy (bool): If True, use greedy decoding (argmax), otherwise use sampling
        
        Returns:
            torch.Tensor: Generated token IDs of shape (B, max_new_tokens)
        """
        B = input_ids.size(0)
        device = input_ids.device
        
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            B = 1
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        left_attention_mask = self._build_left_tower_attention_mask(input_ids, attention_mask)
        
        # Process left tower to get image KV cache
        kv_cache = self._run_left_tower_prefill(
            input_ids=input_ids,
            images=images,
            left_attention_mask=left_attention_mask,
        )
        kv_cache = self._annotate_left_kv_cache(kv_cache, input_ids, attention_mask)
        
        # Prefill phase: process the FULL sequence with image KV cache
        # The right tower will internally handle replacing image K/V with cached values
        # See modified GQA implementation above for this `LanguageModelGroupedQueryAttention`

        # TODO: below token embedding code is the bypass attempt on self.cfg.lm_use_token, future work may include `if self.cfg.lm_use_tokens` for better conditional in various cases.
        full_embd = self.right_tower.token_embedding(input_ids)
        prompt_output, kv_cache = self.right_tower.forward(
            x=full_embd,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=0  # we start prefill from pos 0
        )
        
        last_output = prompt_output[:, -1, :]
        
        # Get logits from the last token output
        if not self.right_tower.lm_use_tokens:
            current_logits = self.right_tower.head(last_output)
        else:
            current_logits = last_output
        
        newly_generated_ids_list = []
        current_attention_mask = attention_mask.clone()
        
        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            # Sample next token
            if greedy or temperature <= 0:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.right_tower.token_embedding(next_token_id)  # [B, 1, D_lm]

            # Decode position must follow KV-cache index space (includes any padded prefix positions).
            current_token_start_pos = kv_cache[0]["key"].size(2)
            
            # Update attention mask
            new_token_mask = torch.ones((B, 1), 
                                        dtype=current_attention_mask.dtype, 
                                        device=device)
            current_attention_mask = torch.cat([current_attention_mask, new_token_mask], dim=1)
            
            # With KV cache: only process the new token
            decode_step_output, kv_cache = self.right_tower.forward(
                x=next_token_embed,
                attention_mask=current_attention_mask,
                kv_cache=kv_cache,
                start_pos=current_token_start_pos
            )
            
            last_token_output = decode_step_output[:, -1, :]
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.right_tower.lm_use_tokens:
                current_logits = self.right_tower.head(last_token_output)
            else:
                current_logits = last_token_output
        
        # Concatenate all generated tokens
        if not newly_generated_ids_list:
            return torch.empty((B, 0), dtype=torch.long, device=device)
        
        generated_ids = torch.cat(newly_generated_ids_list, dim=1)
        
        # Post-process to handle EOS token
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0:
            seq_len = generated_ids.size(1)
            eos_mask = (generated_ids == self.tokenizer.eos_token_id)  # Create a boolean mask for EOS tokens
            col_indices_for_min = torch.arange(seq_len, device=device)  # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1)
            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len)
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)
            
            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids
    
    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        save_model(self, os.path.join(save_directory, "model.safetensors"))

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        config_path: str | None = None,
        device: torch.device | str | None = None,
        load_backbone: bool = False,
        **model_kwargs,
    ):
        """Load a DualTowerVLM from a local checkpoint directory/weights file or HF repo."""
        if device is None:
            device = torch.device("cpu")

        if os.path.isdir(source):
            resolved_config_path = config_path or os.path.join(source, "config.json")
            weights_path = os.path.join(source, "model.safetensors")
        elif os.path.isfile(source):
            weights_path = source
            resolved_config_path = config_path or os.path.join(os.path.dirname(source), "config.json")
        else:
            resolved_config_path = hf_hub_download(repo_id=source, filename="config.json")
            weights_path = hf_hub_download(repo_id=source, filename="model.safetensors")

        if not os.path.exists(resolved_config_path):
            raise FileNotFoundError("config.json not found; pass config_path for local weights.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError("model.safetensors not found for the provided source.")

        with open(resolved_config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = VLMConfig(**cfg_dict)

        model = cls(cfg, load_backbone=load_backbone, **model_kwargs)
        load_safetensors(model, weights_path)
        model = model.to(device)
        return model

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
    ) -> None:
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        rank0_print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            self.save_pretrained(save_path)

            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(DUAL_TOWER_MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload DualTowerVLM using push_to_hub",
            )


DUAL_TOWER_MODEL_CARD_TEMPLATE = """---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: dualtowervlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - dual-tower
  - research
---

**DualTowerVLM** is a dual-tower Vision-Language Model (VLM) architecture that processes images and text through separate towers before combining their representations.

For more information, check out the repository.

**Usage:**

```python
from models.dual_tower.dual_tower import DualTowerVLM
from models.config import VLMConfig

cfg = VLMConfig()
model = DualTowerVLM.from_pretrained("{repo_id}")
```
"""
