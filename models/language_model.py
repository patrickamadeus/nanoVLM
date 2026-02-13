import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activation_checkpointing import (
    normalize_activation_checkpointing_mode,
    run_activation_checkpoint,
)
from models.momh_attention import (
    create_momh_block_mask,
    create_momh_block_mask_from_modality,
    flex_attention_compiled,
    flex_attention_compiled_dynamic,
    generate_momh_score_mod_with_offset,
)


@torch.compiler.disable
def _build_momh_block_mask_prefill(
    *,
    n_q_heads: int,
    seq_len: int,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor,
    pct_v: float,
    pct_t: float,
    device: str,
):
    # Build once per forward and reuse across blocks.
    seq_len = int(seq_len)
    return create_momh_block_mask_from_modality(
        n_q_heads=n_q_heads,
        q_len=seq_len,
        kv_len=seq_len,
        is_vision=is_vision,
        attention_mask=attention_mask,
        pct_v=pct_v,
        pct_t=pct_t,
        device=device,
    )

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states. 
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute inverse of RMS: square the tensor element-wise, mean is computed across lm_hidden_dim.
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # inverse of RMS
        x = x * irms * self.weight

        return x

# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
class RotaryEmbedding(nn.Module):
    """
        Compute Rotary Embedding to introduce positional dependency to input sequence without additional training parameters and 
        relative distance of token position ids through angle rotation.

        Args:
            cfg: Configuration object containing:
                - lm_hidden_dim (int): Hidden dimension size.
                - lm_n_heads (int): Number of attention heads.
                - lm_re_base (float): Base for rotary embedding frequencies.
                - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
                - lm_attn_scaling (float): Attention scaling factor.
        """
    
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        batch_size, seq_len = position_ids.shape
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq
            
        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()
        
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)
        
        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input by dividing the hidden dimension to two, then swapping and negating dimensions.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim:int=1)-> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.

    Rotary positional embeddings inject position-dependent rotations into query and key vectors,
    enabling transformers to encode positional information effectively without explicit positional encoding.

    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`), 
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)

    """

    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    
    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout
        self.momh_enabled = getattr(cfg, "momh_enabled", False)
        self.momh_pct_vision = getattr(cfg, "momh_head_pct_vision", 0.4)
        self.momh_pct_text = getattr(cfg, "momh_head_pct_text", 0.4)
        self.S_V = getattr(cfg, "mp_image_token_length", 64)

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

        # MoMH decode support (legacy span-based mode): captured tensor buffers + score_mod.
        self._momh_decode_score_mod = None
        self._momh_content_starts_buffer = None
        self._momh_position_offset_buffer = None

    def _get_momh_decode_score_mod(self, device: torch.device):
        if self._momh_decode_score_mod is None:
            self._momh_content_starts_buffer = torch.zeros(1, dtype=torch.int64, device=device)
            self._momh_position_offset_buffer = torch.tensor(0, dtype=torch.int64, device=device)
            self._momh_decode_score_mod = generate_momh_score_mod_with_offset(
                n_q_heads=self.n_heads,
                S_V=self.S_V,
                content_starts=self._momh_content_starts_buffer,
                position_offset=self._momh_position_offset_buffer,
                pct_v=self.momh_pct_vision,
                pct_t=self.momh_pct_text,
            )
        return self._momh_decode_score_mod

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask=None,
        block_kv_cache=None,
        block_mask=None,
        content_starts=None,
        is_vision=None,
        position_offset: int | torch.Tensor = 0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, T_curr, C), where
                        B = batch size,
                        T_curr = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, T_curr, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        is_prefill = block_kv_cache is None
        B, T_curr, C = x.size() # T_curr is the sequence length of the current input x

        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T_curr, head_dim)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)

        # Apply rotary embeddings to the current q and k
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        if block_kv_cache is None:
            # No cache, this is the first pass (prefill)
            k = k_rotated
            v = v_curr
            block_kv_cache = {'key': k, 'value': v}
        else:
            k_cached = block_kv_cache.get("key")
            v_cached = block_kv_cache.get("value")
            needs_dual_prefill = bool(block_kv_cache.get("needs_dual_prefill", False))
            img_mask = block_kv_cache.get("img_mask")

            # Optional dual-prefill mode:
            # replace cached K/V only on selected visual-token positions.
            if needs_dual_prefill:
                if k_cached is None or v_cached is None:
                    raise ValueError("Dual prefill requires existing cached `key` and `value` tensors.")
                if k_cached.size(2) != T_curr:
                    raise ValueError(
                        f"Dual prefill expects cached length == current length, got cache={k_cached.size(2)}, curr={T_curr}."
                    )
                if img_mask is None or img_mask.shape != (B, T_curr):
                    raise ValueError(
                        f"Dual prefill `img_mask` shape must be {(B, T_curr)}, got {None if img_mask is None else tuple(img_mask.shape)}."
                    )

                mask_4d = img_mask.to(device=x.device, dtype=torch.bool).unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
                k = torch.where(mask_4d, k_cached, k_rotated)
                v = torch.where(mask_4d, v_cached, v_curr)
                block_kv_cache['key'] = k
                block_kv_cache['value'] = v
                block_kv_cache['needs_dual_prefill'] = False
            elif k_cached is not None and v_cached is not None:
                # Standard decode path: append new K/V to cached prefix.
                k = torch.cat([k_cached, k_rotated], dim=2)
                v = torch.cat([v_cached, v_curr], dim=2)
                block_kv_cache['key'] = k
                block_kv_cache['value'] = v
            else:
                # Fallback to prefill semantics if an empty cache container is provided.
                k = k_rotated
                v = v_curr
                block_kv_cache['key'] = k
                block_kv_cache['value'] = v

        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        
        T_kv = k_exp.size(2) # Total sequence length of keys/values

        # Preferred MoMH mode: explicit per-token modality mask.
        use_momh_modality = (
            self.momh_enabled
            and (is_vision is not None)
            and (attention_mask is not None)
            and x.device.type == "cuda"
        )

        # Legacy MoMH span mode (content_starts + fixed S_V), kept for compatibility.
        use_momh_span = (
            (not use_momh_modality)
            and self.momh_enabled
            and (content_starts is not None)
            and x.device.type == "cuda"
        )

        if use_momh_modality:
            if block_mask is None:
                is_vision_kv = is_vision[:, :T_kv]
                attn_mask_kv = attention_mask[:, :T_kv]
                block_mask = create_momh_block_mask_from_modality(
                    n_q_heads=self.n_heads,
                    q_len=T_curr,
                    kv_len=T_kv,
                    is_vision=is_vision_kv,
                    attention_mask=attn_mask_kv,
                    pct_v=self.momh_pct_vision,
                    pct_t=self.momh_pct_text,
                    device=str(x.device),
                )

            target_dtype = q.dtype
            k_exp = k_exp.to(target_dtype)
            v_exp = v_exp.to(target_dtype)
            if is_prefill:
                y = flex_attention_compiled(q, k_exp, v_exp, block_mask=block_mask)
            else:
                y = flex_attention_compiled_dynamic(q, k_exp, v_exp, block_mask=block_mask)

        elif use_momh_span and is_prefill:
            block_mask = create_momh_block_mask(
                n_q_heads=self.n_heads,
                seq_len=T_kv,
                S_V=self.S_V,
                content_starts=content_starts,
                pct_v=self.momh_pct_vision,
                pct_t=self.momh_pct_text,
                device=str(x.device),
            )
            target_dtype = q.dtype
            k_exp = k_exp.to(target_dtype)
            v_exp = v_exp.to(target_dtype)
            y = flex_attention_compiled(q, k_exp, v_exp, block_mask=block_mask)

        elif use_momh_span and (not is_prefill):
            score_mod = self._get_momh_decode_score_mod(x.device)

            if self._momh_content_starts_buffer.shape[0] != content_starts.shape[0]:
                self._momh_content_starts_buffer = content_starts.clone()
                self._momh_decode_score_mod = generate_momh_score_mod_with_offset(
                    n_q_heads=self.n_heads,
                    S_V=self.S_V,
                    content_starts=self._momh_content_starts_buffer,
                    position_offset=self._momh_position_offset_buffer,
                    pct_v=self.momh_pct_vision,
                    pct_t=self.momh_pct_text,
                )
                score_mod = self._momh_decode_score_mod
            else:
                self._momh_content_starts_buffer.copy_(content_starts)

            if isinstance(position_offset, torch.Tensor):
                position_offset = int(position_offset.item())
            self._momh_position_offset_buffer.fill_(int(position_offset))

            target_dtype = q.dtype
            k_exp = k_exp.to(target_dtype)
            v_exp = v_exp.to(target_dtype)
            y = flex_attention_compiled_dynamic(q, k_exp, v_exp, score_mod=score_mod)

        else:
            # Standard attention path.
            additive_attn_mask = None
            if attention_mask is not None:
                mask_for_keys = attention_mask[:, :T_kv]
                additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min

            if self.sdpa and x.device.type != 'mps':
                is_causal = (T_curr == T_kv and T_curr > 1)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=additive_attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal
                )
            else:
                attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, T_curr, T_kv)
                if T_curr == T_kv and T_curr > 1:
                    causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                    attn = attn.masked_fill(~causal_mask_val, float('-inf'))

                if additive_attn_mask is not None:
                    attn = attn + additive_attn_mask

                attn = F.softmax(attn, dim=-1)
                attn = self.attn_dropout(attn)
                y = attn @ v_exp

        y = y.to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    """
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg) # Input Norm
        self.norm2 = RMSNorm(cfg) # Post Attention Norm
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor = None,
        block_kv_cache: dict = None,
        block_mask=None,
        content_starts: torch.Tensor = None,
        is_vision: torch.Tensor = None,
        position_offset: int | torch.Tensor = 0,
    ):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.

        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(
            x,
            cos,
            sin,
            attention_mask=attention_mask,
            block_kv_cache=block_kv_cache,
            block_mask=block_mask,
            content_starts=content_starts,
            is_vision=is_vision,
            position_offset=position_offset,
        )
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights
        self.activation_checkpointing = bool(getattr(cfg, "activation_checkpointing", False))
        self.activation_checkpointing_mode = normalize_activation_checkpointing_mode(
            getattr(cfg, "activation_checkpointing_mode", "regular")
        )

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])
        self.norm = RMSNorm(cfg) # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        kv_cache: list[dict] = None,
        start_pos: int = 0,
        content_starts: torch.Tensor = None,
        is_vision: torch.Tensor = None,
        prefill_block_mask=None,
        position_offset: int | torch.Tensor | None = None,
    ):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary.
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        # T_curr is the length of the current input sequence
        B, T_curr, _ = x.size()
        
        # Create position_ids for the current sequence based on start_pos
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(current_position_ids) # Get rotary position embeddings for current tokens

        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        if position_offset is None:
            position_offset = start_pos

        if (
            prefill_block_mask is None
            and attention_mask is not None
            and is_vision is not None
            and x.device.type == "cuda"
            and len(self.blocks) > 0
            and self.blocks[0].attn.momh_enabled
            and start_pos == 0
            and T_curr > 1
        ):
            prefill_block_mask = _build_momh_block_mask_prefill(
                n_q_heads=int(self.blocks[0].attn.n_heads),
                seq_len=T_curr,
                is_vision=is_vision[:, :T_curr],
                attention_mask=attention_mask[:, :T_curr],
                pct_v=float(self.blocks[0].attn.momh_pct_vision),
                pct_t=float(self.blocks[0].attn.momh_pct_text),
                device=str(x.device),
            )

        for i, block in enumerate(self.blocks):
            block_kv_cache = kv_cache[i]
            if self.activation_checkpointing and self.training and block_kv_cache is None:
                # Bind `block` into the closure to avoid late-bound recompute issues.
                def _run_block(
                    x_in,
                    cos_in,
                    sin_in,
                    _block=block,
                    _block_kv_cache=block_kv_cache,
                ):
                    return _block(
                        x_in,
                        cos_in,
                        sin_in,
                        attention_mask=attention_mask,
                        block_kv_cache=_block_kv_cache,
                        block_mask=prefill_block_mask,
                        content_starts=content_starts,
                        is_vision=is_vision,
                        position_offset=position_offset,
                    )

                x, kv_cache[i] = run_activation_checkpoint(
                    _run_block,
                    x,
                    cos,
                    sin,
                    mode=self.activation_checkpointing_mode,
                )
            else:
                x, kv_cache[i] = block(
                    x,
                    cos,
                    sin,
                    attention_mask=attention_mask,
                    block_kv_cache=block_kv_cache,
                    block_mask=prefill_block_mask,
                    content_starts=content_starts,
                    is_vision=is_vision,
                    position_offset=position_offset,
                )

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens: 
            x = self.head(x) 

        return x, kv_cache


    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int=20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs, 
            attention_mask=None,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            
            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1: 
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output, 
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :] 
    
        return generated_outputs

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError

        requested_lm_max_position_embeddings = getattr(cfg, "lm_max_position_embeddings", None)
        requested_lm_max_length = getattr(cfg, "lm_max_length", None)

        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        
        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")
        
        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(hf_config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            raise ValueError(
                f"Could not resolve rope_theta from model config '{cfg.lm_model_type}'."
            )
        cfg.lm_re_base = rope_theta
        if requested_lm_max_position_embeddings is None:
            cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        else:
            cfg.lm_max_position_embeddings = int(requested_lm_max_position_embeddings)
        if requested_lm_max_length is None:
            cfg.lm_max_length = cfg.lm_max_position_embeddings
        else:
            cfg.lm_max_length = int(requested_lm_max_length)
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")
        
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers
        
        # Create our model with potentially larger vocabulary
        model = cls(cfg)
        
        try:
            index_path = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors.index.json")
            with open(index_path, 'r') as f:
                index = json.load(f)
            # Get unique filenames from weight map
            safetensors_filenames = sorted(list(set(index['weight_map'].values())))
            # Download all the sharded files
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename=fn) for fn in safetensors_filenames]
        except EntryNotFoundError:
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")]

        sd = model.state_dict()
        
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }
        
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'
            
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
        
        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        loaded_keys = set()
        
        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue
                    
                    if hf_key in f.keys() and our_key in sd:
                        tensor = f.get_tensor(hf_key)
                        
                        # Special handling for token embeddings if vocab sizes differ
                        if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                            has_extended_embeddings = True
                            print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                            
                            # Copy existing embeddings to the beginning of our larger embedding matrix
                            sd[our_key][:tensor.shape[0]].copy_(tensor)
                            
                            # Initialize the new embeddings using the same approach as the original model
                            std = 0.02  # Common value, but you might want to adjust based on model
                            init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                            
                            print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                            sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well
                        elif tensor.shape == sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                        
                        loaded_keys.add(our_key)

        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                if our_key in sd:
                    print(f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})")
        
        # Load the state dict
        model.load_state_dict(sd)
        
        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if 'lm_head.weight' in f.keys():
                        lm_head = f.get_tensor('lm_head.weight')
                        if lm_head.shape[0] != sd['head.weight'].shape[0]:
                            print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                            # Copy existing weights
                            sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                            # Load updated weights
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break
        
        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
