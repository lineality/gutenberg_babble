# perseid_model.py
# Adapted for the Perseids Architectures


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

"""
Maybe 1:
embedding_params = vocab_size * emb_dim


"""

"""
Maybe 2

## Target Configurations

Here are optimized configurations for your target parameter counts:

### 256M Parameters (256,123,651 params)
```python
PERSEID_256M_CONFIG = {
    "vocab_size": 259,
    "context_length": 32_768,
    "emb_dim": 768,        # Increased from 640
    "n_heads": 12,         # Increased from 4
    "n_layers": 12,        # Reduced from 18
    "hidden_dim": 2048,    # Same
    "head_dim": 64,        # Reduced from 256 (768/12)
    "qk_norm": True,
    "n_kv_groups": 4,      # Increased for GQA efficiency
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 64,  # Matches head_dim
    "tie_weights": True
}
```

### 288M Parameters (287,847,939 params)
```python
PERSEID_288M_CONFIG = {
    "vocab_size": 259,
    "context_length": 32_768,
    "emb_dim": 832,        # 832 = 8 * 104 (good for quantization)
    "n_heads": 13,         # 832/64 = 13
    "n_layers": 12,
    "hidden_dim": 2304,    # 832 * 2.77 ≈ 2304 (divisible by 8)
    "head_dim": 64,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 64,
    "tie_weights": True
}
```

### 320M Parameters (319,928,579 params)
```python
PERSEID_320M_CONFIG = {
    "vocab_size": 259,
    "context_length": 32_768,
    "emb_dim": 896,        # 896 = 8 * 112 (excellent for quantization)
    "n_heads": 14,         # 896/64 = 14
    "n_layers": 12,
    "hidden_dim": 2560,    # 896 * 2.86 ≈ 2560 (divisible by 8)
    "head_dim": 64,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 64,
    "tie_weights": True
}
```

## Parameter Count Verification

```python
def verify_configs():
    '''''Verify the parameter counts for all configurations'''''

    configs = {
        "256M": PERSEID_256M_CONFIG,
        "288M": PERSEID_288M_CONFIG,
        "320M": PERSEID_320M_CONFIG
    }

    for name, config in configs.items():
        params = calculate_perseid_parameters(config)
        print(f"\n{name} Configuration:")
        print(f"  Total Parameters: {params['total']:,}")
        print(f"  Embedding: {params['embedding']:,}")
        print(f"  Layers ({config['n_layers']}): {params['layers']:,}")
        print(f"  Per Layer: {params['per_layer']:,}")
        print(f"  Final Norm: {params['final_norm']:,}")
        print(f"  Quantization friendly: {params['total'] % 8 == 0}")

# Run verification
verify_configs()
```
"""


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        # Match HF Gemma3: compute norm in float32, then scale by (1 + w)
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(input_dtype)


def compute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in,
        num_heads,
        num_kv_groups,
        head_dim=None,
        qk_norm=False,
        query_pre_attn_scalar=None,
        dtype=None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Scale queries
        queries = queries * self.scaling

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, attn_type):
        super().__init__()
        self.attn_type = attn_type

        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x,
        mask_global,
        mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x


class PerseidByteModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert (
            cfg["layer_types"] is not None
            and len(cfg["layer_types"]) == cfg["n_layers"]
        )

        # Main model parameters
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )
        self.cfg = cfg

        # Reusable utilities
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)

        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T

        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids):
        # Forward pass
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    # def get_embeddings(self, input_ids, method="last_token"):
    #     """
    #     Get embedding vectors instead of logits.

    #     Args:
    #         input_ids: Input token IDs (batch, seq_len)
    #         method: "last_token" or "mean" (future)

    #     Returns:
    #         embeddings: (batch, emb_dim) embedding vectors
    #     """
    #     # Run the same forward pass as generation, but stop before out_head
    #     b, seq_len = input_ids.shape
    #     x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
    #     mask_global, mask_local = self._create_masks(seq_len, x.device)

    #     for block in self.blocks:
    #         x = block(
    #             x,
    #             mask_global=mask_global,
    #             mask_local=mask_local,
    #             cos_global=self.cos_global,
    #             sin_global=self.sin_global,
    #             cos_local=self.cos_local,
    #             sin_local=self.sin_local,
    #         )

    #     x = self.final_norm(x)  # (batch, seq_len, emb_dim)

    #     # Pool to single vector
    #     if method == "last_token":
    #         embeddings = x[:, -1, :]  # (batch, emb_dim)
    #     elif method == "mean":
    #         embeddings = x.mean(dim=1)  # (batch, emb_dim)
    #     else:
    #         raise ValueError(f"Unknown embedding method: {method}")

    #     return embeddings

    def get_embeddings(self, input_ids, pooling_method="last_token"):
        """
        Generate dense embedding vectors from input sequences using the trained transformer.

        This method performs the same forward pass as text generation but extracts
        semantic embeddings instead of next-token predictions. The embeddings can be
        used for similarity search, clustering, classification, or other downstream tasks.

        The method processes input through all transformer layers, applies final normalization,
        then pools the sequence-level representations into a single dense vector per input.
        No additional parameters are used beyond the pre-trained transformer weights.

        Args:
            input_ids (torch.Tensor):
                Input token IDs with shape (batch_size, sequence_length).
                Should contain valid token IDs in range [0, vocab_size).
                Sequences can be of different lengths but will be processed
                according to the specified pooling method.

            pooling_method (str, optional):
                Strategy for combining sequence-level hidden states into single embeddings.
                Supported values:
                - "last_token": Uses the hidden state of the final token position.
                            Leverages the autoregressive nature where the last token
                            has attended to all previous tokens. Recommended for
                            decoder-only architectures like this model.
                - "mean": Averages hidden states across all token positions.
                        Provides representation of entire sequence content.
                        Good for document-level or content-based similarity.
                Default: "last_token"

        Returns:
            torch.Tensor:
                Dense embedding vectors with shape (batch_size, embedding_dimension).
                Each row represents the semantic embedding for one input sequence.
                Embeddings are normalized by the final RMSNorm layer and suitable
                for cosine similarity or other distance metrics.
                Data type matches model configuration (typically bfloat16).

        Raises:
            ValueError:
                If pooling_method is not one of the supported values.

            RuntimeError:
                If input_ids contains invalid token IDs or has incorrect dimensions.
                If GPU memory is insufficient for the forward pass.

            TypeError:
                If input_ids is not a torch.Tensor or has wrong dtype.

        Example:
            >>> import torch
            >>> from perseid_model import PerseidByteModel, PERSEID_BYTE_CONFIG_BASE
            >>>
            >>> # Initialize model
            >>> model = PerseidByteModel(PERSEID_BYTE_CONFIG_BASE)
            >>> model.eval()
            >>>
            >>> # Prepare input sequences (batch of 2 sequences)
            >>> input_ids = torch.tensor([
            ...     [72, 101, 108, 108, 111],  # "Hello" in byte tokens
            ...     [87, 111, 114, 108, 100]   # "World" in byte tokens
            >>> ])
            >>>
            >>> # Generate embeddings using last token pooling
            >>> with torch.no_grad():
            ...     embeddings = model.get_embeddings(input_ids, pooling_method="last_token")
            >>>
            >>> print(f"Embedding shape: {embeddings.shape}")  # (2, 640)
            >>>
            >>> # Compute cosine similarity between embeddings
            >>> import torch.nn.functional as F
            >>> similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
            >>> print(f"Similarity: {similarity.item():.4f}")
            >>>
            >>> # Alternative: Use mean pooling for content-based embeddings
            >>> content_embeddings = model.get_embeddings(input_ids, pooling_method="mean")
            >>> print(f"Content embedding shape: {content_embeddings.shape}")  # (2, 640)

        Technical Notes:
            - The method reuses the exact same computational path as text generation,
            ensuring consistency with the model's trained representations.
            - Attention masks (global/local sliding window) are applied identically
            to the generation forward pass, preserving the model's learned attention patterns.
            - RoPE (Rotary Position Encoding) is applied with the same parameters
            used during training, maintaining positional understanding.
            - The final RMSNorm layer provides proper normalization for downstream use.
            - No gradient computation is required for inference; wrap calls in torch.no_grad().

        Performance Considerations:
            - Memory usage scales with sequence_length * batch_size * embedding_dimension.
            - For large batches or long sequences, consider processing in chunks.
            - GPU memory requirements are identical to generation mode for same input size.
            - The method has the same computational complexity as a single generation step.

        Integration Notes:
            - Embeddings can be cached and stored for later similarity computations.
            - Compatible with standard similarity metrics (cosine, euclidean, dot product).
            - Suitable for vector databases and semantic search applications.
            - Can be combined with dimensionality reduction techniques if needed.
        """
        try:
            # Input validation with detailed error messages
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(
                    f"input_ids must be torch.Tensor, got {type(input_ids)}. "
                    f"Convert your input using torch.tensor() or similar method."
                )

            if input_ids.dtype not in [torch.long, torch.int, torch.int64]:
                raise TypeError(
                    f"input_ids must have integer dtype for token IDs, got {input_ids.dtype}. "
                    f"Use input_ids.long() to convert to appropriate dtype."
                )

            if len(input_ids.shape) != 2:
                raise ValueError(
                    f"input_ids must have shape (batch_size, sequence_length), "
                    f"got shape {input_ids.shape}. Reshape your input appropriately."
                )

            batch_size, sequence_length = input_ids.shape

            if sequence_length == 0:
                raise ValueError(
                    f"sequence_length cannot be zero. Input sequences must contain at least one token."
                )

            if sequence_length > self.cfg["context_length"]:
                raise ValueError(
                    f"sequence_length ({sequence_length}) exceeds model's maximum context length "
                    f"({self.cfg['context_length']}). Truncate your input sequences."
                )

            # Validate token IDs are within vocabulary range
            if torch.any(input_ids < 0) or torch.any(
                input_ids >= self.cfg["vocab_size"]
            ):
                raise ValueError(
                    f"input_ids contains invalid token IDs. All values must be in range "
                    f"[0, {self.cfg['vocab_size']}), but found min={input_ids.min().item()}, "
                    f"max={input_ids.max().item()}."
                )

            # Validate pooling method
            supported_pooling_methods = ["last_token", "mean"]
            if pooling_method not in supported_pooling_methods:
                raise ValueError(
                    f"pooling_method '{pooling_method}' is not supported. "
                    f"Choose from: {supported_pooling_methods}"
                )

            # Perform identical forward pass as generation mode up to final normalization
            # This ensures consistency with the model's trained representation space

            # Token embedding with scaling (matches training procedure)
            hidden_states = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

            # Create attention masks for global and local (sliding window) attention
            # These masks preserve the model's trained attention patterns
            attention_mask_global, attention_mask_local = self._create_masks(
                sequence_length, hidden_states.device
            )

            # Pass through all transformer layers with proper attention masking
            for transformer_layer_index, transformer_block in enumerate(self.blocks):
                try:
                    hidden_states = transformer_block(
                        hidden_states,
                        mask_global=attention_mask_global,
                        mask_local=attention_mask_local,
                        cos_global=self.cos_global,
                        sin_global=self.sin_global,
                        cos_local=self.cos_local,
                        sin_local=self.sin_local,
                    )
                except Exception as transformer_error:
                    raise RuntimeError(
                        f"Error in transformer layer {transformer_layer_index}: {str(transformer_error)}. "
                        f"This may indicate GPU memory issues or corrupted model weights."
                    ) from transformer_error

            # Apply final normalization (produces clean, normalized representations)
            normalized_hidden_states = self.final_norm(
                hidden_states
            )  # Shape: (batch_size, sequence_length, embedding_dim)

            # Pool sequence-level representations into single vectors per input
            if pooling_method == "last_token":
                # Extract the hidden state from the final token position
                # In decoder-only models, this token has attended to all previous tokens
                sequence_embeddings = normalized_hidden_states[
                    :, -1, :
                ]  # Shape: (batch_size, embedding_dim)

            elif pooling_method == "mean":
                # Average all token representations to capture overall sequence content
                # This provides a content-based representation of the entire sequence
                sequence_embeddings = normalized_hidden_states.mean(
                    dim=1
                )  # Shape: (batch_size, embedding_dim)

            # Verify output tensor properties
            expected_embedding_shape = (batch_size, self.cfg["emb_dim"])
            if sequence_embeddings.shape != expected_embedding_shape:
                raise RuntimeError(
                    f"Internal error: embedding shape mismatch. Expected {expected_embedding_shape}, "
                    f"got {sequence_embeddings.shape}. This indicates a bug in the pooling logic."
                )

            return sequence_embeddings

        except Exception as embedding_extraction_error:
            # Log the full error context for debugging
            import traceback

            error_traceback = traceback.format_exc()

            # Re-raise with additional context about the embedding extraction process
            raise RuntimeError(
                f"Failed to extract embeddings from input sequences. "
                f"Input shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'unknown'}, "
                f"Pooling method: {pooling_method}, "
                f"Model config: emb_dim={self.cfg.get('emb_dim', 'unknown')}, "
                f"vocab_size={self.cfg.get('vocab_size', 'unknown')}. "
                f"Original error: {str(embedding_extraction_error)}\n"
                f"Full traceback:\n{error_traceback}"
            ) from embedding_extraction_error


PERSEID_BYTE_CONFIG_BASE = {
    "vocab_size": 259,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 1024,  #  512
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}


def load_weights_into_perseid(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )
        return torch.nn.Parameter(
            right.clone().detach()
            if isinstance(right, torch.Tensor)
            else torch.tensor(right)
        )

    # Embedding weights
    if "model.embed_tokens.weight" in params:
        model.tok_emb.weight = assign(
            model.tok_emb.weight,
            params["model.embed_tokens.weight"],
            "model.embed_tokens.weight",
        )

    # Iterate over transformer layers
    for l in range(param_config["n_layers"]):
        block = model.blocks[l]
        att = block.att
        # Attention projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight",
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )
        # QK normalization weights
        att.q_norm.scale = assign(
            att.q_norm.scale,
            params[f"model.layers.{l}.self_attn.q_norm.weight"],
            f"model.layers.{l}.self_attn.q_norm.weight",
        )
        att.k_norm.scale = assign(
            att.k_norm.scale,
            params[f"model.layers.{l}.self_attn.k_norm.weight"],
            f"model.layers.{l}.self_attn.k_norm.weight",
        )
        # Feed forward weights
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight",
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight",
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight",
        )
        # LayerNorm weights
        block.input_layernorm.scale = assign(
            block.input_layernorm.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )
        block.post_attention_layernorm.scale = assign(
            block.post_attention_layernorm.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )
        # Pre‑ and post‑feed forward norms
        pre_key = f"model.layers.{l}.pre_feedforward_layernorm.weight"
        post_key = f"model.layers.{l}.post_feedforward_layernorm.weight"
        if pre_key in params:
            block.pre_feedforward_layernorm.scale = assign(
                block.pre_feedforward_layernorm.scale,
                params[pre_key],
                pre_key,
            )
        if post_key in params:
            block.post_feedforward_layernorm.scale = assign(
                block.post_feedforward_layernorm.scale,
                params[post_key],
                post_key,
            )

    # Final LayerNorm
    if "model.norm.weight" in params:
        model.final_norm.scale = assign(
            model.final_norm.scale,
            params["model.norm.weight"],
            "model.norm.weight",
        )
    # Output head
    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight,
            params["lm_head.weight"],
            "lm_head.weight",
        )
    elif "model.embed_tokens.weight" in params:
        # Weight tying: reuse the embedding weights
        model.out_head.weight = assign(
            model.out_head.weight,
            params["model.embed_tokens.weight"],
            "model.embed_tokens.weight",
        )


def calculate_perseid_parameters(config):
    """Calculate total parameters for Perseid model"""

    vocab_size = config["vocab_size"]
    emb_dim = config["emb_dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_groups = config["n_kv_groups"]
    hidden_dim = config["hidden_dim"]
    head_dim = config["head_dim"]

    # 1. Token embedding
    token_embedding = vocab_size * emb_dim

    # 2. Per-layer parameters
    per_layer_params = 0

    # Attention weights
    q_proj = emb_dim * (n_heads * head_dim)  # Query projection
    k_proj = emb_dim * (n_kv_groups * head_dim)  # Key projection
    v_proj = emb_dim * (n_kv_groups * head_dim)  # Value projection
    o_proj = (n_heads * head_dim) * emb_dim  # Output projection

    # QK normalization (if enabled)
    qk_norm = 2 * head_dim if config.get("qk_norm", False) else 0

    attention_params = q_proj + k_proj + v_proj + o_proj + (qk_norm * n_kv_groups)

    # Feed-forward weights
    ff_gate = emb_dim * hidden_dim  # Gate projection (fc1)
    ff_up = emb_dim * hidden_dim  # Up projection (fc2)
    ff_down = hidden_dim * emb_dim  # Down projection (fc3)
    ff_params = ff_gate + ff_up + ff_down

    # Layer normalization weights (4 per layer in Perseid)
    layernorm_params = 4 * emb_dim  # input, post_attn, pre_ff, post_ff

    per_layer_params = attention_params + ff_params + layernorm_params

    # 3. Final components
    final_norm = emb_dim
    output_head = vocab_size * emb_dim  # Usually tied to embedding

    # Total calculation
    total_params = (
        token_embedding + (per_layer_params * n_layers) + final_norm + output_head
    )

    # With weight tying, we don't double count embedding/output
    if config.get("tie_weights", True):
        total_params -= output_head

    return {
        "total": total_params,
        "embedding": token_embedding,
        "layers": per_layer_params * n_layers,
        "per_layer": per_layer_params,
        "final_norm": final_norm,
        "output_head": output_head if not config.get("tie_weights", True) else 0,
    }


if __name__ == "__main__":
    print("\nbase")
    outness = calculate_perseid_parameters(PERSEID_BYTE_CONFIG_BASE)
    print(outness)

    # print("\nbase")
    # outness = calculate_perseid_parameters(PERSEID_BYTE_CONFIG_BASE)
    # print(outness)
