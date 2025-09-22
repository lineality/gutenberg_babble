#!/usr/bin/env python
# coding: utf-8

# In[1]:


# turn this into a .py file
get_ipython().system('jupyter nbconvert --to script perseids_vs_geminids_train_generate_v12.ipynb')


# # Perseids vs. Gemma(nids)
# 
# 1. perseid_256, perseid_288: resize the architecture for better quantizing e.g. 8bit, 4bit
# - resize mode, try 256, 288, 320 (divisible by 8), 288 is divisible by 8 & 6 (another common quantizing)
# - maybe start with 288...
# 
# Note: (as per current understanding)
# To define such a Perseid model:
# these three configurations are changed from gemma 270m, that is all:
# 1.    emb_dim
# 2.    hidden_dim
# 3.    n_layers
# 
# 
# 2. Dropouts "I have a truant been to chivalry, I speak it to my shame." (Prince Hal, Henvy IV.I)
# - note: the *can use the original gemma weights, though the end goal (after tests) is to not use those weights.
# - because 'with-dropouts' will take longer to train, this will be done later when more time exists
# 

# In[ ]:


# helper functions



# # Training Gemma
# 
# ### see: new .py file -> train_gemma.py
# 
# ### Key Differences from GPT-2 Section:
# 
# 1. **Tokenizer Handling**
#    - Handle much larger vocabulary (262k vs 50k tokens)
#    - Special token handling for chat format
# 
# 2. **Memory Management Focus**
#    - More emphasis on memory profiling
#    - Gradient accumulation demonstration
#    - Context length vs batch size tradeoffs
# 
# 3. **Model-Specific Features**
#    - Sliding window attention
#    - RoPE positioning visualization (optional)
#    - Chat template usage for instruct model
# 
# 4. **Data Requirements**
#    - Need more tokens due to larger vocabulary
#    - Different optimal context window (512 vs 256)
#    - May need multiple Gutenberg texts
# 

# Maybe:
# 
# 
# 
# would it be possible / possibly-meaningful to do a basic test where both gemma 270m and perseid-dropout are compared
# maybe for loss function behavior training from scratch on the same gutenberg text.
# yes, dropout might slow training, but still.
# 
# 
# experiment B:
# perseid-small
# is there a size that is best adapted to later quantized versions of the model? (often 8bit quantized is nearly as performant) 
# either slightly smaller or perhaps slightly larger than the original gemma 270m
# 
# create a new model architecture that is a slight variation on the gemma 270m architecture, the new family of model will be call Perseids, as in comparing the Perseids vs. Geminids meteors: here we are comparing families of models.
# 
# With the aim of testing a comparison of:
# 1. train gemma 270m from scratch on a given corpus
# 2. train versions of Perseids and see how they compare for performance: which changes to architecture affect learning/performance help or hinder.
# 
# 
# for example, the first change may be to re-introduce drop-out in Perseids (vs. Gemma, which I think removed it )
# this may not be architecture per se... but a change
# 
# 
# Experiment 2 Size & later quantization:
# e.g.
# Quantization Considerations:
# 
# 8-bit quantization: Works best with dimensions divisible by 8
# 4-bit quantization: Benefits from dimensions divisible by 32
# 
# If true, the low hanging fruit is to try.
# Perseid_256
# 
# or (if 270 is the smallest for better quality, then)
# Perseid_288
# Perseid_320
# 

# # Proposed experimental design for dropout comparison
# ```
# DROPOUT_EXPERIMENT_CONFIGS = {
#     "baseline_gemma": {
#         "dropout_rate": 0.0,  # Original Gemma (no dropout)
#         "name": "Gemma-270M-Baseline"
#     },
#     "perseid_light_dropout": {
#         "dropout_rate": 0.05,  # Very light dropout
#         "name": "Perseid-270M-Light"
#     },
#     "perseid_medium_dropout": {
#         "dropout_rate": 0.1,   # Traditional dropout rate
#         "name": "Perseid-270M-Medium"
#     },
#     "perseid_heavy_dropout": {
#         "dropout_rate": 0.2,   # Aggressive dropout
#         "name": "Perseid-270M-Heavy"
#     }
# }
# # Specific dropout placement strategies to test
# PERSEID_DROPOUT_ARCHITECTURES = {
#     "attention_only": "dropout_after_attention_output_only", 
#     "feedforward_only": "dropout_in_mlp_layers_only",
#     "full_traditional": "dropout_after_each_sublayer",
#     "adaptive_schedule": "dropout_rate_decay_during_training"
# }
# ```
# 

# PERSEID_SIZE_CONFIGS = {
#     "perseid_256": {
#         "emb_dim": 512,      # 256M params (8-bit friendly)
#         "vocab_size": 262_144,
#         "rationale": "Optimized for 8-bit quantization"
#     },
#     "perseid_288": {
#         "emb_dim": 576,      # ~288M params
#         "vocab_size": 262_144,
#         "rationale": "Balanced size maintaining quality"
#     },
#     "perseid_320": {
#         "emb_dim": 640,      # ~320M params (32-divisible)
#         "vocab_size": 262_144,
#         "rationale": "4-bit quantization optimized"
#     }
# }

# 
# ```
# PERSEID_SIZE_EXPERIMENTS = {
#     "perseid_256": {
#         "emb_dim": 640,  # Keep same as Gemma for fair comparison
#         "hidden_dim": 2048,  # Already divisible by 32
#         "context_length": 256,  # More aggressive reduction
#         "rationale": "aggressive_memory_optimization"
#     },
#     
#     "perseid_288": {
#         "emb_dim": 672,  # 640 + 32 (divisible by 32) 
#         "hidden_dim": 2048,
#         "context_length": 288,  # 32-aligned
#         "rationale": "quantization_optimized_slight_increase"
#     },
#     
#     "perseid_320": {
#         "emb_dim": 640,
#         "hidden_dim": 2048, 
#         "context_length": 320,  # 32-aligned, moderate increase
#         "rationale": "balanced_size_increase"
#     }
# }
# ```

# possible comparison metrics
# ```
# COMPARISON_METRICS = {
#     "training_efficiency": {
#         "convergence_speed": "epochs_to_plateau",
#         "compute_efficiency": "flops_per_improvement",
#         "memory_efficiency": "peak_memory_usage"
#     },
#     
#     "model_quality": {
#         "language_modeling": "perplexity_on_test_set", 
#         "generation_quality": "human_eval_scores",
#         "factual_accuracy": "qa_benchmark_performance",
#         "coherence": "automated_coherence_metrics"
#     },
#     
#     "generalization": {
#         "domain_transfer": "cross_corpus_performance",
#         "few_shot_learning": "in_context_learning_ability",
#         "robustness": "adversarial_text_handling"
#     },
#     
#     "practical_deployment": {
#         "inference_speed": "tokens_per_second",
#         "quantization_friendliness": "post_quant_accuracy",
#         "hardware_efficiency": "utilization_metrics"
#     }
# }
# 
# ?
# EXPERIMENTAL_DESIGN = {
#     "replication": "3_random_seeds_minimum",
#     "significance_testing": "paired_t_tests_with_correction",
#     "effect_size": "cohens_d_calculation", 
#     "confidence_intervals": "bootstrap_95_percent"
# }
# 
# EXPERIMENTAL_PROTOCOL = {
#     "baseline_establishment": {
#         "gemma_270m_from_scratch": "train_original_architecture_on_corpus",
#         "gemma_270m_pretrained_finetune": "finetune_hf_weights_on_corpus",
#         "statistical_significance": "run_each_experiment_3_times_different_seeds"
#     },
#     
#     "controlled_comparisons": {
#         "single_variable_changes": "modify_only_one_architectural_element_at_a_time",
#         "same_training_data": "identical_gutenberg_corpus_across_all_experiments", 
#         "same_hyperparameters": "learning_rate_schedule_optimizer_settings_constant",
#         "same_evaluation_protocol": "identical_test_sets_and_metrics"
#     }
# }
# ```

# In[ ]:





# Phase 1: Dropout Study (Most straightforward to implement)
# 
# Implement Perseid-270M with configurable dropout
# Train 4 variants (0.0, 0.05, 0.1, 0.2 dropout) on same Gutenberg corpus
# Compare training dynamics and final performance
# Document findings before moving to sizing experiments
# 
# Success Criteria:
# 
# Clear performance ranking of dropout rates
# Identified trade-offs between training speed and generalization
# Reproducible training curves across multiple runs

# In[ ]:





# # TODO List
# 
# ### Perseids vs. Geminids
# - make a modified gemma architecture called: Perseid or the Perseids family of models
# - add dropout backin?
# - design for...later quantizing?
# 
# 
# 
# ### 1. 
# - gutenberg txt fine tuning
# - epub-conversion fine-tuning
# - alpaca/instruct fine tuning
# - synthetic data fine tuning
#   1. "Who's coming to the party"
# - testing from-nothing training
# - base model trimming, pruning
# - classification head
# - embedding head
# - larger gemma-type models
# - CPU-only version
# - quantizing model after training
# - dynamic-embeddings
# - IoT Data
# - Biological Signals & Behavior
#   1. birdsong
#   2. ants
#   3. population stability
# 
# ### 2. 
# - full model training vs. lora-layer addition
# - saving an reloading weights
# 
# ### 3. 
# - using the gemma 270m archetecture (or slight modification)
#   1. adjust archetecture to compared across same synthetic data training
# - making a public or crowd-sourced open MIT/APACHE2 weight set.
# - STEM-Net Benchmarks: a range of synthetic training data types
# 

# # Based on rasbt, Sebastian Raschhka's (fabuloustastic) notebooks and book:
# 
# "
# Supplementary code for the Build a Large Language Model From Scratch book by Sebastian Raschka
# 
# Code repository: https://github.com/rasbt/LLMs-from-scratch
# "

# # Gemma 3 270M From Scratch (A Standalone Notebook)

# - This notebook is purposefully minimal and focuses on the code to re-implement Gemma 3 270M in pure PyTorch without relying on other external LLM libraries
# - For more information, see the official [Gemma 3 270M model card](https://huggingface.co/google/gemma-3-270m)
# 
# - Below is a side-by-side comparison with Qwen3 0.6B as a reference model; if you are interested in the Qwen3 0.6B standalone notebook, you can find it [here](../11_qwen3)
# <br>
# 
# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp?1">
#   
#   
# - About the code:
#   - all code is my own code, mapping the Gemma 3 architecture onto the model code implemented in my [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) book; the code is released under a permissive open-source Apache 2.0 license (see [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt))

# In[25]:


# pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch05/07_gpt_to_llama/requirements-extra.txt


# In[26]:


from importlib.metadata import version

# modified
pkgs = [
"torch",  # to implement the model
"numpy",
"jupyter",  # to run this notebook
"huggingface-hub",  # to download pretrained weights
"tokenizers",  # to implement the tokenizer
"safetensors",
]

for p in pkgs:
    print(f"{p} version: {version(p)}")


# - This notebook supports both the base model and the instructmodel; which model to use can be controlled via the following flag:

# In[27]:


USE_INSTRUCT_MODEL = True


# &nbsp;
# # 1. Gemma Architecture code

# In[28]:


import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


# In[29]:


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


# In[30]:


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

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


# In[31]:


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
        query_pre_attn_scalar=None, dtype=None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

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
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

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

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


# In[32]:


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


# In[33]:


class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"]

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, attn_type)for attn_type in cfg["layer_types"]
        ])

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
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


# &nbsp;
# # 2. Initialize gemma model

# In[34]:


GEMMA3_CONFIG_270M = {
    "vocab_size": 262_144,
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
    "sliding_window": 512,
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
        "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}


# In[35]:


torch.manual_seed(123)
model = Gemma3Model(GEMMA3_CONFIG_270M)


# In[36]:


model


# - A quick check that the forward pass works before continuing:

# In[37]:


model(torch.tensor([1, 2, 3]).unsqueeze(0))


# In[38]:


total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# Account for weight tying
total_params_normalized = total_params - model.tok_emb.weight.numel()
print(f"\nTotal number of unique parameters: {total_params_normalized:,}")


# In[39]:


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb

print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")


# In[40]:


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device);


# &nbsp;
# # 4. Load pretrained weights

# In[41]:


def load_weights_into_gemma(model, param_config, params):

    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

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


# - Please note that Google requires that you accept the Gemma 3 licensing terms before you can download the files; to do this, you have to create a Hugging Face Hub account and visit the [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) repository to accept the terms
# - Next, you will need to create an access token; to generate an access token with READ permissions, click on the profile picture in the upper right and click on "Settings"
# 
# 
# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/settings.webp?1" width="300px">
# 
# - Then, create and copy the access token so you can copy & paste it into the next code cell
# 
# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/access-token.webp?1" width="600px">

# In[42]:


# Uncomment and run the following code if you are executing the notebook for the first time

#from huggingface_hub import login
#login()


# In[43]:


import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download

CHOOSE_MODEL = "270m"

if USE_INSTRUCT_MODEL:
    repo_id = f"google/gemma-3-{CHOOSE_MODEL}-it"
else:
    repo_id = f"google/gemma-3-{CHOOSE_MODEL}"


local_dir = Path(repo_id).parts[-1]

if CHOOSE_MODEL == "270m":
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    weights_dict = load_file(weights_file)
else:
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in set(index["weight_map"].values()):
        shard_path = os.path.join(repo_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)

load_weights_into_gemma(model, GEMMA3_CONFIG_270M, weights_dict)
model.to(device)
del weights_dict


# &nbsp;
# # 4. Load tokenizer

# In[45]:


from tokenizers import Tokenizer


class GemmaTokenizer:
    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        # Attempt to identify EOS and padding tokens
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def apply_chat_template(user_text):
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


# In[46]:


tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
if not os.path.exists(tokenizer_file_path):
    try:
        tokenizer_file_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
    except Exception as e:
        print(f"Warning: failed to download tokenizer.json: {e}")
        tokenizer_file_path = "tokenizer.json"

tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)


# In[47]:


prompt = "Give me a short introduction to large language models."
prompt = apply_chat_template("Give me a short introduction to large language models.")


input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)
text


# &nbsp;
# # 5. Generate text

# In[28]:


# Optionally use torch.compile for an extra speed-up
# model = torch.compile(model)


# In[29]:


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


# In[30]:


input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)


for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=500,
    eos_token_id=tokenizer.encode("<end_of_turn>")[-1]
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )


# &nbsp;
# # What's next?

# - Check out the [README.md](./README.md), to use this model via the `llms_from_scratch` package
# - For those interested in a comprehensive guide on building a large language model from scratch and gaining a deeper understanding of its mechanics, you might like my [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)
# 
# <a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>

# "end of original code"

# # Mod: Training Gemma
# 
# Points to Emphasize:
# 
# Architecture Differences:
# 
# RoPE vs learned positional embeddings
# Sliding window attention
# RMSNorm vs LayerNorm
# Larger vocabulary impact
# 
# 
# Memory Management:
# 
# Why we reduce context length
# Gradient accumulation explanation
# bfloat16 precision benefits
# 
# 
# Tokenization Differences:
# 
# Larger vocabulary effects
# Different special tokens
# Chat template for instruct models
# 
# 
# Training Adaptations:
# 
# No dropout in Gemma
# Different learning rate
# Gradient clipping importance

# In[6]:


# !python3 -m pip install matplotlib


# In[10]:


# !python3 -m pip install GPUtil


# In[ ]:





# In[9]:


# ## Training Gemma 3 270M
# 
# This section demonstrates how to train/fine-tune the Gemma 3 270M model using the training infrastructure
# we've developed. We'll walk through data preparation, model initialization, and the training process.

# ### Setup and Environment Check

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import urllib.request
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import time
import psutil
import GPUtil


# In[11]:


# Check GPU availability and memory
def check_gpu_memory():
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU: {gpu.name}")
        print(f"Total memory: {gpu.memoryTotal:.1f} MB")
        print(f"Free memory: {gpu.memoryFree:.1f} MB")
        print(f"Used memory: {gpu.memoryUsed:.1f} MB")

        # PyTorch's view
        device_id = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        print(f"\nPyTorch allocated: {allocated:.2f} GB")
        print(f"PyTorch reserved: {reserved:.2f} GB")
    else:
        print("No GPU available. Training will be slow on CPU.")
        print(f"CPU cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

check_gpu_memory()


# In[12]:


# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()  # Clear any cached memory
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\nUsing device: {device}")


# In[13]:


# We'll use Project Gutenberg texts for training. This provides more data than "The Verdict" 
# used in the GPT-2 example, which is necessary for the larger vocabulary of Gemma.

# Download multiple Gutenberg texts for more training data
def download_gutenberg_texts():
    """Download a collection of public domain texts for training"""

    texts = {
        # "shakespeare.txt": "https://www.gutenberg.org/files/100/100-0.txt",  # Complete Shakespeare
        "alice.txt": "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
        # "pride_prejudice.txt": "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        # "frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
    }

    combined_text = ""

    for filename, url in texts.items():
        file_path = f"data/{filename}"
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            print(f"Loading existing {filename}...")
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        # Clean and combine texts
        # Remove Gutenberg headers/footers if present
        start_marker = "*** START OF"
        end_marker = "*** END OF"

        start_idx = text_data.find(start_marker)
        if start_idx != -1:
            start_idx = text_data.find("\n", start_idx) + 1
            text_data = text_data[start_idx:]

        end_idx = text_data.find(end_marker)
        if end_idx != -1:
            text_data = text_data[:end_idx]

        combined_text += text_data + "\n\n"

    return combined_text

# Load the training data
print("Loading training data...")
text_data = download_gutenberg_texts()

print(f"Total characters loaded: {len(text_data):,}")
print(f"First 200 characters:\n{text_data[:200]}")
print(f"\nLast 200 characters:\n{text_data[-200:]}")


# In[14]:


### Tokenizer Setup and Data Analysis
# 
# Gemma uses a much larger vocabulary (262k tokens) compared to GPT-2 (50k tokens).
# Let's set up the tokenizer and analyze our data.

# Download and setup Gemma tokenizer
USE_INSTRUCT_MODEL = False  # Set to True if you want to use the instruct variant

repo_id = "google/gemma-3-270m-it" if USE_INSTRUCT_MODEL else "google/gemma-3-270m"
local_dir = Path(repo_id).parts[-1]

# Download tokenizer
tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
if not os.path.exists(tokenizer_file_path):
    print("Downloading Gemma tokenizer...")
    # Note: You may need to authenticate with Hugging Face here
    # from huggingface_hub import login
    # login()
    tokenizer_file_path = hf_hub_download(
        repo_id=repo_id, 
        filename="tokenizer.json", 
        local_dir=local_dir
    )

# Initialize tokenizer
class GemmaTokenizer:
    def __init__(self, tokenizer_file_path: str):
        self._tok = Tokenizer.from_file(str(tokenizer_file_path))
        self.eos_token = "<end_of_turn>"
        self.pad_token = "<end_of_turn>"

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)

# Analyze tokenization
print("Tokenizing data...")
start_time = time.time()
all_tokens = tokenizer.encode(text_data)
tokenization_time = time.time() - start_time

print(f"Tokenization completed in {tokenization_time:.2f} seconds")
print(f"Total tokens: {len(all_tokens):,}")
print(f"Compression ratio: {len(text_data) / len(all_tokens):.2f} characters per token")
print(f"Unique tokens in data: {len(set(all_tokens)):,}")

# Sample tokenization
sample_text = "The quick brown fox jumps over the lazy dog."
sample_tokens = tokenizer.encode(sample_text)
print(f"\nSample text: '{sample_text}'")
print(f"Token count: {len(sample_tokens)}")
print(f"Token IDs: {sample_tokens[:10]}...")  # Show first 10
print(f"Decoded back: '{tokenizer.decode(sample_tokens)}'")


# In[15]:


# ### Model Configuration and Memory Planning
# 
# Let's configure the model for our available GPU memory (12-13GB target)

# Gemma 3 270M configuration
GEMMA3_CONFIG_270M = {
    "vocab_size": 262_144,
    "context_length": 512,  # Reduced for memory constraints (original: 32,768)
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}

# Training settings optimized for memory constraints
TRAINING_SETTINGS = {
    "use_pretrained": True,  # Start from pretrained weights
    "use_instruct": USE_INSTRUCT_MODEL,
    "context_length": 512,  # Adjust based on your GPU memory
    "batch_size": 1,
    "gradient_accumulation_steps": 4,  # Simulate batch_size=4
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "warmup_steps": 100,
}

# Memory estimation
def estimate_memory_usage(config, settings):
    """Estimate GPU memory requirements"""
    # Model parameters
    total_params = (
        config["vocab_size"] * config["emb_dim"] +  # Token embeddings
        config["n_layers"] * (
            4 * config["emb_dim"]**2 +  # Attention weights (Q,K,V,O)
            3 * config["emb_dim"] * config["hidden_dim"]  # FFN weights
        ) +
        config["emb_dim"] * config["vocab_size"]  # Output layer
    )

    model_memory_gb = (total_params * 2) / (1024**3)  # 2 bytes for bfloat16

    # Optimizer states (Adam uses 2x model memory for momentum and variance)
    optimizer_memory_gb = model_memory_gb * 2

    # Activation memory (rough estimate)
    seq_len = settings["context_length"]
    batch_size = settings["batch_size"]
    activation_memory_gb = (
        batch_size * seq_len * config["emb_dim"] * config["n_layers"] * 4
    ) / (1024**3)

    total_memory_gb = model_memory_gb + optimizer_memory_gb + activation_memory_gb

    print(f"Memory Estimation:")
    print(f"  Model weights: {model_memory_gb:.2f} GB")
    print(f"  Optimizer states: {optimizer_memory_gb:.2f} GB")
    print(f"  Activations (approx): {activation_memory_gb:.2f} GB")
    print(f"  Total estimated: {total_memory_gb:.2f} GB")

    return total_memory_gb

estimated_memory = estimate_memory_usage(GEMMA3_CONFIG_270M, TRAINING_SETTINGS)

if estimated_memory > 12:
    print(f"\n⚠️ Warning: Estimated memory usage ({estimated_memory:.1f} GB) exceeds target (12 GB)")
    print("Consider reducing context_length or using gradient checkpointing")


# In[16]:


# ### Data Validation for Training
# 
# Ensure we have enough data for our context windows

min_tokens_needed = TRAINING_SETTINGS["context_length"] * 10  # At least 10 sequences

if len(all_tokens) < min_tokens_needed:
    print(f"⚠️ Warning: You have {len(all_tokens):,} tokens, but {min_tokens_needed:,} is recommended")
    print(f"Consider adding more text data or reducing context_length")
else:
    print(f"✓ Sufficient tokens for training: {len(all_tokens):,}")

# Calculate train/val split
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_text = text_data[:split_idx]
val_text = text_data[split_idx:]

train_tokens = len(tokenizer.encode(train_text))
val_tokens = len(tokenizer.encode(val_text))

print(f"\nData split:")
print(f"  Training: {train_tokens:,} tokens ({train_ratio*100:.0f}%)")
print(f"  Validation: {val_tokens:,} tokens ({(1-train_ratio)*100:.0f}%)")

# Check if we have enough tokens for even one batch
contexts_in_train = train_tokens // TRAINING_SETTINGS["context_length"]
contexts_in_val = val_tokens // TRAINING_SETTINGS["context_length"]

print(f"\nPossible training sequences: {contexts_in_train:,}")
print(f"Possible validation sequences: {contexts_in_val:,}")

if contexts_in_train < 1 or contexts_in_val < 1:
    print("❌ Error: Not enough data for the specified context length!")
    print("Please reduce context_length or add more training data")


# In[39]:


# ### Initialize Model
# 
# Now let's initialize the Gemma model and optionally load pretrained weights

print("Initializing Gemma 3 270M model...")

# Initialize model
model = Gemma3Model(GEMMA3_CONFIG_270M)

# Load pretrained weights if specified
if TRAINING_SETTINGS["use_pretrained"]:
    print("Loading pretrained weights from Hugging Face...")

    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )

    weights_dict = load_file(weights_file)
    load_weights_into_gemma(model, GEMMA3_CONFIG_270M, weights_dict)
    del weights_dict  # Free memory immediately

    print("✓ Pretrained weights loaded successfully")
else:
    print("Initializing with random weights (training from scratch)")

# Move model to device
model = model.to(device)

# Model statistics
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size (bfloat16): {total_params * 2 / 1e9:.2f} GB")

# Check GPU memory after model loading
if torch.cuda.is_available():
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory allocated: {allocated:.2f} GB")


# In[41]:


# ### Pre-training Evaluation
# ### Pre-training Evaluation
# 
# Let's see how the model performs before training

def generate_text_simple(model, tokenizer, prompt, max_new_tokens=50):
    """Simple generation for evaluation using the working Gemma generation code"""
    model.eval()

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    # Get EOS token ID if it exists
    try:
        eos_token_id = tokenizer.encode("<end_of_turn>")[-1]
    except:
        eos_token_id = None

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for last token
            logits = model(input_ids)[:, -1, :]

            # Get next token (greedy)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Check for EOS
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            generated_tokens.append(next_token.item())

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Truncate if too long
            if input_ids.shape[1] > model.cfg["context_length"]:
                input_ids = input_ids[:, -model.cfg["context_length"]:]

    # Decode the full sequence
    full_sequence = token_ids + generated_tokens
    return tokenizer.decode(full_sequence)

# Alternative: Use the streaming version from the original notebook
def generate_text_stream(model, tokenizer, prompt, max_new_tokens=50):
    """Using the exact generation code from the working Gemma notebook"""
    model.eval()

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    # Get EOS token ID
    try:
        eos_token_id = tokenizer.encode("<end_of_turn>")[-1]
    except:
        eos_token_id = None

    generated = token_ids.copy()  # Start with the prompt

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Handle context overflow
            if input_ids.shape[1] > model.cfg["context_length"]:
                input_ids = input_ids[:, -model.cfg["context_length"]:]

    return tokenizer.decode(generated)

# Test generation before training
test_prompts = [
    "The meaning of life is",
    "Once upon a time",
    "In the beginning",
]


print("Model generation before training:\n")
for prompt in test_prompts:
    print(f"Prompt: '{prompt}'")
    output = generate_text_simple(model, tokenizer, prompt, max_new_tokens=30)
    print(f"Output: {output}\n")


# # Training Gemma

# In[42]:


# ### Create Data Loaders
# 
# Create PyTorch data loaders for training

from torch.utils.data import Dataset, DataLoader

class GemmaDataset(Dataset):
    """Dataset for Gemma training"""
    def __init__(self, text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Tokenize all text
        self.tokens = tokenizer.encode(text)

        # Create sliding windows
        self.windows = []
        for i in range(0, len(self.tokens) - max_length, stride):
            self.windows.append(self.tokens[i:i + max_length + 1])  # +1 for target

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)  # All but last
        target_ids = torch.tensor(window[1:], dtype=torch.long)   # All but first
        return input_ids, target_ids

# Create datasets
print("Creating data loaders...")

train_dataset = GemmaDataset(
    train_text, 
    tokenizer, 
    TRAINING_SETTINGS["context_length"],
    TRAINING_SETTINGS["context_length"]  # Non-overlapping windows
)

val_dataset = GemmaDataset(
    val_text,
    tokenizer,
    TRAINING_SETTINGS["context_length"],
    TRAINING_SETTINGS["context_length"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAINING_SETTINGS["batch_size"],
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=TRAINING_SETTINGS["batch_size"],
    shuffle=False,
    drop_last=False
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Verify data loader
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"\nFirst batch shape - Inputs: {inputs.shape}, Targets: {targets.shape}")
    print(f"First few tokens: {inputs[0, :10].tolist()}")
    break


# In[43]:


# ### Calculate Initial Loss
# 
# Establish baseline performance metrics

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over data loader"""
    model.eval()
    total_loss = 0.

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

    return total_loss / num_batches if num_batches > 0 else float("nan")

# Calculate initial losses
print("Calculating initial losses...")
train_loss_initial = calc_loss_loader(train_loader, model, device, num_batches=10)
val_loss_initial = calc_loss_loader(val_loader, model, device, num_batches=10)

print(f"Initial training loss: {train_loss_initial:.4f}")
print(f"Initial validation loss: {val_loss_initial:.4f}")
print(f"Initial perplexity: {torch.exp(torch.tensor(val_loss_initial)):.2f}")


# In[44]:


# ### Training Loop
# 
# Now we'll train the model with gradient accumulation to work within memory constraints

import time
from tqdm import tqdm

def train_gemma(model, train_loader, val_loader, settings, device):
    """Training loop with gradient accumulation"""

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler (optional)
    total_steps = len(train_loader) * settings["num_epochs"] // settings["gradient_accumulation_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": []
    }

    # Training loop
    print(f"\nStarting training for {settings['num_epochs']} epochs...")
    print(f"Total optimization steps: {total_steps}")

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(settings["num_epochs"]):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{settings['num_epochs']}")

        optimizer.zero_grad()

        for batch_idx, (input_batch, target_batch) in enumerate(progress_bar):
            # Forward pass
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss = loss / settings["gradient_accumulation_steps"]

            # Backward pass
            loss.backward()

            epoch_loss += loss.item() * settings["gradient_accumulation_steps"]

            # Update weights after accumulation
            if (batch_idx + 1) % settings["gradient_accumulation_steps"] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * settings["gradient_accumulation_steps"]:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

        # Calculate epoch metrics
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=20)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "gemma3_best.pth")
            print("  ✓ Saved best model")

        # Generate sample
        print(f"\n  Sample generation:")
        prompt = "Once upon a time"
        output = generate_text_simple(model, tokenizer, prompt, max_new_tokens=40)
        print(f"  '{output}'")

        # Check GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"\n  GPU memory: {allocated:.2f} GB")

    return history

# Run training
torch.manual_seed(42)  # For reproducibility
history = train_gemma(model, train_loader, val_loader, TRAINING_SETTINGS, device)


# In[45]:


# ### Training Results Visualization
# 
# Plot the training curves to analyze performance

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curves
axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
axes[0].plot(history["val_loss"], label="Val Loss", marker='s')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Perplexity
train_perplexity = [torch.exp(torch.tensor(loss)).item() for loss in history["train_loss"]]
val_perplexity = [torch.exp(torch.tensor(loss)).item() for loss in history["val_loss"]]

axes[1].plot(train_perplexity, label="Train Perplexity", marker='o')
axes[1].plot(val_perplexity, label="Val Perplexity", marker='s')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Perplexity")
axes[1].set_title("Perplexity Over Time")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning rate
axes[2].plot(history["learning_rates"], marker='o', color='green')
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Learning Rate Schedule")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gemma_training_results.png", dpi=150, bbox_inches='tight')
plt.show()

# Print final metrics
print("\nTraining Complete!")
print(f"Initial val loss: {val_loss_initial:.4f} → Final val loss: {history['val_loss'][-1]:.4f}")
print(f"Improvement: {(val_loss_initial - history['val_loss'][-1]) / val_loss_initial * 100:.1f}%")


# In[46]:


# ### Post-Training Evaluation
# 
# Compare model performance before and after training

# Load best model
model.load_state_dict(torch.load("gemma3_best.pth", map_location=device))
model.eval()

print("Model generation after training:\n")

# Test with same prompts as before
test_prompts = [
    "The meaning of life is",
    "Once upon a time",
    "In the beginning",
    "To be or not to be",
]

for prompt in test_prompts:
    print(f"Prompt: '{prompt}'")
    output = generate_text_simple(model, tokenizer, prompt, max_new_tokens=50)
    print(f"Output: {output}\n")
    print("-" * 80)


# In[21]:


# ### Advanced Generation with Temperature and Top-k
# 
# Test more sophisticated generation strategies

def generate_advanced(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    """Advanced generation with temperature and top-k sampling"""
    model.eval()

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    # Get EOS token ID
    try:
        eos_token_id = tokenizer.encode("<end_of_turn>")[-1]
    except:
        eos_token_id = None

    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits
            logits = model(input_ids)[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            generated.append(next_token.item())

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Truncate if needed
            if input_ids.shape[1] > model.cfg["context_length"]:
                input_ids = input_ids[:, -model.cfg["context_length"]:]

    # Decode full sequence
    full_sequence = token_ids + generated
    return tokenizer.decode(full_sequence)

# Test different generation settings
generation_configs = [
    {"temperature": 0.5, "top_k": 10, "name": "Conservative"},
    {"temperature": 0.8, "top_k": 50, "name": "Balanced"},
    {"temperature": 1.2, "top_k": 100, "name": "Creative"},
]

prompt = "The future of artificial intelligence"

print("Testing different generation strategies:\n")
for config in generation_configs:
    print(f"{config['name']} (temp={config['temperature']}, top_k={config['top_k']}):")
    output = generate_advanced(
        model, tokenizer, prompt, 
        temperature=config['temperature'], 
        top_k=config['top_k'],
        max_new_tokens=50
    )
    print(f"{output}\n")
    print("-" * 80)


# In[48]:


# for what?
def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print a sample during training"""
    model.eval()

    # Encode prompt
    token_ids = tokenizer.encode(start_context)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    # Get EOS token ID
    try:
        eos_token_id = tokenizer.encode("<end_of_turn>")[-1]
    except:
        eos_token_id = None

    generated = []

    with torch.no_grad():
        for _ in range(50):  # max_new_tokens
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Handle context overflow
            if input_ids.shape[1] > model.cfg["context_length"]:
                input_ids = input_ids[:, -model.cfg["context_length"]:]

    # Decode and print
    full_sequence = token_ids + generated
    decoded_text = tokenizer.decode(full_sequence)
    print(decoded_text.replace("\n", " "))  # Compact print format

    model.train()


# In[49]:


# ### Save Final Model and Training Summary
# 
# Save the trained model and create a summary report

# Save final model with metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'config': GEMMA3_CONFIG_270M,
    'training_settings': TRAINING_SETTINGS,
    'training_history': history,
    'final_metrics': {
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'best_val_loss': min(history['val_loss']),
        'total_epochs': TRAINING_SETTINGS['num_epochs']
    }
}, 'gemma3_trained_complete.pth')

print("Training Summary")
print("=" * 50)
print(f"Model: Gemma 3 270M")
print(f"Training data size: {len(all_tokens):,} tokens")
print(f"Context length: {TRAINING_SETTINGS['context_length']}")
print(f"Batch size: {TRAINING_SETTINGS['batch_size']} (effective: {TRAINING_SETTINGS['batch_size'] * TRAINING_SETTINGS['gradient_accumulation_steps']})")
print(f"Learning rate: {TRAINING_SETTINGS['learning_rate']}")
print(f"Epochs: {TRAINING_SETTINGS['num_epochs']}")
print(f"\nResults:")
print(f"  Initial loss: {val_loss_initial:.4f}")
print(f"  Final loss: {history['val_loss'][-1]:.4f}")
print(f"  Best loss: {min(history['val_loss']):.4f}")
print(f"  Improvement: {(1 - min(history['val_loss'])/val_loss_initial) * 100:.1f}%")

# Memory usage summary
if torch.cuda.is_available():
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak GPU memory usage: {max_allocated:.2f} GB")


# In[ ]:





# In[ ]:





# In[50]:


# ### Next Steps and Recommendations
# 
# Based on the training results, here are some suggestions for further improvement:

print("Recommendations for Further Training:\n")

# Analyze if model is overfitting or underfitting
if len(history['val_loss']) > 1:
    if history['val_loss'][-1] > history['val_loss'][-2]:
        print("⚠️ Validation loss increased in last epoch - possible overfitting")
        print("   Consider: reducing learning rate, adding dropout, or early stopping")
    elif abs(history['train_loss'][-1] - history['val_loss'][-1]) > 0.5:
        print("⚠️ Large gap between train and val loss - likely overfitting")
        print("   Consider: more regularization, data augmentation, or smaller model")
    elif history['train_loss'][-1] > 3.0:
        print("⚠️ Training loss still high - possible underfitting")
        print("   Consider: longer training, higher learning rate, or more complex model")
    else:
        print("✓ Training appears stable and converged well")

print("\nPotential improvements:")
print("1. Data: Add more diverse texts from Project Gutenberg")
print("2. Context: Try longer sequences if memory allows (current: 512)")
print("3. Batch size: Use gradient accumulation to simulate larger batches")
print("4. Learning rate: Experiment with warmup and different schedules")
print("5. Fine-tuning: Try task-specific datasets for better performance")

# Provide code snippet for loading the model later
print("\n" + "="*50)
print("To load this model later, use:\n")
print("""
# Load the trained model
checkpoint = torch.load('gemma3_trained_complete.pth', map_location=device)
model = Gemma3Model(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# View training history
history = checkpoint['training_history']
print(f"Best validation loss: {checkpoint['final_metrics']['best_val_loss']:.4f}")
""")


# In[18]:


# Reload the complete model for continued training or inference
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# model.to(device);


# In[19]:


# Reload the complete model for continued training or inference
import torch

# Load the checkpoint
checkpoint = torch.load('gemma3_trained_complete.pth', map_location=device)

# Recreate the model with the same configuration
model = Gemma3Model(checkpoint['config'])

# Load the state dictionary
model.load_state_dict(checkpoint['model_state_dict'])

# Move to device
model = model.to(device)



# In[ ]:


# If you want to continue training, set to train mode
# model.train()

# # Access training history if needed
# training_history = checkpoint['training_history']
# final_metrics = checkpoint['final_metrics']

# # Print out some information
# print("Model loaded successfully")
# print(f"Training epochs completed: {final_metrics['total_epochs']}")
# print(f"Best validation loss: {final_metrics['best_val_loss']:.4f}")




# In[ ]:


# # Initialize model
# model = Gemma3Model(GEMMA3_CONFIG_270M)

# Or for inference
model.eval()


# In[48]:


# Test different generation settings
generation_configs = [
    {"temperature": 0.5, "top_k": 10, "name": "Conservative"},
    {"temperature": 0.8, "top_k": 50, "name": "Balanced"},
    {"temperature": 1.2, "top_k": 100, "name": "Creative"},
]

prompt = "The future of artificial intelligence"

print("Testing different generation strategies:\n")
for config in generation_configs:
    print(f"{config['name']} (temp={config['temperature']}, top_k={config['top_k']}):")
    output = generate_advanced(
        model, tokenizer, prompt, 
        temperature=config['temperature'], 
        top_k=config['top_k'],
        max_new_tokens=50
    )
    print(f"{output}\n")
    print("-" * 80)


# 

# In[ ]:


PERSEID_DROPOUT_STRATEGIES = {
    "baseline_gemma": {
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.0, 
        "residual_dropout": 0.0,
        "dropout_after_attention": False,
        "dropout_after_feedforward": False,
        "dropout_on_residuals": False,
        "name": "Gemma-270M-Original"
    },

    "perseid_conservative": {
        "attention_dropout": 0.05,
        "feedforward_dropout": 0.05,
        "residual_dropout": 0.0,
        "dropout_after_attention": True,
        "dropout_after_feedforward": True, 
        "dropout_on_residuals": False,
        "name": "Perseid-270M-Conservative"
    },

    "perseid_traditional": {
        "attention_dropout": 0.1,
        "feedforward_dropout": 0.1,
        "residual_dropout": 0.1,
        "dropout_after_attention": True,
        "dropout_after_feedforward": True,
        "dropout_on_residuals": True,
        "name": "Perseid-270M-Traditional"
    },

    "perseid_attention_only": {
        "attention_dropout": 0.15,
        "feedforward_dropout": 0.0,
        "residual_dropout": 0.0,
        "dropout_after_attention": True,
        "dropout_after_feedforward": False,
        "dropout_on_residuals": False,
        "name": "Perseid-270M-AttentionFocus"
    },

    "perseid_feedforward_only": {
        "attention_dropout": 0.0,
        "feedforward_dropout": 0.15,
        "residual_dropout": 0.0,
        "dropout_after_attention": False,
        "dropout_after_feedforward": True,
        "dropout_on_residuals": False,
        "name": "Perseid-270M-FeedforwardFocus"
    }
}


# In[ ]:


def load_pretrained_weights_with_dropout_extension(
    perseid_model, 
    gemma_weights_dict, 
    dropout_config
):
    """
    Load original Gemma weights into Perseid model with dropout layers

    Key insight: Dropout layers have no learned parameters,
    so we can initialize them randomly and load everything else from Gemma
    """

    # Load all original Gemma weights exactly as before
    load_weights_into_gemma(perseid_model, GEMMA3_CONFIG_270M, gemma_weights_dict)

    # Dropout layers initialize themselves (no pretrained weights needed)
    print(f"✓ Loaded pretrained Gemma weights")
    print(f"✓ Initialized dropout layers with config: {dropout_config}")

    # Verify architecture compatibility
    original_param_count = sum(p.numel() for p in perseid_model.parameters() 
                              if not isinstance(p, nn.Dropout))
    print(f"✓ Preserved {original_param_count:,} parameters from original Gemma")

    return perseid_model


# In[ ]:


FINE_TUNING_PROTOCOL = {
    "data_preparation": {
        "corpus": "same_gutenberg_books_across_all_experiments",
        "split": "90_train_10_validation_consistent_across_variants",
        "preprocessing": "identical_tokenization_and_chunking"
    },

    "training_settings": {
        "initial_model": "identical_gemma_270m_pretrained_weights",
        "learning_rate": 5e-5,  # Conservative for fine-tuning
        "batch_size": 4,  # Smaller for fine-tuning
        "epochs": 3,  # Fewer epochs needed with pretrained start
        "warmup_steps": 100,
        "weight_decay": 0.01
    },

    "evaluation_schedule": {
        "eval_frequency": "every_50_training_steps", 
        "metrics_logged": ["train_loss", "val_loss", "perplexity", "generation_samples"],
        "early_stopping": "patience_of_3_evaluations_without_improvement"
    }
}


# FINE_TUNING_COMPARISON_METRICS = {
#     "adaptation_efficiency": {
#         "steps_to_baseline_performance": "how_quickly_each_variant_matches_original_gemma",
#         "final_convergence_quality": "best_validation_loss_achieved",
#         "training_stability": "variance_in_loss_across_training_steps",
#         "overfitting_resistance": "gap_between_train_and_validation_curves"
#     },
#     
#     "generalization_quality": {
#         "held_out_book_performance": "loss_on_books_not_seen_during_fine_tuning",
#         "genre_transfer": "performance_degradation_on_different_gutenberg_genres", 
#         "generation_diversity": "unique_n_gram_ratios_in_generated_samples",
#         "coherence_preservation": "how_well_long_form_generation_maintains_context"
#     },
#     
#     "practical_deployment": {
#         "inference_speed": "tokens_per_second_during_generation",
#         "memory_efficiency": "peak_gpu_memory_during_fine_tuning",
#         "robustness_to_hyperparameters": "sensitivity_to_learning_rate_changes"
#     }
# }

# ??? DROPOUT_FINE_TUNING_HYPOTHESES = {
#     "conservative_hypothesis": {
#         "prediction": "light_dropout_0.05_improves_generalization_minimal_training_cost",
#         "reasoning": "prevents_overfitting_to_fine_tuning_corpus_without_disrupting_pretrained_knowledge"
#     },
#     
#     "traditional_hypothesis": {
#         "prediction": "moderate_dropout_0.1_helps_domain_adaptation_but_slows_convergence", 
#         "reasoning": "classic_regularization_benefits_but_pretrained_weights_already_robust"
#     },
#     
#     "selective_hypothesis": {
#         "prediction": "feedforward_dropout_more_beneficial_than_attention_dropout",
#         "reasoning": "attention_patterns_from_pretraining_valuable_feedforward_overfits_to_new_domain"
#     }
# }

# # 256

# In[ ]:


# CHANGE: Create new config (copy + modify)
PERSEID_CONFIG_256M = GEMMA3_CONFIG_270M.copy()
PERSEID_CONFIG_256M.update({
    "emb_dim": 576,      # Reduced from 640 for ~256M params
    "hidden_dim": 1536,  # Reduced from 2048, keeps 8-bit friendly
    "n_layers": 16,      # Reduced from 18 layers
})

# Alternative sizing for exact 256M parameters:
PERSEID_CONFIG_256M_ALT = GEMMA3_CONFIG_270M.copy()
PERSEID_CONFIG_256M_ALT.update({
    "emb_dim": 512,      # More aggressive reduction
    "hidden_dim": 1536,  # 8-bit quantization friendly
    "n_layers": 18,      # Keep same depth
})

# TODO
# 288?


# In[ ]:


# Dropout experiment
model_baseline = Gemma3Model(PERSEID_DROPOUT_CONFIGS["baseline"])
model_dropout = Gemma3Model(PERSEID_DROPOUT_CONFIGS["medium"])

# Size experiment  
model_256m = Gemma3Model(PERSEID_CONFIG_256M)


# # v1
# 
# Part 1: Core Architecture Modifications
# 1. Modified Transformer Block with Dropout (perseid_blocks.py)

# In[ ]:


#!/usr/bin/env python
"""
Perseid Transformer Blocks with Dropout Support
File: perseid_blocks.py

This module implements the Perseid variant of Gemma transformer blocks
with configurable dropout for architecture comparison experiments.
"""

import torch
import torch.nn as nn
import traceback
from typing import Dict, Optional, Tuple


class PerseidTransformerBlock(nn.Module):
    """
    Perseid variant of Gemma transformer block with configurable dropout reintroduction.

    This block extends the original Gemma architecture by adding strategic dropout
    layers at key positions to test regularization effects during fine-tuning.

    Args:
        cfg (Dict): Model configuration dictionary containing architecture parameters
        attn_type (str): Type of attention mechanism ('sliding_attention' or 'full_attention')
        dropout_config (Dict): Dropout configuration with rates and placement strategy
    """

    def __init__(self, cfg: Dict, attn_type: str, dropout_config: Optional[Dict] = None):
        """
        Initialize Perseid transformer block with optional dropout layers.

        Args:
            cfg: Configuration dictionary with model parameters
            attn_type: Attention type for this layer
            dropout_config: Optional dropout configuration, defaults to no dropout
        """
        super().__init__()

        try:
            self.attn_type = attn_type

            # Default dropout config (no dropout - matches original Gemma)
            if dropout_config is None:
                dropout_config = {
                    "attention_dropout": 0.0,
                    "feedforward_dropout": 0.0,
                    "residual_dropout": 0.0,
                    "dropout_after_attention": False,
                    "dropout_after_feedforward": False,
                    "dropout_on_residuals": False
                }

            self.dropout_config = dropout_config

            # Import necessary components from original Gemma implementation
            from gemma_model import GroupedQueryAttention, FeedForward, RMSNorm

            # Initialize original Gemma components
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

            # Layer normalization components
            self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
            self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
            self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
            self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

            # NEW: Configurable dropout layers for Perseid experiments
            self.attention_dropout = nn.Dropout(dropout_config["attention_dropout"])
            self.feedforward_dropout = nn.Dropout(dropout_config["feedforward_dropout"])
            self.residual_dropout = nn.Dropout(dropout_config["residual_dropout"])

            print(f"Initialized PerseidTransformerBlock with dropout config: "
                  f"attn={dropout_config['attention_dropout']}, "
                  f"ff={dropout_config['feedforward_dropout']}, "
                  f"residual={dropout_config['residual_dropout']}")

        except Exception as e:
            print(f"Error initializing PerseidTransformerBlock: {str(e)}")
            traceback.print_exc()
            raise


    def forward(
        self,
        x: torch.Tensor,
        mask_global: torch.Tensor,
        mask_local: torch.Tensor,
        cos_global: torch.Tensor,
        sin_global: torch.Tensor,
        cos_local: torch.Tensor,
        sin_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with strategic dropout insertion for regularization experiments.

        Args:
            x: Input tensor of shape (batch_size, seq_len, emb_dim)
            mask_global: Global attention mask
            mask_local: Local/sliding window attention mask
            cos_global: Cosine values for global RoPE
            sin_global: Sine values for global RoPE
            cos_local: Cosine values for local RoPE
            sin_local: Sine values for local RoPE

        Returns:
            Output tensor with same shape as input
        """
        try:
            # Attention block with optional dropout
            shortcut = x
            x_norm = self.input_layernorm(x)

            # Select attention type and corresponding mask/RoPE parameters
            if self.attn_type == "sliding_attention":
                x_attn = self.att(x_norm, mask_local, cos_local, sin_local)
            else:
                x_attn = self.att(x_norm, mask_global, cos_global, sin_global)

            x_attn = self.post_attention_layernorm(x_attn)

            # NEW: Apply attention dropout if configured
            if self.dropout_config["dropout_after_attention"] and self.training:
                x_attn = self.attention_dropout(x_attn)

            x = shortcut + x_attn

            # Feed forward block with optional dropout
            shortcut = x
            x_ffn = self.pre_feedforward_layernorm(x)
            x_ffn = self.ff(x_ffn)
            x_ffn = self.post_feedforward_layernorm(x_ffn)

            # NEW: Apply feedforward dropout if configured
            if self.dropout_config["dropout_after_feedforward"] and self.training:
                x_ffn = self.feedforward_dropout(x_ffn)

            x = shortcut + x_ffn

            # NEW: Apply residual dropout if configured (more aggressive regularization)
            if self.dropout_config["dropout_on_residuals"] and self.training:
                x = self.residual_dropout(x)

            return x

        except Exception as e:
            print(f"Error in PerseidTransformerBlock forward pass: {str(e)}")
            traceback.print_exc()
            raise


# 2. Perseid Model Class (perseid_model.py)

# In[ ]:


#!/usr/bin/env python
"""
Perseid Model Implementation
File: perseid_model.py

This module implements the Perseid family of models as variations
on the Gemma architecture for comparative experiments.
"""

import torch
import torch.nn as nn
import traceback
from typing import Dict, Optional, List
from pathlib import Path


class PerseidModel(nn.Module):
    """
    Perseid model - a variant of Gemma 270M for architecture comparison experiments.

    Key differences from Gemma:
    1. Configurable dropout at multiple positions
    2. Alternative sizing options for quantization optimization
    3. Modular design for experiment tracking

    Args:
        cfg (Dict): Model configuration
        dropout_strategy (str): Name of dropout strategy to use
    """

    def __init__(self, cfg: Dict, dropout_strategy: str = "baseline"):
        """
        Initialize Perseid model with specified configuration and dropout strategy.

        Args:
            cfg: Model configuration dictionary
            dropout_strategy: One of the predefined dropout strategies
        """
        super().__init__()

        try:
            self.cfg = cfg
            self.dropout_strategy = dropout_strategy

            # Import necessary components
            from gemma_model import compute_rope_params, RMSNorm
            from perseid_blocks import PerseidTransformerBlock

            # Get dropout configuration for this strategy
            dropout_config = self._get_dropout_config(dropout_strategy)

            # Store model variant name for tracking
            self.model_variant = dropout_config.get("name", f"Perseid-{dropout_strategy}")

            # Token embeddings
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

            # Create transformer blocks with dropout configuration
            self.blocks = nn.ModuleList([
                PerseidTransformerBlock(cfg, attn_type, dropout_config)
                for attn_type in cfg["layer_types"]
            ])

            # Final normalization and output head
            self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

            # Precompute RoPE parameters
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

            # Register as buffers (not trainable parameters)
            self.register_buffer("cos_local", cos_local, persistent=False)
            self.register_buffer("sin_local", sin_local, persistent=False)
            self.register_buffer("cos_global", cos_global, persistent=False)
            self.register_buffer("sin_global", sin_global, persistent=False)

            print(f"Initialized {self.model_variant} with {len(self.blocks)} layers")

        except Exception as e:
            print(f"Error initializing PerseidModel: {str(e)}")
            traceback.print_exc()
            raise

    def _get_dropout_config(self, strategy: str) -> Dict:
        """
        Get dropout configuration for specified strategy.

        Args:
            strategy: Name of dropout strategy

        Returns:
            Dictionary with dropout configuration parameters
        """
        dropout_strategies = {
            "baseline": {
                "attention_dropout": 0.0,
                "feedforward_dropout": 0.0,
                "residual_dropout": 0.0,
                "dropout_after_attention": False,
                "dropout_after_feedforward": False,
                "dropout_on_residuals": False,
                "name": "Perseid-Baseline(Gemma-equivalent)"
            },
            "conservative": {
                "attention_dropout": 0.05,
                "feedforward_dropout": 0.05,
                "residual_dropout": 0.0,
                "dropout_after_attention": True,
                "dropout_after_feedforward": True,
                "dropout_on_residuals": False,
                "name": "Perseid-Conservative-Dropout"
            },
            "moderate": {
                "attention_dropout": 0.1,
                "feedforward_dropout": 0.1,
                "residual_dropout": 0.05,
                "dropout_after_attention": True,
                "dropout_after_feedforward": True,
                "dropout_on_residuals": True,
                "name": "Perseid-Moderate-Dropout"
            },
            "aggressive": {
                "attention_dropout": 0.2,
                "feedforward_dropout": 0.2,
                "residual_dropout": 0.1,
                "dropout_after_attention": True,
                "dropout_after_feedforward": True,
                "dropout_on_residuals": True,
                "name": "Perseid-Aggressive-Dropout"
            },
            "attention_only": {
                "attention_dropout": 0.15,
                "feedforward_dropout": 0.0,
                "residual_dropout": 0.0,
                "dropout_after_attention": True,
                "dropout_after_feedforward": False,
                "dropout_on_residuals": False,
                "name": "Perseid-AttentionDropout-Only"
            },
            "feedforward_only": {
                "attention_dropout": 0.0,
                "feedforward_dropout": 0.15,
                "residual_dropout": 0.0,
                "dropout_after_attention": False,
                "dropout_after_feedforward": True,
                "dropout_on_residuals": False,
                "name": "Perseid-FeedforwardDropout-Only"
            }
        }

        if strategy not in dropout_strategies:
            print(f"Warning: Unknown dropout strategy '{strategy}', using baseline")
            return dropout_strategies["baseline"]

        return dropout_strategies[strategy]

    def _create_masks(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create attention masks for global and local attention patterns.

        Args:
            seq_len: Sequence length
            device: Device to create tensors on

        Returns:
            Tuple of (global_mask, local_mask)
        """
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # Global mask: mask future tokens
        mask_global = torch.triu(ones, diagonal=1)

        # Local mask: sliding window attention
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
        mask_local = mask_global | far_past

        return mask_global, mask_local

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Perseid model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        try:
            b, seq_len = input_ids.shape

            # Token embeddings with scaling
            x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

            # Create attention masks
            mask_global, mask_local = self._create_masks(seq_len, x.device)

            # Pass through transformer blocks
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

            # Final normalization and output projection
            x = self.final_norm(x)
            logits = self.out_head(x.to(self.cfg["dtype"]))

            return logits

        except Exception as e:
            print(f"Error in PerseidModel forward pass: {str(e)}")
            traceback.print_exc()
            raise

    def get_num_params(self) -> int:
        """
        Get total number of parameters in model.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters())


# # 3. Comparison Training Script (train_comparison.py)
# 

# In[ ]:


#!/usr/bin/env python
"""
Perseid vs. Gemma Comparison Training Script
File: train_comparison.py

This script trains and compares Gemma baseline with Perseid variants
to evaluate architectural modifications on fine-tuning performance.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

# Import models and utilities
from gemma_model import Gemma3Model, GEMMA3_CONFIG_270M, load_weights_into_gemma
from perseid_model import PerseidModel
from train_gemma_module import (
    GemmaTokenizer, 
    create_gemma_dataloader,
    calc_loss_batch,
    calc_loss_loader,
    generate_text_simple
)


class ComparisonExperiment:
    """
    Manages comparison experiments between Gemma and Perseid model variants.

    This class handles:
    - Model initialization with pretrained weights
    - Parallel training with identical conditions
    - Metric tracking and comparison
    - Result visualization and reporting
    """

    def __init__(self, base_config: Dict, experiment_name: str = "perseid_vs_gemma"):
        """
        Initialize comparison experiment.

        Args:
            base_config: Base model configuration (Gemma 270M config)
            experiment_name: Name for this experiment run
        """
        self.base_config = base_config
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"experiments/{experiment_name}_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("Warning: Using CPU - training will be slow")

        # Storage for experiment results
        self.models = {}
        self.training_histories = {}
        self.evaluation_metrics = {}

        print(f"Experiment initialized: {experiment_name}")
        print(f"Results will be saved to: {self.results_dir}")

    def initialize_models(self, dropout_strategies: List[str], pretrained_weights_path: Optional[str] = None):
        """
        Initialize Gemma baseline and Perseid variants with same pretrained weights.

        Args:
            dropout_strategies: List of dropout strategies to test
            pretrained_weights_path: Path to pretrained Gemma weights
        """
        try:
            print("\n" + "="*50)
            print("Initializing models for comparison")
            print("="*50)

            # Load pretrained weights if provided
            pretrained_weights = None
            if pretrained_weights_path:
                print(f"Loading pretrained weights from {pretrained_weights_path}")
                from safetensors.torch import load_file
                pretrained_weights = load_file(pretrained_weights_path)

            # Initialize Gemma baseline (original architecture)
            print("\n1. Initializing Gemma baseline model...")
            gemma_model = Gemma3Model(self.base_config)
            if pretrained_weights:
                load_weights_into_gemma(gemma_model, self.base_config, pretrained_weights)
            gemma_model = gemma_model.to(self.device)
            self.models["gemma_baseline"] = gemma_model
            print(f"   Gemma baseline: {gemma_model.get_num_params():,} parameters")

            # Initialize Perseid variants with different dropout strategies
            for strategy in dropout_strategies:
                print(f"\n2. Initializing Perseid variant: {strategy}")
                perseid_model = PerseidModel(self.base_config, dropout_strategy=strategy)

                # Load same pretrained weights into Perseid
                if pretrained_weights:
                    self._load_weights_into_perseid(perseid_model, pretrained_weights)

                perseid_model = perseid_model.to(self.device)
                model_key = f"perseid_{strategy}"
                self.models[model_key] = perseid_model
                print(f"   {perseid_model.model_variant}: {perseid_model.get_num_params():,} parameters")

            print(f"\nTotal models initialized: {len(self.models)}")

        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            traceback.print_exc()
            raise

    def _load_weights_into_perseid(self, perseid_model: PerseidModel, weights_dict: Dict):
        """
        Load Gemma pretrained weights into Perseid model.

        Dropout layers have no weights, so we only load the base architecture weights.

        Args:
            perseid_model: Perseid model instance
            weights_dict: Dictionary of pretrained weights
        """
        try:
            # Use the same loading function as Gemma since base architecture is identical
            load_weights_into_gemma(perseid_model, self.base_config, weights_dict)
            print(f"   Loaded pretrained weights into {perseid_model.model_variant}")

        except Exception as e:
            print(f"Error loading weights into Perseid: {str(e)}")
            traceback.print_exc()
            raise

    def run_comparison_training(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        eval_interval: int = 50,
        gradient_accumulation_steps: int = 4
    ):
        """
        Run parallel training on all model variants with identical conditions.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for all models
            eval_interval: Steps between evaluations
            gradient_accumulation_steps: Gradient accumulation for memory efficiency
        """
        try:
            print("\n" + "="*50)
            print("Starting comparison training")
            print("="*50)
            print(f"Epochs: {num_epochs}, LR: {learning_rate}, Eval interval: {eval_interval}")

            # Initialize optimizers for each model
            optimizers = {}
            for model_name, model in self.models.items():
                optimizers[model_name] = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=0.01,
                    betas=(0.9, 0.95)
                )
                self.training_histories[model_name] = {
                    "train_losses": [],
                    "val_losses": [],
                    "steps": [],
                    "epoch_times": []
                }

            # Training loop
            global_step = 0
            for epoch in range(num_epochs):
                epoch_start = time.time()
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 30)

                # Train each model on same batches
                for batch_idx, (input_batch, target_batch) in enumerate(train_loader):

                    # Train each model variant
                    for model_name, model in self.models.items():
                        model.train()
                        optimizer = optimizers[model_name]

                        # Calculate loss
                        loss = calc_loss_batch(input_batch, target_batch, model, self.device)
                        loss = loss / gradient_accumulation_steps
                        loss.backward()

                        # Update weights after accumulation
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()

                    # Evaluation at intervals
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        global_step += 1

                        if global_step % eval_interval == 0:
                            self._evaluate_all_models(
                                train_loader, val_loader, global_step, epoch
                            )

                # End of epoch timing
                epoch_time = time.time() - epoch_start
                for model_name in self.models:
                    self.training_histories[model_name]["epoch_times"].append(epoch_time)

                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

            print("\n" + "="*50)
            print("Training completed!")
            print("="*50)

        except Exception as e:
            print(f"Error during comparison training: {str(e)}")
            traceback.print_exc()
            raise

    def _evaluate_all_models(self, train_loader, val_loader, step: int, epoch: int):
        """
        Evaluate all models and record metrics.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            step: Current global step
            epoch: Current epoch
        """
        print(f"\nEvaluation at step {step} (epoch {epoch + 1}):")

        for model_name, model in self.models.items():
            model.eval()

            with torch.no_grad():
                # Calculate losses
                train_loss = calc_loss_loader(train_loader, model, self.device, num_batches=10)
                val_loss = calc_loss_loader(val_loader, model, self.device, num_batches=10)

                # Store metrics
                self.training_histories[model_name]["train_losses"].append(train_loss)
                self.training_histories[model_name]["val_losses"].append(val_loss)
                self.training_histories[model_name]["steps"].append(step)

                # Calculate perplexity
                val_perplexity = torch.exp(torch.tensor(val_loss)).item()

                print(f"  {model_name:30s} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                      f"Perplexity: {val_perplexity:.2f}")

    def analyze_results(self):
        """
        Analyze and compare training results across all model variants.

        Generates comparison metrics and identifies best performing variant.
        """
        print("\n" + "="*50)
        print("Results Analysis")
        print("="*50)

        analysis = {}

        for model_name, history in self.training_histories.items():
            if len(history["val_losses"]) > 0:
                final_val_loss = history["val_losses"][-1]
                best_val_loss = min(history["val_losses"])
                convergence_speed = self._calculate_convergence_speed(history["val_losses"])
                overfitting_gap = history["train_losses"][-1] - history["val_losses"][-1]

                analysis[model_name] = {
                    "final_val_loss": final_val_loss,
                    "best_val_loss": best_val_loss,
                    "final_perplexity": torch.exp(torch.tensor(final_val_loss)).item(),
                    "convergence_speed": convergence_speed,
                    "overfitting_gap": overfitting_gap,
                    "total_training_time": sum(history["epoch_times"])
                }

                print(f"\n{model_name}:")
                print(f"  Final validation loss: {final_val_loss:.4f}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                print(f"  Final perplexity: {analysis[model_name]['final_perplexity']:.2f}")
                print(f"  Convergence speed: {convergence_speed:.4f}")
                print(f"  Overfitting gap: {overfitting_gap:.4f}")
                print(f"  Total training time: {analysis[model_name]['total_training_time']:.2f}s")

        # Identify best model
        if analysis:
            best_model = min(analysis.keys(), key=lambda x: analysis[x]["best_val_loss"])
            print(f"\n🏆 Best performing model: {best_model}")
            print(f"   with validation loss: {analysis[best_model]['best_val_loss']:.4f}")

        self.evaluation_metrics = analysis

        # Save analysis to file
        analysis_path = self.results_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to {analysis_path}")

    def _calculate_convergence_speed(self, losses: List[float]) -> float:
        """
        Calculate convergence speed as rate of improvement.

        Args:
            losses: List of loss values

        Returns:
            Convergence speed metric
        """
        if len(losses) < 2:
            return 0.0

        # Calculate average rate of improvement
        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(improvement)

        return np.mean(improvements) if improvements else 0.0

    def plot_comparison(self):
        """
        Generate comparison plots for all model variants.

        Creates visualizations for loss curves, perplexity, and convergence.
        """
        try:
            print("\nGenerating comparison plots...")

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Perseid vs. Gemma Comparison - {self.experiment_name}", fontsize=16)

            colors = plt.cm.Set1(np.linspace(0, 1, len(self.models)))

            for idx, (model_name, history) in enumerate(self.training_histories.items()):
                if len(history["steps"]) == 0:
                    continue

                color = colors[idx]

                # Training loss
                axes[0, 0].plot(history["steps"], history["train_losses"], 
                              label=model_name, color=color, linestyle='-', alpha=0.7)

                # Validation loss
                axes[0, 1].plot(history["steps"], history["val_losses"],
                              label=model_name, color=color, linestyle='-')

                # Perplexity
                perplexities = [torch.exp(torch.tensor(loss)).item() 
                               for loss in history["val_losses"]]
                axes[1, 0].plot(history["steps"], perplexities,
                              label=model_name, color=color, linestyle='-')

                # Overfitting gap
                if len(history["train_losses"]) == len(history["val_losses"]):
                    gaps = [val - train for train, val in 
                           zip(history["train_losses"], history["val_losses"])]
                    axes[1, 1].plot(history["steps"], gaps,
                                  label=model_name, color=color, linestyle='-')

            # Configure subplots
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Steps")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].set_title("Validation Loss")
            axes[0, 1].set_xlabel("Steps")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].set_title("Validation Perplexity")
            axes[1, 0].set_xlabel("Steps")
            axes[1, 0].set_ylabel("Perplexity")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].set_title("Generalization Gap (Val - Train)")
            axes[1, 1].set_xlabel("Steps")
            axes[1, 1].set_ylabel("Loss Difference")
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = self.results_dir / "comparison_plots.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved to {plot_path}")
            plt.show()

        except Exception as e:
            print(f"Error generating plots: {str(e)}")
            traceback.print_exc()

    def save_results(self):
        """
        Save all experimental results and model checkpoints.

        Saves training histories, metrics, and best model weights.
        """
        try:
            print("\nSaving experimental results...")

            # Save training histories
            histories_path = self.results_dir / "training_histories.json"
            with open(histories_path, "w") as f:
                json.dump(self.training_histories, f, indent=2, default=str)
            print(f"Training histories saved to {histories_path}")

            # Save best model checkpoints
            checkpoints_dir = self.results_dir / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)

            for model_name, model in self.models.items():
                checkpoint_path = checkpoints_dir / f"{model_name}_final.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': self.base_config,
                    'model_variant': model_name,
                    'training_history': self.training_histories.get(model_name, {}),
                    'final_metrics': self.evaluation_metrics.get(model_name, {})
                }, checkpoint_path)
                print(f"  {model_name} checkpoint saved")

            # Save experiment summary
            summary = {
                "experiment_name": self.experiment_name,
                "timestamp": self.timestamp,
                "base_config": self.base_config,
                "models_tested": list(self.models.keys()),
                "evaluation_metrics": self.evaluation_metrics,
                "best_model": min(self.evaluation_metrics.keys(), 
                                 key=lambda x: self.evaluation_metrics[x]["best_val_loss"])
                                 if self.evaluation_metrics else None
            }

            summary_path = self.results_dir / "experiment_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Experiment summary saved to {summary_path}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            traceback.print_exc()

    def generate_comparison_samples(self, tokenizer, prompts: List[str], max_tokens: int = 50):
        """
        Generate text samples from all models for qualitative comparison.

        Args:
            tokenizer: Tokenizer instance
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
        """
        print("\n" + "="*50)
        print("Generating comparison samples")
        print("="*50)

        samples = {}

        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            print("-" * 30)
            samples[prompt] = {}

            for model_name, model in self.models.items():
                model.eval()

                # Generate text
                with torch.no_grad():
                    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
                    generated_ids = generate_text_simple(
                        model, input_ids, max_tokens,
                        eos_token_id=tokenizer.encode(tokenizer.eos_token)[-1] if tokenizer.eos_token else None
                    )
                    generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist())

                samples[prompt][model_name] = generated_text
                print(f"\n{model_name}:")
                print(f"  {generated_text}")

        # Save samples
        samples_path = self.results_dir / "generation_samples.json"
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"\nGeneration samples saved to {samples_path}")

        return samples


def main():
    """
    Main function to run Perseid vs. Gemma comparison experiment.

    This function orchestrates the complete experimental pipeline:
    1. Setup and configuration
    2. Data preparation
    3. Model initialization
    4. Training comparison
    5. Results analysis and visualization
    """
    try:
        print("\n" + "="*70)
        print(" PERSEID vs. GEMMA ARCHITECTURE COMPARISON EXPERIMENT")
        print("="*70)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Configuration
        GEMMA3_CONFIG_270M = {
            "vocab_size": 262_144,
            "context_length": 512,  # Reduced for memory efficiency
            "emb_dim": 640,
            "n_heads": 4,
            "n_layers": 18,
            "hidden_dim": 2048,
            "head_dim": 256,
            "qk_norm": True,
            "n_kv_groups": 1,
            "rope_local_base": 10_000.0,
            "rope_base": 1_000_000.0,
            "sliding_window": 512,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "dtype": torch.bfloat16,
            "query_pre_attn_scalar": 256,
        }

        # Training settings
        TRAINING_CONFIG = {
            "num_epochs": 3,
            "batch_size": 1,  # Small for memory constraints
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "eval_interval": 20,  # Evaluate every N steps
            "context_length": 512,
            "max_tokens_generate": 50
        }

        # Dropout strategies to test
        DROPOUT_STRATEGIES = [
            "baseline",      # No dropout (Gemma equivalent)
            "conservative",  # Light dropout (0.05)
            "moderate",      # Traditional dropout (0.1)
            "attention_only" # Dropout only on attention
        ]

        print("\nExperiment Configuration:")
        print(f"  Models to compare: Gemma baseline + {len(DROPOUT_STRATEGIES)} Perseid variants")
        print(f"  Training epochs: {TRAINING_CONFIG['num_epochs']}")
        print(f"  Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
        print(f"  Context length: {TRAINING_CONFIG['context_length']}")

        # Initialize experiment
        experiment = ComparisonExperiment(
            base_config=GEMMA3_CONFIG_270M,
            experiment_name="perseid_dropout_comparison"
        )

        # Setup tokenizer
        print("\n" + "="*50)
        print("Setting up tokenizer")
        print("="*50)

        from huggingface_hub import hf_hub_download

        repo_id = "google/gemma-3-270m"
        local_dir = Path(repo_id).parts[-1]

        tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file_path):
            print("Downloading Gemma tokenizer...")
            tokenizer_file_path = hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=local_dir
            )

        tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)
        print("Tokenizer loaded successfully")

        # Load training data
        print("\n" + "="*50)
        print("Loading training data")
        print("="*50)

        import urllib.request

        # Download sample text data
        file_path = "data/alice.txt"
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(file_path):
            url = "https://www.gutenberg.org/files/11/11-0.txt"
            print(f"Downloading training data from {url}")
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            print(f"Loading existing data from {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        # Clean text data (remove Gutenberg headers/footers)
        start_marker = "*** START OF"
        end_marker = "*** END OF"

        start_idx = text_data.find(start_marker)
        if start_idx != -1:
            start_idx = text_data.find("\n", start_idx) + 1
            text_data = text_data[start_idx:]

        end_idx = text_data.find(end_marker)
        if end_idx != -1:
            text_data = text_data[:end_idx]

        print(f"Loaded {len(text_data):,} characters")

        # Tokenize and check data size
        all_tokens = tokenizer.encode(text_data)
        print(f"Total tokens: {len(all_tokens):,}")

        # Create data loaders
        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))

        train_loader = create_gemma_dataloader(
            text_data[:split_idx],
            tokenizer=tokenizer,
            batch_size=TRAINING_CONFIG["batch_size"],
            max_length=TRAINING_CONFIG["context_length"],
            stride=TRAINING_CONFIG["context_length"],
            drop_last=True,
            shuffle=True
        )

        val_loader = create_gemma_dataloader(
            text_data[split_idx:],
            tokenizer=tokenizer,
            batch_size=TRAINING_CONFIG["batch_size"],
            max_length=TRAINING_CONFIG["context_length"],
            stride=TRAINING_CONFIG["context_length"],
            drop_last=False,
            shuffle=False
        )

        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        # Download pretrained weights
        print("\n" + "="*50)
        print("Downloading pretrained weights")
        print("="*50)

        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        print(f"Weights downloaded to {weights_file}")

        # Initialize models
        experiment.initialize_models(
            dropout_strategies=DROPOUT_STRATEGIES,
            pretrained_weights_path=weights_file
        )

        # Run comparison training
        experiment.run_comparison_training(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=TRAINING_CONFIG["num_epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            eval_interval=TRAINING_CONFIG["eval_interval"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"]
        )

        # Analyze results
        experiment.analyze_results()

        # Generate comparison plots
        experiment.plot_comparison()

        # Generate text samples for qualitative comparison
        test_prompts = [
            "Alice was beginning to get very tired",
            "The Queen of Hearts",
            "Down the rabbit hole",
            "Once upon a time"
        ]

        experiment.generate_comparison_samples(
            tokenizer=tokenizer,
            prompts=test_prompts,
            max_tokens=TRAINING_CONFIG["max_tokens_generate"]
        )

        # Save all results
        experiment.save_results()

        print("\n" + "="*70)
        print(" EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Results saved to: {experiment.results_dir}")

        # Print final summary
        if experiment.evaluation_metrics:
            print("\nFinal Performance Summary:")
            print("-" * 40)

            # Sort models by validation loss
            sorted_models = sorted(
                experiment.evaluation_metrics.items(),
                key=lambda x: x[1]["best_val_loss"]
            )

            for rank, (model_name, metrics) in enumerate(sorted_models, 1):
                print(f"{rank}. {model_name:30s} - Val Loss: {metrics['best_val_loss']:.4f}, "
                      f"Perplexity: {metrics['final_perplexity']:.2f}")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

