# generate_text_tool_perseid.py

# def generate_text_simple(model, tokenizer, prompt, max_new_tokens=50):
#     """Simple generation for evaluation using the working Gemma generation code"""
#     model.eval()

#     # Encode prompt
#     token_ids = tokenizer.encode(prompt)
#     input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

#     # Get EOS token ID if it exists
#     try:
#         eos_token_id = tokenizer.encode("<end_of_turn>")[-1]
#     except:
#         eos_token_id = None

#     generated_tokens = []

#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             # Get logits for last token
#             logits = model(input_ids)[:, -1, :]

#             # Get next token (greedy)
#             next_token = torch.argmax(logits, dim=-1, keepdim=True)

#             # Check for EOS
#             if eos_token_id is not None and torch.all(next_token == eos_token_id):
#                 break

#             generated_tokens.append(next_token.item())

#             # Append
#             input_ids = torch.cat([input_ids, next_token], dim=1)

#             # Truncate if too long
#             if input_ids.shape[1] > model.cfg["context_length"]:
#                 input_ids = input_ids[:, -model.cfg["context_length"]:]

#     # Decode the full sequence
#     full_sequence = token_ids + generated_tokens
#     return tokenizer.decode(full_sequence)

# # Alternative: Use the streaming version from the original notebook
# def generate_text_stream(model, tokenizer, prompt, max_new_tokens=50):
#     """Using the exact generation code from the working Gemma notebook"""
#     model.eval()

#     # Encode prompt
#     token_ids = tokenizer.encode(prompt)
#     input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

#     # Get EOS token ID
#     try:
#         eos_token_id = tokenizer.encode("<end_of_turn>")[-1]
#     except:
#         eos_token_id = None

#     generated = token_ids.copy()  # Start with the prompt

#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             out = model(input_ids)[:, -1]
#             next_token = torch.argmax(out, dim=-1, keepdim=True)

#             if eos_token_id is not None and torch.all(next_token == eos_token_id):
#                 break

#             generated.append(next_token.item())
#             input_ids = torch.cat([input_ids, next_token], dim=1)

#             # Handle context overflow
#             if input_ids.shape[1] > model.cfg["context_length"]:
#                 input_ids = input_ids[:, -model.cfg["context_length"]:]

#     return tokenizer.decode(generated)

# # Test generation before training
# test_prompts = [
#     "The meaning of life is",
#     "Once upon a time",
#     "In the beginning",
# ]


# print("Model generation before training:\n")
# for prompt in test_prompts:
#     print(f"Prompt: '{prompt}'")
#     output = generate_text_simple(model, tokenizer, prompt, max_new_tokens=30)
#     print(f"Output: {output}\n")

# # ### Advanced Generation with Temperature and Top-k
# #
# # Test more sophisticated generation strategies

# def generate_advanced(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
#     """Advanced generation with temperature and top-k sampling"""
#     model.eval()

#     # Encode prompt
#     token_ids = tokenizer.encode(prompt)
#     input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

#     # Get EOS token ID
#     try:
#         eos_token_id = tokenizer.encode("<end_of_turn>")[-1]
#     except:
#         eos_token_id = None

#     generated = []

#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             # Get logits
#             logits = model(input_ids)[:, -1, :]

#             # Apply temperature
#             if temperature > 0:
#                 logits = logits / temperature

#             # Top-k filtering
#             if top_k > 0:
#                 indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#                 logits[indices_to_remove] = float('-inf')

#             # Sample
#             probs = torch.softmax(logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)

#             # Check for EOS
#             if eos_token_id is not None and torch.all(next_token == eos_token_id):
#                 break

#             generated.append(next_token.item())

#             # Append
#             input_ids = torch.cat([input_ids, next_token], dim=1)

#             # Truncate if needed
#             if input_ids.shape[1] > model.cfg["context_length"]:
#                 input_ids = input_ids[:, -model.cfg["context_length"]:]

#     # Decode full sequence
#     full_sequence = token_ids + generated
#     return tokenizer.decode(full_sequence)

# # Test different generation settings
# generation_configs = [
#     {"temperature": 0.5, "top_k": 10, "name": "Conservative"},
#     {"temperature": 0.8, "top_k": 50, "name": "Balanced"},
#     {"temperature": 1.2, "top_k": 100, "name": "Creative"},
# ]

# prompt = "The future of artificial intelligence"

# print("Testing different generation strategies:\n")
# for config in generation_configs:
#     print(f"{config['name']} (temp={config['temperature']}, top_k={config['top_k']}):")
#     output = generate_advanced(
#         model, tokenizer, prompt,
#         temperature=config['temperature'],
#         top_k=config['top_k'],
#         max_new_tokens=50
#     )
#     print(f"{output}\n")
#     print("-" * 80)


# # # Initialize model
# # model = Gemma3Model(GEMMA3_CONFIG_270M)

# # Reload the complete model for continued training or inference
# import torch

# # Load the checkpoint
# checkpoint = torch.load('gemma3_trained_complete.pth', map_location=device)

# # Recreate the model with the same configuration
# model = Gemma3Model(checkpoint['config'])

# # Load the state dictionary
# model.load_state_dict(checkpoint['model_state_dict'])

# # Move to device
# model = model.to(device)



# # Or for inference
# model.eval()

# # Test different generation settings
# generation_configs = [
#     {"temperature": 0.5, "top_k": 10, "name": "Conservative"},
#     {"temperature": 0.8, "top_k": 50, "name": "Balanced"},
#     {"temperature": 1.2, "top_k": 100, "name": "Creative"},
# ]

# prompt = "The future of artificial intelligence"

# print("Testing different generation strategies:\n")
# for config in generation_configs:
#     print(f"{config['name']} (temp={config['temperature']}, top_k={config['top_k']}):")
#     output = generate_advanced(
#         model, tokenizer, prompt,
#         temperature=config['temperature'],
#         top_k=config['top_k'],
#         max_new_tokens=50
#     )
#     print(f"{output}\n")
#     print("-" * 80)

"""
generate_text_tool_perseid.py

Simple text generation using trained Perseid models.
Point to your saved model and generate text.
"""

import torch
from pathlib import Path
from gemma_model import Gemma3Model
from tokenizers import Tokenizer

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Point to your trained model
# CHECKPOINT_PATH = "./models/perseid_256m_alice/checkpoint_best.pth"  # <- MODIFY THIS
CHECKPOINT_PATH = "./models/perseid_256m_alice/perseid_model_final.pth"  # <- MODIFY THIS
# or: "./models/perseid_256m_my_document/perseid_model_final.pth"

# Generation settings
PROMPTS = [
    "Once upon a time",
    "The meaning of life is",
    "In the beginning",
]
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50

# ============================================================================

def load_perseid_model(checkpoint_path):
    """Load trained Perseid model from checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_config' in checkpoint:
        # Full checkpoint with config
        model = Gemma3Model(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Just state dict - need to load config from same directory
        config_path = checkpoint_path.parent / "model_config.json"
        import json
        with open(config_path) as f:
            config = json.load(f)
        model = Gemma3Model(config)
        model.load_state_dict(checkpoint)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device), device

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
    """Generate text from prompt"""
    model.eval()
    device = next(model.parameters()).device

    token_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if input_ids.shape[1] > model.cfg["context_length"]:
                input_ids = input_ids[:, -model.cfg["context_length"]:]

    return tokenizer.decode(token_ids + generated)

def main():
    # Load model
    model, device = load_perseid_model(CHECKPOINT_PATH)
    print(f"Model loaded on {device}")

    # Setup tokenizer (assumes tokenizer.json in same directory as checkpoint)
    tokenizer_path = Path(CHECKPOINT_PATH).parent / "tokenizer.json"
    if not tokenizer_path.exists():
        # Fallback to download if needed
        print("Downloading tokenizer...")
        from huggingface_hub import hf_hub_download
        tokenizer_path = hf_hub_download("google/gemma-3-270m", "tokenizer.json", local_dir=".")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Generate from prompts
    print("\n" + "="*60)
    print("Generating Text")
    print("="*60)

    for prompt in PROMPTS:
        print(f"\nPrompt: '{prompt}'")
        output = generate_text(model, tokenizer, prompt, MAX_NEW_TOKENS, TEMPERATURE, TOP_K)
        print(f"Output: {output}")
        print("-"*40)

import sys
if __name__ == "__main__":

    # Run
    main()
