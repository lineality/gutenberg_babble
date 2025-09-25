# generate_text_perseid_byte.py

"""
generate_text_tool_perseid_byte.py

Simple text generation using trained PerseidByte models.
Point to your saved model and generate text using ByteTokenizer.
"""

import torch
import traceback
import sys
from pathlib import Path
from typing import Optional, Tuple

# Import PerseidByte architecture and tokenizer
try:
    from perseid_model import PerseidByteModel, PERSEID_BYTE_CONFIG_BASE
    from byte_tokenizer import ByteTokenizer
    from perseid_config_tools import validate_config, calculate_model_params
except ImportError as import_error:
    print(f"Error importing required modules: {import_error}")
    print("Ensure all required files are in the same directory:")
    print("- perseid_model.py")
    print("- byte_tokenizer.py")
    print("- perseid_config_tools.py")
    sys.exit(1)

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Point to your trained PerseidByte model
CHECKPOINT_PATH = "./models/perseid_256m_alice/perseid_model_final.pth"  # <- MODIFY THIS
# Alternative examples:
# CHECKPOINT_PATH = "./models/perseid_256m_alice/checkpoint_best.pth"
# CHECKPOINT_PATH = "./models/perseid_256m_my_document/perseid_model_final.pth"

# Generation settings
PROMPTS = [
    "Once upon a time",
    "The meaning of life is",
    "In the beginning",
    "Alice was beginning to get very tired",
    "The quick brown fox",
]

# Generation parameters
MAX_NEW_TOKENS = 100        # Maximum tokens to generate
TEMPERATURE = 0.8           # Sampling temperature (0.0 = greedy, higher = more random)
TOP_K = 50                 # Top-k sampling (0 = disabled)
TOP_P = 0.9                # Nucleus sampling (0.0 = disabled)

# ============================================================================

def infer_config_from_checkpoint(checkpoint_data: dict) -> dict:
    """
    Automatically infer PerseidByte model configuration from checkpoint weight shapes.

    Args:
        checkpoint_data: Loaded checkpoint dictionary

    Returns:
        dict: Inferred model configuration

    Raises:
        RuntimeError: If configuration inference fails
    """
    try:
        # Get state dict (handle different checkpoint formats)
        if "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        elif "model" in checkpoint_data:
            state_dict = checkpoint_data["model"]
        else:
            state_dict = checkpoint_data

        # Infer dimensions from weight shapes
        tok_emb_shape = state_dict["tok_emb.weight"].shape
        vocab_size, emb_dim = tok_emb_shape

        # Infer from attention weights
        q_proj_shape = state_dict["blocks.0.att.W_query.weight"].shape
        total_q_dim, _ = q_proj_shape

        k_proj_shape = state_dict["blocks.0.att.W_key.weight"].shape
        total_kv_dim, _ = k_proj_shape

        # Calculate heads and head dimensions
        head_dim = total_kv_dim  # For GQA, this works
        n_kv_groups = 1  # Assuming MHA for now
        n_heads = total_q_dim // head_dim

        # Infer hidden_dim from feedforward weights
        ff_shape = state_dict["blocks.0.ff.fc1.weight"].shape
        hidden_dim, _ = ff_shape

        # Count transformer layers
        n_layers = 0
        while f"blocks.{n_layers}.att.W_query.weight" in state_dict:
            n_layers += 1

        # Create reasonable layer types pattern
        layer_types = []
        for i in range(n_layers):
            if (i + 1) % 6 == 0:  # Every 6th layer is full attention
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")

        # Build configuration from inferred values
        inferred_config = PERSEID_BYTE_CONFIG_BASE.copy()
        inferred_config.update({
            "vocab_size": int(vocab_size),
            "emb_dim": int(emb_dim),
            "n_heads": int(n_heads),
            "n_kv_groups": int(n_kv_groups),
            "head_dim": int(head_dim),
            "n_layers": int(n_layers),
            "hidden_dim": int(hidden_dim),
            "layer_types": layer_types,
        })

        print(f"✓ Inferred model configuration:")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - emb_dim: {emb_dim}")
        print(f"  - n_heads: {n_heads}")
        print(f"  - head_dim: {head_dim}")
        print(f"  - n_layers: {n_layers}")
        print(f"  - hidden_dim: {hidden_dim}")

        return inferred_config

    except Exception as inference_error:
        raise RuntimeError(f"Failed to infer config from checkpoint: {str(inference_error)}") from inference_error


def load_perseid_byte_model(checkpoint_path: str) -> Tuple[PerseidByteModel, torch.device, dict]:
    """
    Load trained PerseidByte model from checkpoint for text generation.

    Args:
        checkpoint_path: Path to the model checkpoint file

    Returns:
        Tuple containing:
        - model: Loaded PerseidByteModel ready for generation
        - device: PyTorch device where model is loaded
        - config: Model configuration dictionary

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    try:
        checkpoint_file_path = Path(checkpoint_path)

        if not checkpoint_file_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading PerseidByte model from: {checkpoint_path}")
        print(f"Checkpoint size: {checkpoint_file_path.stat().st_size / (1024*1024):.1f} MB")

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load checkpoint
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            print("✓ Checkpoint loaded successfully")
        except Exception as load_error:
            raise RuntimeError(f"Failed to load checkpoint: {str(load_error)}") from load_error

        # Extract or infer model configuration
        if "config" in checkpoint_data or "model_config" in checkpoint_data:
            model_config = checkpoint_data.get("config", checkpoint_data.get("model_config"))
            print("✓ Using model configuration from checkpoint")
        else:
            print("⚠ Config not found in checkpoint, inferring from weights...")
            model_config = infer_config_from_checkpoint(checkpoint_data)

        # Validate configuration
        is_config_valid, validation_issues = validate_config(model_config)
        if not is_config_valid:
            print("⚠ Configuration validation issues:")
            for issue in validation_issues:
                print(f"  - {issue}")

        # Create model instance
        try:
            model = PerseidByteModel(model_config)
            print("✓ Model architecture created")

            # Display model statistics
            param_info = calculate_model_params(model_config)
            print(f"  - Parameters: {param_info['total_millions']:.1f}M")
            print(f"  - Embedding dim: {model_config['emb_dim']}")

        except Exception as model_error:
            raise RuntimeError(f"Failed to create model: {str(model_error)}") from model_error

        # Load model weights
        try:
            if "model_state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state_dict"])
            elif "model" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model"])
            else:
                model.load_state_dict(checkpoint_data)
            print("✓ Model weights loaded")
        except Exception as weight_error:
            raise RuntimeError(f"Failed to load weights: {str(weight_error)}") from weight_error

        # Move to device and set to evaluation mode
        model = model.to(device)
        model.eval()

        print(f"✓ Model ready for generation on {device}")
        return model, device, model_config

    except Exception as load_error:
        error_traceback = traceback.format_exc()
        raise RuntimeError(
            f"Failed to load PerseidByte model from {checkpoint_path}. "
            f"Error: {str(load_error)}\n"
            f"Traceback:\n{error_traceback}"
        ) from load_error


def setup_byte_tokenizer() -> ByteTokenizer:
    """
    Initialize ByteTokenizer for text generation.

    Returns:
        ByteTokenizer: Configured tokenizer

    Raises:
        RuntimeError: If tokenizer setup fails
    """
    try:
        print("Initializing ByteTokenizer...")
        tokenizer = ByteTokenizer()

        # Test tokenizer functionality
        test_text = "Hello, world!"
        test_tokens = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_tokens)

        if test_decoded == test_text:
            print("✓ ByteTokenizer ready")
            print(f"  - Vocab size: {tokenizer.vocab_size}")
            print(f"  - Special tokens: PAD={tokenizer.PAD_ID}, EOS={tokenizer.EOS_ID}")
        else:
            print(f"⚠ Tokenizer test failed: '{test_text}' != '{test_decoded}'")

        return tokenizer

    except Exception as tokenizer_error:
        raise RuntimeError(f"Failed to setup ByteTokenizer: {str(tokenizer_error)}") from tokenizer_error


def generate_text_perseid_byte(
    model: PerseidByteModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate text from a prompt using PerseidByte model and ByteTokenizer.

    Args:
        model: PerseidByte model in evaluation mode
        tokenizer: ByteTokenizer for encoding/decoding
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (0 = disabled)
        top_p: Nucleus sampling parameter (0.0 = disabled)
        device: PyTorch device for computation

    Returns:
        str: Generated text including the original prompt

    Raises:
        ValueError: If prompt is empty or parameters are invalid
        RuntimeError: If generation fails
    """
    try:
        # Input validation
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt must be a non-empty string")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if top_k < 0:
            raise ValueError("top_k must be non-negative (0 = disabled)")

        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")

        # Set device
        if device is None:
            device = next(model.parameters()).device

        # Ensure model is in evaluation mode
        model.eval()

        # Tokenize prompt
        try:
            token_ids = tokenizer.encode(prompt.strip())
            if len(token_ids) == 0:
                raise ValueError(f"Prompt tokenized to empty sequence: '{prompt}'")
        except Exception as tokenize_error:
            raise RuntimeError(f"Failed to tokenize prompt: {str(tokenize_error)}") from tokenize_error

        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        original_length = input_ids.shape[1]

        print(f"  Generating from prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print(f"  Prompt tokens: {len(token_ids)}")

        # Generation loop
        generated_tokens = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                try:
                    # Forward pass
                    logits = model(input_ids)[:, -1, :]  # Shape: (batch_size, vocab_size)

                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature

                    # Apply top-k filtering
                    if top_k > 0:
                        # Find top-k values and set others to -inf
                        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits_filtered = torch.full_like(logits, float('-inf'))
                        logits_filtered.scatter_(1, top_k_indices, top_k_values)
                        logits = logits_filtered

                    # Apply top-p (nucleus) sampling
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0  # Keep at least one token

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits.scatter_(1, indices_to_remove.unsqueeze(0), float('-inf'))

                    # Sample from the filtered distribution
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_token_id = next_token.item()

                    # Add to generated sequence
                    generated_tokens.append(next_token_id)
                    input_ids = torch.cat([input_ids, next_token], dim=1)

                    # Handle context window overflow
                    if input_ids.shape[1] > model.cfg["context_length"]:
                        # Keep the most recent tokens within context window
                        input_ids = input_ids[:, -model.cfg["context_length"]:]

                    # Optional: Stop on EOS token
                    if next_token_id == tokenizer.EOS_ID:
                        print(f"  Generation stopped at EOS token (step {step + 1})")
                        break

                except Exception as generation_step_error:
                    raise RuntimeError(
                        f"Generation failed at step {step}: {str(generation_step_error)}"
                    ) from generation_step_error

        # Decode the complete sequence
        try:
            all_token_ids = token_ids + generated_tokens
            generated_text = tokenizer.decode(all_token_ids)

            print(f"  Generated {len(generated_tokens)} new tokens")
            return generated_text

        except Exception as decode_error:
            raise RuntimeError(f"Failed to decode generated tokens: {str(decode_error)}") from decode_error

    except Exception as generation_error:
        error_traceback = traceback.format_exc()
        raise RuntimeError(
            f"Text generation failed for prompt: '{prompt[:50]}...'. "
            f"Error: {str(generation_error)}\n"
            f"Traceback:\n{error_traceback}"
        ) from generation_error


def main():
    """
    Main function to load model and generate text from configured prompts.
    """
    try:
        print("="*80)
        print("PerseidByte Text Generation")
        print("="*80)

        # Step 1: Load model
        print("\nStep 1: Loading Model")
        print("-" * 40)
        model, device, model_config = load_perseid_byte_model(CHECKPOINT_PATH)

        # Step 2: Setup tokenizer
        print("\nStep 2: Setting Up Tokenizer")
        print("-" * 40)
        tokenizer = setup_byte_tokenizer()

        # Step 3: Generate text from prompts
        print("\nStep 3: Generating Text")
        print("-" * 40)

        for i, prompt in enumerate(PROMPTS, 1):
            print(f"\n--- Generation {i}/{len(PROMPTS)} ---")

            try:
                generated_text = generate_text_perseid_byte(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    device=device
                )

                print(f"\nPrompt: '{prompt}'")
                print(f"Generated: {generated_text}")
                print("-" * 60)

            except Exception as prompt_error:
                print(f"✗ Failed to generate text for prompt '{prompt}': {str(prompt_error)}")
                continue

        print("\n" + "="*80)
        print("Text Generation Complete!")
        print("="*80)

        # Display generation settings used
        print(f"\nGeneration Settings:")
        print(f"  - Max new tokens: {MAX_NEW_TOKENS}")
        print(f"  - Temperature: {TEMPERATURE}")
        print(f"  - Top-k: {TOP_K}")
        print(f"  - Top-p: {TOP_P}")
        print(f"  - Model: {CHECKPOINT_PATH}")

    except KeyboardInterrupt:
        print("\n⚠ Generation interrupted by user")
        sys.exit(1)

    except Exception as main_error:
        error_traceback = traceback.format_exc()
        print(f"\n✗ Fatal error during text generation:")
        print(f"Error: {str(main_error)}")
        print(f"Full traceback:\n{error_traceback}")
        sys.exit(1)


def interactive_mode():
    """
    Interactive text generation mode - prompts user for input.
    """
    try:
        print("="*80)
        print("PerseidByte Interactive Text Generation")
        print("="*80)
        print("Type 'quit' to exit, 'settings' to change parameters")

        # Load model once
        print("\nLoading model...")
        model, device, model_config = load_perseid_byte_model(CHECKPOINT_PATH)
        tokenizer = setup_byte_tokenizer()

        # Interactive settings
        interactive_max_tokens = MAX_NEW_TOKENS
        interactive_temperature = TEMPERATURE
        interactive_top_k = TOP_K
        interactive_top_p = TOP_P

        while True:
            try:
                # Get user input
                user_prompt = input("\nEnter your prompt: ").strip()

                if user_prompt.lower() == 'quit':
                    print("Goodbye!")
                    break

                elif user_prompt.lower() == 'settings':
                    print(f"\nCurrent settings:")
                    print(f"  Max tokens: {interactive_max_tokens}")
                    print(f"  Temperature: {interactive_temperature}")
                    print(f"  Top-k: {interactive_top_k}")
                    print(f"  Top-p: {interactive_top_p}")

                    try:
                        new_max = input(f"Max tokens ({interactive_max_tokens}): ").strip()
                        if new_max:
                            interactive_max_tokens = int(new_max)

                        new_temp = input(f"Temperature ({interactive_temperature}): ").strip()
                        if new_temp:
                            interactive_temperature = float(new_temp)

                        new_k = input(f"Top-k ({interactive_top_k}): ").strip()
                        if new_k:
                            interactive_top_k = int(new_k)

                        new_p = input(f"Top-p ({interactive_top_p}): ").strip()
                        if new_p:
                            interactive_top_p = float(new_p)

                        print("✓ Settings updated")

                    except ValueError as setting_error:
                        print(f"Invalid setting: {setting_error}")

                    continue

                elif len(user_prompt) == 0:
                    print("Please enter a prompt or 'quit' to exit")
                    continue

                # Generate text
                print(f"Generating... (max {interactive_max_tokens} tokens)")

                generated_text = generate_text_perseid_byte(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_prompt,
                    max_new_tokens=interactive_max_tokens,
                    temperature=interactive_temperature,
                    top_k=interactive_top_k,
                    top_p=interactive_top_p,
                    device=device
                )

                print(f"\n{'='*60}")
                print(f"Generated Text:")
                print(f"{'='*60}")
                print(generated_text)
                print(f"{'='*60}")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit properly")
                continue

            except Exception as interactive_error:
                print(f"Error: {str(interactive_error)}")
                continue

    except Exception as interactive_mode_error:
        print(f"Interactive mode failed: {str(interactive_mode_error)}")
        sys.exit(1)


def batch_generate_from_file(input_file_path: str, output_file_path: str):
    """
    Generate text from prompts in a file and save results.

    Args:
        input_file_path: Path to text file with one prompt per line
        output_file_path: Path to save generated results
    """
    try:
        input_path = Path(input_file_path)
        output_path = Path(output_file_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        print(f"Batch generation from: {input_file_path}")
        print(f"Output will be saved to: {output_file_path}")

        # Load model
        model, device, model_config = load_perseid_byte_model(CHECKPOINT_PATH)
        tokenizer = setup_byte_tokenizer()

        # Read prompts
        with open(input_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Found {len(prompts)} prompts to process")

        # Generate for each prompt
        results = []

        for i, prompt in enumerate(prompts, 1):
            print(f"\nProcessing prompt {i}/{len(prompts)}: '{prompt[:50]}...'")

            try:
                generated_text = generate_text_perseid_byte(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    device=device
                )

                results.append({
                    "prompt": prompt,
                    "generated": generated_text,
                    "success": True
                })

            except Exception as batch_error:
                print(f"  ✗ Failed: {str(batch_error)}")
                results.append({
                    "prompt": prompt,
                    "generated": None,
                    "success": False,
                    "error": str(batch_error)
                })

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("PerseidByte Batch Generation Results\n")
            f.write("=" * 60 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"--- Generation {i} ---\n")
                f.write(f"Prompt: {result['prompt']}\n")

                if result['success']:
                    f.write(f"Generated: {result['generated']}\n")
                else:
                    f.write(f"Error: {result['error']}\n")

                f.write("\n" + "-" * 40 + "\n\n")

        successful = sum(1 for r in results if r['success'])
        print(f"\nBatch generation complete!")
        print(f"  Successful: {successful}/{len(prompts)}")
        print(f"  Results saved to: {output_path}")

    except Exception as batch_generation_error:
        raise RuntimeError(f"Batch generation failed: {str(batch_generation_error)}") from batch_generation_error


if __name__ == "__main__":
    # Parse command line arguments for different modes
    if len(sys.argv) == 1:
        # Default mode - generate from configured prompts
        main()

    elif len(sys.argv) == 2:
        if sys.argv[1].lower() == "interactive":
            # Interactive mode
            interactive_mode()
        else:
            print("Usage:")
            print("  python generate_text_tool_perseid_byte.py                    # Standard mode")
            print("  python generate_text_tool_perseid_byte.py interactive        # Interactive mode")
            print("  python generate_text_tool_perseid_byte.py batch input.txt output.txt  # Batch mode")

    elif len(sys.argv) == 4 and sys.argv[1].lower() == "batch":
        # Batch mode
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        batch_generate_from_file(input_file, output_file)

    else:
        print("Invalid arguments. Usage:")
        print("  python generate_text_tool_perseid_byte.py                    # Standard mode")
        print("  python generate_text_tool_perseid_byte.py interactive        # Interactive mode")
        print("  python generate_text_tool_perseid_byte.py batch input.txt output.txt  # Batch mode")
        sys.exit(1)
