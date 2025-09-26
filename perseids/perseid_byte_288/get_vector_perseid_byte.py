"""
get_embedding_vector_perseidbyte.py

Script for extracting semantic embeddings from trained PerseidByte models.
Supports both last-token and mean pooling strategies for generating dense vector
representations suitable for similarity search, clustering, and other NLP tasks.

This script provides a complete pipeline from text input to embedding vectors,
including model loading, tokenization, and embedding extraction with proper
error handling and validation.

Usage:
    python get_embedding_vector_perseidbyte.py

Configuration:
    Modify the USER CONFIGURATION section below to:
    - Point to your trained model checkpoint
    - Specify input texts for embedding extraction
    - Choose pooling method and output format
"""

import torch
import numpy as np
from pathlib import Path
import traceback
import json
from typing import List, Dict, Tuple, Optional, Union
import argparse
import sys
import glob

# Import model architecture and configuration tools
try:
    from byte_tokenizer import ByteTokenizer
    from perseid_config_tools import (
        create_perseid_config,
        calculate_model_params,
        validate_config,
    )
    from perseid_model import PerseidByteModel, PERSEID_BYTE_CONFIG_BASE
except ImportError as import_error:
    print(f"Error importing required modules: {import_error}")
    print("Ensure all required files are in the same directory:")
    print("- byte_tokenizer.py")
    print("- perseid_config_tools.py")
    print("- perseid_model.py")
    sys.exit(1)

# ============================================================================
# USER CONFIGURATION
# ============================================================================


# Example path with wildcard
pattern = "./models/perseid_*/perseid_model_final.pth"

# Find all matching paths
matching_paths = glob.glob(pattern)

print("Found...")
print(matching_paths)
print("default to using first option")

CHECKPOINT_PATH = matching_paths[0]

# Point to your trained model checkpoint
# CHECKPOINT_PATH = "./models/perseid_256m_alice/perseid_model_final.pth"  # <- MODIFY THIS
# Alternative examples:
# CHECKPOINT_PATH = "./models/perseid_256m_alice/checkpoint_best.pth"
# CHECKPOINT_PATH = "./models/perseid_256m_my_document/perseid_model_final.pth"

# Input texts for embedding extraction
INPUT_TEXTS = [
    "Once upon a time, in a land far away, there lived a wise old wizard.",
    "The lazy dog in the moonlit pub jumped over the slow brown fox.",
    "Knead the bread and put it in the oven.",
    "Bake the bread, then send it to the table.",
    "1 + 1 = one and one.",
    "The quick brown fox jumps over the lazy dog in the moonlit forest.",
]

from datetime import datetime, UTC as datetime_UTC

# get time
sample_time = datetime.now(datetime_UTC)
# make readable string
readable_timesatamp = sample_time.strftime("%Y_%m_%d__%H_%M_%S%f")

# Embedding extraction settings
POOLING_METHOD = "last_token"  # Options: "last_token", "mean"
OUTPUT_FORMAT = "numpy"  # Options: "numpy", "torch", "json"
SAVE_EMBEDDINGS = True  # Whether to save embeddings to file
OUTPUT_FILE = f"./embeddings_output_{readable_timesatamp}.json"  # Output file path

# Model configuration (will be loaded from checkpoint if available)
MODEL_CONFIG = PERSEID_BYTE_CONFIG_BASE.copy()

# ============================================================================


def infer_config_from_checkpoint(checkpoint_data: dict) -> dict:
    """
    Automatically infer model configuration from checkpoint weight shapes.
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

        # Calculate heads
        head_dim = (
            total_kv_dim  # For this checkpoint, looks like head_dim = total_kv_dim
        )
        n_kv_groups = 1  # Assuming MHA for now
        n_heads = total_q_dim // head_dim

        # Infer hidden_dim from FF weights
        ff_shape = state_dict["blocks.0.ff.fc1.weight"].shape
        hidden_dim, _ = ff_shape

        # Count layers
        n_layers = 0
        while f"blocks.{n_layers}.att.W_query.weight" in state_dict:
            n_layers += 1

        # Create layer types (simple pattern)
        layer_types = []
        for i in range(n_layers):
            if (i + 1) % 6 == 0:  # Every 6th layer
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")

        inferred_config = PERSEID_BYTE_CONFIG_BASE.copy()
        inferred_config.update(
            {
                "vocab_size": int(vocab_size),
                "emb_dim": int(emb_dim),
                "n_heads": int(n_heads),
                "n_kv_groups": int(n_kv_groups),
                "head_dim": int(head_dim),
                "n_layers": int(n_layers),
                "hidden_dim": int(hidden_dim),
                "layer_types": layer_types,
            }
        )

        print(f"✓ Inferred configuration from checkpoint:")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - emb_dim: {emb_dim}")
        print(f"  - n_heads: {n_heads}")
        print(f"  - head_dim: {head_dim}")
        print(f"  - n_layers: {n_layers}")
        print(f"  - hidden_dim: {hidden_dim}")

        return inferred_config

    except Exception as inference_error:
        raise RuntimeError(
            f"Failed to infer config from checkpoint: {str(inference_error)}"
        ) from inference_error


def load_perseid_model_for_embeddings(
    checkpoint_path: str,
) -> Tuple[PerseidByteModel, torch.device, Dict]:
    """
    Load a trained PerseidByte model from checkpoint for embedding extraction.

    Args:
        checkpoint_path (str): Path to the model checkpoint file

    Returns:
        Tuple containing:
        - model: Loaded PerseidByteModel ready for inference
        - device: PyTorch device (CPU or CUDA) where model is loaded
        - config: Model configuration dictionary used during training

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails due to compatibility issues
        ValueError: If checkpoint contains invalid configuration
    """
    try:
        checkpoint_file_path = Path(checkpoint_path)

        if not checkpoint_file_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {checkpoint_path}. "
                f"Please verify the path exists and contains a valid .pth file."
            )

        print(f"Loading model checkpoint from: {checkpoint_path}")
        print(
            f"Checkpoint file size: {checkpoint_file_path.stat().st_size / (1024 * 1024):.1f} MB"
        )

        # Determine device for model loading
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU (GPU not available)")

        # Load checkpoint with proper device mapping
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            print("✓ Checkpoint loaded successfully")
        except Exception as load_error:
            raise RuntimeError(
                f"Failed to load checkpoint file. This may indicate file corruption "
                f"or incompatible PyTorch versions. Original error: {str(load_error)}"
            ) from load_error

        # # Extract model configuration from checkpoint or use default
        # if "config" in checkpoint_data:
        #     model_configuration = checkpoint_data["config"]
        #     print("✓ Using model configuration from checkpoint")
        # else:
        #     model_configuration = MODEL_CONFIG
        #     print("⚠ Using default configuration (config not found in checkpoint)")

        if "config" in checkpoint_data or "model_config" in checkpoint_data:
            model_configuration = checkpoint_data.get(
                "config", checkpoint_data.get("model_config")
            )
            print("✓ Using model configuration from checkpoint")
        else:
            print("⚠ Config not found in checkpoint, inferring from weights...")
            model_configuration = infer_config_from_checkpoint(checkpoint_data)

        # Validate configuration before model creation
        is_config_valid, validation_issues = validate_config(model_configuration)
        if not is_config_valid:
            print("⚠ Configuration validation issues found:")
            for issue in validation_issues:
                print(f"  - {issue}")
            print("Attempting to proceed anyway...")

        # Create model instance with loaded configuration
        try:
            model = PerseidByteModel(model_configuration)
            print(f"✓ Model architecture created")

            # Calculate and display model parameters
            parameter_info = calculate_model_params(model_configuration)
            print(f"  - Total parameters: {parameter_info['total_millions']:.1f}M")
            print(f"  - Embedding dimension: {model_configuration['emb_dim']}")
            print(f"  - Context length: {model_configuration['context_length']}")

        except Exception as model_creation_error:
            raise RuntimeError(
                f"Failed to create model with loaded configuration. "
                f"Configuration may be incompatible with current code version. "
                f"Error: {str(model_creation_error)}"
            ) from model_creation_error

        # Load model weights from checkpoint
        try:
            if "model_state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state_dict"])
                print("✓ Model weights loaded from 'model_state_dict'")
            elif "model" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model"])
                print("✓ Model weights loaded from 'model'")
            else:
                # Assume the entire checkpoint is the state dict
                model.load_state_dict(checkpoint_data)
                print("✓ Model weights loaded from checkpoint root")

        except Exception as weight_loading_error:
            raise RuntimeError(
                f"Failed to load model weights. This may indicate architecture mismatch "
                f"between checkpoint and current model definition. "
                f"Error: {str(weight_loading_error)}"
            ) from weight_loading_error

        # Move model to appropriate device and set to evaluation mode
        model = model.to(device)
        model.eval()

        print(f"✓ Model ready for inference on {device}")
        return model, device, model_configuration

    except Exception as model_loading_error:
        error_traceback = traceback.format_exc()
        raise RuntimeError(
            f"Failed to load PerseidByte model for embedding extraction. "
            f"Checkpoint path: {checkpoint_path}. "
            f"Error: {str(model_loading_error)}\n"
            f"Full traceback:\n{error_traceback}"
        ) from model_loading_error


def setup_byte_tokenizer() -> ByteTokenizer:
    """
    Initialize and configure the ByteTokenizer for text preprocessing.

    Returns:
        ByteTokenizer: Configured tokenizer ready for encoding text

    Raises:
        ImportError: If ByteTokenizer class cannot be imported
        RuntimeError: If tokenizer initialization fails
    """
    try:
        print("Initializing ByteTokenizer...")
        tokenizer = ByteTokenizer()
        print("✓ ByteTokenizer ready")

        # Test tokenizer with a simple example
        test_text = "Hello, World!"
        test_tokens = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_tokens)

        if test_decoded != test_text:
            print(f"⚠ Tokenizer round-trip test failed:")
            print(f"  Original: '{test_text}'")
            print(f"  Decoded:  '{test_decoded}'")
        else:
            print("✓ Tokenizer round-trip test passed")

        return tokenizer

    except Exception as tokenizer_error:
        raise RuntimeError(
            f"Failed to setup ByteTokenizer. Error: {str(tokenizer_error)}"
        ) from tokenizer_error


def extract_text_embedding(
    model: PerseidByteModel,
    tokenizer: ByteTokenizer,
    input_text: str,
    pooling_method: str = "last_token",
    device: torch.device = None,
) -> np.ndarray:
    """
    Extract a dense embedding vector from input text using the trained model.

    Args:
        model: Trained PerseidByteModel in evaluation mode
        tokenizer: ByteTokenizer for text preprocessing
        input_text: Raw text to encode as embedding vector
        pooling_method: Strategy for sequence pooling ("last_token" or "mean")
        device: PyTorch device for computation

    Returns:
        numpy.ndarray: Dense embedding vector of shape (embedding_dimension,)

    Raises:
        ValueError: If input text is empty or pooling method is invalid
        RuntimeError: If model forward pass fails
        TypeError: If inputs have incorrect types
    """
    try:
        # Input validation
        if not isinstance(input_text, str):
            raise TypeError(f"input_text must be string, got {type(input_text)}")

        if len(input_text.strip()) == 0:
            raise ValueError("input_text cannot be empty or whitespace-only")

        if pooling_method not in ["last_token", "mean"]:
            raise ValueError(
                f"pooling_method must be 'last_token' or 'mean', got '{pooling_method}'"
            )

        # Tokenize input text to token IDs
        try:
            token_ids = tokenizer.encode(input_text)
        except Exception as tokenization_error:
            raise RuntimeError(
                f"Failed to tokenize input text: '{input_text[:50]}...'. "
                f"Error: {str(tokenization_error)}"
            ) from tokenization_error

        if len(token_ids) == 0:
            raise ValueError(
                f"Tokenization resulted in empty token sequence for text: '{input_text[:50]}...'"
            )

        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor([token_ids], dtype=torch.long)

        if device is not None:
            input_tensor = input_tensor.to(device)

        print(f"  Text length: {len(input_text)} chars")
        print(f"  Token count: {len(token_ids)} tokens")
        print(f"  Input shape: {input_tensor.shape}")

        # Extract embedding using model's get_embeddings method
        with torch.no_grad():  # Disable gradient computation for inference
            try:
                embedding_tensor = model.get_embeddings(
                    input_tensor, pooling_method=pooling_method
                )
                print(f"  Embedding shape: {embedding_tensor.shape}")

            except Exception as embedding_error:
                raise RuntimeError(
                    f"Model embedding extraction failed for pooling method '{pooling_method}'. "
                    f"Input shape: {input_tensor.shape}. "
                    f"Error: {str(embedding_error)}"
                ) from embedding_error

        # # Convert to numpy array (remove batch dimension)
        # # Convert bfloat16 to float32 first since NumPy doesn't support bfloat16
        # embedding_vector = embedding_tensor.squeeze(0).cpu().float().numpy()

        # Convert to numpy array (remove batch dimension)
        embedding_tensor_squeezed = embedding_tensor.squeeze(0).cpu()

        # Convert bfloat16 to float32 for NumPy compatibility
        # NumPy doesn't natively support bfloat16 precision
        if embedding_tensor_squeezed.dtype == torch.bfloat16:
            print(f"  Converting from bfloat16 to float32 for NumPy compatibility")
            embedding_tensor_squeezed = embedding_tensor_squeezed.float()

        embedding_vector = embedding_tensor_squeezed.numpy()

        # Validate output shape
        expected_embedding_dim = model.cfg["emb_dim"]
        if embedding_vector.shape != (expected_embedding_dim,):
            raise RuntimeError(
                f"Unexpected embedding shape. Expected ({expected_embedding_dim},), "
                f"got {embedding_vector.shape}"
            )

        print(f"  ✓ Embedding extracted successfully")
        return embedding_vector

    except Exception as extraction_error:
        error_traceback = traceback.format_exc()
        raise RuntimeError(
            f"Failed to extract embedding from text: '{input_text[:50]}...'. "
            f"Pooling method: {pooling_method}. "
            f"Error: {str(extraction_error)}\n"
            f"Full traceback:\n{error_traceback}"
        ) from extraction_error


def compute_similarity_matrix(embeddings_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarities between embedding vectors.

    Args:
        embeddings_list: List of embedding vectors as numpy arrays

    Returns:
        numpy.ndarray: Similarity matrix of shape (n_embeddings, n_embeddings)
                      where entry (i,j) is cosine similarity between embedding i and j
    """
    try:
        if len(embeddings_list) == 0:
            raise ValueError("embeddings_list cannot be empty")

        # Stack embeddings into matrix
        embeddings_matrix = np.stack(
            embeddings_list, axis=0
        )  # Shape: (n_embeddings, embedding_dim)

        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings_matrix / np.linalg.norm(
            embeddings_matrix, axis=1, keepdims=True
        )

        # Compute similarity matrix
        similarity_matrix = embeddings_normalized @ embeddings_normalized.T

        print(f"✓ Similarity matrix computed: {similarity_matrix.shape}")
        return similarity_matrix

    except Exception as similarity_error:
        raise RuntimeError(
            f"Failed to compute similarity matrix: {str(similarity_error)}"
        ) from similarity_error


def format_embeddings_for_output(
    embeddings_list: List[np.ndarray], format_type: str
) -> Union[List, np.ndarray, str]:
    """
    Convert embeddings to specified output format.

    Args:
        embeddings_list: List of embedding vectors
        format_type: Output format ("numpy", "torch", "json")

    Returns:
        Formatted embeddings in requested format
    """
    try:
        if format_type == "numpy":
            return np.stack(embeddings_list, axis=0)

        elif format_type == "torch":
            return torch.tensor(np.stack(embeddings_list, axis=0))

        elif format_type == "json":
            # Convert to JSON-serializable format
            embeddings_json = []
            for i, embedding in enumerate(embeddings_list):
                embeddings_json.append(
                    {
                        "index": i,
                        "vector": embedding.tolist(),
                        "dimension": len(embedding),
                    }
                )
            return json.dumps(embeddings_json, indent=2)

        else:
            raise ValueError(f"Unsupported format_type: {format_type}")

    except Exception as format_error:
        raise RuntimeError(
            f"Failed to format embeddings: {str(format_error)}"
        ) from format_error


def save_embeddings_to_file(
    embeddings_data: Dict, output_file_path: str, include_similarity_matrix: bool = True
) -> None:
    """
    Save embeddings and metadata to JSON file.

    Args:
        embeddings_data: Dictionary containing embeddings and metadata
        output_file_path: Path where to save the output file
        include_similarity_matrix: Whether to include similarity computations
    """
    try:
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(embeddings_data, output_file, indent=2, ensure_ascii=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Embeddings saved to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")

    except Exception as save_error:
        raise RuntimeError(
            f"Failed to save embeddings to {output_file_path}: {str(save_error)}"
        ) from save_error


def main():
    """
    Main function to orchestrate embedding extraction from input texts.

    Loads the trained model, processes input texts, extracts embeddings,
    computes similarities, and optionally saves results to file.
    """
    try:
        print("=" * 80)
        print("PerseidByte Embedding Vector Extraction")
        print("=" * 80)

        # Load trained model
        print("\nStep 1: Loading Model")
        print("-" * 40)
        model, device, model_config = load_perseid_model_for_embeddings(CHECKPOINT_PATH)

        # Setup tokenizer
        print("\nStep 2: Setting Up Tokenizer")
        print("-" * 40)
        tokenizer = setup_byte_tokenizer()

        # Extract embeddings for all input texts
        print("\nStep 3: Extracting Embeddings")
        print("-" * 40)
        all_embeddings = []
        embedding_metadata = []

        for text_index, input_text in enumerate(INPUT_TEXTS):
            print(f"\nProcessing text {text_index + 1}/{len(INPUT_TEXTS)}:")
            print(
                f"Text preview: '{input_text[:60]}{'...' if len(input_text) > 60 else ''}'"
            )

            try:
                embedding_vector = extract_text_embedding(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=input_text,
                    pooling_method=POOLING_METHOD,
                    device=device,
                )

                all_embeddings.append(embedding_vector)
                embedding_metadata.append(
                    {
                        "index": text_index,
                        "text": input_text,
                        "text_length": len(input_text),
                        "embedding_dimension": len(embedding_vector),
                        "embedding_norm": float(np.linalg.norm(embedding_vector)),
                        "pooling_method": POOLING_METHOD,
                    }
                )

            except Exception as text_processing_error:
                print(
                    f"  ✗ Failed to process text {text_index + 1}: {str(text_processing_error)}"
                )
                continue

        if len(all_embeddings) == 0:
            raise RuntimeError(
                "No embeddings were successfully extracted from input texts"
            )

        print(f"\n✓ Successfully extracted {len(all_embeddings)} embeddings")

        # Compute similarity matrix
        print("\nStep 4: Computing Similarities")
        print("-" * 40)
        similarity_matrix = compute_similarity_matrix(all_embeddings)

        # Display similarity results
        print("\nPairwise Cosine Similarities:")
        for i in range(len(all_embeddings)):
            for j in range(i + 1, len(all_embeddings)):
                similarity_score = similarity_matrix[i, j]
                print(f"  Text {i + 1} ↔ Text {j + 1}: {similarity_score:.4f}")

        # Format embeddings for output
        print(f"\nStep 5: Formatting Output ({OUTPUT_FORMAT})")
        print("-" * 40)
        formatted_embeddings = format_embeddings_for_output(
            all_embeddings, OUTPUT_FORMAT
        )
        print(f"✓ Embeddings formatted as {OUTPUT_FORMAT}")

        # Save results if requested
        if SAVE_EMBEDDINGS:
            print("\nStep 6: Saving Results")
            print("-" * 40)

            # Convert model_config to JSON-serializable format
            json_safe_model_config = {}
            for key, value in model_config.items():
                if hasattr(value, "__module__") and "torch" in str(type(value)):
                    # Convert torch objects to string representation
                    json_safe_model_config[key] = str(value)
                else:
                    json_safe_model_config[key] = value

            # Prepare comprehensive output data
            output_data = {
                "metadata": {
                    "model_checkpoint": str(CHECKPOINT_PATH),
                    "pooling_method": POOLING_METHOD,
                    "embedding_dimension": model_config["emb_dim"],
                    "num_embeddings": len(all_embeddings),
                    "model_config": json_safe_model_config,
                },
                "embeddings": [
                    {
                        "text": meta["text"],
                        "vector": embedding.tolist(),
                        "metadata": meta,
                    }
                    for embedding, meta in zip(all_embeddings, embedding_metadata)
                ],
                "similarity_matrix": similarity_matrix.tolist(),
            }

            save_embeddings_to_file(output_data, OUTPUT_FILE)

        print("\n" + "=" * 80)
        print("Embedding Extraction Complete!")
        print("=" * 80)
        print(f"✓ Processed {len(INPUT_TEXTS)} input texts")
        print(f"✓ Extracted {len(all_embeddings)} embeddings")
        print(f"✓ Embedding dimension: {len(all_embeddings[0])}")
        print(f"✓ Pooling method: {POOLING_METHOD}")
        if SAVE_EMBEDDINGS:
            print(f"✓ Results saved to: {OUTPUT_FILE}")

        return all_embeddings, embedding_metadata, similarity_matrix

    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        sys.exit(1)

    except Exception as main_execution_error:
        error_traceback = traceback.format_exc()
        print(f"\n✗ Fatal error during embedding extraction:")
        print(f"Error: {str(main_execution_error)}")
        print(f"Full traceback:\n{error_traceback}")
        sys.exit(1)


def parse_command_line_arguments():
    """
    Parse command line arguments for script configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract embedding vectors from trained PerseidByte models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_embedding_vector_perseidbyte.py
  python get_embedding_vector_perseidbyte.py --checkpoint ./models/my_model.pth
  python get_embedding_vector_perseidbyte.py --pooling mean --output embeddings.json
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help="Path to model checkpoint file",
    )

    parser.add_argument(
        "--pooling",
        type=str,
        choices=["last_token", "mean"],
        default=POOLING_METHOD,
        help="Pooling method for sequence embedding",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help="Output file path for saving embeddings",
    )

    parser.add_argument(
        "--no-save", action="store_true", help="Don't save embeddings to file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments if provided
    try:
        args = parse_command_line_arguments()

        # Override global settings with command line arguments
        CHECKPOINT_PATH = args.checkpoint
        POOLING_METHOD = args.pooling
        OUTPUT_FILE = args.output
        SAVE_EMBEDDINGS = not args.no_save

    except Exception as args_error:
        print(f"Error parsing command line arguments: {args_error}")
        sys.exit(1)

    # Run main embedding extraction process
    main()
