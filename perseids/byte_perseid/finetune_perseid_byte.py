"""
finetune_perseid_byte.py

Minimal viable fine-tuning script for Perseid byte-level language models.

This script provides a straightforward approach to fine-tuning a pre-trained
Perseid model on a smaller, task-specific dataset. It focuses on:
- Loading pre-trained model weights
- Selective layer freezing to preserve base knowledge
- Training on new data with reduced learning rates
- Checkpoint management for fine-tuned models

Key differences from pre-training:
- Uses existing model weights as starting point
- Freezes lower layers to prevent catastrophic forgetting
- Uses smaller learning rates (10-100x lower)
- Typically requires much less training data and time
- Saves fine-tuned model separately from base model

Usage:
    1. Set configuration parameters below
    2. Ensure pre-trained model checkpoint exists
    3. Prepare your fine-tuning dataset
    4. Run: python finetune_perseid_byte.py

Example:
    # Fine-tune on a specific domain corpus
    python finetune_perseid_byte.py
"""

# from decimal import HAVE_THREADS
# import os
# import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import model and tokenizer
from byte_tokenizer import ByteTokenizer
from perseid_model import PerseidByteModel
from perseidbyte_256_288_320_config_tools import calculate_model_params

# ============================================================================
#  USER CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

# Pre-trained model to fine-tune from
PRETRAINED_MODEL_PATH = "/home/oops/code/gutenberg_babble/perseids/byte_perseid/models/perseid_288m_alice/perseid_model_final.pth"

# Fine-tuning data
FINETUNING_DATA_PATH = (
    "/home/oops/code/gutenberg_babble/perseids/byte_perseid/data/204.txt"
)

# Output directory for fine-tuned model
OUTPUT_DIR = "./models/perseid_finetuned/"

# Layer freezing strategy
FREEZE_LOWER_LAYERS = True  # Whether to freeze lower layers
FREEZE_PERCENTAGE = 0.5  # What percentage of layers to freeze (0.0 to 1.0)
# 0.5 = freeze bottom 50% of layers, train top 50%

# Fine-tuning hyperparameters
FINETUNING_CONFIG = {
    "context_length": 1024,  # Match pre-training or use shorter
    "batch_size": 2,  # Typically smaller than pre-training
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-5,  # Much lower than pre-training (was 5e-4)
    "num_epochs": 3,  # Usually fewer epochs needed
    "weight_decay": 0.01,
    "warmup_steps": 50,  # Shorter warmup
    "eval_every": 50,
    "eval_batches": 5,
    "save_every": 100,
    "chunk_overlap": 0.1,
}

# Data split for validation
TRAIN_VAL_SPLIT = 0.9

# Hardware settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BFLOAT16 = torch.cuda.is_available()

# ============================================================================
# END USER CONFIGURATION
# ============================================================================


# def load_pretrained_model(checkpoint_path, device):
#     """
#     Load a pre-trained Perseid model from checkpoint.

#     This function loads model weights and configuration from a saved checkpoint,
#     preparing the model for fine-tuning. It handles both old-style checkpoints
#     (just state dict) and new-style checkpoints (with metadata).

#     Args:
#         checkpoint_path (str or Path): Path to the checkpoint file (.pth)
#         device (str): Device to load model onto ('cuda' or 'cpu')

#     Returns:
#         tuple: (model, model_config, checkpoint_metadata)
#             - model: Loaded PerseidByteModel instance
#             - model_config: Configuration dictionary used to create model
#             - checkpoint_metadata: Dict with training history (step, loss, etc.)

#     Raises:
#         FileNotFoundError: If checkpoint file doesn't exist
#         RuntimeError: If checkpoint is corrupted or incompatible

#     Example:
#         >>> model, config, metadata = load_pretrained_model(
#         ...     "./models/checkpoint_best.pth",
#         ...     device="cuda"
#         ... )
#         >>> print(f"Loaded model from step {metadata['step']}")
#     """
#     try:
#         checkpoint_path = Path(checkpoint_path)

#         if not checkpoint_path.exists():
#             raise FileNotFoundError(
#                 f"Checkpoint not found: {checkpoint_path}\n"
#                 f"Please verify the path to your pre-trained model."
#             )

#         print(f"\n{'=' * 60}")
#         print("Loading Pre-trained Model")
#         print(f"{'=' * 60}")
#         print(f"Checkpoint: {checkpoint_path}")

#         # Load checkpoint from disk
#         print("Reading checkpoint file...")
#         checkpoint = torch.load(checkpoint_path, map_location=device)

#         # Extract model configuration
#         if "model_config" in checkpoint:
#             # New-style checkpoint with metadata
#             model_config = checkpoint["model_config"]
#             print("  ✓ Found model configuration in checkpoint")
#         else:
#             # Old-style checkpoint - need to infer or provide config
#             raise RuntimeError(
#                 "Checkpoint does not contain model configuration.\n"
#                 "This appears to be an old-style checkpoint. Please provide\n"
#                 "the model configuration separately or use a newer checkpoint."
#             )

#         # Initialize model with the saved configuration
#         print("\nInitializing model architecture...")
#         model = PerseidByteModel(model_config)

#         # Load the trained weights
#         if "model_state_dict" in checkpoint:
#             # New-style checkpoint
#             model.load_state_dict(checkpoint["model_state_dict"])
#             print("  ✓ Loaded model weights (new format)")
#         else:
#             # Old-style checkpoint (just state dict)
#             model.load_state_dict(checkpoint)
#             print("  ✓ Loaded model weights (legacy format)")

#         # Move model to target device
#         model = model.to(device)

#         # Extract metadata about the checkpoint
#         checkpoint_metadata = {
#             "global_step": checkpoint.get("step", 0),
#             "validation_loss": checkpoint.get("val_loss", None),
#             "epoch": checkpoint.get("epoch", None),
#             "tokens_seen": checkpoint.get("tokens_seen", 0),
#         }

#         # Print model information
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#         print(f"\n✅ Model loaded successfully")
#         print(f"  Total parameters: {total_params:,}")
#         print(f"  Trainable parameters: {trainable_params:,}")
#         print(f"  Device: {device}")

#         if checkpoint_metadata["global_step"] > 0:
#             print(f"\nCheckpoint training history:")
#             print(f"  Training step: {checkpoint_metadata['global_step']:,}")
#             if checkpoint_metadata["validation_loss"] is not None:
#                 print(
#                     f"  Validation loss: {checkpoint_metadata['validation_loss']:.4f}"
#                 )
#             if checkpoint_metadata["tokens_seen"] > 0:
#                 print(f"  Tokens seen: {checkpoint_metadata['tokens_seen']:,}")

#         return model, model_config, checkpoint_metadata

#     except Exception as e:
#         print(f"\n❌ Error loading pre-trained model: {e}")
#         traceback.print_exc()
#         raise


def load_pretrained_model(checkpoint_path, device):
    """
    Load a pre-trained Perseid model from checkpoint.

    This function loads model weights and configuration from a saved checkpoint,
    preparing the model for fine-tuning. It handles both old-style checkpoints
    (just state dict) and new-style checkpoints (with metadata).

    If the checkpoint doesn't contain configuration, it looks for a
    model_config.json file in the same directory.

    Args:
        checkpoint_path (str or Path): Path to the checkpoint file (.pth)
        device (str): Device to load model onto ('cuda' or 'cpu')

    Returns:
        tuple: (model, model_config, checkpoint_metadata)
            - model: Loaded PerseidByteModel instance
            - model_config: Configuration dictionary used to create model
            - checkpoint_metadata: Dict with training history (step, loss, etc.)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint is corrupted or incompatible

    Example:
        >>> model, config, metadata = load_pretrained_model(
        ...     "./models/checkpoint_best.pth",
        ...     device="cuda"
        ... )
        >>> print(f"Loaded model from step {metadata['step']}")
    """
    try:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Please verify the path to your pre-trained model."
            )

        print(f"\n{'=' * 60}")
        print("Loading Pre-trained Model")
        print(f"{'=' * 60}")
        print(f"Checkpoint: {checkpoint_path}")

        # Load checkpoint from disk
        print("Reading checkpoint file...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model configuration
        model_config = None

        if "model_config" in checkpoint:
            # New-style checkpoint with metadata
            model_config = checkpoint["model_config"]
            print("  ✓ Found model configuration in checkpoint")
        else:
            # Old-style checkpoint - look for model_config.json in same directory
            print("  ! Checkpoint does not contain embedded configuration")

            config_json_path = checkpoint_path.parent / "model_config.json"

            if config_json_path.exists():
                print(f"  ✓ Found external configuration file: {config_json_path.name}")

                try:
                    with open(config_json_path, "r", encoding="utf-8") as config_file:
                        model_config = json.load(config_file)

                    print("  ✓ Loaded configuration from JSON file")

                    # Convert dtype string back to torch dtype
                    if "dtype" in model_config and isinstance(
                        model_config["dtype"], str
                    ):
                        dtype_string = model_config["dtype"]

                        # Parse common dtype strings
                        if "bfloat16" in dtype_string:
                            model_config["dtype"] = torch.bfloat16
                        elif "float16" in dtype_string:
                            model_config["dtype"] = torch.float16
                        elif "float32" in dtype_string:
                            model_config["dtype"] = torch.float32
                        else:
                            # Default to float32 if unknown
                            print(
                                f"  ! Unknown dtype '{dtype_string}', defaulting to float32"
                            )
                            model_config["dtype"] = torch.float32

                except json.JSONDecodeError as json_error:
                    raise RuntimeError(
                        f"Failed to parse model_config.json: {json_error}\n"
                        f"The configuration file may be corrupted."
                    )
            else:
                # No config available - provide helpful error
                raise RuntimeError(
                    f"Checkpoint does not contain model configuration.\n"
                    f"Looked for external config at: {config_json_path}\n"
                    f"Please ensure model_config.json exists in the same directory\n"
                    f"as the checkpoint file, or use a newer checkpoint format."
                )

        # Verify we have a valid configuration
        if model_config is None:
            raise RuntimeError("Failed to load model configuration from any source")

        # Print configuration summary
        print("\nModel Configuration:")
        print(f"  Vocabulary size: {model_config.get('vocab_size', 'unknown')}")
        print(f"  Embedding dimension: {model_config.get('emb_dim', 'unknown')}")
        print(f"  Number of layers: {model_config.get('n_layers', 'unknown')}")
        print(f"  Number of heads: {model_config.get('n_heads', 'unknown')}")
        print(f"  Context length: {model_config.get('context_length', 'unknown')}")

        # Initialize model with the loaded configuration
        print("\nInitializing model architecture...")
        model = PerseidByteModel(model_config)

        # Load the trained weights
        if "model_state_dict" in checkpoint:
            # New-style checkpoint
            model.load_state_dict(checkpoint["model_state_dict"])
            print("  ✓ Loaded model weights (new format)")
        else:
            # Old-style checkpoint (just state dict)
            model.load_state_dict(checkpoint)
            print("  ✓ Loaded model weights (legacy format)")

        # Move model to target device
        model = model.to(device)

        # Extract metadata about the checkpoint
        checkpoint_metadata = {
            "global_step": checkpoint.get("step", 0),
            "validation_loss": checkpoint.get("val_loss", None),
            "epoch": checkpoint.get("epoch", None),
            "tokens_seen": checkpoint.get("tokens_seen", 0),
        }

        # Print model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n✅ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {device}")

        if checkpoint_metadata["global_step"] > 0:
            print(f"\nCheckpoint training history:")
            print(f"  Training step: {checkpoint_metadata['global_step']:,}")
            if checkpoint_metadata["validation_loss"] is not None:
                print(
                    f"  Validation loss: {checkpoint_metadata['validation_loss']:.4f}"
                )
            if checkpoint_metadata["tokens_seen"] > 0:
                print(f"  Tokens seen: {checkpoint_metadata['tokens_seen']:,}")

        return model, model_config, checkpoint_metadata

    except Exception as e:
        print(f"\n❌ Error loading pre-trained model: {e}")
        traceback.print_exc()
        raise


def freeze_model_layers(model, freeze_percentage, verbose=True):
    """
    Freeze lower layers of the model to preserve pre-trained knowledge.

    Freezing layers prevents their weights from being updated during fine-tuning,
    which helps preserve general knowledge learned during pre-training while
    allowing upper layers to adapt to the new task.

    Strategy: Freeze bottom N% of transformer layers, keep embeddings trainable,
    and keep final layers (norm, output head) trainable. This is a common
    approach that balances stability and adaptability.

    Args:
        model (PerseidByteModel): Model to apply freezing to
        freeze_percentage (float): Percentage of layers to freeze (0.0 to 1.0)
            - 0.0: No freezing, train everything (full fine-tuning)
            - 0.5: Freeze bottom 50% of layers
            - 1.0: Freeze all transformer layers (only train embeddings/head)
        verbose (bool): Whether to print detailed freezing information

    Returns:
        dict: Statistics about freezing with keys:
            - total_params: Total parameters in model
            - frozen_params: Number of frozen parameters
            - trainable_params: Number of trainable parameters
            - frozen_layers: Number of frozen transformer layers
            - trainable_layers: Number of trainable transformer layers

    Example:
        >>> model = PerseidByteModel(config)
        >>> stats = freeze_model_layers(model, freeze_percentage=0.5)
        >>> print(f"Frozen {stats['frozen_params']:,} parameters")
    """
    try:
        if verbose:
            print(f"\n{'=' * 60}")
            print("Applying Layer Freezing Strategy")
            print(f"{'=' * 60}")
            print(f"Freeze percentage: {freeze_percentage:.0%}")

        # Calculate how many layers to freeze
        total_transformer_layers = len(model.blocks)
        num_layers_to_freeze = int(total_transformer_layers * freeze_percentage)

        if verbose:
            print(f"Total transformer layers: {total_transformer_layers}")
            print(f"Layers to freeze: {num_layers_to_freeze}")
            print(f"Layers to train: {total_transformer_layers - num_layers_to_freeze}")

        # Freeze the lower transformer layers
        for layer_index in range(num_layers_to_freeze):
            transformer_block = model.blocks[layer_index]

            # Freeze all parameters in this block
            for parameter in transformer_block.parameters():
                parameter.requires_grad = False

            if verbose and layer_index < 3:  # Show first few for brevity
                print(f"  ✓ Frozen layer {layer_index}")

        if verbose and num_layers_to_freeze > 3:
            print(f"  ... (frozen {num_layers_to_freeze} layers total)")

        # Keep embeddings trainable (optional - could freeze these too)
        # Embeddings often benefit from task-specific adaptation
        for parameter in model.tok_emb.parameters():
            parameter.requires_grad = True

        # Keep final normalization trainable
        for parameter in model.final_norm.parameters():
            parameter.requires_grad = True

        # Keep output head trainable
        for parameter in model.out_head.parameters():
            parameter.requires_grad = True

        # Calculate statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        freezing_stats = {
            "total_params": total_params,
            "frozen_params": frozen_params,
            "trainable_params": trainable_params,
            "frozen_layers": num_layers_to_freeze,
            "trainable_layers": total_transformer_layers - num_layers_to_freeze,
            "freeze_percentage": freeze_percentage,
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print("Freezing Summary")
            print(f"{'=' * 60}")
            print(f"Total parameters: {total_params:,}")
            print(
                f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)"
            )
            print(
                f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)"
            )
            print(
                f"\nMemory savings: ~{frozen_params * 4 / 1e9:.2f} GB (no gradients for frozen params)"
            )

        return freezing_stats

    except Exception as e:
        print(f"\n❌ Error freezing model layers: {e}")
        traceback.print_exc()
        raise


def load_finetuning_data(file_path):
    """
    Load text data for fine-tuning from a file.

    Simple text file loader that reads UTF-8 encoded text. For fine-tuning,
    datasets are typically smaller and more focused than pre-training data.

    Args:
        file_path (str or Path): Path to text file containing fine-tuning data

    Returns:
        str: The loaded text content

    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file encoding is not UTF-8 compatible

    Example:
        >>> text = load_finetuning_data("./data/medical_texts.txt")
        >>> print(f"Loaded {len(text):,} characters")
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fine-tuning data not found: {file_path}\n"
                f"Please provide a valid path to your training data."
            )

        print(f"\n{'=' * 60}")
        print("Loading Fine-tuning Data")
        print(f"{'=' * 60}")
        print(f"File: {file_path}")

        # Get file size
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

        # Load the text
        with open(file_path, "r", encoding="utf-8") as file_handle:
            text_content = file_handle.read()

        print(f"✅ Loaded {len(text_content):,} characters")

        # Basic statistics
        num_lines = text_content.count("\n")
        print(f"Lines: {num_lines:,}")

        return text_content

    except Exception as e:
        print(f"\n❌ Error loading fine-tuning data: {e}")
        traceback.print_exc()
        raise


class FineTuningDataset(Dataset):
    """
    Dataset class for fine-tuning on text data.

    Creates overlapping windows of text for training, similar to pre-training
    but typically with smaller data. Handles tokenization and chunking.

    Attributes:
        tokenizer (ByteTokenizer): Tokenizer for encoding text
        max_length (int): Maximum sequence length for model input
        stride (int): Stride between consecutive windows (controls overlap)
        windows (list): Pre-computed training windows of token IDs
    """

    def __init__(self, text, tokenizer, max_length, stride, verbose=True):
        """
        Initialize fine-tuning dataset from text.

        Args:
            text (str): Input text to create training windows from
            tokenizer (ByteTokenizer): Tokenizer for encoding
            max_length (int): Maximum sequence length
            stride (int): Stride between windows (smaller = more overlap)
            verbose (bool): Whether to print progress information
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        try:
            if verbose:
                print(f"\n{'=' * 60}")
                print("Creating Fine-tuning Dataset")
                print(f"{'=' * 60}")

            # Tokenize the entire text
            if verbose:
                print("Tokenizing text...")

            import time

            start_time = time.perf_counter()

            self.tokens = tokenizer.encode(text, add_eos=False)

            tokenization_time = time.perf_counter() - start_time

            if verbose:
                tokens_per_second = len(self.tokens) / tokenization_time
                print(f"  ✓ Tokenized {len(self.tokens):,} tokens")
                print(f"  Time: {tokenization_time:.2f} seconds")
                print(f"  Speed: {tokens_per_second:,.0f} tokens/second")

            # Create overlapping windows
            if verbose:
                print("\nCreating training windows...")

            self.windows = []
            for start_index in range(0, len(self.tokens) - max_length, stride):
                # Extract window with one extra token for target
                window = self.tokens[start_index : start_index + max_length + 1]

                # Only include complete windows
                if len(window) == max_length + 1:
                    self.windows.append(window)

            if verbose:
                overlap_percent = (max_length - stride) / max_length * 100
                print(f"  ✓ Created {len(self.windows):,} training windows")
                print(f"  Window size: {max_length} tokens")
                print(f"  Stride: {stride} tokens")
                print(f"  Overlap: {overlap_percent:.1f}%")
                print(f"\n{'=' * 60}")
                print("Dataset Ready")
                print(f"{'=' * 60}")

        except Exception as e:
            print(f"\n❌ Error creating dataset: {e}")
            traceback.print_exc()
            raise

    def __len__(self):
        """Return the number of training windows."""
        return len(self.windows)

    def __getitem__(self, index):
        """
        Get a training sample at the given index.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            tuple: (input_ids, target_ids) as torch.Tensor
                - input_ids: Token IDs for model input (length: max_length)
                - target_ids: Token IDs for targets, shifted by 1 (length: max_length)
        """
        window = self.windows[index]

        # Input is all tokens except the last
        input_ids = torch.tensor(window[:-1], dtype=torch.long)

        # Target is all tokens except the first (shifted by 1 position)
        target_ids = torch.tensor(window[1:], dtype=torch.long)

        return input_ids, target_ids


def create_finetuning_dataloaders(text, tokenizer, config, train_ratio=0.9):
    """
    Create training and validation data loaders for fine-tuning.

    Splits the text data into training and validation sets, then creates
    PyTorch DataLoader objects for batch processing during training.

    Args:
        text (str): Full text corpus for fine-tuning
        tokenizer (ByteTokenizer): Tokenizer for encoding
        config (dict): Training configuration with keys:
            - context_length: Maximum sequence length
            - chunk_overlap: Overlap ratio between chunks (0.0 to 1.0)
            - batch_size: Batch size for training
        train_ratio (float): Ratio of data to use for training (rest for validation)

    Returns:
        tuple: (train_loader, val_loader)
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data

    Example:
        >>> train_loader, val_loader = create_finetuning_dataloaders(
        ...     text=corpus,
        ...     tokenizer=tokenizer,
        ...     config=FINETUNING_CONFIG,
        ...     train_ratio=0.9
        ... )
        >>> print(f"Training batches: {len(train_loader)}")
    """
    try:
        print(f"\n{'=' * 60}")
        print("Creating Data Loaders")
        print(f"{'=' * 60}")
        print(f"Train/val split: {train_ratio:.0%} / {(1 - train_ratio):.0%}")

        # Split text into train and validation
        split_index = int(len(text) * train_ratio)
        train_text = text[:split_index]
        val_text = text[split_index:]

        print(f"Training text: {len(train_text):,} characters")
        print(f"Validation text: {len(val_text):,} characters")

        # Calculate stride from overlap
        stride = int(config["context_length"] * (1 - config["chunk_overlap"]))

        # Create datasets
        print("\n" + "-" * 60)
        print("Creating TRAINING dataset...")
        train_dataset = FineTuningDataset(
            train_text, tokenizer, config["context_length"], stride, verbose=True
        )

        print("\n" + "-" * 60)
        print("Creating VALIDATION dataset...")
        val_dataset = FineTuningDataset(
            val_text,
            tokenizer,
            config["context_length"],
            config["context_length"],  # No overlap for validation
            verbose=True,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,  # Shuffle for better training
            drop_last=True,  # Drop incomplete batches
            num_workers=0,  # Single-threaded for simplicity
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,  # No shuffling for validation
            drop_last=False,  # Keep all validation data
            num_workers=0,
        )

        print(f"\n{'=' * 60}")
        print("Data Loaders Ready")
        print(f"{'=' * 60}")
        print(f"Training batches: {len(train_loader):,}")
        print(f"Validation batches: {len(val_loader):,}")
        print(f"Batch size: {config['batch_size']}")

        return train_loader, val_loader

    except Exception as e:
        print(f"\n❌ Error creating data loaders: {e}")
        traceback.print_exc()
        raise


def calculate_loss(input_batch, target_batch, model, device):
    """
    Calculate cross-entropy loss for a batch.

    Standard language modeling loss: compare model predictions against
    true next tokens. Lower loss indicates better prediction accuracy.

    Args:
        input_batch (torch.Tensor): Input token IDs, shape (batch, seq_len)
        target_batch (torch.Tensor): Target token IDs, shape (batch, seq_len)
        model (PerseidByteModel): Model to evaluate
        device (str): Device to run computation on

    Returns:
        torch.Tensor: Scalar loss value (requires_grad=True for training)

    Raises:
        RuntimeError: If shapes are incompatible or computation fails
    """
    try:
        # Move data to correct device
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Forward pass through model
        logits = model(input_batch)  # Shape: (batch, seq_len, vocab_size)

        # Flatten for cross-entropy calculation
        # Cross-entropy expects: (batch * seq_len, vocab_size) and (batch * seq_len)
        logits_flat = logits.flatten(0, 1)  # Combine batch and sequence dimensions
        targets_flat = target_batch.flatten()

        # Calculate cross-entropy loss
        loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return loss

    except Exception as e:
        print(f"Error calculating loss: {e}")
        traceback.print_exc()
        raise


def evaluate_model(model, data_loader, device, num_batches=None):
    """
    Evaluate model performance on a dataset.

    Runs model in evaluation mode (no gradient computation) and calculates
    average loss across the dataset. Used to monitor validation performance
    during fine-tuning.

    Args:
        model (PerseidByteModel): Model to evaluate
        data_loader (DataLoader): Data loader with evaluation data
        device (str): Device to run evaluation on
        num_batches (int, optional): Maximum number of batches to evaluate.
            If None, evaluates entire dataset.

    Returns:
        float: Average loss across evaluated batches

    Example:
        >>> val_loss = evaluate_model(model, val_loader, device="cuda")
        >>> print(f"Validation loss: {val_loss:.4f}")
    """
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    total_loss = 0.0
    evaluated_batches = 0

    # Determine how many batches to evaluate
    max_batches = (
        len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    )

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for batch_index, (input_batch, target_batch) in enumerate(data_loader):
            if batch_index >= max_batches:
                break

            # Calculate loss for this batch
            loss = calculate_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
            evaluated_batches += 1

    # Return to training mode
    model.train()

    # Calculate average loss
    average_loss = (
        total_loss / evaluated_batches if evaluated_batches > 0 else float("nan")
    )

    return average_loss


def save_finetuned_checkpoint(
    model,
    optimizer,
    scheduler,
    global_step,
    validation_loss,
    output_dir,
    checkpoint_name,
    metadata=None,
):
    """
    Save a checkpoint of the fine-tuned model.

    Saves model weights, optimizer state, scheduler state, and training metadata
    for resuming training or deployment. Creates checkpoint file compatible with
    the pre-training format.

    Args:
        model (PerseidByteModel): Model to save
        optimizer: PyTorch optimizer with current state
        scheduler: Learning rate scheduler with current state
        global_step (int): Current training step number
        validation_loss (float): Current validation loss
        output_dir (str or Path): Directory to save checkpoint in
        checkpoint_name (str): Name for checkpoint file (without .pth extension)
        metadata (dict, optional): Additional metadata to save (e.g., freeze info)

    Returns:
        Path: Path to the saved checkpoint file

    Example:
        >>> checkpoint_path = save_finetuned_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     global_step=1000,
        ...     validation_loss=0.8234,
        ...     output_dir="./models/finetuned/",
        ...     checkpoint_name="checkpoint_best"
        ... )
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / f"{checkpoint_name}.pth"

        # Prepare checkpoint dictionary
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": global_step,
            "val_loss": validation_loss,
            "model_config": model.cfg,
            "timestamp": datetime.now().isoformat(),
        }

        # Add any additional metadata
        if metadata is not None:
            checkpoint["metadata"] = metadata

        # Save to disk
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        traceback.print_exc()
        raise


def finetune_model(
    model, train_loader, val_loader, config, device, output_dir, freezing_stats=None
):
    """
    Main fine-tuning loop for Perseid model.

    Performs supervised fine-tuning on a pre-trained model using the provided
    data loaders. Implements gradient accumulation, learning rate scheduling,
    periodic evaluation, and checkpoint saving.

    Key differences from pre-training:
    - Uses lower learning rate (10-100x smaller)
    - Shorter warmup period
    - Typically fewer epochs
    - Some layers may be frozen
    - Preserves pre-trained knowledge while adapting to new data

    Args:
        model (PerseidByteModel): Pre-trained model to fine-tune
        train_loader (DataLoader): Training data
        val_loader (DataLoader): Validation data
        config (dict): Fine-tuning configuration with keys:
            - learning_rate: Learning rate for fine-tuning
            - num_epochs: Number of training epochs
            - weight_decay: L2 regularization strength
            - warmup_steps: Learning rate warmup steps
            - gradient_accumulation_steps: Steps before optimizer update
            - eval_every: Steps between validation evaluations
            - eval_batches: Number of validation batches per evaluation
            - save_every: Steps between checkpoint saves
        device (str): Device to train on
        output_dir (str or Path): Directory to save checkpoints and results
        freezing_stats (dict, optional): Statistics about frozen layers

    Returns:
        dict: Training history with keys:
            - train_loss: List of training losses
            - val_loss: List of validation losses
            - learning_rates: List of learning rates at each step
            - step: List of step numbers

    Example:
        >>> history = finetune_model(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=FINETUNING_CONFIG,
        ...     device="cuda",
        ...     output_dir="./models/finetuned/"
        ... )
    """
    try:
        print(f"\n{'=' * 60}")
        print("Starting Fine-tuning")
        print(f"{'=' * 60}")

        # Setup optimizer - only optimize trainable parameters
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),  # Standard values for transformers
        )

        print(f"Optimizer: AdamW")
        print(f"  Learning rate: {config['learning_rate']:.2e}")
        print(f"  Weight decay: {config['weight_decay']}")
        print(
            f"  Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}"
        )

        # Calculate total training steps
        steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
        total_steps = steps_per_epoch * config["num_epochs"]

        print(f"\nTraining schedule:")
        print(f"  Total epochs: {config['num_epochs']}")
        print(f"  Steps per epoch: {steps_per_epoch:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Warmup steps: {config['warmup_steps']:,}")
        print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(
            f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}"
        )

        # Setup learning rate scheduler with warmup
        def get_learning_rate_multiplier(step):
            """
            Learning rate schedule with linear warmup and cosine decay.

            Args:
                step (int): Current training step

            Returns:
                float: Learning rate multiplier (0.0 to 1.0)
            """
            if step < config["warmup_steps"]:
                # Linear warmup phase
                if config["warmup_steps"] == 0:
                    return 1.0
                return step / config["warmup_steps"]
            else:
                # Cosine decay phase
                decay_steps = max(1, total_steps - config["warmup_steps"])
                progress = (step - config["warmup_steps"]) / decay_steps
                progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, get_learning_rate_multiplier
        )

        # Initialize training state
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "step": [],
        }

        global_step = 0
        best_validation_loss = float("inf")

        print(f"\n{'=' * 60}")
        print("Beginning Training Loop")
        print(f"{'=' * 60}")

        # Training loop - iterate over epochs
        for epoch_number in range(config["num_epochs"]):
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch_number + 1}/{config['num_epochs']}")
            print(f"{'=' * 40}")

            model.train()  # Set model to training mode
            epoch_loss_accumulator = 0.0
            batches_processed_in_epoch = 0

            # Iterate over training batches
            for batch_index, (input_batch, target_batch) in enumerate(train_loader):
                # Forward pass and loss calculation
                loss = calculate_loss(input_batch, target_batch, model, device)

                # Scale loss by accumulation steps (for correct gradient magnitude)
                scaled_loss = loss / config["gradient_accumulation_steps"]

                # Backward pass - compute gradients
                scaled_loss.backward()

                # Accumulate loss for logging (unscaled)
                epoch_loss_accumulator += loss.item()
                batches_processed_in_epoch += 1

                # Update weights after accumulating gradients
                if (batch_index + 1) % config["gradient_accumulation_steps"] == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)

                    # Optimizer step - update weights
                    optimizer.step()

                    # Learning rate scheduler step
                    scheduler.step()

                    # Clear gradients for next accumulation
                    optimizer.zero_grad()

                    # Increment global step counter
                    global_step += 1

                    # Periodic evaluation
                    if global_step % config["eval_every"] == 0:
                        # Calculate current training loss
                        current_train_loss = (
                            epoch_loss_accumulator / batches_processed_in_epoch
                        )

                        # Evaluate on validation set
                        validation_loss = evaluate_model(
                            model,
                            val_loader,
                            device,
                            num_batches=config["eval_batches"],
                        )

                        # Get current learning rate
                        current_learning_rate = scheduler.get_last_lr()[0]

                        # Record in history
                        training_history["train_loss"].append(current_train_loss)
                        training_history["val_loss"].append(validation_loss)
                        training_history["learning_rates"].append(current_learning_rate)
                        training_history["step"].append(global_step)

                        # Print progress
                        print(
                            f"Step {global_step:5d} | "
                            f"Train Loss: {current_train_loss:.4f} | "
                            f"Val Loss: {validation_loss:.4f} | "
                            f"LR: {current_learning_rate:.2e}"
                        )

                        # Save best model based on validation loss
                        if validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss

                            # Prepare metadata for checkpoint
                            checkpoint_metadata = {
                                "fine_tuning": True,
                                "epoch": epoch_number,
                            }
                            if freezing_stats is not None:
                                checkpoint_metadata["freezing_stats"] = freezing_stats

                            # Save checkpoint
                            checkpoint_path = save_finetuned_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                validation_loss=validation_loss,
                                output_dir=output_dir,
                                checkpoint_name="checkpoint_best",
                                metadata=checkpoint_metadata,
                            )

                            print(
                                f"  → Saved best model (val_loss: {validation_loss:.4f})"
                            )

                    # Periodic checkpoint saving
                    if global_step % config["save_every"] == 0:
                        checkpoint_metadata = {
                            "fine_tuning": True,
                            "epoch": epoch_number,
                        }
                        if freezing_stats is not None:
                            checkpoint_metadata["freezing_stats"] = freezing_stats

                        checkpoint_path = save_finetuned_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            validation_loss=validation_loss,
                            output_dir=output_dir,
                            checkpoint_name=f"checkpoint_step_{global_step}",
                            metadata=checkpoint_metadata,
                        )
                        print(f"  → Saved checkpoint at step {global_step}")

            # End-of-epoch evaluation and reporting
            average_epoch_loss = epoch_loss_accumulator / batches_processed_in_epoch
            final_validation_loss = evaluate_model(model, val_loader, device)

            print(f"\nEpoch {epoch_number + 1} Summary:")
            print(f"  Average training loss: {average_epoch_loss:.4f}")
            print(f"  Validation loss: {final_validation_loss:.4f}")
            print(f"  Best validation loss so far: {best_validation_loss:.4f}")

        # Training complete
        print(f"\n{'=' * 60}")
        print("Fine-tuning Complete!")
        print(f"{'=' * 60}")
        print(f"Total steps: {global_step:,}")
        print(f"Best validation loss: {best_validation_loss:.4f}")

        return training_history

    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        traceback.print_exc()
        raise


def save_finetuning_results(
    model, model_config, history, output_dir, freezing_stats=None
):
    """
    Save final fine-tuned model and training artifacts.

    Saves the complete fine-tuned model state, configuration, training history,
    and generates visualization plots. Creates a comprehensive record of the
    fine-tuning process for future reference and deployment.

    Args:
        model (PerseidByteModel): Fine-tuned model to save
        model_config (dict): Model configuration dictionary
        history (dict): Training history with losses and learning rates
        output_dir (str or Path): Directory to save all artifacts
        freezing_stats (dict, optional): Statistics about layer freezing

    Returns:
        None

    Side Effects:
        Creates the following files in output_dir:
        - perseid_finetuned_final.pth: Final model weights
        - model_config.json: Model architecture configuration
        - finetuning_history.json: Complete training history
        - finetuning_curves.png: Training visualization plots
        - finetuning_info.json: Fine-tuning metadata and statistics
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("Saving Fine-tuning Results")
        print(f"{'=' * 60}")
        print(f"Output directory: {output_dir}")

        # =====================================================================
        # Save final model weights
        # =====================================================================
        try:
            model_weights_path = output_dir / "perseid_finetuned_final.pth"
            torch.save(model.state_dict(), model_weights_path)
            print(f"  ✓ Model weights saved: {model_weights_path.name}")
        except Exception as weights_error:
            print(f"  ✗ Failed to save model weights: {weights_error}")

        # =====================================================================
        # Save model configuration
        # =====================================================================
        try:
            config_path = output_dir / "model_config.json"

            # Make config JSON-serializable
            serializable_config = {}
            for key, value in model_config.items():
                if key == "dtype":
                    serializable_config[key] = str(value)
                else:
                    serializable_config[key] = value

            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(serializable_config, config_file, indent=2)

            print(f"  ✓ Configuration saved: {config_path.name}")
        except Exception as config_error:
            print(f"  ✗ Failed to save configuration: {config_error}")

        # =====================================================================
        # Save training history
        # =====================================================================
        try:
            history_path = output_dir / "finetuning_history.json"

            # Convert any tensors to Python types
            serializable_history = {}
            for key, values in history.items():
                if isinstance(values, list):
                    converted_values = []
                    for value in values:
                        if hasattr(value, "cpu"):
                            converted_values.append(value.cpu().item())
                        elif hasattr(value, "item"):
                            converted_values.append(value.item())
                        else:
                            converted_values.append(value)
                    serializable_history[key] = converted_values
                else:
                    serializable_history[key] = values

            with open(history_path, "w", encoding="utf-8") as history_file:
                json.dump(serializable_history, history_file, indent=2)

            print(f"  ✓ Training history saved: {history_path.name}")
        except Exception as history_error:
            print(f"  ✗ Failed to save training history: {history_error}")

        # =====================================================================
        # Save fine-tuning information and statistics
        # =====================================================================
        try:
            info_path = output_dir / "finetuning_info.json"

            finetuning_info = {
                "timestamp": datetime.now().isoformat(),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "frozen_parameters": sum(
                    p.numel() for p in model.parameters() if not p.requires_grad
                ),
            }

            if freezing_stats is not None:
                finetuning_info["freezing_strategy"] = freezing_stats

            if len(history.get("val_loss", [])) > 0:
                # Add best metrics
                val_losses = history["val_loss"]
                best_val_loss = min(val_losses)
                best_val_step = history["step"][val_losses.index(best_val_loss)]

                finetuning_info["best_validation_loss"] = float(best_val_loss)
                finetuning_info["best_validation_step"] = int(best_val_step)

            with open(info_path, "w", encoding="utf-8") as info_file:
                json.dump(finetuning_info, info_file, indent=2)

            print(f"  ✓ Fine-tuning info saved: {info_path.name}")
        except Exception as info_error:
            print(f"  ✗ Failed to save fine-tuning info: {info_error}")

        # =====================================================================
        # Generate and save training visualization plots
        # =====================================================================
        try:
            if len(history.get("train_loss", [])) > 0:
                import matplotlib.pyplot as plt

                # Convert data for plotting (handle tensors)
                steps = []
                train_losses = []
                val_losses = []
                learning_rates = []

                for value in history["step"]:
                    if hasattr(value, "cpu"):
                        steps.append(value.cpu().item())
                    elif hasattr(value, "item"):
                        steps.append(value.item())
                    else:
                        steps.append(value)

                for value in history["train_loss"]:
                    if hasattr(value, "cpu"):
                        train_losses.append(value.cpu().item())
                    elif hasattr(value, "item"):
                        train_losses.append(value.item())
                    else:
                        train_losses.append(value)

                for value in history["val_loss"]:
                    if hasattr(value, "cpu"):
                        val_losses.append(value.cpu().item())
                    elif hasattr(value, "item"):
                        val_losses.append(value.item())
                    else:
                        val_losses.append(value)

                for value in history["learning_rates"]:
                    if hasattr(value, "cpu"):
                        learning_rates.append(value.cpu().item())
                    elif hasattr(value, "item"):
                        learning_rates.append(value.item())
                    else:
                        learning_rates.append(value)

                # Create figure with subplots
                figure = plt.figure(figsize=(15, 5))

                # Plot 1: Training and Validation Loss
                plt.subplot(1, 3, 1)
                plt.plot(
                    steps,
                    train_losses,
                    label="Training Loss",
                    color="blue",
                    linewidth=1.5,
                )
                plt.plot(
                    steps,
                    val_losses,
                    label="Validation Loss",
                    color="orange",
                    linewidth=1.5,
                )
                plt.xlabel("Training Step")
                plt.ylabel("Loss")
                plt.title("Fine-tuning Loss Curves")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Plot 2: Validation Perplexity
                plt.subplot(1, 3, 2)
                val_perplexity = [
                    torch.exp(torch.tensor(loss)).item() for loss in val_losses
                ]
                # Cap perplexity for visualization
                val_perplexity = [min(p, 1000.0) for p in val_perplexity]
                plt.plot(steps, val_perplexity, color="green", linewidth=1.5)
                plt.xlabel("Training Step")
                plt.ylabel("Perplexity")
                plt.title("Validation Perplexity")
                plt.grid(True, alpha=0.3)

                # Plot 3: Learning Rate Schedule
                plt.subplot(1, 3, 3)
                plt.plot(steps, learning_rates, color="red", linewidth=1.5)
                plt.xlabel("Training Step")
                plt.ylabel("Learning Rate")
                plt.title("Learning Rate Schedule")
                plt.grid(True, alpha=0.3)

                # Adjust layout and save
                plt.tight_layout()
                plot_path = output_dir / "finetuning_curves.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(figure)

                print(f"  ✓ Training curves saved: {plot_path.name}")
            else:
                print("  ⚠ No training data available for plotting")

        except Exception as plot_error:
            print(f"  ✗ Failed to create training plots: {plot_error}")
            plt.close("all")  # Ensure cleanup

        print(f"\n{'=' * 60}")
        print("All results saved successfully!")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n❌ Error saving fine-tuning results: {e}")
        traceback.print_exc()
        raise


def main():
    """
    Main fine-tuning pipeline for Perseid models.

    Orchestrates the complete fine-tuning process:
    1. Load pre-trained model
    2. Apply layer freezing strategy
    3. Load fine-tuning data
    4. Create data loaders
    5. Fine-tune the model
    6. Save results

    Returns:
        tuple: (model, training_history)
            - model: Fine-tuned PerseidByteModel
            - training_history: Dict with training metrics
    """
    try:
        print(f"\n{'=' * 60}")
        print("Perseid Model Fine-tuning Pipeline")
        print(f"{'=' * 60}")
        print(f"Pre-trained model: {PRETRAINED_MODEL_PATH}")
        print(f"Fine-tuning data: {FINETUNING_DATA_PATH}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Device: {DEVICE}")

        # Create output directory
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # =====================================================================
        # Step 1: Load pre-trained model
        # =====================================================================
        print(f"\n{'=' * 40}")
        print("Step 1: Loading Pre-trained Model")
        print(f"{'=' * 40}")

        model, model_config, checkpoint_metadata = load_pretrained_model(
            PRETRAINED_MODEL_PATH, DEVICE
        )

        # =====================================================================
        # Step 2: Apply layer freezing strategy
        # =====================================================================
        print(f"\n{'=' * 40}")
        print("Step 2: Applying Layer Freezing")
        print(f"{'=' * 40}")

        freezing_stats = None
        if FREEZE_LOWER_LAYERS:
            freezing_stats = freeze_model_layers(
                model, freeze_percentage=FREEZE_PERCENTAGE, verbose=True
            )
        else:
            print("Layer freezing disabled - training all parameters")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {total_params:,}")

        # =====================================================================
        # Step 3: Setup tokenizer
        # =====================================================================
        print(f"\n{'=' * 40}")
        print("Step 3: Setting Up Tokenizer")
        print(f"{'=' * 40}")

        tokenizer = ByteTokenizer()
        print(f"  ✓ ByteTokenizer initialized")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")

        # =====================================================================
        # Step 4: Load fine-tuning data
        # =====================================================================
        print(f"\n{'=' * 40}")
        print("Step 4: Loading Fine-tuning Data")
        print(f"{'=' * 40}")

        finetuning_text = load_finetuning_data(FINETUNING_DATA_PATH)

        # =====================================================================
        # Step 5: Create data loaders
        # =====================================================================
        print(f"\n{'=' * 40}")
        print("Step 5: Creating Data Loaders")
        print(f"{'=' * 40}")

        train_loader, val_loader = create_finetuning_dataloaders(
            finetuning_text, tokenizer, FINETUNING_CONFIG, train_ratio=TRAIN_VAL_SPLIT
        )

        # =====================================================================
        # Step 6: Fine-tune the model
        # =====================================================================
        print(f"\n{'=' * 40}")
        print("Step 6: Fine-tuning Model")
        print(f"{'=' * 40}")

        training_history = finetune_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=FINETUNING_CONFIG,
            device=DEVICE,
            output_dir=output_dir,
            freezing_stats=freezing_stats,
        )

        # =====================================================================
        # Step 7: Save results
        # =====================================================================
        print(f"\n{'=' * 40}")
        print("Step 7: Saving Results")
        print(f"{'=' * 40}")

        save_finetuning_results(
            model,
            model_config,
            training_history,
            output_dir,
            freezing_stats=freezing_stats,
        )

        # =====================================================================
        # Pipeline complete
        # =====================================================================
        print(f"\n{'=' * 60}")
        print("Fine-tuning Pipeline Complete!")
        print(f"{'=' * 60}")
        print(f"Fine-tuned model saved to: {output_dir}")

        # Print summary statistics
        if len(training_history.get("val_loss", [])) > 0:
            best_val_loss = min(training_history["val_loss"])
            final_val_loss = training_history["val_loss"][-1]

            print(f"\nFinal Statistics:")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            print(f"  Final validation loss: {final_val_loss:.4f}")
            print(
                f"  Improvement: {((final_val_loss - best_val_loss) / best_val_loss * 100):.2f}%"
            )

        return model, training_history

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("Fine-tuning Pipeline Failed")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the fine-tuning pipeline
    finetuned_model, history = main()
