"""
(this is broken, or is under construction,
probably should be one corpus, not separate books retrained)
experimental_docs_train_perseidbyte.py

Training module for Perseid models on text document corpus.
Handles single document input with configurable train/val split.
Trains from scratch (no pretrained weights),
unless existing weights are found, then it should continue training.

Usage:
    1. Set configuration parameters at top of file
    2. Run: python train_perseid_byte.py
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Import model architecture and configuration tools
from byte_tokenizer import ByteTokenizer
from perseidbyte_256_288_320_config_tools import (
    create_perseid_config,
    calculate_model_params,
    validate_config,
)

# from generate_text_tools_perseid import generate_text_simple
from perseid_model import PerseidByteModel

from perseid_model import PERSEID_BYTE_CONFIG_BASE

# Hardware settings (move this up before any function definitions)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BFLOAT16 = torch.cuda.is_available() and DEVICE != "cpu"


def setup_tokenizer():
    """Setup ByteTokenizer for training."""
    print("\nInitializing ByteTokenizer...")
    tokenizer = ByteTokenizer()

    print(f"  ✓ Vocabulary size: {tokenizer.vocab_size}")
    print(
        f"  ✓ Special tokens: PAD={tokenizer.PAD_ID}, EOS={tokenizer.EOS_ID}, MASKUNK={tokenizer.MASKUNK_ID}"
    )

    return tokenizer


def generate_text_simple(model, tokenizer, prompt, max_new_tokens=50, device=None):
    """Simple generation for evaluation"""
    if device is None:
        device = next(model.parameters()).device  # Get device from model parameters

    model.eval()
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if input_ids.shape[1] > model.cfg["context_length"]:
                input_ids = input_ids[:, -model.cfg["context_length"] :]

    model.train()
    return tokenizer.decode(input_ids.squeeze(0).tolist())


# Model configuration
MODEL_SIZE = 288  # Options: 256, 288, 320 (millions of parameters)
MODEL_STRATEGY = "balanced"  # Options: "balanced", "deep", "wide"

# Training continuation settings
TRAINING_MODE = "continue"  # Options: "new", "continue", "force_restart"
# - "new": Start new model if no checkpoint exists, error if checkpoint exists
# - "continue": Resume from checkpoint if exists, start new if doesn't exist
# - "force_restart": Always start fresh (WARNING: overwrites existing model!)

CHECKPOINT_PATH = None  # Set to specific checkpoint file, or None to auto-find
# If None, looks for: {OUTPUT_DIR}/checkpoint_best.pth

# Data split
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation (modify as needed)

# # Training settings
# TRAINING_CONFIG = {
#     "context_length": 512,  # Context window for training
#     "batch_size": 2,  # Batch size (increase if memory allows)
#     "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
#     "learning_rate": 2e-4,  # 5e-4 is too fast! superfastness,  # Learning rate
#     "num_epochs": 8,  # Number of training epochs
#     "weight_decay": 0.05,  # 0.01 may be too low  # Weight decay for AdamW
#     "warmup_steps": 200,  # Warmup steps for learning rate (100 too short)
#     "eval_every": 50,  # Evaluate every N steps
#     "eval_batches": 10,  # Number of batches for evaluation
#     "save_every": 500,  # Save checkpoint every N steps
#     "chunk_overlap": 0.1,  # Overlap between text chunks (0.0 to 0.5)
# }

# TRAINING_CONFIG = {
#     "context_length": 512,
#     "batch_size": 2,  # Slightly larger for stability
#     "gradient_accumulation_steps": 4,  # Total effective batch = 8
#     "learning_rate": 3e-4,  # Moderate starting LR
#     "num_epochs": 10,  # Can train longer with early stopping
#     "weight_decay": 0.1,  # Strong regularization
#     "warmup_steps": 200,  # Good warmup period
#     "eval_every": 50,  # Regular evaluation
#     "eval_batches": 20,  # More validation batches for stable estimate
#     "save_every": 500,
#     "chunk_overlap": 0.15,  # Slightly more overlap for context
#     # New parameters for better control
#     "gradient_clip": 0.5,  # More aggressive clipping
#     "dropout": 0.1,  # If your model supports it
#     "label_smoothing": 0.1,  # Helps prevent overconfidence
# }
# FIXED TRAINING CONFIGURATION
TRAINING_CONFIG = {
    "context_length": 512,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,  # Total effective batch = 8
    "learning_rate": 3e-4,  # Starting LR
    "num_epochs": 10,
    "weight_decay": 0.1,  # Regularization
    "warmup_steps": 200,
    "eval_every": 50,
    "eval_batches": 20,
    "save_every": 500,
    "chunk_overlap": 0.15,
    # Training control parameters
    "gradient_clip": 1.0,  # Standard clipping value
    "label_smoothing": 0.1,
    # Early stopping parameters - MUCH MORE PERMISSIVE
    "early_stop_patience": 5,  # Wait 5 evaluations before stopping (not 3)
    "early_stop_min_delta": 0.01,  # Require 1% improvement (not 0.1%)
    "early_stop_baseline": None,  # Don't use a hard baseline - let it train!
    "min_epochs_before_stopping": 2,  # Don't allow early stopping before epoch 2
}

# ============================================================================
# END USER CONFIGURATION
# ============================================================================


# class EarlyStopping:
#     def __init__(
#         self,
#         patience=3,
#         min_delta=0.001,
#         restore_best_weights=True,
#         verbose=True,
#         mode="min",
#         baseline=None,
#     ):
#         """
#         Enhanced early stopping with better tracking.

#         Args:
#             patience: Number of checks to wait after last improvement
#             min_delta: Minimum change to qualify as improvement (can be percentage if < 1)
#             restore_best_weights: Whether to restore best weights when stopping
#             verbose: Print messages
#             mode: 'min' for loss (lower is better) or 'max' for accuracy
#             baseline: Stop immediately if worse than this baseline
#         """
#         self.patience = patience
#         self.min_delta = min_delta
#         self.restore_best_weights = restore_best_weights
#         self.verbose = verbose
#         self.mode = mode
#         self.baseline = baseline

#         self.best_score = float("inf") if mode == "min" else float("-inf")
#         self.counter = 0
#         self.best_weights = None
#         self.early_stop = False
#         self.best_epoch = 0
#         self.history = []

#     def __call__(self, current_score, model, epoch=None):
#         self.history.append(current_score)

#         # Check if we should use percentage-based min_delta
#         if self.min_delta < 1 and self.best_score != float("inf"):
#             threshold = self.min_delta * abs(self.best_score)
#         else:
#             threshold = self.min_delta

#         # Check for improvement
#         if self.mode == "min":
#             improved = current_score < self.best_score - threshold
#         else:
#             improved = current_score > self.best_score + threshold

#         # Check baseline
#         if self.baseline is not None:
#             if self.mode == "min" and current_score > self.baseline:
#                 if self.verbose:
#                     print(
#                         f"  ⚠ Score {current_score:.4f} worse than baseline {self.baseline:.4f}"
#                     )
#                 self.early_stop = True
#                 return

#         if improved:
#             if self.verbose and self.counter > 0:
#                 print(
#                     f"  ✓ Validation improved: {self.best_score:.4f} → {current_score:.4f}"
#                 )
#             self.best_score = current_score
#             self.counter = 0
#             self.best_epoch = epoch if epoch is not None else len(self.history)
#             if self.restore_best_weights:
#                 # Deep copy the state dict
#                 import copy

#                 self.best_weights = copy.deepcopy(model.state_dict())
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"  → No improvement for {self.counter}/{self.patience} checks")

#             if self.counter >= self.patience:
#                 self.early_stop = True
#                 if self.restore_best_weights and self.best_weights is not None:
#                     model.load_state_dict(self.best_weights)
#                     if self.verbose:
#                         print(
#                             f"  ✓ Restored best weights from epoch {self.best_epoch} (score: {self.best_score:.4f})"
#                         )

#         # Additional check: if validation is getting much worse, stop immediately
#         if len(self.history) > 1:
#             recent_degradation = current_score - self.best_score
#             if self.mode == "min" and recent_degradation > 0.5:  # Loss increased by 0.5
#                 if self.verbose:
#                     print(
#                         f"  ⚠ Severe degradation detected: {recent_degradation:.4f} increase"
#                     )
#                 self.early_stop = True


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting during model training.

    This class monitors validation metrics and stops training when the model
    stops improving, while being permissive enough to allow for natural
    training fluctuations and exploration of the loss landscape.

    The implementation includes multiple safeguards to prevent premature stopping:
    - Minimum training period before stopping is allowed
    - Patience counter for temporary degradations
    - Percentage-based or absolute improvement thresholds
    - Optional weight restoration to best checkpoint
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.01,
        restore_best_weights=True,
        verbose=True,
        mode="min",
        baseline=None,
        min_epochs=2,
    ):
        """
        Initialize early stopping monitor with training-friendly defaults.

        Args:
            patience (int): Number of evaluations to wait after last improvement
                          before stopping. Higher values are more permissive.
                          Default: 5 (allows for exploration)

            min_delta (float): Minimum change to qualify as improvement.
                             If < 1.0, treated as percentage of current best.
                             Default: 0.01 (1% improvement required)

            restore_best_weights (bool): Whether to restore model weights from
                                       best checkpoint when stopping.
                                       Default: True

            verbose (bool): Print detailed status messages during training.
                          Default: True

            mode (str): Optimization direction - 'min' for loss (lower is better)
                       or 'max' for accuracy (higher is better).
                       Default: 'min'

            baseline (float, optional): Hard threshold - stop immediately if
                                      score is worse than this. Use None to
                                      disable. Default: None (disabled)

            min_epochs (int): Minimum number of epochs before early stopping
                            is allowed to trigger. Prevents premature stopping.
                            Default: 2
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.min_epochs = min_epochs

        # Initialize tracking variables
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.best_epoch = 0
        self.history = []
        self.epoch_count = 0

    # def __call__(self, current_score, model, epoch=None):
    #     """
    #     Check if training should stop based on current validation score.

    #     Args:
    #         current_score (float): Current validation metric value
    #         model (torch.nn.Module): Model to potentially save weights from
    #         epoch (int, optional): Current epoch number for tracking

    #     Returns:
    #         None (sets self.early_stop flag)
    #     """
    #     # Track history
    #     self.history.append(current_score)

    #     # Update epoch count
    #     if epoch is not None:
    #         self.epoch_count = epoch + 1  # Convert 0-indexed to 1-indexed

    #     # Don't allow early stopping before minimum epochs
    #     if self.epoch_count < self.min_epochs:
    #         if self.verbose:
    #             print(
    #                 f"  → Epoch {self.epoch_count}/{self.min_epochs} - "
    #                 f"Early stopping disabled during warmup"
    #             )
    #         return

    #     # Calculate improvement threshold
    #     if (
    #         self.min_delta < 1.0
    #         and self.best_score != float("inf")
    #         and self.best_score != float("-inf")
    #     ):
    #         # Percentage-based threshold
    #         threshold = self.min_delta * abs(self.best_score)
    #     else:
    #         # Absolute threshold
    #         threshold = self.min_delta

    #     # Check for improvement based on mode
    #     if self.mode == "min":
    #         improved = current_score < (self.best_score - threshold)
    #     else:
    #         improved = current_score > (self.best_score + threshold)

    #     # Optional baseline check (disabled by default)
    #     if self.baseline is not None:
    #         if self.mode == "min" and current_score > self.baseline:
    #             if self.verbose:
    #                 print(
    #                     f"  ⚠ Score {current_score:.4f} worse than baseline {self.baseline:.4f}"
    #                 )
    #             self.early_stop = True
    #             return

    #     # Handle improvement or degradation
    #     if improved:
    #         if self.verbose and self.counter > 0:
    #             print(
    #                 f"  ✓ Validation improved: {self.best_score:.4f} → {current_score:.4f}"
    #             )

    #         self.best_score = current_score
    #         self.counter = 0  # Reset patience counter
    #         self.best_epoch = epoch if epoch is not None else len(self.history)

    #         # Save best weights
    #         if self.restore_best_weights:
    #             import copy

    #             self.best_weights = copy.deepcopy(model.state_dict())
    #     else:
    #         self.counter += 1

    #         if self.verbose:
    #             improvement_needed = (
    #                 self.best_score - threshold
    #                 if self.mode == "min"
    #                 else self.best_score + threshold
    #             )
    #             print(
    #                 f"  → No improvement for {self.counter}/{self.patience} checks "
    #                 f"(need {improvement_needed:.4f} or better)"
    #             )

    #         # Check if patience is exhausted
    #         if self.counter >= self.patience:
    #             self.early_stop = True

    #             if self.restore_best_weights and self.best_weights is not None:
    #                 model.load_state_dict(self.best_weights)
    #                 if self.verbose:
    #                     print(f"  ✓ Early stopping triggered!")
    #                     print(
    #                         f"  ✓ Restored best weights from epoch {self.best_epoch} "
    #                         f"(score: {self.best_score:.4f})"
    #                     )

    def __call__(self, current_score, model, epoch=None):
        """
        Check if training should stop based on current validation score.

        FIXED: Now correctly tracks whether current score is better than
        the best score seen so far, not whether it beats itself minus threshold.
        """
        # Track history
        self.history.append(current_score)

        # Update epoch count
        if epoch is not None:
            self.epoch_count = epoch + 1  # Convert 0-indexed to 1-indexed

        # Don't allow early stopping before minimum epochs
        if self.epoch_count < self.min_epochs:
            if self.verbose:
                print(
                    f"  → Epoch {self.epoch_count}/{self.min_epochs} - "
                    f"Early stopping disabled during warmup"
                )
            return

        # Check baseline if configured (disabled by default)
        if self.baseline is not None:
            if self.mode == "min" and current_score > self.baseline:
                if self.verbose:
                    print(
                        f"  ⚠ Score {current_score:.4f} worse than baseline {self.baseline:.4f}"
                    )
                self.early_stop = True
                return

        # THE FIX: Check if current score beats the best score
        # We compare against best_score directly, not best_score minus threshold
        if self.mode == "min":
            # For loss: lower is better
            is_better = current_score < self.best_score
        else:
            # For accuracy: higher is better
            is_better = current_score > self.best_score

        # Now check if the improvement is significant enough (using threshold)
        if is_better:
            # Calculate how much we improved
            improvement = abs(self.best_score - current_score)

            # Calculate required improvement threshold
            if (
                self.min_delta < 1.0
                and self.best_score != float("inf")
                and self.best_score != float("-inf")
            ):
                # Percentage-based threshold
                required_improvement = self.min_delta * abs(self.best_score)
            else:
                # Absolute threshold
                required_improvement = self.min_delta

            # Check if improvement is significant enough
            if improvement >= required_improvement:
                # Significant improvement found!
                if self.verbose and self.counter > 0:
                    print(
                        f"  ✓ Validation improved: {self.best_score:.4f} → {current_score:.4f}"
                    )

                self.best_score = current_score
                self.counter = 0  # Reset patience counter
                self.best_epoch = epoch if epoch is not None else len(self.history)

                # Save best weights
                if self.restore_best_weights:
                    import copy

                    self.best_weights = copy.deepcopy(model.state_dict())
            else:
                # Improvement exists but is too small
                self.counter += 1
                if self.verbose:
                    print(
                        f"  → Minor improvement ({improvement:.4f} < {required_improvement:.4f} required), "
                        f"patience {self.counter}/{self.patience}"
                    )
        else:
            # No improvement at all
            self.counter += 1

            if self.verbose:
                # Show what score would be needed for improvement
                if self.mode == "min":
                    target_score = self.best_score
                else:
                    target_score = self.best_score

                print(
                    f"  → No improvement for {self.counter}/{self.patience} checks "
                    f"(need {target_score:.4f} or better, got {current_score:.4f})"
                )

        # Check if patience is exhausted
        if self.counter >= self.patience:
            self.early_stop = True

            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"  ✓ Early stopping triggered!")
                    print(
                        f"  ✓ Restored best weights from epoch {self.best_epoch} "
                        f"(score: {self.best_score:.4f})"
                    )


class DocumentDataset(Dataset):
    """
    Dataset for document-based training.
    Handles chunking and tokenization of text documents.
    """

    def __init__(self, text, tokenizer, max_length, stride, verbose=True):
        """
        Initialize document dataset.

        Args:
            text (str): Raw text to process
            tokenizer: Tokenizer object with encode method
            max_length (int): Maximum sequence length
            stride (int): Stride between chunks (for overlap)
            verbose (bool): Print statistics
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Tokenize full text
        if verbose:
            print(f"Tokenizing text of length {len(text):,} characters...")

        try:
            self.tokens = tokenizer.encode(text)

            # Create overlapping windows
            self.windows = []
            for i in range(0, len(self.tokens) - max_length, stride):
                window = self.tokens[i : i + max_length + 1]  # +1 for target
                if len(window) == max_length + 1:
                    self.windows.append(window)

            if verbose:
                print(f"Created {len(self.windows):,} training windows")
                print(f"Total tokens: {len(self.tokens):,}")
                print(f"Window size: {max_length}, Stride: {stride}")
                print(
                    f"Effective overlap: {((max_length - stride) / max_length * 100):.1f}%"
                )

        except Exception as e:
            print(f"Error during tokenization: {e}")
            traceback.print_exc()
            raise

    def __len__(self):
        """Return number of training windows."""
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Get a training example.

        Returns:
            tuple: (input_ids, target_ids) tensors
        """
        window = self.windows[idx]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        target_ids = torch.tensor(window[1:], dtype=torch.long)
        return input_ids, target_ids


def load_document(file_path):
    """
    Load text document from file.

    Args:
        file_path (str): Path to text file

    Returns:
        str: Document text content
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Get file size for reporting
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"\nLoading document: {file_path}")
        print(f"File size: {file_size_mb:.2f} MB")

        # Load with encoding detection
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                print(f"Successfully loaded with {encoding} encoding")
                print(f"Document length: {len(text):,} characters")
                return text
            except UnicodeDecodeError:
                continue

        raise ValueError(
            f"Could not decode file with any supported encoding: {encodings_to_try}"
        )

    except Exception as e:
        print(f"Error loading document: {e}")
        traceback.print_exc()
        raise


def create_data_loaders(text, tokenizer, config, train_ratio=0.9):
    """
    Create training and validation data loaders.

    Args:
        text (str): Full document text
        tokenizer: Tokenizer object
        config (dict): Training configuration
        train_ratio (float): Proportion of data for training

    Returns:
        tuple: (train_loader, val_loader)
    """
    try:
        print(
            f"\nCreating data loaders with {train_ratio:.0%} train / {(1 - train_ratio):.0%} validation split"
        )

        # Calculate split point
        split_idx = int(train_ratio * len(text))
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        print(f"Train text: {len(train_text):,} chars")
        print(f"Val text: {len(val_text):,} chars")

        # Calculate stride from overlap
        stride = int(config["context_length"] * (1 - config["chunk_overlap"]))

        # Create datasets
        train_dataset = DocumentDataset(
            train_text, tokenizer, config["context_length"], stride, verbose=True
        )

        val_dataset = DocumentDataset(
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
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        print(f"\nTrain batches: {len(train_loader):,}")
        print(f"Val batches: {len(val_loader):,}")

        return train_loader, val_loader

    except Exception as e:
        print(f"Error creating data loaders: {e}")
        traceback.print_exc()
        raise


# def setup_model(model_size, strategy, config, device):
#     """
#     Initialize Perseid model with specified configuration.

#     Args:
#         model_size (int): Target model size in millions
#         strategy (str): Model strategy (balanced/deep/wide)
#         config (dict): Training configuration
#         device: Device to move model to

#     Returns:
#         tuple: (model, model_config)
#     """
#     try:
#         print(f"\n{'='*60}")
#         print(f"Setting up Perseid-{model_size}M ({strategy} strategy)")
#         print(f"{'='*60}")

#         # Generate Perseid configuration
#         model_config = create_perseid_config(
#             target_size_millions=model_size,
#             strategy=strategy
#         )

#         # Override context length for training
#         model_config["context_length"] = config["context_length"]

#         # Set dtype
#         if USE_BFLOAT16:
#             model_config["dtype"] = torch.bfloat16
#         else:
#             model_config["dtype"] = torch.float32

#         # Validate configuration
#         is_valid, issues = validate_config(model_config)
#         if not is_valid:
#             print("Configuration validation issues:")
#             for issue in issues:
#                 print(f"  - {issue}")
#             if len([i for i in issues if "Warning" not in i]) > 0:
#                 raise ValueError("Critical configuration issues found")

#         # Initialize model
#         print("\nInitializing model...")
#         model = Gemma3Model(model_config)

#         # Move to device
#         model = model.to(device)

#         # Print model statistics
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#         print(f"Model initialized successfully")
#         print(f"Total parameters: {total_params:,}")
#         print(f"Trainable parameters: {trainable_params:,}")
#         print(f"Model size (bfloat16): {total_params * 2 / 1e9:.3f} GB")
#         print(f"Device: {device}")

#         return model, model_config

#     except Exception as e:
#         print(f"Error setting up model: {e}")
#         traceback.print_exc()
#         raise


def setup_model(
    model_size,
    strategy,
    config,
    device,
    output_dir,
    training_mode,
    checkpoint_path=None,
):
    """
    Initialize Perseid model with specified configuration.
    Handles resuming from checkpoint or starting fresh.

    Args:
        model_size (int): Target model size in millions
        strategy (str): Model strategy (balanced/deep/wide)
        config (dict): Training configuration
        device: Device to move model to
        output_dir (str/Path): Output directory for checkpoints
        training_mode (str): "new", "continue", or "force_restart"
        checkpoint_path (str/Path): Specific checkpoint to load, or None for auto

    Returns:
        tuple: (model, model_config, training_state)
    """
    try:
        output_dir = Path(output_dir)

        # Determine checkpoint to load
        if checkpoint_path is None:
            # Look for best checkpoint in output directory
            best_checkpoint = output_dir / "checkpoint_best.pth"
            if best_checkpoint.exists():
                checkpoint_path = best_checkpoint
            else:
                # Try final checkpoint as fallback
                final_checkpoint = output_dir / "perseid_model_final.pth"
                if final_checkpoint.exists():
                    checkpoint_path = final_checkpoint

        # Check if we're resuming or starting fresh
        checkpoint_exists = checkpoint_path and Path(checkpoint_path).exists()

        # Validate training mode
        if training_mode == "new" and checkpoint_exists:
            raise ValueError(
                f"Training mode is 'new' but checkpoint already exists at {checkpoint_path}!\n"
                f"Either:\n"
                f"  1. Change OUTPUT_DIR to a new location\n"
                f"  2. Set TRAINING_MODE = 'continue' to resume training\n"
                f"  3. Set TRAINING_MODE = 'force_restart' to overwrite (WARNING: loses existing model!)"
            )

        elif training_mode == "force_restart":
            if checkpoint_exists:
                print(
                    f"WARNING: Force restart mode - existing checkpoint will be overwritten!"
                )
                print(f"Existing checkpoint: {checkpoint_path}")
                response = input("Are you sure? Type 'yes' to continue: ")
                if response.lower() != "yes":
                    print("Aborting to preserve existing model.")
                    sys.exit(1)
            checkpoint_exists = False  # Treat as new training

        # Load or create model configuration
        if checkpoint_exists and training_mode == "continue":
            print(f"\n{'=' * 60}")
            print(f"Resuming Training from Checkpoint")
            print(f"{'=' * 60}")
            print(f"Loading from: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Extract configuration from checkpoint
            if "model_config" in checkpoint:
                model_config = checkpoint["model_config"]
                print("  ✓ Loaded model configuration from checkpoint")
            else:
                # Fallback: generate config if not in checkpoint
                print("  ! No config in checkpoint, generating from parameters")
                model_config = create_perseid_config(
                    target_size_millions=model_size, strategy=strategy
                )

            # Override context length for current training
            model_config["context_length"] = config["context_length"]

            # Set dtype
            if USE_BFLOAT16 and device != "cpu":
                model_config["dtype"] = torch.bfloat16
            else:
                model_config["dtype"] = torch.float32

            # Initialize model with configuration
            model = PerseidByteModel(model_config)

            # Load weights
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("  ✓ Loaded model weights")
            else:
                # Old-style checkpoint with just state dict
                model.load_state_dict(checkpoint)
                print("  ✓ Loaded model weights (legacy format)")

            # Move to device
            model = model.to(device)

            # Extract training state if available
            training_state = {
                "global_step": checkpoint.get("step", 0),
                "best_val_loss": checkpoint.get("val_loss", float("inf")),
                "optimizer_state": checkpoint.get("optimizer_state_dict", None),
                "scheduler_state": checkpoint.get("scheduler_state_dict", None),
                "epoch": 0,  # <-- Always start from epoch 0 for continued training
                "tokens_seen": checkpoint.get("tokens_seen", 0),
            }

            print(f"\nResuming from:")
            print(f"  - Step: {training_state['global_step']:,}")
            print(f"  - Epoch: {training_state['epoch']}")
            print(f"  - Best validation loss: {training_state['best_val_loss']:.4f}")
            print(f"  - Tokens seen: {training_state['tokens_seen']:,}")

        else:
            # Starting fresh
            print(f"\n{'=' * 60}")
            print(f"Initializing New Model")
            print(f"{'=' * 60}")
            print(f"Creating Perseid-{model_size}M ({strategy} strategy)")

            # Generate Perseid configuration
            model_config = create_perseid_config(
                target_size_millions=model_size,
                strategy=strategy,
                base_config=PERSEID_BYTE_CONFIG_BASE,
            )

            # Override context length for training
            model_config["context_length"] = config["context_length"]

            # Set dtype
            if USE_BFLOAT16 and device != "cpu":
                model_config["dtype"] = torch.bfloat16
            else:
                model_config["dtype"] = torch.float32

            # Validate configuration
            is_valid, issues = validate_config(model_config)
            if not is_valid:
                print("Configuration validation issues:")
                for issue in issues:
                    print(f"  - {issue}")
                if len([i for i in issues if "Warning" not in i]) > 0:
                    raise ValueError("Critical configuration issues found")

            # Initialize model with random weights
            print("\nInitializing model with random weights...")

            # TODO HERE
            model = PerseidByteModel(model_config)

            # Move to device
            model = model.to(device)

            # Fresh training state
            training_state = {
                "global_step": 0,
                "best_val_loss": float("inf"),
                "optimizer_state": None,
                "scheduler_state": None,
                "epoch": 0,
                "tokens_seen": 0,
            }

            print("  ✓ Model initialized with random weights")

        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size (bfloat16): {total_params * 2 / 1e9:.3f} GB")
        print(f"  Device: {device}")

        return model, model_config, training_state

    except Exception as e:
        print(f"Error setting up model: {e}")
        traceback.print_exc()
        raise


def calculate_loss(input_batch, target_batch, model, device):
    """
    Calculate loss for a batch.

    Args:
        input_batch: Input token IDs
        target_batch: Target token IDs
        model: Model to evaluate
        device: Device to use

    Returns:
        torch.Tensor: Loss value
    """
    try:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Forward pass
        logits = model(input_batch)

        # Calculate cross entropy loss
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

        return loss

    except Exception as e:
        print(f"Error calculating loss: {e}")
        traceback.print_exc()
        raise


def evaluate_model(model, data_loader, device, num_batches=None):
    """
    Evaluate model on data loader.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        num_batches: Maximum number of batches to evaluate

    Returns:
        float: Average loss
    """
    model.eval()
    total_loss = 0.0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calculate_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()

    model.train()
    return total_loss / num_batches if num_batches > 0 else float("nan")


def get_lr_with_restarts(step, warmup_steps, total_steps, num_restarts=2):
    """Cosine annealing with warm restarts to escape local minima."""
    if step < warmup_steps:
        return step / warmup_steps

    # Calculate which cycle we're in
    steps_after_warmup = step - warmup_steps
    steps_per_cycle = (total_steps - warmup_steps) / (num_restarts + 1)
    cycle = int(steps_after_warmup / steps_per_cycle)

    if cycle >= num_restarts + 1:
        # After all restarts, stay at minimum
        return 0.05

    # Progress within current cycle
    cycle_progress = (steps_after_warmup % steps_per_cycle) / steps_per_cycle

    # Cosine annealing within cycle, with decreasing amplitude per cycle
    cycle_amplitude = 1.0 / (2**cycle)  # Each restart has half the amplitude
    min_lr = 0.05

    return min_lr + cycle_amplitude * (1.0 - min_lr) * 0.5 * (
        1.0 + torch.cos(torch.tensor(cycle_progress * 3.14159))
    )


def calculate_loss_with_label_smoothing(
    input_batch, target_batch, model, device, smoothing=0.1
):
    """Calculate loss with label smoothing for regularization."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    vocab_size = logits.size(-1)

    # Standard loss
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size), target_batch.flatten(), reduction="none"
    )

    # Add label smoothing
    if smoothing > 0:
        smooth_loss = -logits.reshape(-1, vocab_size).mean(dim=-1)
        loss = (1 - smoothing) * loss + smoothing * smooth_loss

    return loss.mean()


# def train_model(
#     model, train_loader, val_loader, config, device, output_dir, training_state
# ):
#     """
#     Main training loop for Perseid model.

#     Args:
#         model: Model to train
#         train_loader: Training data loader
#         val_loader: Validation data loader
#         config: Training configuration
#         device: Device to use
#         output_dir: Directory to save outputs
#         training_state: Dict with resume information (global_step, optimizer_state, etc.)

#     Returns:
#         dict: Training history
#     """
#     try:
#         print(f"\n{'=' * 60}")
#         print("Starting Training")
#         print(f"{'=' * 60}")

#         # Initialize early stopping with tighter constraints
#         early_stopping = EarlyStopping(
#             patience=3,  # Stop after 3 checks without improvement
#             min_delta=0.001,  # 0.1% improvement required
#             restore_best_weights=True,
#             verbose=True,
#             baseline=2.5,  # Stop immediately if val loss exceeds this
#         )

#         # Track training metrics
#         train_loss_ma = []  # Moving average of training loss
#         val_loss_history = []

#         # Setup optimizer
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=config["learning_rate"],
#             weight_decay=config["weight_decay"],
#             betas=(0.9, 0.95),
#         )

#         # Restore optimizer state if resuming
#         if training_state["optimizer_state"] is not None:
#             optimizer.load_state_dict(training_state["optimizer_state"])
#             print("  ✓ Restored optimizer state")

#         # Calculate total steps
#         steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
#         total_steps = steps_per_epoch * config["num_epochs"]

#         # Setup scheduler with warmup
#         def get_lr(step):
#             """Learning rate schedule with warmup."""
#             if step < config["warmup_steps"]:
#                 return step / config["warmup_steps"]
#             else:
#                 progress = (step - config["warmup_steps"]) / (
#                     total_steps - config["warmup_steps"]
#                 )
#                 return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

#         scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

#         # Restore scheduler state if resuming
#         if training_state["scheduler_state"] is not None:
#             scheduler.load_state_dict(training_state["scheduler_state"])
#             print("  ✓ Restored scheduler state")

#         # Training state
#         history = {"train_loss": [], "val_loss": [], "learning_rates": [], "step": []}

#         # global_step = 0
#         # best_val_loss = float('inf')
#         # tokens_seen = 0
#         global_step = training_state["global_step"]
#         best_val_loss = training_state["best_val_loss"]
#         tokens_seen = training_state["tokens_seen"]
#         start_epoch = training_state["epoch"]

#         print(f"\nStarting from epoch {start_epoch + 1}, step {global_step}")

#         print(f"Total training steps: {total_steps:,}")
#         print(f"Warmup steps: {config['warmup_steps']:,}")
#         print(
#             f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}"
#         )

#         # # Training loop
#         # # for epoch in range(config["num_epochs"]):
#         # for epoch in range(start_epoch, config["num_epochs"]):
#         #     print(f"\n{'=' * 40}")
#         #     print(f"Epoch {epoch + 1}/{config['num_epochs']}")
#         #     print(f"{'=' * 40}")

#         #     model.train()
#         #     epoch_loss = 0
#         #     epoch_tokens = 0

#         #     for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
#         #         # Calculate loss
#         #         loss = calculate_loss(input_batch, target_batch, model, device)
#         #         loss = loss / config["gradient_accumulation_steps"]
#         #         loss.backward()

#         #         epoch_loss += loss.item() * config["gradient_accumulation_steps"]
#         #         epoch_tokens += input_batch.numel()

#         #         # Update weights after gradient accumulation
#         #         if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
#         #             # Gradient clipping
#         #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#         #             optimizer.step()
#         #             scheduler.step()
#         #             optimizer.zero_grad()
#         #             global_step += 1
#         #             tokens_seen += epoch_tokens
#         #             epoch_tokens = 0

#         #             # Periodic evaluation
#         #             if global_step % config["eval_every"] == 0:
#         #                 val_loss = evaluate_model(
#         #                     model,
#         #                     val_loader,
#         #                     device,
#         #                     num_batches=config["eval_batches"],
#         #                 )

#         #                 train_loss = epoch_loss / (batch_idx + 1)
#         #                 current_lr = scheduler.get_last_lr()[0]

#         #                 history["train_loss"].append(train_loss)
#         #                 history["val_loss"].append(val_loss)
#         #                 history["learning_rates"].append(current_lr)
#         #                 history["step"].append(global_step)

#         #                 print(
#         #                     f"Step {global_step:5d} | "
#         #                     f"Train Loss: {train_loss:.4f} | "
#         #                     f"Val Loss: {val_loss:.4f} | "
#         #                     f"LR: {current_lr:.2e} | "
#         #                     f"Tokens: {tokens_seen:,}"
#         #                 )

#         #                 # Save best model
#         #                 if val_loss < best_val_loss:
#         #                     best_val_loss = val_loss
#         #                     save_checkpoint(
#         #                         model,
#         #                         optimizer,
#         #                         scheduler,
#         #                         global_step,
#         #                         best_val_loss,
#         #                         output_dir,
#         #                         "best",
#         #                     )
#         #                     print(f"  → Saved best model (val_loss: {val_loss:.4f})")

#         #             # Periodic checkpoint
#         #             if global_step % config["save_every"] == 0:
#         #                 save_checkpoint(
#         #                     model,
#         #                     optimizer,
#         #                     scheduler,
#         #                     global_step,
#         #                     val_loss,
#         #                     output_dir,
#         #                     f"step_{global_step}",
#         #                 )

#         #     # End of epoch evaluation
#         #     avg_epoch_loss = epoch_loss / len(train_loader)
#         #     val_loss = evaluate_model(model, val_loader, device)

#         #     print(f"\nEpoch {epoch + 1} Summary:")
#         #     print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
#         #     print(f"  Validation Loss: {val_loss:.4f}")
#         #     print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

#         # print(f"\n{'=' * 60}")
#         # print("Training Complete!")
#         # print(f"{'=' * 60}")
#         # print(f"Best validation loss: {best_val_loss:.4f}")
#         # print(f"Total tokens seen: {tokens_seen:,}")

#         # return history
#         #

#         # Use the enhanced learning rate schedule
#         scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer,
#             lambda step: get_lr_with_restarts(
#                 step, config["warmup_steps"], total_steps
#             ),
#         )

#         for epoch in range(start_epoch, config["num_epochs"]):
#             print(f"\n{'=' * 40}")
#             print(f"Epoch {epoch + 1}/{config['num_epochs']}")
#             print(f"{'=' * 40}")

#             # TODO where??
#             # # Update training_state for next checkpoint
#             # training_state["global_step"] = global_step
#             # training_state["epoch"] = epoch
#             # training_state["tokens_seen"] = tokens_seen
#             # training_state["best_val_loss"] = best_val_loss

#             model.train()
#             epoch_loss = 0
#             epoch_tokens = 0
#             batch_losses = []

#             for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
#                 # Calculate loss with label smoothing if specified
#                 if config.get("label_smoothing", 0) > 0:
#                     loss = calculate_loss_with_label_smoothing(
#                         input_batch,
#                         target_batch,
#                         model,
#                         device,
#                         smoothing=config["label_smoothing"],
#                     )
#                 else:
#                     loss = calculate_loss(input_batch, target_batch, model, device)

#                 loss = loss / config["gradient_accumulation_steps"]
#                 loss.backward()

#                 batch_losses.append(loss.item() * config["gradient_accumulation_steps"])
#                 epoch_loss += batch_losses[-1]
#                 epoch_tokens += input_batch.numel()

#                 # Update weights after gradient accumulation
#                 if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
#                     # More aggressive gradient clipping
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), max_norm=config.get("gradient_clip", 1.0)
#                     )

#                     optimizer.step()
#                     scheduler.step()
#                     optimizer.zero_grad()
#                     global_step += 1
#                     tokens_seen += epoch_tokens
#                     epoch_tokens = 0

#                     # Periodic evaluation with enhanced monitoring
#                     if global_step % config["eval_every"] == 0:
#                         val_loss = evaluate_model(
#                             model,
#                             val_loader,
#                             device,
#                             num_batches=config["eval_batches"],
#                         )

#                         # Calculate moving average of recent training loss
#                         recent_train_loss = (
#                             np.mean(batch_losses[-20:])
#                             if len(batch_losses) >= 20
#                             else np.mean(batch_losses)
#                         )

#                         # Calculate gap
#                         train_val_gap = recent_train_loss - val_loss

#                         current_lr = scheduler.get_last_lr()[0]

#                         print(
#                             f"Step {global_step:5d} | "
#                             f"Train Loss: {recent_train_loss:.4f} | "
#                             f"Val Loss: {val_loss:.4f} | "
#                             f"Gap: {abs(train_val_gap):.3f} | "
#                             f"LR: {current_lr:.2e} | "
#                             f"Tokens: {tokens_seen:,}"
#                         )

#                         # Warning for large gap
#                         if train_val_gap < -0.2:  # Validation much worse than training
#                             print(f"  ⚠ Overfitting detected! Gap: {train_val_gap:.3f}")

#                             # Optionally reduce learning rate on overfitting
#                             # if current_lr > 1e-5:
#                             #     for param_group in optimizer.param_groups:
#                             #         param_group["lr"] *= 0.5
#                             if current_lr > 1e-4:  # Higher threshold
#                                 for param_group in optimizer.param_groups:
#                                     param_group["lr"] *= (
#                                         0.8  # Less aggressive reduction
#                                     )
#                                 print(
#                                     f"  → Reduced learning rate to {param_group['lr']:.2e}"
#                                 )

#                         # Check early stopping
#                         early_stopping(val_loss, model, epoch)

#                         if early_stopping.early_stop:
#                             print(f"\n{'=' * 40}")
#                             print(f"Early stopping triggered at step {global_step}")
#                             print(
#                                 f"Best validation loss: {early_stopping.best_score:.4f}"
#                             )
#                             print(f"Training stopped to prevent overfitting")
#                             print(f"{'=' * 40}")

#                             # Save final model before returning
#                             save_checkpoint(
#                                 model,
#                                 optimizer,
#                                 scheduler,
#                                 global_step,
#                                 early_stopping.best_score,
#                                 output_dir,
#                                 "early_stopped",
#                                 epoch=epoch,
#                             )
#                             return history

#                         # Save best model
#                         if val_loss < best_val_loss:
#                             best_val_loss = val_loss
#                             save_checkpoint(
#                                 model,
#                                 optimizer,
#                                 scheduler,
#                                 global_step,
#                                 best_val_loss,
#                                 output_dir,
#                                 "best",
#                                 epoch=epoch,
#                             )
#                             print(f"  → Saved best model (val_loss: {val_loss:.4f})")

#                         history["train_loss"].append(recent_train_loss)
#                         history["val_loss"].append(val_loss)
#                         history["learning_rates"].append(current_lr)
#                         history["step"].append(global_step)

#             # End of epoch evaluation
#             avg_epoch_loss = epoch_loss / len(train_loader)
#             val_loss = evaluate_model(model, val_loader, device)
#             val_loss_history.append(val_loss)

#             print(f"\nEpoch {epoch + 1} Summary:")
#             print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
#             print(f"  Validation Loss: {val_loss:.4f}")
#             print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

#             # Check if validation is getting worse over epochs
#             if len(val_loss_history) >= 2:
#                 if val_loss_history[-1] > val_loss_history[-2]:
#                     print(
#                         f"  ⚠ Validation loss increased from {val_loss_history[-2]:.4f} to {val_loss_history[-1]:.4f}"
#                     )

#             # Early stopping check at epoch level
#             early_stopping(val_loss, model, epoch)
#             if early_stopping.early_stop:
#                 print(f"\nEarly stopping after epoch {epoch + 1}")
#                 break

#         return history

#     except Exception as e:
#         print(f"Error during training: {e}")
#         traceback.print_exc()
#         raise


def train_model(
    model, train_loader, val_loader, config, device, output_dir, training_state
):
    """
    Main training loop for Perseid model with early stopping and checkpointing.

    This function implements the complete training pipeline including:
    - Gradient accumulation for effective batch size scaling
    - Learning rate scheduling with warmup and cosine annealing with restarts
    - Early stopping to prevent overfitting
    - Periodic evaluation and checkpointing
    - Label smoothing regularization
    - Automatic mixed precision training (if supported)
    - Comprehensive logging and monitoring

    The training loop supports both starting from scratch and resuming from
    a checkpoint, with all necessary state preserved for exact reproduction.

    Args:
        model (torch.nn.Module): The Perseid model to train. Must have .cfg attribute
                                containing model configuration dictionary.

        train_loader (torch.utils.data.DataLoader): DataLoader providing training
                                                    batches. Each batch should return
                                                    (input_ids, target_ids) tensors.

        val_loader (torch.utils.data.DataLoader): DataLoader providing validation
                                                  batches for evaluation.

        config (dict): Training configuration dictionary containing:
                      - learning_rate (float): Initial/maximum learning rate
                      - weight_decay (float): L2 regularization strength
                      - num_epochs (int): Maximum number of training epochs
                      - warmup_steps (int): Number of linear warmup steps
                      - gradient_accumulation_steps (int): Steps before weight update
                      - eval_every (int): Evaluation frequency in steps
                      - eval_batches (int): Number of validation batches to evaluate
                      - save_every (int): Checkpoint frequency in steps
                      - gradient_clip (float): Maximum gradient norm for clipping
                      - label_smoothing (float, optional): Label smoothing factor

        device (str or torch.device): Device to run training on ('cuda', 'cpu', etc.)

        output_dir (str or Path): Directory for saving checkpoints and logs.

        training_state (dict): Dictionary containing training state for resumption:
                              - global_step (int): Starting global step
                              - best_val_loss (float): Best validation loss so far
                              - optimizer_state (dict or None): Optimizer state dict
                              - scheduler_state (dict or None): Scheduler state dict
                              - epoch (int): Starting epoch (0-indexed)
                              - tokens_seen (int): Total tokens processed so far

    Returns:
        dict: Training history containing:
              - train_loss (list): Training loss values at each evaluation
              - val_loss (list): Validation loss values at each evaluation
              - learning_rates (list): Learning rate values at each evaluation
              - step (list): Global step numbers at each evaluation

    Raises:
        RuntimeError: If CUDA out of memory or other PyTorch runtime errors.
        KeyboardInterrupt: Re-raised to allow graceful training interruption.
        Exception: Other exceptions are logged with traceback and re-raised.

    Side Effects:
        - Saves model checkpoints to output_dir
        - Modifies model weights in-place
        - Prints training progress to stdout
        - May modify learning rate if overfitting detected

    Example:
        >>> training_state = {
        ...     "global_step": 0,
        ...     "best_val_loss": float('inf'),
        ...     "optimizer_state": None,
        ...     "scheduler_state": None,
        ...     "epoch": 0,
        ...     "tokens_seen": 0
        ... }
        >>> history = train_model(
        ...     model, train_loader, val_loader,
        ...     config, device, "./outputs", training_state
        ... )
    """
    try:
        print(f"\n{'=' * 60}")
        print("Starting Training")
        print(f"{'=' * 60}")

        # # Initialize early stopping with configured parameters
        # early_stopping = EarlyStopping(
        #     patience=3,  # Stop after 3 checks without improvement
        #     min_delta=0.001,  # 0.1% improvement required
        #     restore_best_weights=True,
        #     verbose=True,
        #     baseline=2.5,  # Stop immediately if val loss exceeds this
        # )

        # # Initialize early stopping with REASONABLE parameters
        # early_stopping = EarlyStopping(
        #     patience=config.get("early_stop_patience", 5),  # More patience
        #     min_delta=config.get("early_stop_min_delta", 0.01),  # 1% improvement
        #     restore_best_weights=True,
        #     verbose=True,
        #     baseline=config.get("early_stop_baseline", None),  # No hard limit
        #     min_epochs=config.get(
        #         "min_epochs_before_stopping", 2
        #     ),  # Wait before stopping
        # )

        # Initialize early stopping with VERY permissive parameters
        # This should only trigger after significant training
        early_stopping = EarlyStopping(
            patience=10,  # Very patient - 10 checks without improvement
            min_delta=0.005,  # Only require 0.5% improvement
            restore_best_weights=True,
            verbose=True,
            baseline=None,  # No baseline at all
            min_epochs=3,  # Don't even consider stopping before epoch 3
        )

        # Initialize tracking lists for metrics
        train_loss_ma = []  # Moving average of training loss
        val_loss_history = []  # Full validation loss history

        # Setup AdamW optimizer with configured parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),  # Beta values from GPT-3 paper
        )

        # Restore optimizer state if resuming from checkpoint
        if training_state["optimizer_state"] is not None:
            optimizer.load_state_dict(training_state["optimizer_state"])
            print("  ✓ Restored optimizer state")

        # Calculate total training steps for scheduler
        steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
        total_steps = steps_per_epoch * config["num_epochs"]

        # Setup learning rate scheduler with cosine annealing and warm restarts
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: get_lr_with_restarts(
                step, config["warmup_steps"], total_steps, num_restarts=2
            ),
        )

        # Restore scheduler state if resuming from checkpoint
        if training_state["scheduler_state"] is not None:
            scheduler.load_state_dict(training_state["scheduler_state"])
            print("  ✓ Restored scheduler state")

        # Initialize history tracking
        history = {"train_loss": [], "val_loss": [], "learning_rates": [], "step": []}

        # Extract current training state
        global_step = training_state["global_step"]
        best_val_loss = training_state["best_val_loss"]
        tokens_seen = training_state["tokens_seen"]
        start_epoch = training_state["epoch"]

        print(f"\nStarting from epoch {start_epoch + 1}, step {global_step}")
        print(f"Total training steps: {total_steps:,}")
        print(f"Warmup steps: {config['warmup_steps']:,}")
        print(
            f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}"
        )

        # Main training loop over epochs
        for epoch in range(start_epoch, config["num_epochs"]):
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            print(f"{'=' * 40}")

            # Update training_state with current epoch
            training_state["epoch"] = epoch

            model.train()  # Set model to training mode
            epoch_loss = 0
            epoch_tokens = 0
            batch_losses = []  # Store individual batch losses

            # Iterate through training batches
            for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                # Calculate loss with optional label smoothing
                if config.get("label_smoothing", 0) > 0:
                    loss = calculate_loss_with_label_smoothing(
                        input_batch,
                        target_batch,
                        model,
                        device,
                        smoothing=config["label_smoothing"],
                    )
                else:
                    loss = calculate_loss(input_batch, target_batch, model, device)

                # Scale loss for gradient accumulation
                loss = loss / config["gradient_accumulation_steps"]
                loss.backward()

                # Track batch statistics
                batch_losses.append(loss.item() * config["gradient_accumulation_steps"])
                epoch_loss += batch_losses[-1]
                epoch_tokens += input_batch.numel()

                # Update weights after accumulating gradients
                if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.get("gradient_clip", 1.0)
                    )

                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Update global counters
                    global_step += 1
                    tokens_seen += epoch_tokens
                    epoch_tokens = 0

                    # Update training_state with current progress
                    training_state["global_step"] = global_step
                    training_state["tokens_seen"] = tokens_seen

                    # Periodic evaluation
                    if global_step % config["eval_every"] == 0:
                        # Evaluate on validation set
                        val_loss = evaluate_model(
                            model,
                            val_loader,
                            device,
                            num_batches=config["eval_batches"],
                        )

                        # Calculate recent training loss (moving average)
                        recent_train_loss = (
                            np.mean(batch_losses[-20:])
                            if len(batch_losses) >= 20
                            else np.mean(batch_losses)
                        )

                        # Calculate training-validation gap
                        train_val_gap = recent_train_loss - val_loss
                        current_lr = scheduler.get_last_lr()[0]

                        # Log current metrics
                        print(
                            f"\nStep {global_step:5d} | "
                            f"Train Loss: {recent_train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Gap: {abs(train_val_gap):.3f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens: {tokens_seen:,}"
                        )

                        # bunk
                        # # Check for overfitting
                        # if train_val_gap < -0.2:  # Validation much worse than training
                        #     print(f"  ⚠ Overfitting detected! Gap: {train_val_gap:.3f}")

                        #     # Optionally reduce learning rate
                        #     if current_lr > 1e-4:
                        #         for param_group in optimizer.param_groups:
                        #             param_group["lr"] *= 0.8
                        #         print(
                        #             f"  → Reduced learning rate to {param_group['lr']:.2e}"
                        #         )

                        # # Check early stopping criteria
                        # early_stopping(val_loss, model, epoch)

                        # if early_stopping.early_stop:
                        #     print(f"\n{'=' * 40}")
                        #     print(f"Early stopping triggered at step {global_step}")
                        #     print(
                        #         f"Best validation loss: {early_stopping.best_score:.4f}"
                        #     )
                        #     print(f"Training stopped to prevent overfitting")
                        #     print(f"{'=' * 40}")

                        #     # Save final checkpoint before stopping
                        #     save_checkpoint(
                        #         model,
                        #         optimizer,
                        #         scheduler,
                        #         global_step,
                        #         early_stopping.best_score,
                        #         output_dir,
                        #         "early_stopped",
                        #         epoch=epoch,
                        #         tokens_seen=tokens_seen,
                        #         training_state=training_state,
                        #     )
                        #     return history

                        # The gap is INFORMATION ONLY - not a trigger for action
                        if epoch >= 3:  # Only after substantial training
                            if train_val_gap < -0.5:  # Only if gap is really large
                                print(
                                    f"  Note: Large train/val gap of {abs(train_val_gap):.3f}"
                                )
                                # BUT DON'T DO ANYTHING ABOUT IT!
                                # No learning rate changes!
                                # No stopping!
                                # Just note it for the user

                        # Early stopping check - but won't trigger early
                        if epoch >= early_stopping.min_epochs:
                            early_stopping(val_loss, model, epoch)

                            if early_stopping.early_stop:
                                print(f"\n{'=' * 40}")
                                print(f"Early stopping triggered at step {global_step}")
                                print(
                                    f"Best validation loss: {early_stopping.best_score:.4f}"
                                )
                                print(
                                    f"This happened after {epoch + 1} full epochs of training"
                                )
                                print(f"{'=' * 40}")

                                save_checkpoint(
                                    model,
                                    optimizer,
                                    scheduler,
                                    global_step,
                                    early_stopping.best_score,
                                    output_dir,
                                    "early_stopped",
                                    epoch=epoch,
                                    tokens_seen=tokens_seen,
                                    training_state=training_state,
                                )
                                return history
                        else:
                            print(
                                f"  → Epoch {epoch + 1}/{early_stopping.min_epochs} - "
                                f"Early stopping disabled during initial training"
                            )

                        # Save best model checkpoint
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            training_state["best_val_loss"] = best_val_loss

                            save_checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                global_step,
                                best_val_loss,
                                output_dir,
                                "best",
                                epoch=epoch,
                                tokens_seen=tokens_seen,
                                training_state=training_state,
                            )
                            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

                        # Record history
                        history["train_loss"].append(recent_train_loss)
                        history["val_loss"].append(val_loss)
                        history["learning_rates"].append(current_lr)
                        history["step"].append(global_step)

                    # Periodic checkpoint saving
                    if global_step % config.get("save_every", 500) == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            val_loss,
                            output_dir,
                            f"step_{global_step}",
                            epoch=epoch,
                            tokens_seen=tokens_seen,
                            training_state=training_state,
                        )

            # End of epoch evaluation
            avg_epoch_loss = epoch_loss / len(train_loader)
            val_loss = evaluate_model(model, val_loader, device)
            val_loss_history.append(val_loss)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

            # Check validation degradation
            if len(val_loss_history) >= 2:
                if val_loss_history[-1] > val_loss_history[-2]:
                    print(
                        f"  ⚠ Validation loss increased from "
                        f"{val_loss_history[-2]:.4f} to {val_loss_history[-1]:.4f}"
                    )

            # Epoch-level early stopping check
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f"\nEarly stopping after epoch {epoch + 1}")
                break

        # Training completed
        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Total tokens seen: {tokens_seen:,}")

        return history

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving emergency checkpoint...")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            global_step,
            val_loss,
            output_dir,
            "interrupted",
            epoch=epoch,
            tokens_seen=tokens_seen,
            training_state=training_state,
        )
        raise

    except RuntimeError as runtime_error:
        if "out of memory" in str(runtime_error).lower():
            print("\n⚠ CUDA out of memory error!")
            print("Consider reducing batch_size or gradient_accumulation_steps")
        print(f"Runtime error during training: {runtime_error}")
        traceback.print_exc()
        raise

    except Exception as unexpected_error:
        print(f"Unexpected error during training: {unexpected_error}")
        traceback.print_exc()
        raise


# def save_checkpoint(model, optimizer, scheduler, step, val_loss, output_dir, tag):
#     """
#     Save model checkpoint with all training state.

#     Args:
#         model: Model to save
#         optimizer: Optimizer state
#         scheduler: Scheduler state
#         step: Current training step
#         val_loss: Current validation loss
#         output_dir: Output directory
#         tag: Checkpoint tag (e.g., "best", "step_1000")
#     """
#     try:
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         checkpoint_path = output_dir / f"checkpoint_{tag}.pth"

#         torch.save(
#             {
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "scheduler_state_dict": scheduler.state_dict(),
#                 "step": step,
#                 "val_loss": val_loss,
#                 "model_config": model.cfg,
#             },
#             checkpoint_path,
#         )

#     except Exception as e:
#         print(f"Error saving checkpoint: {e}")
#         traceback.print_exc()


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    val_loss,
    output_dir,
    tag,
    epoch=None,
    tokens_seen=None,
    training_state=None,
):
    """
    Save model checkpoint with complete training state for resumption.

    This function creates a checkpoint file containing all necessary information
    to resume training from the exact point where it was saved. This includes
    model weights, optimizer state, learning rate scheduler state, and training
    metadata.

    The checkpoint is saved in PyTorch's standard format and can be loaded
    using torch.load() for training resumption or inference.

    Args:
        model (torch.nn.Module): The model whose state_dict will be saved.
                                Must have a .cfg attribute containing model configuration.

        optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
                                          This preserves momentum and other optimizer
                                          internal states for proper training resumption.

        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
                                                          whose state will be saved.
                                                          Preserves the learning rate
                                                          schedule position.

        step (int): Current global training step number. This is the total number
                   of gradient updates performed across all epochs.

        val_loss (float): Current validation loss value. Used to track best model
                         performance and for early stopping decisions.

        output_dir (str or Path): Directory where checkpoint file will be saved.
                                 Will be created if it doesn't exist.

        tag (str): Checkpoint identifier tag used in filename.
                  Examples: "best", "final", "early_stopped", "step_1000"
                  Results in filename: checkpoint_{tag}.pth

        epoch (int, optional): Current epoch number (0-indexed). Used for resuming
                              training from the correct epoch. Default: None.

        tokens_seen (int, optional): Total number of tokens processed so far.
                                    Useful for tracking data efficiency and sample
                                    complexity. Default: None (will attempt to
                                    get from training_state if provided).

        training_state (dict, optional): Dictionary containing additional training
                                        state information. If provided and tokens_seen
                                        is None, will extract tokens_seen from this.
                                        Default: None.

    Returns:
        None

    Raises:
        OSError: If checkpoint file cannot be written due to permissions or disk space.
        AttributeError: If model doesn't have required .cfg attribute.
        Exception: Re-raises any unexpected exceptions after logging.

    Side Effects:
        Creates a checkpoint file at: {output_dir}/checkpoint_{tag}.pth
        Creates output_dir if it doesn't exist.

    Example:
        >>> # During training loop
        >>> save_checkpoint(
        ...     model, optimizer, scheduler,
        ...     step=1000, val_loss=1.234,
        ...     output_dir="./checkpoints",
        ...     tag="best",
        ...     epoch=5,
        ...     tokens_seen=1_000_000
        ... )
        # Creates: ./checkpoints/checkpoint_best.pth

    Notes:
        - Checkpoint files can be large (several GB for large models)
        - Uses torch.save() which pickles Python objects
        - Compatible with distributed training (each rank should save separately)
        - For production, consider adding checkpoint validation after saving
    """
    try:
        # Convert output_dir to Path for robust path handling
        output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct checkpoint file path
        checkpoint_path = output_dir / f"checkpoint_{tag}.pth"

        # Extract tokens_seen from training_state if not explicitly provided
        if tokens_seen is None and training_state is not None:
            tokens_seen = training_state.get("tokens_seen", 0)
        elif tokens_seen is None:
            tokens_seen = 0  # Default to 0 if not available

        # Prepare checkpoint dictionary with all training state
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "model_config": model.cfg,
            "epoch": epoch,
            "tokens_seen": tokens_seen,
            "checkpoint_tag": tag,  # Store tag for reference
            "checkpoint_timestamp": datetime.now().isoformat(),  # Add timestamp
        }

        # Save checkpoint to disk
        torch.save(checkpoint_dict, checkpoint_path)

        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    except AttributeError as attr_error:
        print(f"Error: Model missing required attributes: {attr_error}")
        traceback.print_exc()
        raise

    except OSError as os_error:
        print(f"Error: Failed to save checkpoint to disk: {os_error}")
        traceback.print_exc()
        raise

    except Exception as unexpected_error:
        print(f"Unexpected error saving checkpoint: {unexpected_error}")
        traceback.print_exc()
        raise


def save_training_results(model, model_config, history, output_dir):
    """
    Save final model weights, configuration, training history, and visualization plots.

    This function performs a complete export of all training artifacts including:
    - Final model weights (PyTorch state dict)
    - Model configuration (JSON)
    - Training history with loss and learning rate data (JSON)
    - Training curve visualizations (PNG)
    - Training hyperparameters (JSON)

    All tensor values are automatically converted from GPU to CPU before saving
    or plotting to ensure compatibility with numpy/matplotlib and JSON serialization.

    Args:
        model (torch.nn.Module): The trained PyTorch model whose weights will be saved.
                                Must have a state_dict() method.

        model_config (dict): Dictionary containing model architecture configuration.
                            Should include all parameters needed to recreate the model.
                            The 'dtype' field will be converted to string for JSON compatibility.

        history (dict): Dictionary containing training history with keys:
                       - 'train_loss': List of training loss values per evaluation step
                       - 'val_loss': List of validation loss values per evaluation step
                       - 'learning_rates': List of learning rate values per evaluation step
                       - 'step': List of global training steps at each evaluation
                       Values can be Python floats or PyTorch tensors (CPU or CUDA).

        output_dir (str or Path): Directory path where all outputs will be saved.
                                 Will be created if it doesn't exist.

    Returns:
        None

    Raises:
        Exception: Re-raises any exceptions after logging them with full traceback.
                  Partial saves may occur if error happens mid-function.

    Side Effects:
        Creates the following files in output_dir:
        - perseid_model_final.pth: PyTorch model weights
        - model_config.json: Model architecture configuration
        - training_history.json: Complete training history data
        - training_curves.png: Three-panel visualization of training progress
        - training_config.json: Training hyperparameter configuration

    Example:
        >>> model = PerseidByteModel(config)
        >>> history = {'train_loss': [...], 'val_loss': [...], ...}
        >>> save_training_results(model, config, history, './outputs/')
    """
    try:
        # Convert output_dir to Path object for consistent path operations
        output_dir = Path(output_dir)

        # Create output directory and all parent directories if they don't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving training results to {output_dir}")

        # =====================================================================
        # SECTION 1: Save Final Model Weights
        # =====================================================================
        try:
            model_weights_path = output_dir / "perseid_model_final.pth"

            # Save model state dictionary containing all learned parameters
            torch.save(model.state_dict(), model_weights_path)

            print(f"  ✓ Model weights saved to {model_weights_path}")

        except Exception as model_save_error:
            print(f"  ✗ Failed to save model weights: {model_save_error}")
            traceback.print_exc()
            # Continue with other saves even if model save fails

        # =====================================================================
        # SECTION 2: Save Model Configuration
        # =====================================================================
        try:
            config_json_path = output_dir / "model_config.json"

            # Create a JSON-serializable version of the configuration
            # We must handle the dtype field specially as torch dtypes aren't JSON serializable
            config_for_json_serialization = {}

            for key, value in model_config.items():
                if key == "dtype":
                    # Convert PyTorch dtype to string representation
                    config_for_json_serialization[key] = str(value)
                else:
                    # Copy all other fields as-is
                    config_for_json_serialization[key] = value

            # Write configuration to JSON file with readable formatting
            with open(config_json_path, "w", encoding="utf-8") as config_file:
                json.dump(config_for_json_serialization, config_file, indent=2)

            print(f"  ✓ Configuration saved to {config_json_path}")

        except Exception as config_save_error:
            print(f"  ✗ Failed to save model configuration: {config_save_error}")
            traceback.print_exc()

        # =====================================================================
        # SECTION 3: Save Training History (with tensor-to-CPU conversion)
        # =====================================================================
        try:
            history_json_path = output_dir / "training_history.json"

            # Convert all history values to JSON-serializable format
            # This handles PyTorch tensors that may be on GPU
            history_for_json_serialization = {}

            for history_key, history_values in history.items():
                if isinstance(history_values, list) and len(history_values) > 0:
                    # Process each value in the list
                    converted_values_list = []

                    for individual_value in history_values:
                        # Check if value is a PyTorch tensor on GPU
                        if hasattr(individual_value, "cpu"):
                            # Move tensor to CPU and extract Python scalar
                            cpu_tensor = individual_value.cpu()
                            if cpu_tensor.numel() == 1:
                                # Single element tensor - convert to Python scalar
                                converted_values_list.append(cpu_tensor.item())
                            else:
                                # Multi-element tensor - convert to Python list
                                converted_values_list.append(cpu_tensor.tolist())

                        # Check if value is a scalar tensor already on CPU
                        elif hasattr(individual_value, "item"):
                            converted_values_list.append(individual_value.item())

                        # Value is already a regular Python number
                        else:
                            converted_values_list.append(individual_value)

                    history_for_json_serialization[history_key] = converted_values_list

                else:
                    # Non-list values (e.g., metadata) - copy as-is
                    history_for_json_serialization[history_key] = history_values

            # Write training history to JSON file
            with open(history_json_path, "w", encoding="utf-8") as history_file:
                json.dump(history_for_json_serialization, history_file, indent=2)

            print(f"  ✓ Training history saved to {history_json_path}")

        except Exception as history_save_error:
            print(f"  ✗ Failed to save training history: {history_save_error}")
            traceback.print_exc()

        # =====================================================================
        # SECTION 4: Generate and Save Training Visualization Plots
        # =====================================================================
        try:
            # Only create plots if we have training data
            if len(history.get("train_loss", [])) > 0:
                # Convert all plotting data from potential GPU tensors to CPU values
                # This ensures matplotlib/numpy compatibility

                steps_for_plotting = []
                for step_value in history["step"]:
                    if hasattr(step_value, "cpu"):
                        steps_for_plotting.append(step_value.cpu().item())
                    elif hasattr(step_value, "item"):
                        steps_for_plotting.append(step_value.item())
                    else:
                        steps_for_plotting.append(step_value)

                train_loss_for_plotting = []
                for loss_value in history["train_loss"]:
                    if hasattr(loss_value, "cpu"):
                        train_loss_for_plotting.append(loss_value.cpu().item())
                    elif hasattr(loss_value, "item"):
                        train_loss_for_plotting.append(loss_value.item())
                    else:
                        train_loss_for_plotting.append(loss_value)

                val_loss_for_plotting = []
                for loss_value in history["val_loss"]:
                    if hasattr(loss_value, "cpu"):
                        val_loss_for_plotting.append(loss_value.cpu().item())
                    elif hasattr(loss_value, "item"):
                        val_loss_for_plotting.append(loss_value.item())
                    else:
                        val_loss_for_plotting.append(loss_value)

                learning_rates_for_plotting = []
                for lr_value in history["learning_rates"]:
                    if hasattr(lr_value, "cpu"):
                        learning_rates_for_plotting.append(lr_value.cpu().item())
                    elif hasattr(lr_value, "item"):
                        learning_rates_for_plotting.append(lr_value.item())
                    else:
                        learning_rates_for_plotting.append(lr_value)

                # Create figure with three subplots
                figure_handle = plt.figure(figsize=(12, 4))

                # -------------------------
                # Subplot 1: Loss Curves
                # -------------------------
                plt.subplot(1, 3, 1)
                plt.plot(
                    steps_for_plotting,
                    train_loss_for_plotting,
                    label="Train Loss",
                    color="blue",
                    linewidth=1.5,
                )
                plt.plot(
                    steps_for_plotting,
                    val_loss_for_plotting,
                    label="Val Loss",
                    color="orange",
                    linewidth=1.5,
                )
                plt.xlabel("Training Step")
                plt.ylabel("Loss")
                plt.title("Training Progress")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # -------------------------
                # Subplot 2: Perplexity
                # -------------------------
                plt.subplot(1, 3, 2)

                # Calculate perplexity from validation loss
                # Perplexity = exp(loss) for language modeling
                validation_perplexity_values = []
                for loss in val_loss_for_plotting:
                    perplexity = torch.exp(torch.tensor(loss)).item()
                    # Cap perplexity at a reasonable value for visualization
                    perplexity = min(perplexity, 1000.0)
                    validation_perplexity_values.append(perplexity)

                plt.plot(
                    steps_for_plotting,
                    validation_perplexity_values,
                    color="green",
                    linewidth=1.5,
                )
                plt.xlabel("Training Step")
                plt.ylabel("Perplexity")
                plt.title("Validation Perplexity")
                plt.grid(True, alpha=0.3)

                # -------------------------
                # Subplot 3: Learning Rate Schedule
                # -------------------------
                plt.subplot(1, 3, 3)
                plt.plot(
                    steps_for_plotting,
                    learning_rates_for_plotting,
                    color="red",
                    linewidth=1.5,
                )
                plt.xlabel("Training Step")
                plt.ylabel("Learning Rate")
                plt.title("Learning Rate Schedule")
                plt.grid(True, alpha=0.3)

                # Adjust layout to prevent overlap
                plt.tight_layout()

                # Save figure to file
                plot_output_path = output_dir / "training_curves.png"
                plt.savefig(plot_output_path, dpi=150, bbox_inches="tight")

                # Close figure to free memory
                plt.close(figure_handle)

                print(f"  ✓ Training curves saved to {plot_output_path}")

            else:
                print("  ⚠ No training data available for plotting")

        except Exception as plotting_error:
            print(f"  ✗ Failed to create training plots: {plotting_error}")
            traceback.print_exc()
            # Ensure any open plots are closed
            plt.close("all")

        # =====================================================================
        # SECTION 5: Save Training Hyperparameters Configuration
        # =====================================================================
        try:
            training_config_path = output_dir / "training_config.json"

            # Save the global training configuration
            with open(training_config_path, "w", encoding="utf-8") as config_file:
                json.dump(TRAINING_CONFIG, config_file, indent=2)

            print(f"  ✓ Training config saved to {training_config_path}")

        except Exception as training_config_error:
            print(f"  ✗ Failed to save training config: {training_config_error}")
            traceback.print_exc()

        # Final summary message
        print(f"\nAll outputs saved to: {output_dir}")

    except Exception as unexpected_error:
        # Catch any unexpected errors not handled by specific try-except blocks
        print(f"Unexpected error in save_training_results: {unexpected_error}")
        traceback.print_exc()
        # Re-raise to alert calling code
        raise


def main():
    """
    Main training pipeline for Perseid document training.
    """
    try:
        print(f"\n{'=' * 60}")
        print("Perseid Document Training Pipeline")
        print(f"{'=' * 60}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Output directory: {OUTPUT_DIR}")

        # Create output directory
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # 1. Load document
        print(f"\n{'=' * 40}")
        print("Step 1: Loading Document")
        print(f"{'=' * 40}")
        document_text = load_document(DOCUMENT_PATH)

        # 2. Setup model and tokenizer
        print(f"\n{'=' * 40}")
        print("Step 2: Setting Up Model")
        print(f"{'=' * 40}")

        ###############################
        # Setup ByteTokenizer
        ###############################

        tokenizer = setup_tokenizer()

        # Initialize model
        model, model_config, training_state = setup_model(
            MODEL_SIZE,
            MODEL_STRATEGY,
            TRAINING_CONFIG,
            DEVICE,
            OUTPUT_DIR,
            TRAINING_MODE,
            CHECKPOINT_PATH,
        )

        # 3. Create data loaders
        print(f"\n{'=' * 40}")
        print("Step 3: Preparing Data")
        print(f"{'=' * 40}")
        train_loader, val_loader = create_data_loaders(
            document_text, tokenizer, TRAINING_CONFIG, train_ratio=TRAIN_VAL_SPLIT
        )

        # 4. Train model
        print(f"\n{'=' * 40}")
        print("Step 4: Training Model")
        print(f"{'=' * 40}")
        history = train_model(
            model,
            train_loader,
            val_loader,
            TRAINING_CONFIG,
            DEVICE,
            output_dir,
            training_state,
        )

        # 5. Save results
        print(f"\n{'=' * 40}")
        print("Step 5: Saving Results")
        print(f"{'=' * 40}")
        save_training_results(model, model_config, history, output_dir)

        print(f"\n{'=' * 60}")
        print("Training Pipeline Complete!")

        # 5.5 Generate sample text with trained model
        print(f"\n{'=' * 40}")
        print("Step 5.5: Sample Generation")
        print(f"{'=' * 40}")

        test_prompts = [
            "Once upon a time",
            "The meaning of life is",
            "In the beginning",
        ]
        for prompt in test_prompts:
            # output = generate_text_simple(model, tokenizer, prompt, max_new_tokens=50)
            output = generate_text_simple(
                model, tokenizer, prompt, max_new_tokens=50, device=DEVICE
            )
            print(f"Prompt: '{prompt}'")
            print(f"Output: {output}\n")

        print(f"{'=' * 60}")
        print(f"Model and results saved to: {output_dir}")

        return model, history

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("Training Pipeline Failed")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        traceback.print_exc()
        raise


def test_integration():
    """Test ByteTokenizer integration with model config."""
    print("\n" + "=" * 60)
    print("Testing ByteTokenizer Integration")
    print("=" * 60)

    # Test tokenizer
    tokenizer = setup_tokenizer()

    # Test encoding/decoding
    test_text = "Hello, World! 🌍"
    tokens = tokenizer.encode(test_text, add_eos=True)
    decoded = tokenizer.decode(tokens)

    print(f"\nTokenizer test:")
    print(f"  Original: {repr(test_text)}")
    print(f"  Tokens: {tokens} ({len(tokens)} tokens)")
    print(f"  Decoded: {repr(decoded)}")
    print(f"  Match: {test_text + tokenizer.eos_token == decoded}")

    # Test model config
    model_config = create_perseid_config(256, strategy="balanced")
    print(f"\nModel config test:")
    print(f"  Vocab size: {model_config['vocab_size']}")
    print(f"  Expected: 259")
    print(f"  Match: {model_config['vocab_size'] == 259}")

    # Test parameter calculation
    params = calculate_model_params(model_config)
    print(f"  Model size: {params['total_millions']:.2f}M parameters")

    return tokenizer, model_config


# if __name__ == "__main__":
#     # Add this line before the main training pipeline
#     test_integration()

#     # inspect
#     print("Arguments passed to the script:")
#     for i, arg in enumerate(sys.argv):
#         print(f"\tArgument {i}: {arg}")

#     # ============================================================================
#     # USER CONFIGURATION SECTION - MODIFY THESE SETTINGS
#     # ============================================================================


#     # preset/reset
#     file_path = None

#     # get path if supplied
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <path>")

#         # Download sample text data
#         demo_file_path = "data/alice.txt"
#         os.makedirs("data", exist_ok=True)

#         if not os.path.exists(demo_file_path):
#             url = "https://www.gutenberg.org/files/11/11-0.txt"
#             print(f"Downloading training data from {url}")
#             with urllib.request.urlopen(url) as response:
#                 text_data = response.read().decode('utf-8')
#             with open(demo_file_path, "w", encoding="utf-8") as file:
#                 file.write(text_data)
#         else:
#             print(f"Loading existing data from {demo_file_path}")
#             with open(demo_file_path, "r", encoding="utf-8") as file:
#                 text_data = file.read()

#         # Q&A
#         user_path_or_demo_choice = input("\nEnter a file path to a .txt file or for a demo say 'demo'\n")

#         # use demo if demo is selected
#         if user_path_or_demo_choice.lower().strip() == 'demo':
#             file_path = demo_file_path

#         # use Q&A input path if selected
#         else:
#             file_path = user_path_or_demo_choice

#     # use argument input path if supplied by user
#     elif len(sys.argv) == 2:

#         file_path = sys.argv[1]
#         print(f"path argument found... {file_path}")

#     else:
#         print("Edge case, defaulting to demo.")

#     # Document input
#     DOCUMENT_PATH = file_path  # "./data/my_document.txt"  # Path to your text file

#     # Output configuration
#     OUTPUT_DIR = f"./models/perseid_{MODEL_SIZE}m_{Path(DOCUMENT_PATH).stem}/"
#     EXPERIMENT_NAME = f"perseid_{MODEL_SIZE}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#     # Hardware settings
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     USE_BFLOAT16 = torch.cuda.is_available()  # Use bfloat16 if on GPU

#     # Run the training pipeline
#     model, history = main()

if __name__ == "__main__":
    # Add this line before the main training pipeline
    test_integration()

    # inspect
    print("Arguments passed to the script:")
    for i, arg in enumerate(sys.argv):
        print(f"\tArgument {i}: {arg}")

    # ============================================================================
    # USER CONFIGURATION SECTION - MODIFY THESE SETTINGS
    # ============================================================================

    # preset/reset
    file_path = None

    # Helper function to extract book ID from Gutenberg URL
    def extract_book_id(url):
        """Extract book ID from Gutenberg URL"""
        # Remove base URL and file extension
        book_id = url.replace("https://www.gutenberg.org/ebooks/", "")
        book_id = book_id.replace("https://www.gutenberg.org/files/", "")
        book_id = book_id.replace("https://www.gutenberg.org/cache/epub/", "")
        book_id = book_id.split("/")[0]  # Handle paths like "11/11-0.txt"
        book_id = book_id.replace(".txt.utf-8", "").replace(".txt", "")
        book_id = book_id.replace("pg", "")  # Remove 'pg' prefix if present
        return book_id

    # Helper function to download a book if it doesn't exist
    def download_book(url, save_path):
        """Download a book from URL if it doesn't already exist"""
        if os.path.exists(save_path):
            print(f"Book already exists: {save_path}")
            with open(save_path, "r", encoding="utf-8") as file:
                return file.read()
        else:
            print(f"Downloading: {url}")
            try:
                with urllib.request.urlopen(url) as response:
                    text_data = response.read().decode("utf-8")
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(text_data)
                print(f"Saved to: {save_path}")
                return text_data
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                return None

    # get path if supplied
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")

        # Download sample text data
        demo_file_path = "data/alice.txt"
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(demo_file_path):
            url = "https://www.gutenberg.org/files/11/11-0.txt"
            print(f"Downloading training data from {url}")
            text_data = download_book(url, demo_file_path)
        else:
            print(f"Loading existing data from {demo_file_path}")
            with open(demo_file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        # Q&A
        user_path_or_demo_choice = input(
            """\nEnter a file path to a .txt file or for a demo say 'demo'
Or say "url" to enter a gutenberg.org url, e.g.

shakespeare first folio...
https://www.gutenberg.org/cache/epub/2270/pg2270.txt

chaucer in six volumes...
https://www.gutenberg.org/cache/epub/43089/pg43089.txt
https://www.gutenberg.org/cache/epub/44833/pg44833.txt
https://www.gutenberg.org/cache/epub/45027/pg45027.txt
https://www.gutenberg.org/cache/epub/22120/pg22120.txt
https://www.gutenberg.org/cache/epub/43016/pg43016.txt
https://www.gutenberg.org/cache/epub/43097/pg43097.txt

Poetry:
https://www.gutenberg.org/cache/epub/30235/pg30235.txt

# homer & odyssey
https://www.gutenberg.org/ebooks/3160.txt.utf-8
https://www.gutenberg.org/ebooks/1727.txt.utf-8
https://www.gutenberg.org/ebooks/348.txt.utf-8
https://www.gutenberg.org/ebooks/65000.txt.utf-8
https://www.gutenberg.org/ebooks/53004.txt.utf-8
https://www.gutenberg.org/ebooks/49324.txt.utf-8
https://www.gutenberg.org/ebooks/48895.txt.utf-8
https://www.gutenberg.org/ebooks/16338.txt.utf-8
https://www.gutenberg.org/ebooks/47356.txt.utf-8
https://www.gutenberg.org/ebooks/26275.txt.utf-8
https://www.gutenberg.org/ebooks/45896.txt.utf-8
https://www.gutenberg.org/ebooks/12651.txt.utf-8
https://www.gutenberg.org/ebooks/65461.txt.utf-8
https://www.gutenberg.org/ebooks/49858.txt.utf-8
https://www.gutenberg.org/ebooks/65381.txt.utf-8
https://www.gutenberg.org/ebooks/13725.txt.utf-8
https://www.gutenberg.org/ebooks/53646.txt.utf-8
https://www.gutenberg.org/ebooks/24856.txt.utf-8

Enter a file path to a .txt file or for a demo say 'demo'
Or say "url" to enter a gutenberg.org url:\n
"""
        )

        # use demo if demo is selected
        if user_path_or_demo_choice.lower().strip() == "demo":
            file_path = demo_file_path

        elif user_path_or_demo_choice.lower().strip() == "url":
            url = input("url is... -> \n")
            book_id = extract_book_id(url)
            book_file_path = f"data/{book_id}.txt"
            print(f"Downloading training data from {url}")
            text_data = download_book(url, book_file_path)
            if text_data:
                file_path = book_file_path
            else:
                print("Download failed, using demo file")
                file_path = demo_file_path

    # use argument input path if supplied by user
    elif len(sys.argv) == 2:
        file_path = sys.argv[1]
        print(f"path argument found... {file_path}")

    else:
        print("Edge case, defaulting to demo.")
        file_path = "data/alice.txt"

    asklongtrain = input("if longtrain say: longtrain -> ")

    if asklongtrain.lower().strip() == "longtrain":
        train_list = [
            "https://www.gutenberg.org/ebooks/3160.txt.utf-8",
            "https://www.gutenberg.org/ebooks/1727.txt.utf-8",
            "https://www.gutenberg.org/ebooks/348.txt.utf-8",
            "https://www.gutenberg.org/ebooks/65000.txt.utf-8",
            "https://www.gutenberg.org/ebooks/53004.txt.utf-8",
            "https://www.gutenberg.org/ebooks/49324.txt.utf-8",
            "https://www.gutenberg.org/ebooks/48895.txt.utf-8",
            "https://www.gutenberg.org/ebooks/16338.txt.utf-8",
            "https://www.gutenberg.org/ebooks/47356.txt.utf-8",
            "https://www.gutenberg.org/ebooks/26275.txt.utf-8",
            "https://www.gutenberg.org/ebooks/45896.txt.utf-8",
            "https://www.gutenberg.org/ebooks/12651.txt.utf-8",
            "https://www.gutenberg.org/ebooks/65461.txt.utf-8",
            "https://www.gutenberg.org/ebooks/49858.txt.utf-8",
            "https://www.gutenberg.org/ebooks/65381.txt.utf-8",
            "https://www.gutenberg.org/ebooks/13725.txt.utf-8",
            "https://www.gutenberg.org/ebooks/53646.txt.utf-8",
            "https://www.gutenberg.org/ebooks/24856.txt.utf-8",
        ]

        os.makedirs("data", exist_ok=True)

        for i, this_url in enumerate(train_list):
            print(f"\n{'=' * 60}")
            print(f"Processing book {i + 1}/{len(train_list)}")
            print(f"{'=' * 60}")

            # Extract book ID and create unique filename
            book_id = extract_book_id(this_url)
            book_file_path = f"data/{book_id}.txt"

            # Download book if needed
            text_data = download_book(this_url, book_file_path)

            if text_data:
                # Set file path for this iteration
                file_path = book_file_path

                # Document input
                DOCUMENT_PATH = file_path

                # Output configuration - using book ID in the path
                OUTPUT_DIR = f"./models/perseid_{MODEL_SIZE}m_multibook/"
                EXPERIMENT_NAME = f"perseid_{MODEL_SIZE}m_book{book_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # # Hardware settings
                # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                # USE_BFLOAT16 = torch.cuda.is_available()  # Use bfloat16 if on GPU

                print(f"Training on: {book_file_path}")

                # Run the training pipeline for this book
                model, history = main()
            else:
                print(f"Skipping book {book_id} due to download failure")

    else:
        # Single book training (demo, url, or file path)

        # Document input
        DOCUMENT_PATH = file_path

        # Output configuration
        OUTPUT_DIR = f"./models/perseid_{MODEL_SIZE}m_{Path(DOCUMENT_PATH).stem}/"
        EXPERIMENT_NAME = (
            f"perseid_{MODEL_SIZE}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Hardware settings
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        USE_BFLOAT16 = torch.cuda.is_available()  # Use bfloat16 if on GPU

        # Run the training pipeline
        model, history = main()
