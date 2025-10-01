"""
train_on_corpus_perseid_byte.py

note: GPU status
```bash
nvidia-smi
```
&
```bash
watch -n 1 nvidia-smi
```

This tool, along with aug_nlp.py (NLP Augmentation of text file),
and make_one_corpus_file.py,
is for pre-training a model on one (big-ish) corpus document
(that is made of all individual training documents).

This CAN be, but is NOT recommended to be,
used for continued training or fine-tuning,
as that can disrupt the other weights.

For fine tuning, use:


Note:
There can be long periods when training does not find an improvement,
but, as I understand it, that is ok because only the improvements are saved.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
import urllib.request
import torch
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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


# Training settings
TRAINING_CONFIG = {
    "context_length": 1024,  # Context window for training
    "batch_size": 10,  # ~parallel processsed Batches (increase as memory allows)
    "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
    "learning_rate": 5e-4,  # Learning rate
    "num_epochs": 1,  # Number of training epochs, default 3
    "weight_decay": 0.01,  # Weight decay for AdamW
    "warmup_steps": 100,  # Warmup steps for learning rate
    "eval_every": 200,  # Evaluate every N steps NOTE purely for human-readable console output during training
    "eval_batches": 10,  # Number of batches for evaluation
    "save_every": 500,  # Save checkpoint every N steps
    "chunk_overlap": 0.1,  # Overlap between text chunks (0.0 to 0.5)
}


# Model configuration
MODEL_SIZE = 288  # Options: 256, 288, 320 (millions of parameters)
MODEL_STRATEGY = "balanced"  # Options: "balanced", "deep", "wide"

# Training continuation settings
TRAINING_MODE = "continue"  # "continue"  # Options: "new", "continue", "force_restart"
# - "new": Start new model if no checkpoint exists, error if checkpoint exists
# - "continue": Resume from checkpoint if exists, start new if doesn't exist
# - "force_restart": Always start fresh (WARNING: overwrites existing model!)

CHECKPOINT_PATH = None  # Set to specific checkpoint file, or None to auto-find
# If None, looks for: {OUTPUT_DIR}/checkpoint_best.pth

# Data split
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation (modify as needed)

"""
Aim of Checkpoint Management and Training Continuation System
Current Implementation Status: COMPLETE
This training script (train_on_corpus_perseid_byte.py) includes two major systems that solve critical training infrastructure problems:
System 1: Conservative Checkpoint Retention Strategy
What it does:

Automatically deletes old checkpoint files to prevent disk space exhaustion
Maintains only essential checkpoints using a sliding window approach

How it works:

Keeps checkpoint_best.pth (lowest validation loss) - NEVER deleted
Keeps checkpoint_latest.pth (most recent for crash recovery) - ALWAYS updated
Keeps last 3 checkpoint_step_*.pth files (older ones automatically deleted)
Keeps last 5 checkpoint_epoch_*.pth files (older ones automatically deleted)
Checks available disk space before saving (warns if low, aborts if critical)
Uses atomic write pattern (saves to temp file, then renames) to prevent corruption

Key functions:

save_checkpoint() - Enhanced checkpoint saving with automatic cleanup
cleanup_old_step_checkpoints() - Removes old step checkpoints beyond retention limit
cleanup_old_epoch_checkpoints() - Removes old epoch checkpoints beyond retention limit

System 2: Window-Level Training Continuation
What it does:

Tracks exact position in training data (window-level granularity)
On resume after interruption, skips already-processed windows
Prevents the model from re-training on the same data repeatedly

How it works:

Every checkpoint stores windows_processed counter (total windows seen across all epochs)
On resume, training loop skips the first N windows where N = windows_processed
Ensures training continues from approximately where it left off
Prevents bias toward early corpus sections

Key tracking variables:

windows_processed - Total number of training windows processed
windows_to_skip - Number of windows to skip when resuming
current_window_index - Current position in the data stream

Problem These Systems Solve
Without these systems:

Disk fills up - Every 500 steps creates a ~1-2GB checkpoint file, eventually exhausting all disk space
Training fails - Disk full errors crash training with RuntimeError: File cannot be opened
Data repetition - After crash/resume, model re-trains on beginning of corpus, creating bias
Wasted compute - Redundant gradient updates on already-seen data

With these systems:

Controlled disk usage - Only ~8-10 checkpoint files maintained at any time
Robust training - Can handle interruptions and resume properly
Even data coverage - Each part of corpus seen approximately equally
Efficient compute - No wasted gradient updates on repeated data

Files Created and Maintained
Copyoutput_dir/
├── checkpoint_best.pth           # Best model (lowest val loss) - PERMANENT
├── checkpoint_latest.pth         # Most recent - ALWAYS CURRENT
├── checkpoint_step_156500.pth    # Recent step checkpoints - ROTATING (keep 3)
├── checkpoint_step_157000.pth
├── checkpoint_step_157500.pth
├── checkpoint_epoch_28.pth       # Recent epoch checkpoints - ROTATING (keep 5)
├── checkpoint_epoch_29.pth
├── checkpoint_epoch_30.pth
└── perseid_model_final.pth       # Created at training completion - PERMANENT
How to Resume Training After Interruption
Simply re-run the script with the same configuration:
bashCopypython train_on_corpus_perseid_byte.py
The script will:

Find checkpoint_best.pth or checkpoint_latest.pth
Load model weights, optimizer state, learning rate schedule
Read windows_processed to know where to continue
Skip already-seen windows
Continue training from the correct position

Configuration Parameters

max_step_checkpoints=3 - Number of step checkpoints to keep
max_epoch_checkpoints=5 - Number of epoch checkpoints to keep
TRAINING_MODE="continue" - Automatically resume if checkpoint exists
save_every=500 - Save checkpoint every N steps

Monitoring Training Progress
The console output shows:

When old checkpoints are deleted: 🗑️ Removed old checkpoint: checkpoint_step_155500.pth (1234.5 MB)
Disk space warnings: ⚠️ WARNING: Low disk space! Only 5.2 GB free
Window skipping on resume: Skipping window 5,000 of 12,345
Successful saves: ✓ Checkpoint saved: checkpoint_best.pth (1234.5 MB)
"""

"""

Training module for Perseid models on text document corpus.
Handles single document input with configurable train/val split.
Trains from scratch (no pretrained weights required, or can do continued-train),
unless existing weights are found, then it will continue training.

Usage:
    1. Set configuration parameters at top of file
    2. Run: python train_perseid_byte.py

The innocence of Father Brown by G. K. Chesterton
https://www.gutenberg.org/ebooks/204.txt.utf-8

Pride & Prejudice
https://www.gutenberg.org/ebooks/1342.txt.utf-8

shakespeare first folio...
https://www.gutenberg.org/ebooks/2270.txt.utf-8

chaucer in six volumes...
https://www.gutenberg.org/ebooks/43089.txt.utf-8
https://www.gutenberg.org/ebooks/44833.txt.utf-8
https://www.gutenberg.org/ebooks/45027.txt.utf-8
https://www.gutenberg.org/ebooks/22120.txt.utf-8
https://www.gutenberg.org/ebooks/43016.txt.utf-8
https://www.gutenberg.org/ebooks/43097.txt.utf-8

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

"""


"""
reference

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
#     # parameters for better control
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


# Training settings
TRAINING_CONFIG = {
    "context_length": 1024,  # Context window for training
    "batch_size": 5,  # Batch size (increase if memory allows)
    "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
    "learning_rate": 4e-4,  # 5e-4,  # Learning rate
    "num_epochs": 15,  # Number of training epochs, default 3
    "weight_decay": 0.01,  # Weight decay for AdamW
    "warmup_steps": 100,  # Warmup steps for learning rate
    "eval_every": 50,  # Evaluate every N steps
    "eval_batches": 10,  # Number of batches for evaluation
    "save_every": 500,  # Save checkpoint every N steps
    "chunk_overlap": 0.1,  # Overlap between text chunks (0.0 to 0.5)
}

"""

# # Training settings
# TRAINING_CONFIG = {
#     "context_length": 1024,  # Context window for training
#     "batch_size": 2,  # Batch size (increase if memory allows)
#     "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
#     "learning_rate": 5e-4,  # Learning rate
#     "num_epochs": 3,  # Number of training epochs, default 3
#     "weight_decay": 0.01,  # Weight decay for AdamW
#     "warmup_steps": 100,  # Warmup steps for learning rate
#     "eval_every": 25,  # Evaluate every N steps NOTE purely for human-readable console output during training
#     "eval_batches": 10,  # Number of batches for evaluation
#     "save_every": 500,  # Save checkpoint every N steps
#     "chunk_overlap": 0.1,  # Overlap between text chunks (0.0 to 0.5)
# }

# ============================================================================
# END USER CONFIGURATION
# ============================================================================


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


class DocumentDataset(Dataset):
    """
    Dataset for document-based training with efficient byte-level tokenization.

    Handles chunking and tokenization of documents efficiently by:
    - Using direct byte encoding when possible (avoiding string conversions)
    - Supporting both file paths and text strings
    - Providing clear progress feedback
    """

    def __init__(
        self,
        data_source,
        tokenizer,
        max_length,
        stride,
        verbose=True,
        is_file_path=False,
    ):
        """
        Initialize document dataset with efficient tokenization.

        Args:
            data_source (str or Path): Either:
                - File path to document (when is_file_path=True)
                - Raw text string (when is_file_path=False)
            tokenizer: ByteTokenizer object with encode_bytes/encode_file methods
            max_length (int): Maximum sequence length for training windows
            stride (int): Stride between chunks (for overlap)
            verbose (bool): Print detailed statistics and progress
            is_file_path (bool): If True, data_source is treated as file path.
                                If False, data_source is treated as text string.

        Note:
            When is_file_path=True, uses efficient direct byte encoding
            avoiding unnecessary string conversions.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        try:
            # ================================================================
            # TOKENIZATION PHASE - Use most efficient method
            # ================================================================

            if verbose:
                print("\n" + "=" * 60)
                print("TOKENIZATION STARTING")
                print("=" * 60)

            if is_file_path:
                # EFFICIENT PATH: Direct file-to-tokens conversion
                file_path = Path(data_source)

                if not file_path.exists():
                    raise FileNotFoundError(f"Document file not found: {file_path}")

                file_size_bytes = file_path.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)

                if verbose:
                    print(f"Tokenization method: DIRECT FILE ENCODING (most efficient)")
                    print(f"File: {file_path.name}")
                    print(
                        f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)"
                    )
                    print(f"Starting tokenization...")

                import time

                start_time = time.perf_counter()

                # Use the efficient file encoding method
                # For very large files, could use chunk_size parameter
                if file_size_mb > 100:  # For files larger than 100MB
                    if verbose:
                        print(f"Large file detected - using chunked encoding...")
                    self.tokens = tokenizer.encode_file(
                        file_path,
                        add_eos=False,
                        chunk_size=1024 * 1024,  # 1MB chunks
                    )
                else:
                    # Small enough to load at once
                    self.tokens = tokenizer.encode_file(file_path, add_eos=False)

                tokenization_time = time.perf_counter() - start_time

                if verbose:
                    tokens_per_second = len(self.tokens) / tokenization_time
                    mb_per_second = file_size_mb / tokenization_time
                    print(f"\n✅ FINISH! TOKENIZATION COMPLETE!")
                    print(f"  Time taken: {tokenization_time:.2f} seconds")
                    print(f"  Speed: {mb_per_second:.2f} MB/s")
                    print(f"  Speed: {tokens_per_second:,.0f} tokens/second")
                    print(f"  Total tokens generated: {len(self.tokens):,}")

            else:
                # FALLBACK PATH: String-based encoding (for text strings)
                if verbose:
                    print(f"Tokenization method: TEXT ENCODING")
                    print(f"Text length: {len(data_source):,} characters")
                    print(f"Starting tokenization...")

                import time

                start_time = time.perf_counter()

                # Use the text encoding method
                self.tokens = tokenizer.encode(data_source, add_eos=False)

                tokenization_time = time.perf_counter() - start_time

                if verbose:
                    tokens_per_second = len(self.tokens) / tokenization_time
                    print(f"\n✅ TOKENIZATION COMPLETE!")
                    print(f"  Time taken: {tokenization_time:.2f} seconds")
                    print(f"  Speed: {tokens_per_second:,.0f} tokens/second")
                    print(f"  Total tokens generated: {len(self.tokens):,}")

            # ================================================================
            #  WINDOW CREATION PHASE
            # ================================================================

            if verbose:
                print("\n" + "-" * 60)
                print("Creating training windows...")

            # Create overlapping windows for training
            self.windows = []
            total_possible_windows = (len(self.tokens) - max_length) // stride

            window_creation_start = time.perf_counter()
            last_progress_report = window_creation_start

            for window_idx, token_start_idx in enumerate(
                range(0, len(self.tokens) - max_length, stride)
            ):
                window = self.tokens[
                    token_start_idx : token_start_idx + max_length + 1
                ]  # +1 for target

                if len(window) == max_length + 1:
                    self.windows.append(window)

                # Progress reporting for large datasets
                if verbose and (
                    time.perf_counter() - last_progress_report > 1.0
                ):  # Report every second
                    progress_pct = (window_idx / total_possible_windows) * 100
                    print(
                        f"  Creating windows: {progress_pct:.1f}% ({len(self.windows):,} windows created)"
                    )
                    last_progress_report = time.perf_counter()

            window_creation_time = time.perf_counter() - window_creation_start

            if verbose:
                print(f"\n✅ WINDOW CREATION COMPLETE!")
                print(f"  Time taken: {window_creation_time:.2f} seconds")
                print(f"  Training windows created: {len(self.windows):,}")
                print(f"  Window size: {max_length} tokens")
                print(f"  Stride: {stride} tokens")
                print(
                    f"  Effective overlap: {((max_length - stride) / max_length * 100):.1f}%"
                )
                print(
                    f"  Coverage: {(len(self.windows) * stride + max_length) / len(self.tokens) * 100:.1f}% of tokens"
                )

                print("\n" + "=" * 60)
                print("DATASET PREPARATION COMPLETE")
                print("=" * 60)

        except Exception as e:
            print(f"\n❌ ERROR during dataset preparation: {e}")
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


class DocumentDataset(Dataset):
    """
    Dataset for document-based training.
    Handles chunking and tokenization of text documents.

    Now supports efficient file-based tokenization to avoid
    unnecessary byte->string->byte conversions.
    """

    def __init__(
        self, data_source, tokenizer, max_length, stride, verbose=True, is_file=False
    ):
        """
        Initialize document dataset.

        Args:
            data_source: Either a string of text or a file path (when is_file=True)
            tokenizer: Tokenizer object with encode method
            max_length (int): Maximum sequence length
            stride (int): Stride between chunks (for overlap)
            verbose (bool): Print statistics
            is_file (bool): If True, data_source is a file path to read directly
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        try:
            if verbose:
                print(f"\n{'=' * 60}")
                print("TOKENIZATION STARTING")
                print(f"{'=' * 60}")

            start_time = time.perf_counter()

            if is_file:
                # Use efficient file encoding - direct bytes to tokens
                file_path = Path(data_source)
                file_size = file_path.stat().st_size

                if verbose:
                    print(f"Source: File - {file_path.name}")
                    print(
                        f"Size: {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)"
                    )
                    print(f"Method: Direct byte encoding (no string conversion)")
                    print(f"Tokenizing...")

                # Use efficient file encoding
                # For large files (>100MB), use chunked reading
                if file_size > 100 * 1024 * 1024:
                    self.tokens = tokenizer.encode_file(
                        file_path,
                        chunk_size=1024 * 1024,  # 1MB chunks
                        verbose=verbose,
                    )
                else:
                    self.tokens = tokenizer.encode_file(file_path, verbose=verbose)

            else:
                # Regular text encoding
                if verbose:
                    print(f"Source: Text string")
                    print(f"Length: {len(data_source):,} characters")
                    print(f"Method: UTF-8 text encoding")
                    print(f"Tokenizing...")

                self.tokens = tokenizer.encode(data_source)

            elapsed = time.perf_counter() - start_time

            if verbose:
                tokens_per_sec = len(self.tokens) / elapsed if elapsed > 0 else 0
                print(f"\n✅ Tokenization complete!")
                print(f"  Time: {elapsed:.2f} seconds")
                print(f"  Speed: {tokens_per_sec:,.0f} tokens/second")
                print(f"  Total tokens: {len(self.tokens):,}")

            # Create overlapping windows
            if verbose:
                print(f"\n{'-' * 60}")
                print(f"Creating training windows...")

            window_start = time.perf_counter()
            self.windows = []

            for i in range(0, len(self.tokens) - max_length, stride):
                window = self.tokens[i : i + max_length + 1]  # +1 for target
                if len(window) == max_length + 1:
                    self.windows.append(window)

            window_time = time.perf_counter() - window_start

            if verbose:
                print(f"✅ Window creation complete!")
                print(f"  Time: {window_time:.2f} seconds")
                print(f"  Training windows: {len(self.windows):,}")
                print(f"  Window size: {max_length}")
                print(f"  Stride: {stride}")
                print(f"  Overlap: {((max_length - stride) / max_length * 100):.1f}%")
                print(f"\n{'=' * 60}")
                print(f"DATASET READY")
                print(f"{'=' * 60}")

        except Exception as e:
            print(f"Error during dataset creation: {e}")
            traceback.print_exc()
            raise

    def __len__(self):
        """Return the number of training windows."""
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            tuple: (input_ids, target_ids) where target is input shifted by 1
        """
        window = self.windows[idx]
        # Input is all tokens except the last
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        # Target is all tokens except the first (shifted by 1)
        target_ids = torch.tensor(window[1:], dtype=torch.long)
        return input_ids, target_ids


def create_data_loaders(
    text_or_path, tokenizer, config, train_ratio=0.9, is_file_path=False
):
    """
    Create training and validation data loaders with efficient tokenization.

    Args:
        text_or_path (str or Path): Either a file path or text string
        tokenizer: ByteTokenizer object
        config (dict): Training configuration with keys:
            - context_length: sequence length for model
            - chunk_overlap: overlap between chunks (0.0 to 1.0)
            - batch_size: batch size for data loaders
        train_ratio (float): Proportion of data for training (default 0.9)
        is_file_path (bool): If True, text_or_path is a file path.
                           If False, text_or_path is a text string.

    Returns:
        tuple: (train_loader, val_loader) PyTorch DataLoader objects
    """
    try:
        print(f"\n" + "=" * 70)
        print("DATA LOADER CREATION")
        print(f"=" * 70)
        print(
            f"Split ratio: {train_ratio:.0%} train / {(1 - train_ratio):.0%} validation"
        )

        # Calculate stride from overlap
        stride = int(config["context_length"] * (1 - config["chunk_overlap"]))

        if is_file_path:
            # ============================================================
            # FILE PATH: Use efficient direct byte encoding
            # ============================================================
            print("\nTokenizing file for train/validation split...")

            # Get all tokens first using efficient file encoding
            file_path = Path(text_or_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Use efficient file encoding method
            file_size = file_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # Files > 100MB
                print(
                    f"Large file detected ({file_size / (1024 * 1024):.1f} MB), using chunked encoding..."
                )
                all_tokens = tokenizer.encode_file(
                    file_path,
                    add_eos=False,
                    chunk_size=1024 * 1024,  # 1MB chunks
                    verbose=True,
                )
            else:
                all_tokens = tokenizer.encode_file(
                    file_path, add_eos=False, verbose=True
                )

            # Split tokens for train/validation
            split_idx = int(train_ratio * len(all_tokens))
            train_tokens = all_tokens[:split_idx]
            val_tokens = all_tokens[split_idx:]

            print(
                f"\nToken split: {len(train_tokens):,} train / {len(val_tokens):,} validation"
            )

            # Create datasets from pre-tokenized data
            print("\n" + "-" * 50)
            print("Creating TRAINING dataset from tokens...")
            train_dataset = DocumentDatasetFromTokens(
                train_tokens, config["context_length"], stride, verbose=True
            )

            print("\n" + "-" * 50)
            print("Creating VALIDATION dataset from tokens...")
            val_dataset = DocumentDatasetFromTokens(
                val_tokens,
                config["context_length"],
                config["context_length"],  # No overlap for validation
                verbose=True,
            )

        else:
            # ============================================================
            # TEXT STRING: Split text then tokenize each part
            # ============================================================
            split_idx = int(train_ratio * len(text_or_path))
            train_text = text_or_path[:split_idx]
            val_text = text_or_path[split_idx:]

            print(f"\nText split: {len(train_text):,} / {len(val_text):,} characters")

            # Create datasets using DocumentDataset with is_file=False
            print("\n" + "-" * 50)
            print("Creating TRAINING dataset...")
            train_dataset = DocumentDataset(
                train_text,
                tokenizer,
                config["context_length"],
                stride,
                verbose=True,
                is_file=False,  # Changed from is_file_path to is_file
            )

            print("\n" + "-" * 50)
            print("Creating VALIDATION dataset...")
            val_dataset = DocumentDataset(
                val_text,
                tokenizer,
                config["context_length"],
                config["context_length"],  # No overlap for validation
                verbose=True,
                is_file=False,  # Changed from is_file_path to is_file
            )

        # Create PyTorch data loaders
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

        print(f"\n" + "=" * 70)
        print("✅ DATA PREPARATION COMPLETE")
        print(f"=" * 70)
        print(f"Training batches: {len(train_loader):,}")
        print(f"Validation batches: {len(val_loader):,}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Total training samples: {len(train_dataset):,}")
        print(f"Total validation samples: {len(val_dataset):,}")

        return train_loader, val_loader

    except Exception as e:
        print(f"\n❌ Error creating data loaders: {e}")
        traceback.print_exc()
        raise


class DocumentDatasetFromTokens(Dataset):
    """
    Lightweight dataset that works directly with pre-tokenized data.

    This avoids redundant tokenization when tokens are already available
    (e.g., from file encoding or when splitting pre-tokenized data).
    """

    def __init__(self, tokens, max_length, stride, verbose=True):
        """
        Initialize dataset from pre-tokenized data.

        Args:
            tokens (list[int]): Pre-tokenized data (list of token IDs 0-258)
            max_length (int): Maximum sequence length for model input
            stride (int): Stride between chunks (controls overlap)
            verbose (bool): Print statistics about dataset creation
        """
        super().__init__()
        self.tokens = tokens
        self.max_length = max_length
        self.stride = stride

        # Create overlapping windows for training
        self.windows = []
        for i in range(0, len(self.tokens) - max_length, stride):
            window = self.tokens[i : i + max_length + 1]  # +1 for target
            if len(window) == max_length + 1:
                self.windows.append(window)

        if verbose:
            overlap_percent = (
                ((max_length - stride) / max_length * 100) if max_length > 0 else 0
            )
            print(
                f"  Created {len(self.windows):,} windows from {len(tokens):,} tokens"
            )
            print(f"  Window size: {max_length}, Stride: {stride}")
            print(f"  Overlap: {overlap_percent:.1f}%")

            # Calculate coverage
            if len(tokens) > 0:
                coverage = min(
                    100, (len(self.windows) * stride + max_length) / len(tokens) * 100
                )
                print(f"  Coverage: {coverage:.1f}% of input tokens")

    def __len__(self):
        """Return the number of training windows."""
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Args:
            idx (int): Index of the training window

        Returns:
            tuple: (input_ids, target_ids) as PyTorch tensors
                  where target_ids is input_ids shifted by 1 position
        """
        window = self.windows[idx]
        # Input is all tokens except the last
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        # Target is all tokens except the first (shifted by 1)
        target_ids = torch.tensor(window[1:], dtype=torch.long)
        return input_ids, target_ids


# def create_data_loaders(text, tokenizer, config, train_ratio=0.9):
#     """
#     Create training and validation data loaders.

#     Args:
#         text (str): Full document text
#         tokenizer: Tokenizer object
#         config (dict): Training configuration
#         train_ratio (float): Proportion of data for training

#     Returns:
#         tuple: (train_loader, val_loader)
#     """
#     try:
#         print(
#             f"\nCreating data loaders with {train_ratio:.0%} train / {(1 - train_ratio):.0%} validation split"
#         )

#         # Calculate split point
#         split_idx = int(train_ratio * len(text))
#         train_text = text[:split_idx]
#         val_text = text[split_idx:]

#         print(f"Train text: {len(train_text):,} chars")
#         print(f"Val text: {len(val_text):,} chars")

#         # Calculate stride from overlap
#         stride = int(config["context_length"] * (1 - config["chunk_overlap"]))

#         # Create datasets
#         train_dataset = DocumentDataset(
#             train_text, tokenizer, config["context_length"], stride, verbose=True
#         )

#         val_dataset = DocumentDataset(
#             val_text,
#             tokenizer,
#             config["context_length"],
#             config["context_length"],  # No overlap for validation
#             verbose=True,
#         )

#         # Create data loaders
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=config["batch_size"],
#             shuffle=True,
#             drop_last=True,
#             num_workers=0,
#         )

#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=config["batch_size"],
#             shuffle=False,
#             drop_last=False,
#             num_workers=0,
#         )

#         print(f"\nTrain batches: {len(train_loader):,}")
#         print(f"Val batches: {len(val_loader):,}")

#         return train_loader, val_loader

#     except Exception as e:
#         print(f"Error creating data loaders: {e}")
#         traceback.print_exc()
#         raise


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
                "epoch": checkpoint.get("epoch", 0),
                "tokens_seen": checkpoint.get("tokens_seen", 0),
                "windows_processed": checkpoint.get(
                    "windows_processed", 0
                ),  # Window tracking for proper continuation
            }

            print(f"\nResuming from:")
            print(f"  - Step: {training_state['global_step']:,}")
            print(f"  - Epoch: {training_state['epoch']}")
            print(f"  - Best validation loss: {training_state['best_val_loss']:.4f}")
            print(f"  - Tokens seen: {training_state['tokens_seen']:,}")
            print(f"  - Windows processed: {training_state['windows_processed']:,}")

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


def train_model(
    model, train_loader, val_loader, config, device, output_dir, training_state
):
    """
    Main training loop for Perseid model with window-level continuation support.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to use
        output_dir: Directory to save outputs
        training_state: Dict with resume information (global_step, optimizer_state, etc.)

    Returns:
        dict: Training history

    Specs:
    - Window-level progress tracking for proper continuation
    - Intelligent checkpoint management with automatic cleanup
    - Epoch boundary checkpointing
    """
    try:
        print(f"\n{'=' * 60}")
        print("Starting Training")
        print(f"{'=' * 60}")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),
        )

        # Restore optimizer state if resuming
        if training_state["optimizer_state"] is not None:
            optimizer.load_state_dict(training_state["optimizer_state"])
            print("  ✓ Restored optimizer state")

        # Calculate total steps
        steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
        total_steps = steps_per_epoch * config["num_epochs"]

        # Setup scheduler with warmup
        # def get_lr(step):
        #     """Learning rate schedule with warmup."""
        #     if step < config["warmup_steps"]:
        #         return step / config["warmup_steps"]
        #     else:
        #         progress = (step - config["warmup_steps"]) / (
        #             total_steps - config["warmup_steps"]
        #         )
        #         return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        # def get_lr(step):
        #     """Learning rate schedule with warmup."""
        #     if step < config["warmup_steps"]:
        #         # Warmup phase: linear increase
        #         return step / config["warmup_steps"]
        #     else:
        #         # Cosine decay phase
        #         # SAFETY: Prevent division by zero if warmup_steps >= total_steps
        #         decay_steps = max(1, total_steps - config["warmup_steps"])
        #         progress = (step - config["warmup_steps"]) / decay_steps
        #         return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        def get_lr(step):
            """
            Learning rate schedule with warmup and cosine decay.

            Handles edge case where warmup_steps >= total_steps.
            """
            if step < config["warmup_steps"]:
                # Linear warmup
                if config["warmup_steps"] == 0:
                    return 1.0
                return step / config["warmup_steps"]
            else:
                # Cosine decay
                # Ensure we don't divide by zero
                decay_steps = max(1, total_steps - config["warmup_steps"])
                progress = (step - config["warmup_steps"]) / decay_steps

                # Clamp progress to [0, 1] range
                progress = min(1.0, max(0.0, progress))

                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        # Restore scheduler state if resuming
        if training_state["scheduler_state"] is not None:
            scheduler.load_state_dict(training_state["scheduler_state"])
            print("  ✓ Restored scheduler state")

        # Training state
        history = {"train_loss": [], "val_loss": [], "learning_rates": [], "step": []}

        # global_step = 0
        # best_val_loss = float('inf')
        # tokens_seen = 0
        global_step = training_state["global_step"]
        best_val_loss = training_state["best_val_loss"]
        tokens_seen = training_state["tokens_seen"]
        start_epoch = training_state["epoch"]

        # Track windows for continuation
        windows_processed = training_state.get("windows_processed", 0)
        windows_to_skip = windows_processed  # How many to skip on resume

        print(f"\nStarting from epoch {start_epoch + 1}, step {global_step}")
        if windows_to_skip > 0:
            print(
                f"  ✓ Will skip first {windows_to_skip:,} windows (continuation mode)"
            )

        print(f"Total training steps: {total_steps:,}")
        print(f"Warmup steps: {config['warmup_steps']:,}")
        print(
            f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}"
        )

        # Training loop
        # for epoch in range(config["num_epochs"]):
        for epoch in range(start_epoch, config["num_epochs"]):
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            print(f"{'=' * 40}")

            model.train()
            epoch_loss = 0
            epoch_tokens = 0
            epoch_windows = 0  # Track windows in this epoch

            # Handle continuation by skipping windows
            windows_skipped_this_epoch = 0

            for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                # Skip windows if resuming mid-training
                current_window_index = epoch * len(train_loader) + batch_idx
                if current_window_index < windows_to_skip:
                    windows_skipped_this_epoch += 1
                    if windows_skipped_this_epoch % 1000 == 0:
                        print(
                            f"  Skipping window {windows_skipped_this_epoch:,} of {windows_to_skip:,}"
                        )
                    continue

                # Calculate loss
                loss = calculate_loss(input_batch, target_batch, model, device)
                loss = loss / config["gradient_accumulation_steps"]
                loss.backward()

                epoch_loss += loss.item() * config["gradient_accumulation_steps"]
                epoch_tokens += input_batch.numel()
                epoch_windows += 1
                windows_processed += 1  # Track total windows seen

                # Update weights after gradient accumulation
                if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    tokens_seen += epoch_tokens
                    epoch_tokens = 0

                    # Periodic evaluation
                    if global_step % config["eval_every"] == 0:
                        val_loss = evaluate_model(
                            model,
                            val_loader,
                            device,
                            num_batches=config["eval_batches"],
                        )

                        train_loss = epoch_loss / (batch_idx + 1)
                        current_lr = scheduler.get_last_lr()[0]

                        history["train_loss"].append(train_loss)
                        history["val_loss"].append(val_loss)
                        history["learning_rates"].append(current_lr)
                        history["step"].append(global_step)

                        print(
                            f"Step {global_step:5d} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens: {tokens_seen:,}"
                        )

                        # Save best model with window tracking
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
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
                                windows_processed=windows_processed,
                            )
                            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

                    # Periodic checkpoint with window tracking
                    if global_step % config["save_every"] == 0:
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
                            windows_processed=windows_processed,
                        )

                        # Also save as "latest" for easy resume
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            val_loss,
                            output_dir,
                            "latest",
                            epoch=epoch,
                            tokens_seen=tokens_seen,
                            windows_processed=windows_processed,
                        )

            # # End of epoch evaluation
            # avg_epoch_loss = epoch_loss / len(train_loader)
            # val_loss = evaluate_model(model, val_loader, device)

            # print(f"\nEpoch {epoch + 1} Summary:")
            # print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
            # print(f"  Validation Loss: {val_loss:.4f}")
            # print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

            # End of epoch - save epoch checkpoint
            if epoch_windows > 0:  # Only if we actually processed windows
                avg_epoch_loss = epoch_loss / epoch_windows
                val_loss = evaluate_model(model, val_loader, device)

                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Windows Processed This Epoch: {epoch_windows:,}")
                print(f"  Total Windows Processed: {windows_processed:,}")

                # Save epoch checkpoint
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    val_loss,
                    output_dir,
                    f"epoch_{epoch + 1}",
                    epoch=epoch + 1,
                    tokens_seen=tokens_seen,
                    windows_processed=windows_processed,
                )

        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Total tokens seen: {tokens_seen:,}")

        return history

    except Exception as e:
        print(f"Error during training: {e}")
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
    windows_processed=None,
    max_step_checkpoints=3,
    max_epoch_checkpoints=5,
):
    """
    Save model checkpoint with intelligent file management and training state tracking.

    CRITICAL DESIGN DECISIONS:
    1. Timestamps prevent filename collisions between training runs
    2. Cleanup happens AFTER save to prevent race conditions
    3. File existence checks before operations prevent FileNotFoundError

    Checkpoint naming strategy:
    - checkpoint_best.pth (no timestamp - always overwrites)
    - checkpoint_latest.pth (no timestamp - always overwrites)
    - checkpoint_step_1000_20240115_143022.pth (timestamped - unique)
    - checkpoint_epoch_5_20240115_143022.pth (timestamped - unique)

    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state to preserve
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        step (int): Current global training step
        val_loss (float): Current validation loss
        output_dir (str or Path): Directory for saving checkpoints
        tag (str): Checkpoint identifier (e.g., "best", "step_1000", "epoch_5")
        epoch (int, optional): Current epoch number for tracking
        tokens_seen (int, optional): Total tokens processed so far
        windows_processed (int, optional): Total training windows seen (for continuation)
        max_step_checkpoints (int): Maximum number of step checkpoints to retain
        max_epoch_checkpoints (int): Maximum number of epoch checkpoints to retain

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ================================================================
        # SECTION 1: Check disk space
        # ================================================================
        import shutil

        disk_stats = shutil.disk_usage(output_dir)
        free_space_gb = disk_stats.free / (1024**3)

        # Estimate checkpoint size
        model_size_bytes = sum(p.numel() for p in model.parameters()) * 4
        checkpoint_size_gb = (
            model_size_bytes / (1024**3) * 1.5
        )  # 50% buffer for optimizer

        # Check if we have enough space
        if free_space_gb < checkpoint_size_gb * 1.2:
            print(f"❌ ERROR: Insufficient disk space to save checkpoint")
            print(
                f"   Need at least {checkpoint_size_gb * 1.2:.2f} GB, have {free_space_gb:.2f} GB"
            )
            return False

        if free_space_gb < checkpoint_size_gb * 2:
            print(f"⚠️  WARNING: Low disk space! Only {free_space_gb:.2f} GB free")

        # ================================================================
        # SECTION 2: Prepare checkpoint data
        # ================================================================
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.cfg,
            "step": step,
            "val_loss": val_loss,
            "epoch": epoch if epoch is not None else 0,
            "tokens_seen": tokens_seen if tokens_seen is not None else 0,
            "windows_processed": windows_processed
            if windows_processed is not None
            else 0,
            "checkpoint_version": "v3_timestamped",
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
        }

        # ================================================================
        # SECTION 3: Generate filename with timestamp for uniqueness
        # ================================================================

        # Generate timestamp for unique identification
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :21
        ]  # Include microseconds

        # Determine checkpoint filename based on tag type
        if tag == "best":
            # Best model always overwrites (no timestamp needed)
            checkpoint_filename = "checkpoint_best.pth"
            needs_cleanup = False

        elif tag == "latest":
            # Latest always overwrites (no timestamp needed)
            checkpoint_filename = "checkpoint_latest.pth"
            needs_cleanup = False

        elif tag.startswith("step_"):
            # Step checkpoints need timestamps to prevent collisions
            step_num = tag.split("_")[1]
            checkpoint_filename = f"checkpoint_step_{step_num}_{timestamp_str}.pth"
            needs_cleanup = True
            cleanup_type = "step"

        elif tag.startswith("epoch_"):
            # Epoch checkpoints need timestamps to prevent collisions
            epoch_num = tag.split("_")[1]
            checkpoint_filename = f"checkpoint_epoch_{epoch_num}_{timestamp_str}.pth"
            needs_cleanup = True
            cleanup_type = "epoch"

        else:
            # Generic checkpoint with timestamp
            checkpoint_filename = f"checkpoint_{tag}_{timestamp_str}.pth"
            needs_cleanup = False

        checkpoint_path = output_dir / checkpoint_filename
        temp_path = output_dir / f"temp_{checkpoint_filename}.partial"

        # ================================================================
        # SECTION 4: Save checkpoint with atomic write pattern
        # ================================================================

        # Save to temporary file first
        try:
            torch.save(checkpoint_data, temp_path)

            # Verify temp file was created successfully
            if not temp_path.exists():
                raise IOError(
                    f"Failed to create temporary checkpoint file: {temp_path}"
                )

            # For overwriting checkpoints (best/latest), remove old version
            if checkpoint_path.exists() and tag in ["best", "latest"]:
                try:
                    checkpoint_path.unlink()
                except Exception as remove_error:
                    print(
                        f"⚠️  Warning: Could not remove old {tag} checkpoint: {remove_error}"
                    )

            # Atomic rename from temp to final
            temp_path.rename(checkpoint_path)

            # Verify the final file exists
            if not checkpoint_path.exists():
                raise IOError(
                    f"Checkpoint file not found after save: {checkpoint_path}"
                )

        except Exception as save_error:
            print(f"❌ Failed to save checkpoint: {save_error}")
            traceback.print_exc()

            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    print(f"⚠️  Could not clean up temp file: {cleanup_error}")

            return False

        # ================================================================
        # SECTION 5: Report success (with file size if available)
        # ================================================================

        try:
            file_size_mb = checkpoint_path.stat().st_size / (1024**2)
            print(f"✓ Checkpoint saved: {checkpoint_filename} ({file_size_mb:.1f} MB)")
        except Exception as stat_error:
            print(f"✓ Checkpoint saved: {checkpoint_filename} (size unknown)")

        # ================================================================
        # SECTION 6: Cleanup old checkpoints AFTER successful save
        # ================================================================
        # CRITICAL: This happens AFTER save, preventing deletion of current file

        if needs_cleanup:
            try:
                if cleanup_type == "step":
                    cleanup_old_step_checkpoints(
                        output_dir,
                        current_step=step,
                        max_to_keep=max_step_checkpoints,
                        exclude_file=checkpoint_path,  # Don't delete the file we just created!
                    )
                elif cleanup_type == "epoch":
                    cleanup_old_epoch_checkpoints(
                        output_dir,
                        current_epoch=epoch,
                        max_to_keep=max_epoch_checkpoints,
                        exclude_file=checkpoint_path,  # Don't delete the file we just created!
                    )
            except Exception as cleanup_error:
                print(f"⚠️  Warning: Checkpoint cleanup failed: {cleanup_error}")
                traceback.print_exc()
                # Non-fatal - we still saved the checkpoint successfully

        return True

    except Exception as unexpected_error:
        print(f"❌ Unexpected error in save_checkpoint: {unexpected_error}")
        traceback.print_exc()
        return False


def cleanup_old_step_checkpoints(
    output_dir, current_step, max_to_keep=3, exclude_file=None
):
    """
    Remove old step checkpoint files, keeping only the most recent N.

    CRITICAL IMPROVEMENTS:
    1. Handles timestamped filenames correctly
    2. Excludes the file we just created from deletion
    3. Uses modification time for reliable sorting

    Args:
        output_dir (Path): Directory containing checkpoint files
        current_step (int): Current training step (for logging)
        max_to_keep (int): Maximum number of step checkpoints to retain
        exclude_file (Path, optional): File to exclude from deletion (the one we just created)

    Returns:
        int: Number of files deleted
    """
    try:
        output_dir = Path(output_dir)

        # Find all step checkpoint files
        # Pattern: checkpoint_step_*.pth matches both old and new formats
        all_step_files = list(output_dir.glob("checkpoint_step_*.pth"))

        # Exclude the file we just created
        if exclude_file and exclude_file in all_step_files:
            all_step_files.remove(exclude_file)

        if len(all_step_files) <= max_to_keep:
            return 0  # Nothing to delete

        # Sort by modification time (most recent first)
        sorted_files = sorted(
            all_step_files, key=lambda f: f.stat().st_mtime, reverse=True
        )

        # Select files to delete (oldest ones)
        files_to_delete = sorted_files[max_to_keep:]

        deleted_count = 0
        total_freed_mb = 0

        for old_checkpoint in files_to_delete:
            try:
                file_size_mb = old_checkpoint.stat().st_size / (1024**2)
                old_checkpoint.unlink()
                print(
                    f"  🗑️  Removed old step checkpoint: {old_checkpoint.name} ({file_size_mb:.1f} MB)"
                )
                deleted_count += 1
                total_freed_mb += file_size_mb
            except Exception as delete_error:
                print(f"  ⚠️  Failed to delete {old_checkpoint.name}: {delete_error}")

        if deleted_count > 0:
            print(
                f"  ✓ Cleanup complete: removed {deleted_count} files, freed {total_freed_mb:.1f} MB"
            )

        return deleted_count

    except Exception as cleanup_error:
        print(f"⚠️  Warning: Step checkpoint cleanup failed: {cleanup_error}")
        traceback.print_exc()
        return 0


def cleanup_old_epoch_checkpoints(
    output_dir, current_epoch, max_to_keep=5, exclude_file=None
):
    """
    Remove old epoch checkpoint files, keeping only the most recent N.

    CRITICAL IMPROVEMENTS:
    1. Handles timestamped filenames correctly
    2. Excludes the file we just created from deletion
    3. Uses modification time for reliable sorting

    Args:
        output_dir (Path): Directory containing checkpoint files
        current_epoch (int): Current epoch number (for logging)
        max_to_keep (int): Maximum number of epoch checkpoints to retain
        exclude_file (Path, optional): File to exclude from deletion (the one we just created)

    Returns:
        int: Number of files deleted
    """
    try:
        output_dir = Path(output_dir)

        # Find all epoch checkpoint files
        # Pattern: checkpoint_epoch_*.pth matches both old and new formats
        all_epoch_files = list(output_dir.glob("checkpoint_epoch_*.pth"))

        # Exclude the file we just created
        if exclude_file and exclude_file in all_epoch_files:
            all_epoch_files.remove(exclude_file)

        if len(all_epoch_files) <= max_to_keep:
            return 0  # Nothing to delete

        # Sort by modification time (most recent first)
        sorted_files = sorted(
            all_epoch_files, key=lambda f: f.stat().st_mtime, reverse=True
        )

        # Select files to delete (oldest ones)
        files_to_delete = sorted_files[max_to_keep:]

        deleted_count = 0
        total_freed_mb = 0

        for old_checkpoint in files_to_delete:
            try:
                file_size_mb = old_checkpoint.stat().st_size / (1024**2)
                old_checkpoint.unlink()
                print(
                    f"  🗑️  Removed old epoch checkpoint: {old_checkpoint.name} ({file_size_mb:.1f} MB)"
                )
                deleted_count += 1
                total_freed_mb += file_size_mb
            except Exception as delete_error:
                print(f"  ⚠️  Failed to delete {old_checkpoint.name}: {delete_error}")

        if deleted_count > 0:
            print(
                f"  ✓ Cleanup complete: removed {deleted_count} files, freed {total_freed_mb:.1f} MB"
            )

        return deleted_count

    except Exception as cleanup_error:
        print(f"⚠️  Warning: Epoch checkpoint cleanup failed: {cleanup_error}")
        traceback.print_exc()
        return 0


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
        # print(f"\n{'=' * 40}")
        # print("Step 3: Preparing Data")
        # print(f"{'=' * 40}")
        # train_loader, val_loader = create_data_loaders(
        #     document_text, tokenizer, TRAINING_CONFIG, train_ratio=TRAIN_VAL_SPLIT
        # )

        # 3. Create data loaders
        print(f"\n{'=' * 40}")
        print("Step 3: Preparing Data")
        print(f"{'=' * 40}")

        # Use the efficient file-based tokenization
        train_loader, val_loader = create_data_loaders(
            DOCUMENT_PATH,  # Pass file path directly
            tokenizer,
            TRAINING_CONFIG,
            train_ratio=TRAIN_VAL_SPLIT,
            is_file_path=True,  # Tell it this is a file path, not text
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

chaucer in six volumes...
https://www.gutenberg.org/cache/epub/43089/pg43089.txt
https://www.gutenberg.org/cache/epub/44833/pg44833.txt
https://www.gutenberg.org/cache/epub/45027/pg45027.txt
https://www.gutenberg.org/cache/epub/22120/pg22120.txt
https://www.gutenberg.org/cache/epub/43016/pg43016.txt
https://www.gutenberg.org/cache/epub/43097/pg43097.txt


#v shakespeare first folio...
https://www.gutenberg.org/cache/epub/2270/pg2270.txt

# Aeschylus
https://www.gutenberg.org/ebooks/8714


# Poetry:
https://www.gutenberg.org/cache/epub/30235/pg30235.txt

G. K. Chesterton:
https://www.gutenberg.org/ebooks/204.txt.utf-8
https://www.gutenberg.org/ebooks/18639.txt.utf-8
https://www.gutenberg.org/ebooks/13468.txt.utf-8
https://www.gutenberg.org/ebooks/223.txt.utf-8
https://www.gutenberg.org/ebooks/65688.txt.utf-8
https://www.gutenberg.org/ebooks/2134.txt.utf-8
https://www.gutenberg.org/ebooks/9656.txt.utf-8
https://www.gutenberg.org/ebooks/16769.txt.utf-8
https://www.gutenberg.org/ebooks/20058.txt.utf-8
https://www.gutenberg.org/ebooks/8092.txt.utf-8
https://www.gutenberg.org/ebooks/1695.txt.utf-8
https://www.gutenberg.org/ebooks/1719.txt.utf-8
https://www.gutenberg.org/ebooks/12245.txt.utf-8
https://www.gutenberg.org/ebooks/59239.txt.utf-8
https://www.gutenberg.org/ebooks/22362.txt.utf-8
https://www.gutenberg.org/ebooks/1720.txt.utf-8
https://www.gutenberg.org/ebooks/21522.txt.utf-8
https://www.gutenberg.org/ebooks/60057.txt.utf-8
https://www.gutenberg.org/ebooks/1696.txt.utf-8
https://www.gutenberg.org/ebooks/21525.txt.utf-8
https://www.gutenberg.org/ebooks/70175.txt.utf-8
https://www.gutenberg.org/ebooks/14706.txt.utf-8
https://www.gutenberg.org/ebooks/67639.txt.utf-8
https://www.gutenberg.org/ebooks/61760.txt.utf-8
https://www.gutenberg.org/ebooks/11560.txt.utf-8
https://www.gutenberg.org/ebooks/11554.txt.utf-8
https://www.gutenberg.org/ebooks/25795.txt.utf-8
https://www.gutenberg.org/ebooks/1721.txt.utf-8


https://www.gutenberg.org/ebooks/382.txt.utf-8

# jane austin
https://www.gutenberg.org/ebooks/31100.txt.utf-8

# churchill
https://www.gutenberg.org/ebooks/5400.txt.utf-8

# franklin
https://www.gutenberg.org/ebooks/48136.txt.utf-8
https://www.gutenberg.org/ebooks/48138.txt.utf-8
https://www.gutenberg.org/ebooks/48137.txt.utf-8
https://www.gutenberg.org/ebooks/40236.txt.utf-8
https://www.gutenberg.org/ebooks/36338.txt.utf-8

# dickens
https://www.gutenberg.org/ebooks/1023.txt.utf-8

# Homer & Odyssey
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
            book_file_path = f"data/chesterton/{book_id}.txt"
            print(f"Downloading training data from {url}")
            text_data = download_book(url, book_file_path)
            if text_data:
                file_path = book_file_path
            else:
                print("Download failed, using demo file")
                file_path = demo_file_path

        else:  # if user does no specify url or demo, they are saying what file-path
            file_path = user_path_or_demo_choice

    # use argument input path if supplied by user
    elif len(sys.argv) == 2:
        file_path = sys.argv[1]
        print(f"path argument found... {file_path}")

    else:
        print("Edge case, defaulting to demo.")

    # Document input
    DOCUMENT_PATH = file_path  # "./data/my_document.txt"  # Path to your text file

    # Output configuration
    OUTPUT_DIR = f"./models/perseid_{MODEL_SIZE}m_{Path(DOCUMENT_PATH).stem}/"
    EXPERIMENT_NAME = (
        f"perseid_{MODEL_SIZE}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BFLOAT16 = torch.cuda.is_available()  # Use bfloat16 if on GPU

    print("\nTraining config:")
    for i in TRAINING_CONFIG:
        print(i)

    # Run the training pipeline
    model, history = main()
