"""
dynamic_slide_backweighted_validation_perseidbyte.py

Having a sliding window over larger training files might not
conflict with the idea of 'end of doc' weighted training:

While the early slider-windows do not contain two ||| ||| delimiters,
they can train in the traditional way: rote memorization.

And when the window does contain two delimiters, then we test the outcome
between those two delimitors.

This could allow both for shorter token windows and longer training docs.


For each window:
    if window contains exactly 2 delimiters (|||):
        # Complete answer present
        → Use WEIGHTED validation loss
        → Focus learning on answer tokens (10x weight)

    elif window contains 0 delimiters:
        # Pure context/question tokens
        → Use STANDARD uniform loss
        → Learn general language modeling

    elif window contains 1 delimiter:
        # Partial answer (split across windows)
        → Use STANDARD uniform loss (or maybe skip?)
        → Don't try to weight incomplete answer



...

This is a simpler trainer that allows re-training / continued pre-training.
Possibly should track/log what last-best validation loss was...


Training module for Perseid models on text document corpus.
Handles single document input with configurable train/val split.
Trains from scratch (no pretrained weights),
unless existing weights are found, then it should continue training.

Usage:
    1. Set configuration parameters at top of file
    2. Run: python train_perseid_byte.py



Description:

Weighted validation loss target by one of two mutually exclusive options:
A. last N non-padding tokens
B. delimitor string surrounding target

Raise exception if both modes enabled.
Only applies to validation loss by default, possible option toggle to apply to both training and validation.


## A. last N non-padding tokens
Backweighted validation-loss for specific training chunks is designed
to give the last ~8-16 non-padding tokens (bytes, using a byte-tokenizer) more weight,
as a foot in the door or text-proxy for having the 'outcome' of a 'task'
be focused on above misc token-regurgitation for all tokens uniformly.

Find last N real (non-padding) tokens and weight those
Probably 16 is a safe first N, since six are delimitors,
and the answer flag will be there if the answer itself is short

e.g. this is 16 bytes, which showing the shortest answer
filling that weight span.
# answer\n|||1|||

Here the format of the training/validation data is designed so that the answer to the question
is at the end (before the padding tokens).


## B. delimitor string surrounding target
The delimted target uses a string only used around the target answer.
"|||" tripple pipe will be used by default, because that is unlikely to collide
even with math notation. Only weight what is inside; this is one
of the main differences to test-compare for N-last tokens (including delimitors)
vs. only what is inside.

This is more targeted and more strict.
If there are no delimitors, it is an automatic fail for the validation.

e.g. here the target is inside the delimiters
# answer\n|||1|||


If delimitor not found in results.

Other ~parameters of valiation regime include:
- always weighting validation, or N pct of the time (seeded) randomely
- grace-period before starting weight

What Gets Heavily Weighted: e.g. for |||4|||
- Only the '4' token inside |||4||| gets the 10x multiplier
- Everything else: normal weight (1x) or masked (0x for padding)

e.g.

TRAINING_CONFIG = {
    ...
    "weight_only_validation": True,       # by default only the validation-loss is weighted
    "validation_loss_weight_multiplier": 10.0,     # How much to weight answer tokens
    "training_loss_weight_multiplier": 0.0,        # How much to weight answer tokens
    "use_back_weighted_loss": False,      # Enable back-weighted loss
    "n_answer_tokens": 16,                # Number of answer tokens to weight
    "use_delimited_target_weighted_loss": True,    # Enable delimited weighted loss
    "target_delimiter_string": "|||",     # default, tripple pipe delimiter
    "pct_valiations_weighted": 50,        # how frequently validation uses weight
    "steps_grace_period_before_weighting": 200     # allow some learning before strict requirements
}

e.g.

def calculate_weighted_loss_backweighted_last_n(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    n_answer_tokens: int = 16,
    answer_weight_multiplier: float = 10.0,
    pad_token_id: int = 0,
    pct_valiations_weighted = 100,
    steps_grace_period_before_weighting = 100,
)-> torch.Tensor:


def calculate_weighted_loss_delimited_target(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    target_delimiter_string: str = "|||",
    answer_weight_multiplier: float = 10.0,
    pad_token_id: int = 0,
    tokenizer: ByteTokenizer = None,
    pct_valiations_weighted = 100,
    steps_grace_period_before_weighting = 100,
) -> torch.Tensor:

or
def calculate_weighted_loss_delimited_target(
    input_batch: torch.Tensor,              # [batch_size, seq_len] - model inputs
    target_batch: torch.Tensor,             # [batch_size, seq_len] - ground truth targets
    model: nn.Module,                       # The neural network model
    device: torch.device,                   # 'cuda' or 'cpu'
    target_delimiter_string: str = "|||",   # Delimiter surrounding answer
    answer_weight_multiplier: float = 10.0, # How much to weight answer tokens
    pad_token_id: int = 0,                  # Padding token ID to mask
    tokenizer: ByteTokenizer = None,        # Needed to encode delimiter string
    pct_validations_weighted: int = 100,    # % of time to apply weighting (0-100)
    steps_grace_period_before_weighting: int = 100,  # Steps before weighting starts
    current_step: int = 0,                  # Current training step (for grace period)
    random_seed: int = 42,                  # Seed for reproducible probability sampling
) -> torch.Tensor:                          # Returns: scalar loss value
    '''
    Calculate cross-entropy loss with heavy weighting on tokens between delimiters.

    The goal: Focus learning on the actual answer (between |||delimiters|||) rather
    than uniformly weighting all token predictions including prompt regurgitation.

    This is a "task outcome" proxy - we care most about predicting the answer correctly,
    less about perfectly mimicking formatting/prompt tokens.
    '''


note:
    maybe alternative to grace period?
    Option A (Strict): Set all weights to 0.0 for that sequence (treat as failed validation)
    Option B (Lenient): Fall back to uniform weighting (weight = 1.0 for all non-padding)
"""

"""

# Project Scope Outline: Core Design Principles

1. Each file = One complete semantic unit
- Like Alpaca format: one file is one Q&A pair
- File structure: Question → |||answer||| → Answer
- CANNOT create training windows that span multiple files
 -CANNOT take random subsequences from within a file


2.Batch = Collection of complete files processed together
- batch_size = 10 means "process 10 complete files simultaneously"
- NOT "10 sliding windows from concatenated files"
- Each batch element is one complete file (padded to fixed length)


3. File size variation is expected and acceptable
- Files might be 150 tokens (simple math)
- Files might be 800 tokens (complex problem)
- Files might be 1500 tokens (need truncation)
- This pipeline's job: Handle whatever sizes exist
- NOT this pipeline's job: Enforce uniform file sizes


4. Padding strategy
- Files < context_length → Pad with PAD tokens (256)
- Files > context_length → Truncate (maybe with warning)
- Files = context_length → Use as-is


5. Validation weighting
- Find |||answer||| delimiters in each file
- Weight the answer tokens heavily (10x default)
- Weight padding tokens at 0x (ignored)
- This focuses learning on correct outcomes, not only regurgitating docs

# For each file:
1. Load file → tokenize → get token list (e.g., 300 tokens)
2. If len < 1024: pad to 1024 with PAD tokens
3. If len > 1024: truncate to 1024 (keep first 1024)
4. If len = 1024: use as-is
5. Add to dataset as ONE training example

# Result:
- 500 files → 500 training examples
- batch_size=10 → each batch contains 10 complete files
- Each file maintains its |||answer||| structure intact

"""
import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import random

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


# Check if CUDA (GPU support) is available
print("CUDA available:", torch.cuda.is_available())

# Get the current device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

# Optionally, get the name of the GPU if available
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


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
#     "context_length": 1024,  # Context window for training
#     "batch_size": 11,  # Batch size (increase if memory allows)
#     "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
#     "learning_rate": 5e-4,  # Learning rate
#     "num_epochs": 7,  # Number of training epochs, default 3
#     "weight_decay": 0.01,  # Weight decay for AdamW
#     "warmup_steps": 100,  # Warmup steps for learning rate
#     "eval_every": 2,  # Evaluate every N steps
#     "eval_batches": 10,  # Number of batches for evaluation
#     "save_every": 500,  # Save checkpoint every N steps
#     "chunk_overlap": 0.1,  # Overlap between text chunks (0.0 to 0.5)
# }


TRAINING_CONFIG = {
    "context_length": 512,  # 1024,  (see: sliding_window size in model file)
    "batch_size": 10,  # (number of docs/chunks trained on at one time)
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "num_epochs": 1,  # 7 (one for no repeat training)
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "eval_every": 100,
    "eval_batches": 10,
    "save_every": 50,  # 100
    "chunk_overlap": 0.1,
    # Weighted validation loss parameters
    "use_weighted_validation": True,  # Enable/disable weighted validation
    "target_delimiter_string": "|||",  # Delimiter surrounding answers
    "answer_weight_multiplier": 10.0,  # Weight multiplier for answer tokens
    "pct_validations_weighted": 100,  # Percentage of validations to weight (0-100)
    "steps_grace_period_before_weighting": 600,  # Grace period steps before weighting starts
    "validation_random_seed": 42,  # Seed for reproducible stochastic weighting
    # Sliding window behavior
    "use_sliding_windows": True,  # Enable multi-window per file
    "min_file_length_for_windows": 1024,  # Files shorter than this: single window, padded
}


# ============================================================================
# END USER CONFIGURATION
# ============================================================================


# def load_documents_from_directory(
#     directory_path: str | Path,
#     file_extension: str = ".toml",
#     train_val_split: float = 0.9,
#     shuffle_files: bool = True,
#     random_seed: int = 42,
#     verbose: bool = True,
# ) -> Tuple[str, str]:
#     """
#     Load all text files from directory and split into train/val sets.

#     This function loads multiple training files, shuffles them (optionally),
#     and splits them into training and validation sets at the FILE level
#     (not character level), which is better for validation integrity.

#     Args:
#         directory_path: Path to directory containing training files
#         file_extension: File extension to filter (default: ".toml")
#         train_val_split: Fraction of files for training (default: 0.9 = 90%)
#         shuffle_files: Whether to shuffle files before splitting (default: True)
#         random_seed: Random seed for reproducible shuffling (default: 42)
#         verbose: Print detailed information (default: True)

#     Returns:
#         Tuple[str, str]: (train_text, val_text) - concatenated file contents

#     Raises:
#         FileNotFoundError: If directory doesn't exist
#         ValueError: If no valid files found or split is invalid

#     Example:
#         >>> train_text, val_text = load_documents_from_directory(
#         ...     directory_path="./training_data/",
#         ...     file_extension=".toml",
#         ...     train_val_split=0.9
#         ... )
#         >>> print(f"Train: {len(train_text)} chars, Val: {len(val_text)} chars")
#     """
#     try:
#         directory_path = Path(directory_path)

#         # =====================================================================
#         # SECTION 1: Validate Directory Exists
#         # =====================================================================

#         if not directory_path.exists():
#             error_message = f"Directory not found: {directory_path}"
#             raise FileNotFoundError(error_message)

#         if not directory_path.is_dir():
#             error_message = f"Path is not a directory: {directory_path}"
#             raise ValueError(error_message)

#         if verbose:
#             print(f"\n{'=' * 60}")
#             print(f"Loading documents from: {directory_path}")
#             print(f"{'=' * 60}")

#         # =====================================================================
#         # SECTION 2: Find All Matching Files
#         # =====================================================================

#         # Get all files with matching extension
#         all_files = sorted(directory_path.glob(f"*{file_extension}"))

#         if len(all_files) == 0:
#             error_message = (
#                 f"No files found with extension '{file_extension}' in {directory_path}"
#             )
#             raise ValueError(error_message)

#         if verbose:
#             print(f"Found {len(all_files)} files with extension '{file_extension}'")

#         # =====================================================================
#         # SECTION 3: Shuffle Files (Optional but Recommended)
#         # =====================================================================

#         if shuffle_files:
#             # Use seeded random for reproducibility
#             rng = random.Random(random_seed)
#             all_files = list(all_files)  # Convert to list for shuffling
#             rng.shuffle(all_files)

#             if verbose:
#                 print(f"Files shuffled with seed={random_seed}")

#         # =====================================================================
#         # SECTION 4: Split Files into Train/Val Sets
#         # =====================================================================

#         # Validate split ratio
#         if not (0.0 < train_val_split < 1.0):
#             error_message = (
#                 f"train_val_split must be between 0 and 1, got {train_val_split}"
#             )
#             raise ValueError(error_message)

#         # Calculate split point
#         num_train_files = int(len(all_files) * train_val_split)

#         # Ensure at least one file in each set
#         if num_train_files == 0:
#             num_train_files = 1
#         elif num_train_files == len(all_files):
#             num_train_files = len(all_files) - 1

#         train_files = all_files[:num_train_files]
#         val_files = all_files[num_train_files:]

#         if verbose:
#             print(
#                 f"\nSplit: {len(train_files)} train files, {len(val_files)} val files"
#             )
#             print(
#                 f"Ratio: {len(train_files) / len(all_files):.1%} train, "
#                 f"{len(val_files) / len(all_files):.1%} val"
#             )

#         # =====================================================================
#         # SECTION 5: Load and Concatenate File Contents
#         # =====================================================================

#         def load_and_concatenate_files(file_list, set_name):
#             """
#             Load all files in list and concatenate their contents.

#             Args:
#                 file_list: List of Path objects to load
#                 set_name: Name for logging ("train" or "val")

#             Returns:
#                 str: Concatenated text from all files
#             """
#             concatenated_text = ""
#             total_bytes = 0

#             if verbose:
#                 print(f"\nLoading {set_name} files:")

#             for file_index, file_path in enumerate(file_list):
#                 try:
#                     # Try UTF-8 first, fall back to other encodings
#                     encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
#                     file_content = None

#                     for encoding in encodings_to_try:
#                         try:
#                             with open(file_path, "r", encoding=encoding) as f:
#                                 file_content = f.read()
#                             break  # Success - stop trying encodings
#                         except UnicodeDecodeError:
#                             continue  # Try next encoding

#                     if file_content is None:
#                         print(f"  ⚠ Could not decode: {file_path.name} (skipping)")
#                         continue

#                     # Add file content to concatenated text
#                     # Optionally add separator between files
#                     if concatenated_text:
#                         concatenated_text += "\n\n"  # Double newline separator

#                     concatenated_text += file_content

#                     file_size = len(file_content)
#                     total_bytes += file_size

#                     if verbose:
#                         print(
#                             f"  [{file_index + 1}/{len(file_list)}] "
#                             f"{file_path.name}: {file_size:,} bytes"
#                         )

#                 except Exception as file_load_error:
#                     print(f"  ✗ Error loading {file_path.name}: {file_load_error}")
#                     # Continue with other files

#             if verbose:
#                 print(
#                     f"\n{set_name.capitalize()} total: {total_bytes:,} bytes "
#                     f"({len(concatenated_text):,} characters)"
#                 )

#             return concatenated_text

#         # Load train files
#         train_text = load_and_concatenate_files(train_files, "train")

#         # Load validation files
#         val_text = load_and_concatenate_files(val_files, "val")

#         # =====================================================================
#         # SECTION 6: Final Validation
#         # =====================================================================

#         if len(train_text) == 0:
#             raise ValueError(
#                 "No training data loaded - all files may be empty or unreadable"
#             )

#         if len(val_text) == 0:
#             raise ValueError(
#                 "No validation data loaded - all files may be empty or unreadable"
#             )

#         if verbose:
#             print(f"\n{'=' * 60}")
#             print("Loading complete!")
#             print(
#                 f"  Train: {len(train_text):,} characters from {len(train_files)} files"
#             )
#             print(f"  Val: {len(val_text):,} characters from {len(val_files)} files")
#             print(f"{'=' * 60}\n")

#         return train_text, val_text

#     except Exception as unexpected_error:
#         print(f"Error loading documents from directory: {unexpected_error}")
#         traceback.print_exc()
#         raise


def load_and_tokenize_documents_from_directory(
    directory_path: str | Path,
    tokenizer,
    file_extension: str = ".toml",
    train_val_split: float = 0.9,
    shuffle_files: bool = True,
    random_seed: int = 42,
    verbose: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Load and tokenize all files from directory, returning pre-tokenized data.

    CRITICAL PERFORMANCE: This function uses efficient file-based tokenization
    (direct bytes-to-tokens) instead of loading text strings first.

    Process:
        1. Find all matching files in directory
        2. Shuffle file list (reproducibly)
        3. Split into train/val file lists
        4. Tokenize each file DIRECTLY from bytes (no string conversion)
        5. Return concatenated token lists for train and val

    Args:
        directory_path: Path to directory containing training files
        tokenizer: ByteTokenizer instance with encode_file method
        file_extension: File extension filter (default: ".toml")
        train_val_split: Fraction of files for training (default: 0.9)
        shuffle_files: Whether to shuffle files before splitting (default: True)
        random_seed: Random seed for reproducible shuffling (default: 42)
        verbose: Print detailed progress information (default: True)

    Returns:
        tuple[list[int], list[int]]: (train_tokens, val_tokens)
            Both are lists of integer token IDs ready for dataset creation

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid files found or split ratio invalid

    Performance Note:
        This is 5-10x faster than loading text first because it avoids
        the bytes→string→bytes conversion cycle that happens with text loading.
    """
    try:
        directory_path = Path(directory_path)

        # =====================================================================
        # SECTION 1: Validate Directory
        # =====================================================================

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Loading and tokenizing documents from: {directory_path}")
            print(f"{'=' * 60}")

        # =====================================================================
        # SECTION 2: Find All Matching Files
        # =====================================================================

        all_files = sorted(directory_path.glob(f"*{file_extension}"))

        if len(all_files) == 0:
            raise ValueError(
                f"No files found with extension '{file_extension}' in {directory_path}"
            )

        if verbose:
            print(f"Found {len(all_files)} files with extension '{file_extension}'")

        # =====================================================================
        # SECTION 3: Shuffle Files
        # =====================================================================

        if shuffle_files:
            rng = random.Random(random_seed)
            all_files = list(all_files)
            rng.shuffle(all_files)

            if verbose:
                print(f"Files shuffled with seed={random_seed}")

        # =====================================================================
        # SECTION 4: Split Files into Train/Val Sets
        # =====================================================================

        if not (0.0 < train_val_split < 1.0):
            raise ValueError(
                f"train_val_split must be between 0 and 1, got {train_val_split}"
            )

        num_train_files = int(len(all_files) * train_val_split)

        # Ensure at least one file in each set
        if num_train_files == 0:
            num_train_files = 1
        elif num_train_files == len(all_files):
            num_train_files = len(all_files) - 1

        train_files = all_files[:num_train_files]
        val_files = all_files[num_train_files:]

        if verbose:
            print(
                f"\nSplit: {len(train_files)} train files, {len(val_files)} val files"
            )
            print(
                f"Ratio: {len(train_files) / len(all_files):.1%} train, "
                f"{len(val_files) / len(all_files):.1%} val"
            )

        # =====================================================================
        # SECTION 5: Tokenize Files Efficiently (Direct Byte Encoding)
        # =====================================================================

        def tokenize_file_list_efficiently(
            file_list: list[Path], set_name: str
        ) -> list[int]:
            """
            Tokenize a list of files using efficient direct byte encoding.

            Args:
                file_list: List of Path objects to tokenize
                set_name: Name for logging ("train" or "val")

            Returns:
                list[int]: Concatenated token IDs from all files
            """
            all_tokens = []
            total_bytes = 0
            tokenization_start = time.perf_counter()

            if verbose:
                print(f"\n{'-' * 60}")
                print(f"Tokenizing {set_name} files:")

            for file_idx, file_path in enumerate(file_list):
                try:
                    file_size = file_path.stat().st_size

                    # Use efficient direct byte encoding
                    # This is the key performance improvement!
                    file_tokens = tokenizer.encode_file(
                        file_path,
                        add_eos=False,  # Don't add EOS (we'll add separator)
                        verbose=False,  # Suppress per-file verbosity
                    )

                    # Add file tokens to collection
                    all_tokens.extend(file_tokens)

                    # Add EOS token as separator between files
                    all_tokens.append(tokenizer.EOS_ID)

                    total_bytes += file_size

                    if verbose:
                        print(
                            f"  [{file_idx + 1}/{len(file_list)}] "
                            f"{file_path.name}: {len(file_tokens):,} tokens "
                            f"({file_size:,} bytes)"
                        )

                except Exception as file_error:
                    print(f"  ✗ Error tokenizing {file_path.name}: {file_error}")
                    traceback.print_exc()
                    # Continue with other files
                    continue

            tokenization_time = time.perf_counter() - tokenization_start

            if verbose:
                tokens_per_sec = (
                    len(all_tokens) / tokenization_time if tokenization_time > 0 else 0
                )
                mb_per_sec = (
                    total_bytes / tokenization_time / (1024 * 1024)
                    if tokenization_time > 0
                    else 0
                )

                print(f"\n✅ {set_name.capitalize()} tokenization complete!")
                print(f"  Files processed: {len(file_list)}")
                print(f"  Total tokens: {len(all_tokens):,}")
                print(f"  Total bytes: {total_bytes:,}")
                print(f"  Time: {tokenization_time:.2f} seconds")
                print(
                    f"  Speed: {tokens_per_sec:,.0f} tokens/sec ({mb_per_sec:.2f} MB/s)"
                )

            return all_tokens

        # Tokenize train files
        train_tokens = tokenize_file_list_efficiently(train_files, "train")

        # Tokenize validation files
        val_tokens = tokenize_file_list_efficiently(val_files, "val")

        # =====================================================================
        # SECTION 6: Final Validation
        # =====================================================================

        if len(train_tokens) == 0:
            raise ValueError(
                "No training data tokenized - all files may be empty or unreadable"
            )

        if len(val_tokens) == 0:
            raise ValueError(
                "No validation data tokenized - all files may be empty or unreadable"
            )

        if verbose:
            print(f"\n{'=' * 60}")
            print("Tokenization complete!")
            print(
                f"  Train tokens: {len(train_tokens):,} from {len(train_files)} files"
            )
            print(f"  Val tokens: {len(val_tokens):,} from {len(val_files)} files")
            print(f"{'=' * 60}\n")

        return train_tokens, val_tokens

    except Exception as loading_error:
        print(f"Error loading and tokenizing documents: {loading_error}")
        traceback.print_exc()
        raise


# def create_data_loaders_from_directory(
#     directory_path: str | Path,
#     tokenizer,
#     config,
#     train_ratio: float = 0.9,
#     file_extension: str = ".toml",
#     shuffle_files: bool = True,
#     random_seed: int = 42,
# ):
#     """
#     Create training and validation data loaders from directory of pre-chunked files.

#     IMPORTANT PARADIGM: Each file IS a complete training chunk.
#     ==============================================================
#     This function treats files as pre-made training chunks, not as documents
#     to be chunked. If you have 500 .toml files, you have 500 training chunks.

#     The function:
#     1. Loads ALL files from the directory
#     2. Shuffles the FILE LIST (not content) for randomization
#     3. Splits files into train/val sets (e.g., 450 train, 50 val)
#     4. Concatenates train files into one big training text
#     5. Concatenates val files into one big validation text
#     6. Creates sliding windows WITHIN each concatenated text

#     Why file-level splitting?
#     -------------------------
#     - Prevents data leakage: validation never sees training file content
#     - Natural boundaries: each file is a complete problem/answer pair
#     - Easy management: add/remove files without code changes
#     - Reproducible: seeded shuffle gives same split every time

#     File Structure Example:
#     ----------------------
#     training_data/
#     ├── chunk_001.toml  ← One complete Q&A with |||answer|||
#     ├── chunk_002.toml  ← Another complete Q&A
#     ├── chunk_003.toml
#     ...
#     └── chunk_500.toml

#     With train_ratio=0.9:
#     - Files 1-450 → concatenated into train_text → train windows → train_loader
#     - Files 451-500 → concatenated into val_text → val windows → val_loader

#     Each .toml file format (example):
#     ---------------------------------
#     ```
#     ||expression section||

#     |English|
#     seven minus six

#     |symbolic|
#     7-6

#     ||evaluation section||

#     |answer|
#     |||1|||
#     ```

#     The |||1||| delimiter marks the answer for weighted validation loss.

#     Workflow Detail:
#     ---------------
#     1. Find all files matching file_extension in directory_path
#     2. Optionally shuffle file list (reproducibly with random_seed)
#     3. Split file list: first train_ratio% → train, rest → val
#     4. Load and concatenate all train files → train_text string
#     5. Load and concatenate all val files → val_text string
#     6. Create DocumentDataset from train_text (with overlapping windows)
#     7. Create DocumentDataset from val_text (no overlap)
#     8. Wrap datasets in DataLoader for batching

#     Args:
#         directory_path: Path to directory containing .toml training chunk files.
#                        Each file should be a complete training example with
#                        delimited answer (e.g., |||answer|||).

#         tokenizer: ByteTokenizer instance with encode/decode methods.
#                   Each byte becomes one token (1 byte = 1 token).

#         config: Training configuration dictionary containing:
#                 - 'context_length': Max sequence length (e.g., 1024 or 2048 bytes)
#                 - 'batch_size': Number of sequences per batch
#                 - 'chunk_overlap': Fraction of overlap for training windows (0.0-0.5)

#         train_ratio: Fraction of FILES (not bytes) to use for training.
#                     Default 0.9 means 90% of files → train, 10% → validation.
#                     Example: 500 files × 0.9 = 450 train files, 50 val files.

#         file_extension: File extension to filter. Default ".toml".
#                        Only files matching this extension are loaded.
#                        Change to ".txt" if your chunks are .txt files.

#         shuffle_files: Whether to shuffle the file list before splitting.
#                       Default True (recommended for unbiased splits).
#                       Set False only if files are already randomly ordered.

#         random_seed: Random seed for reproducible file shuffling.
#                     Default 42. Same seed always gives same train/val split.
#                     Change seed to get different split of same files.

#     Returns:
#         tuple: (train_loader, val_loader)

#         train_loader: DataLoader yielding (input_ids, target_ids) batches
#                      from training file set. Shuffled=True for randomness.

#         val_loader: DataLoader yielding (input_ids, target_ids) batches
#                    from validation file set. Shuffled=False for consistency.

#     Raises:
#         FileNotFoundError: If directory_path doesn't exist
#         ValueError: If no files found with file_extension
#         ValueError: If train_ratio not in range (0.0, 1.0)

#     Example Usage:
#         >>> # You have 1000 .toml files, each 1.5KB (1500 bytes)
#         >>> config = {"context_length": 2048, "batch_size": 11, "chunk_overlap": 0.1}
#         >>>
#         >>> train_loader, val_loader = create_data_loaders_from_directory(
#         ...     directory_path="./my_training_chunks/",
#         ...     tokenizer=byte_tokenizer,
#         ...     config=config,
#         ...     train_ratio=0.9,              # 900 files train, 100 val
#         ...     file_extension=".toml",       # Only load .toml files
#         ...     shuffle_files=True,           # Randomize which files go to train/val
#         ...     random_seed=42                # Reproducible split
#         ... )
#         >>>
#         >>> print(len(train_loader))  # Number of training batches
#         >>> print(len(val_loader))    # Number of validation batches

#     Notes:
#         - File size should be ≤ context_length for optimal training
#         - If files are 1.5KB, use context_length=2048 (leaves room for padding)
#         - Byte tokenizer: 1 byte = 1 token, so 2048 tokens = 2048 bytes ≈ 2KB
#         - Files concatenated with "\\n\\n" separator between them
#         - Training windows have chunk_overlap, validation windows have none
#         - Empty or unreadable files are skipped with warning

#     See Also:
#         - load_documents_from_directory(): Does the file loading and concatenation
#         - DocumentDataset: Creates sliding windows from concatenated text
#         - calculate_weighted_loss_delimited_target(): Uses ||| delimiters for weighting
#     """
#     try:
#         print(f"\n{'=' * 60}")
#         print("Creating Data Loaders from Directory")
#         print(f"{'=' * 60}")

#         # Load documents with file-level train/val split
#         train_text, val_text = load_documents_from_directory(
#             directory_path=directory_path,
#             file_extension=file_extension,
#             train_val_split=train_ratio,
#             shuffle_files=shuffle_files,
#             random_seed=random_seed,
#             verbose=True,
#         )

#         # Calculate stride from overlap
#         stride = int(config["context_length"] * (1 - config["chunk_overlap"]))

#         print(f"\nCreating training dataset...")
#         # Create training dataset
#         train_dataset = DocumentDataset(
#             train_text, tokenizer, config["context_length"], stride, verbose=True
#         )

#         print(f"\nCreating validation dataset...")
#         # Create validation dataset (no overlap for validation)
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
#             shuffle=True,  # Shuffle windows within training set
#             drop_last=True,
#             num_workers=0,
#         )

#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=config["batch_size"],
#             shuffle=False,  # Don't shuffle validation
#             drop_last=False,
#             num_workers=0,
#         )

#         print(f"\n{'=' * 60}")
#         print("Data Loader Summary")
#         print(f"{'=' * 60}")
#         print(f"Train batches: {len(train_loader):,}")
#         print(f"Val batches: {len(val_loader):,}")
#         print(f"Total train windows: {len(train_dataset):,}")
#         print(f"Total val windows: {len(val_dataset):,}")
#         print(f"{'=' * 60}\n")

#         return train_loader, val_loader

#     except Exception as loader_creation_error:
#         print(f"Error creating data loaders: {loader_creation_error}")
#         traceback.print_exc()
#         raise


# def create_data_loaders_from_directory(
#     directory_path: str | Path,
#     tokenizer,
#     config: dict,
#     train_ratio: float = 0.9,
#     file_extension: str = ".toml",
#     shuffle_files: bool = True,
#     random_seed: int = 42,
# ) -> tuple[DataLoader, DataLoader]:
#     """
#     Create training and validation data loaders from directory of files.

#     PERFORMANCE OPTIMIZED: Uses efficient file-based tokenization.

#     Args:
#         directory_path: Path to directory containing training chunk files
#         tokenizer: ByteTokenizer instance
#         config: Training configuration dict with keys:
#             - 'context_length': Max sequence length
#             - 'batch_size': Batch size
#             - 'chunk_overlap': Overlap fraction for training windows
#         train_ratio: Fraction of files for training (default: 0.9)
#         file_extension: File extension filter (default: ".toml")
#         shuffle_files: Shuffle files before split (default: True)
#         random_seed: Seed for reproducible shuffling (default: 42)

#     Returns:
#         tuple[DataLoader, DataLoader]: (train_loader, val_loader)

#     Note:
#         This function is 5-10x faster than the string-based approach because
#         it uses direct byte-to-token encoding from files.
#     """
#     try:
#         print(f"\n{'=' * 60}")
#         print("Creating Data Loaders from Directory")
#         print(f"{'=' * 60}")

#         # =====================================================================
#         # SECTION 1: Load and Tokenize Files Efficiently
#         # =====================================================================

#         # This returns PRE-TOKENIZED data (lists of token IDs)
#         # NOT text strings - this is the key performance improvement!
#         train_tokens, val_tokens = load_and_tokenize_documents_from_directory(
#             directory_path=directory_path,
#             tokenizer=tokenizer,
#             file_extension=file_extension,
#             train_val_split=train_ratio,
#             shuffle_files=shuffle_files,
#             random_seed=random_seed,
#             verbose=True,
#         )

#         # =====================================================================
#         # SECTION 2: Calculate Stride from Overlap Configuration
#         # =====================================================================

#         stride = int(config["context_length"] * (1 - config["chunk_overlap"]))

#         # =====================================================================
#         # SECTION 3: Create Datasets from Pre-Tokenized Data
#         # =====================================================================

#         print(f"\n{'-' * 50}")
#         print("Creating TRAINING dataset from tokens...")

#         # Use DocumentDatasetFromTokens (efficient - no re-tokenization!)
#         train_dataset = DocumentDatasetFromTokens(
#             tokens=train_tokens,
#             max_length=config["context_length"],
#             stride=stride,
#             verbose=True,
#         )

#         print(f"\n{'-' * 50}")
#         print("Creating VALIDATION dataset from tokens...")

#         # Validation uses no overlap (stride = context_length)
#         val_dataset = DocumentDatasetFromTokens(
#             tokens=val_tokens,
#             max_length=config["context_length"],
#             stride=config["context_length"],  # No overlap for validation
#             verbose=True,
#         )

#         # =====================================================================
#         # SECTION 4: Create PyTorch Data Loaders
#         # =====================================================================

#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=config["batch_size"],
#             shuffle=True,  # Shuffle windows for training
#             drop_last=True,  # Drop incomplete batches
#             num_workers=0,  # Single-threaded for simplicity
#             pin_memory=torch.cuda.is_available(),  # Speed up GPU transfer
#         )

#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=config["batch_size"],
#             shuffle=False,  # Don't shuffle validation
#             drop_last=False,  # Keep all validation data
#             num_workers=0,
#             pin_memory=torch.cuda.is_available(),
#         )

#         # =====================================================================
#         # SECTION 5: Print Summary Statistics
#         # =====================================================================

#         print(f"\n{'=' * 60}")
#         print("Data Loader Summary")
#         print(f"{'=' * 60}")
#         print(f"Train batches: {len(train_loader):,}")
#         print(f"Val batches: {len(val_loader):,}")
#         print(f"Total train windows: {len(train_dataset):,}")
#         print(f"Total val windows: {len(val_dataset):,}")
#         print(f"Batch size: {config['batch_size']}")
#         print(f"Context length: {config['context_length']}")
#         print(f"{'=' * 60}\n")

#         return train_loader, val_loader

#     except Exception as loader_creation_error:
#         print(f"Error creating data loaders: {loader_creation_error}")
#         traceback.print_exc()
#         raise


def create_data_loaders_from_directory(
    directory_path: str | Path,
    tokenizer,
    config: dict,
    train_ratio: float = 0.9,
    file_extension: str = ".toml",
    shuffle_files: bool = True,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders from directory of files.

    HYBRID WINDOWING SUPPORT: Now supports both single-window-per-file and
    sliding windows for long files, with intelligent answer detection.

    PERFORMANCE OPTIMIZED: Uses efficient file-based tokenization.

    Args:
        directory_path: Path to directory containing training chunk files (.toml)
        tokenizer: ByteTokenizer instance with encode_file() method
        config: Training configuration dict with keys:
            - 'context_length': Max sequence length (e.g., 1024)
            - 'batch_size': Number of examples per batch
            - 'chunk_overlap': Overlap fraction for sliding windows (0.0-1.0)
            - 'use_sliding_windows': Enable multi-window mode for long files
            - 'min_file_length_for_windows': Threshold for multi-window (e.g., 1024)
            - 'target_delimiter_string': Delimiter for answer detection (e.g., "|||")
        train_ratio: Fraction of files for training (default: 0.9)
        file_extension: File extension filter (default: ".toml")
        shuffle_files: Shuffle files before split (default: True)
        random_seed: Seed for reproducible shuffling (default: 42)

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)
            - train_loader: DataLoader with shuffled batches for training
            - val_loader: DataLoader with sequential batches for validation

    Note:
        Files are tokenized directly (bytes → tokens) without intermediate
        string conversion, providing 5-10x speedup over text-based methods.

        Each loader's dataset tracks which windows contain complete answers
        (2 delimiters) for conditional weighted validation loss.
    """
    try:
        print(f"\n{'=' * 60}")
        print("Creating Data Loaders from Directory (Hybrid Mode)")
        print(f"{'=' * 60}")

        # =====================================================================
        # SECTION 1: Load and Tokenize Files Efficiently
        # =====================================================================

        # Load pre-tokenized data (lists of token IDs, NOT text strings)
        # This is the key performance optimization - direct byte→token encoding
        train_tokens, val_tokens = load_and_tokenize_documents_from_directory(
            directory_path=directory_path,
            tokenizer=tokenizer,
            file_extension=file_extension,
            train_val_split=train_ratio,
            shuffle_files=shuffle_files,
            random_seed=random_seed,
            verbose=True,
        )

        # =====================================================================
        # SECTION 2: Prepare Configuration Parameters
        # =====================================================================

        # Calculate stride from overlap configuration
        # Example: context_length=1024, chunk_overlap=0.1 → stride=922
        # This means windows overlap by 102 tokens (10%)
        stride = int(config["context_length"] * (1 - config["chunk_overlap"]))

        # Get sliding window configuration
        use_sliding_windows = config.get("use_sliding_windows", True)
        min_file_length_for_windows = config.get(
            "min_file_length_for_windows", config["context_length"]
        )

        # Prepare delimiter tokens for answer detection
        # Convert delimiter string (e.g., "|||") to token IDs
        delimiter_string = config.get("target_delimiter_string", "|||")
        delimiter_tokens = tokenizer.encode(delimiter_string, add_eos=False)

        print(f"\nDataset Configuration:")
        print(f"  Context length: {config['context_length']}")
        print(
            f"  Stride: {stride} (overlap: {config['context_length'] - stride} tokens)"
        )
        print(f"  Sliding windows: {use_sliding_windows}")
        print(f"  Multi-window threshold: {min_file_length_for_windows} tokens")
        print(f"  Answer delimiter: '{delimiter_string}' → tokens {delimiter_tokens}")

        # =====================================================================
        # SECTION 3: Create Training Dataset with Hybrid Windowing
        # =====================================================================

        print(f"\n{'-' * 50}")
        print("Creating TRAINING dataset from tokens...")

        train_dataset = DocumentDatasetFromTokens(
            tokens=train_tokens,
            max_length=config["context_length"],
            stride=stride,  # NOW USED for sliding windows
            verbose=True,
            eos_token_id=tokenizer.EOS_ID,
            pad_token_id=tokenizer.PAD_ID,
            delimiter_tokens=delimiter_tokens,  # NEW: For answer detection
            use_sliding_windows=use_sliding_windows,  # NEW: Enable hybrid mode
            min_file_length_for_windows=min_file_length_for_windows,  # NEW: Threshold
        )

        # =====================================================================
        # SECTION 4: Create Validation Dataset (No Overlap)
        # =====================================================================

        print(f"\n{'-' * 50}")
        print("Creating VALIDATION dataset from tokens...")

        # Validation uses NO overlap for cleaner evaluation
        # Each validation example is independent
        val_dataset = DocumentDatasetFromTokens(
            tokens=val_tokens,
            max_length=config["context_length"],
            stride=config["context_length"],  # No overlap: stride = window size
            verbose=True,
            eos_token_id=tokenizer.EOS_ID,
            pad_token_id=tokenizer.PAD_ID,
            delimiter_tokens=delimiter_tokens,  # NEW: For answer detection
            use_sliding_windows=use_sliding_windows,  # NEW: Enable hybrid mode
            min_file_length_for_windows=min_file_length_for_windows,  # NEW: Threshold
        )

        # =====================================================================
        # SECTION 5: Create PyTorch Data Loaders
        # =====================================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,  # Shuffle windows for training randomness
            drop_last=True,  # Drop incomplete final batch
            num_workers=0,  # Single-threaded for debugging simplicity
            pin_memory=torch.cuda.is_available(),  # Speed up GPU transfer
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,  # Sequential order for validation
            drop_last=False,  # Keep all validation data
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # =====================================================================
        # SECTION 6: Print Summary Statistics
        # =====================================================================

        # Get answer statistics from datasets
        train_windows_with_answers = sum(
            1
            for idx in range(len(train_dataset))
            if train_dataset.has_complete_answer(idx)
        )
        val_windows_with_answers = sum(
            1 for idx in range(len(val_dataset)) if val_dataset.has_complete_answer(idx)
        )

        print(f"\n{'=' * 60}")
        print("Data Loader Summary")
        print(f"{'=' * 60}")
        print(f"Training:")
        print(f"  Batches: {len(train_loader):,}")
        print(f"  Total windows: {len(train_dataset):,}")
        print(
            f"  Windows with answers: {train_windows_with_answers:,} "
            f"({train_windows_with_answers / len(train_dataset) * 100:.1f}%)"
        )
        print(f"\nValidation:")
        print(f"  Batches: {len(val_loader):,}")
        print(f"  Total windows: {len(val_dataset):,}")
        print(
            f"  Windows with answers: {val_windows_with_answers:,} "
            f"({val_windows_with_answers / len(val_dataset) * 100:.1f}%)"
        )
        print(f"\nConfiguration:")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Context length: {config['context_length']}")
        print(f"  Stride: {stride}")
        print(f"  Sliding windows: {use_sliding_windows}")
        print(f"{'=' * 60}\n")

        return train_loader, val_loader

    except Exception as loader_creation_error:
        print(f"\n❌ Error creating data loaders: {loader_creation_error}")
        traceback.print_exc()
        raise


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


# class DocumentDatasetFromTokens(Dataset):
#     """
#     Dataset where each file segment becomes ONE training example (padded to context_length).

#     CRITICAL DESIGN CHANGE:
#         - Each file is a complete training chunk (e.g., question + |||answer|||)
#         - Files are typically SHORTER than context_length (e.g., 200-500 tokens)
#         - We PAD each file to context_length (1024 tokens)
#         - NO sliding windows across files
#         - NO combining multiple files

#     This preserves the semantic integrity of each file as one complete training example.

#     Args:
#         tokens (list[int]): Pre-tokenized data with EOS markers between files
#         max_length (int): Context length to pad each file to
#         stride (int): IGNORED - kept for API compatibility but not used
#         verbose (bool): Print statistics
#         eos_token_id (int): Token ID marking file boundaries (default: 257)
#         pad_token_id (int): Token ID for padding (default: 256)
#     """

#     def __init__(
#         self,
#         tokens: list[int],
#         max_length: int,
#         stride: int,  # NOW USED!
#         verbose: bool = True,
#         eos_token_id: int = 257,
#         pad_token_id: int = 256,
#         delimiter_tokens: list[int] = None,  # NEW: Tokenized "|||"
#     ):
#         """
#         Args:
#             delimiter_tokens: Token IDs for "|||" delimiter
#                              Used to detect complete answers in windows
#         """

#         # Store delimiter for answer detection
#         if delimiter_tokens is None:
#             # Default: encode "|||" using tokenizer
#             # You'll pass this from outside
#             self.delimiter_tokens = None
#         else:
#             self.delimiter_tokens = delimiter_tokens
#         """
#         Initialize dataset where each file becomes one padded training example.
#         """
#         super().__init__()

#         try:
#             # Store configuration
#             self.tokens = tokens
#             self.max_length = max_length
#             self.eos_token_id = eos_token_id
#             self.pad_token_id = pad_token_id

#             if verbose:
#                 print(
#                     f"\nInitializing DocumentDatasetFromTokens (one file = one example):"
#                 )
#                 print(f"  Total tokens: {len(tokens):,}")
#                 print(f"  Context length (target): {max_length}")
#                 print(f"  Padding token ID: {pad_token_id}")
#                 print(f"  EOS token ID: {eos_token_id}")
#                 print(
#                     f"  Note: stride parameter ignored (not applicable for file-per-example)"
#                 )

#             # =====================================================================
#             # SECTION 1: Split Tokens by EOS Markers (File Boundaries)
#             # =====================================================================

#             file_segments = []
#             current_segment = []

#             for token_id in tokens:
#                 if token_id == self.eos_token_id:
#                     # End of file - save segment (excluding EOS)
#                     if current_segment:
#                         file_segments.append(current_segment)
#                     current_segment = []
#                 else:
#                     current_segment.append(token_id)

#             # Handle any remaining tokens (file without trailing EOS)
#             if current_segment:
#                 file_segments.append(current_segment)

#             if verbose:
#                 print(f"\nFile segments extracted:")
#                 print(f"  Total files: {len(file_segments)}")

#                 if file_segments:
#                     segment_lengths = [len(seg) for seg in file_segments]
#                     print(f"  Min file length: {min(segment_lengths):,} tokens")
#                     print(f"  Max file length: {max(segment_lengths):,} tokens")
#                     print(
#                         f"  Avg file length: {sum(segment_lengths) / len(segment_lengths):.1f} tokens"
#                     )
#                     print(
#                         f"  Median file length: {sorted(segment_lengths)[len(segment_lengths) // 2]:,} tokens"
#                     )

#             # =====================================================================
#             # SECTION 2: Pad or Truncate Each File to context_length
#             # =====================================================================

#             self.windows = []
#             files_truncated = 0
#             files_padded = 0
#             files_exact_length = 0

#             for file_idx, file_tokens in enumerate(file_segments):
#                 file_length = len(file_tokens)

#                 if file_length > max_length:
#                     # File too long - truncate to max_length
#                     # Keep the first max_length tokens
#                     padded_tokens = file_tokens[:max_length]
#                     files_truncated += 1

#                     if verbose and files_truncated <= 5:  # Show first few warnings
#                         print(
#                             f"  ⚠ File {file_idx}: truncated from {file_length} to {max_length} tokens"
#                         )

#                 elif file_length < max_length:
#                     # File too short - pad to max_length
#                     padding_needed = max_length - file_length
#                     padded_tokens = file_tokens + [self.pad_token_id] * padding_needed
#                     files_padded += 1

#                 else:
#                     # File exactly right length
#                     padded_tokens = file_tokens
#                     files_exact_length += 1

#                 # Add +1 for target (next token prediction)
#                 # For padded sequences, the target after last real token is first pad token
#                 padded_tokens_with_target = padded_tokens + [self.pad_token_id]

#                 self.windows.append(padded_tokens_with_target)

#             if verbose:
#                 print(f"\nPadding/Truncation statistics:")
#                 print(
#                     f"  Files padded: {files_padded} ({files_padded / len(file_segments) * 100:.1f}%)"
#                 )
#                 print(
#                     f"  Files truncated: {files_truncated} ({files_truncated / len(file_segments) * 100:.1f}%)"
#                 )
#                 print(
#                     f"  Files exact length: {files_exact_length} ({files_exact_length / len(file_segments) * 100:.1f}%)"
#                 )
#                 print(f"  Total training examples: {len(self.windows):,}")

#                 if files_truncated > 5:
#                     print(
#                         f"  ⚠ {files_truncated} files were truncated (only showed first 5)"
#                     )

#             # =====================================================================
#             # SECTION 3: Validation
#             # =====================================================================

#             if len(self.windows) == 0:
#                 raise ValueError(
#                     "No training examples created - no valid file segments found"
#                 )

#             # Verify all windows are correct length
#             for window_idx, window in enumerate(self.windows):
#                 if len(window) != max_length + 1:
#                     raise RuntimeError(
#                         f"Window {window_idx} has incorrect length: "
#                         f"expected {max_length + 1}, got {len(window)}"
#                     )

#             if verbose:
#                 print(f"\n{'=' * 60}")
#                 print(f"Dataset Ready:")
#                 print(f"  Total training examples: {len(self.windows):,}")
#                 print(f"  Example length: {max_length} tokens (+ 1 target)")
#                 print(f"  Each example is one complete file (padded if needed)")
#                 print(f"{'=' * 60}")

#         except Exception as dataset_creation_error:
#             print(f"\n❌ Error creating DocumentDatasetFromTokens:")
#             print(f"   {dataset_creation_error}")
#             traceback.print_exc()
#             raise

#         # Split by EOS to get file segments
#         file_segments = self._split_by_eos(tokens, eos_token_id)

#         # Create windows from each file segment
#         self.windows = []
#         self.window_metadata = []  # NEW: Track answer presence

#         for segment_idx, segment_tokens in enumerate(file_segments):
#             # Create sliding windows from this segment
#             segment_windows = self._create_windows_from_segment(
#                 segment_tokens=segment_tokens,
#                 max_length=max_length,
#                 stride=stride,
#                 pad_token_id=pad_token_id,
#             )

#             # For each window, check if it contains complete answer
#             for window in segment_windows:
#                 self.windows.append(window)

#                 # Detect if this window has complete answer (2 delimiters)
#                 has_complete_answer = self._window_has_complete_answer(window)

#                 self.window_metadata.append(
#                     {
#                         "segment_idx": segment_idx,
#                         "has_complete_answer": has_complete_answer,
#                     }
#                 )

#         if verbose:
#             windows_with_answers = sum(
#                 1 for meta in self.window_metadata if meta["has_complete_answer"]
#             )
#             print(f"\nWindow statistics:")
#             print(f"  Total windows: {len(self.windows):,}")
#             print(f"  Windows with complete answers: {windows_with_answers:,}")
#             print(
#                 f"  Windows without answers: {len(self.windows) - windows_with_answers:,}"
#             )

#     def _create_windows_from_segment(
#         self,
#         segment_tokens: list[int],
#         max_length: int,
#         stride: int,
#         pad_token_id: int,
#     ) -> list[list[int]]:
#         """
#         Create overlapping windows from one file segment.

#         Returns:
#             List of windows, each max_length + 1 tokens (for target)
#         """
#         windows = []

#         if len(segment_tokens) <= max_length:
#             # Short segment: one window, padded
#             padding_needed = max_length - len(segment_tokens)
#             padded = segment_tokens + [pad_token_id] * padding_needed
#             windows.append(padded + [pad_token_id])  # +1 for target

#         else:
#             # Long segment: multiple overlapping windows
#             for start_idx in range(0, len(segment_tokens), stride):
#                 window_tokens = segment_tokens[start_idx : start_idx + max_length]

#                 # Pad if last window is short
#                 if len(window_tokens) < max_length:
#                     padding_needed = max_length - len(window_tokens)
#                     window_tokens = window_tokens + [pad_token_id] * padding_needed

#                 # Add target token
#                 if start_idx + max_length < len(segment_tokens):
#                     target_token = segment_tokens[start_idx + max_length]
#                 else:
#                     target_token = pad_token_id

#                 windows.append(window_tokens + [target_token])

#         return windows

#     def _window_has_complete_answer(self, window: list[int]) -> bool:
#         """
#         Check if window contains exactly 2 delimiter sequences (|||).

#         Returns:
#             True if complete answer present (2 delimiters found)
#             False otherwise (0, 1, or >2 delimiters)
#         """
#         if self.delimiter_tokens is None:
#             return False

#         # Count delimiter occurrences in window
#         delimiter_count = 0
#         delimiter_len = len(self.delimiter_tokens)

#         for i in range(len(window) - delimiter_len + 1):
#             if window[i : i + delimiter_len] == self.delimiter_tokens:
#                 delimiter_count += 1

#         # Exactly 2 delimiters = complete answer
#         return delimiter_count == 2

#     def has_complete_answer(self, idx: int) -> bool:
#         """
#         Public API: Check if window at index has complete answer.

#         Use this during training to decide weighted vs. standard loss.
#         """
#         return self.window_metadata[idx]["has_complete_answer"]

#     def __len__(self) -> int:
#         """Return the number of training examples (= number of files)."""
#         return len(self.windows)

#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Get a training example (one padded file).

#         Args:
#             idx (int): Index of the file/example

#         Returns:
#             tuple[torch.Tensor, torch.Tensor]: (input_ids, target_ids)
#                 Both tensors have shape [max_length]

#         Note:
#             Padding tokens in targets will have 0 loss weight automatically
#             by the loss function (they're ignored during training).
#         """
#         try:
#             if idx < 0 or idx >= len(self.windows):
#                 raise IndexError(
#                     f"Index {idx} out of range for dataset with {len(self.windows)} examples"
#                 )

#             window = self.windows[idx]

#             # Input: all tokens except last
#             input_ids = torch.tensor(window[:-1], dtype=torch.long)

#             # Target: all tokens except first (shifted by 1)
#             target_ids = torch.tensor(window[1:], dtype=torch.long)

#             return input_ids, target_ids

#         except Exception as getitem_error:
#             print(f"❌ Error getting item at index {idx}: {getitem_error}")
#             traceback.print_exc()
#             raise


class DocumentDatasetFromTokens(Dataset):
    """
    Hybrid dataset supporting both single-file-per-example and sliding windows.

    Strategy:
        - Files ≤ min_file_length: One padded window per file
        - Files > min_file_length: Multiple overlapping windows per file
        - Windows with 2 delimiters (|||): Marked for weighted validation
        - Windows without complete answer: Standard training

    This preserves semantic integrity while efficiently using data from long files.

    Args:
        tokens (list[int]): Pre-tokenized data with EOS (257) markers between files
        max_length (int): Context length for each training window (e.g., 1024)
        stride (int): Number of tokens to advance between windows (controls overlap)
        verbose (bool): Print detailed statistics during initialization
        eos_token_id (int): Token ID marking file boundaries (default: 257)
        pad_token_id (int): Token ID for padding short sequences (default: 256)
        delimiter_tokens (list[int] | None): Token IDs for "|||" delimiter sequence
                                             Used to detect complete answers in windows
        use_sliding_windows (bool): Enable multi-window mode for long files
        min_file_length_for_windows (int): Files longer than this get multiple windows

    Raises:
        ValueError: If no valid training examples can be created
        RuntimeError: If window creation produces invalid sizes
    """

    def __init__(
        self,
        tokens: list[int],
        max_length: int,
        stride: int,
        verbose: bool = True,
        eos_token_id: int = 257,
        pad_token_id: int = 256,
        delimiter_tokens: list[int] | None = None,
        use_sliding_windows: bool = True,
        min_file_length_for_windows: int = 1024,
    ):
        """
        Initialize hybrid dataset with intelligent windowing strategy.

        Process:
            1. Split concatenated tokens by EOS markers (file boundaries)
            2. For each file segment:
               - If short (≤ threshold): Create one padded window
               - If long (> threshold): Create multiple overlapping windows
            3. For each window: Detect if complete answer present (2 delimiters)
            4. Store windows with metadata for conditional weighting during training
        """
        super().__init__()

        try:
            # ================================================================
            # CONFIGURATION STORAGE
            # ================================================================

            self.tokens = tokens
            self.max_length = max_length
            self.stride = stride
            self.eos_token_id = eos_token_id
            self.pad_token_id = pad_token_id
            self.delimiter_tokens = delimiter_tokens
            self.use_sliding_windows = use_sliding_windows
            self.min_file_length_for_windows = min_file_length_for_windows

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Initializing DocumentDatasetFromTokens (Hybrid Mode)")
                print(f"{'=' * 70}")
                print(f"  Total input tokens: {len(tokens):,}")
                print(f"  Context length: {max_length}")
                print(f"  Stride: {stride} (overlap: {max_length - stride} tokens)")
                print(f"  Padding token ID: {pad_token_id}")
                print(f"  EOS token ID: {eos_token_id}")
                print(f"  Sliding windows enabled: {use_sliding_windows}")
                print(f"  Multi-window threshold: {min_file_length_for_windows} tokens")

                if delimiter_tokens:
                    print(f"  Answer delimiter tokens: {delimiter_tokens}")
                else:
                    print(
                        f"  ⚠ No delimiter tokens provided - answer detection disabled"
                    )

            # ================================================================
            # STEP 1: Split Tokens by EOS Markers (File Boundaries)
            # ================================================================

            file_segments = self._split_tokens_by_eos(
                tokens=tokens, eos_token_id=eos_token_id, verbose=verbose
            )

            # ================================================================
            # STEP 2: Create Windows from Each File Segment
            # ================================================================

            self.windows = []
            self.window_metadata = []  # Track segment index and answer presence

            files_with_single_window = 0
            files_with_multiple_windows = 0
            total_windows_created = 0

            for segment_idx, segment_tokens in enumerate(file_segments):
                segment_length = len(segment_tokens)

                # Decide windowing strategy based on file length
                if (
                    segment_length <= min_file_length_for_windows
                    or not use_sliding_windows
                ):
                    # SHORT FILE or SLIDING DISABLED: One padded window
                    segment_windows = self._create_single_padded_window(
                        segment_tokens=segment_tokens,
                        max_length=max_length,
                        pad_token_id=pad_token_id,
                    )
                    files_with_single_window += 1

                else:
                    # LONG FILE: Multiple overlapping windows
                    segment_windows = self._create_sliding_windows(
                        segment_tokens=segment_tokens,
                        max_length=max_length,
                        stride=stride,
                        pad_token_id=pad_token_id,
                    )
                    files_with_multiple_windows += 1

                # Add windows and metadata
                for window in segment_windows:
                    self.windows.append(window)

                    # Detect if this window contains complete answer (2 delimiters)
                    has_complete_answer = self._window_has_complete_answer(
                        window=window, delimiter_tokens=delimiter_tokens
                    )

                    self.window_metadata.append(
                        {
                            "segment_idx": segment_idx,
                            "has_complete_answer": has_complete_answer,
                        }
                    )

                    total_windows_created += 1

            # ================================================================
            # STEP 3: Validation and Statistics
            # ================================================================

            if len(self.windows) == 0:
                raise ValueError(
                    "No training windows could be created! "
                    "All file segments were empty or invalid."
                )

            # Verify all windows have correct length
            for window_idx, window in enumerate(self.windows):
                if len(window) != max_length + 1:
                    raise RuntimeError(
                        f"Window {window_idx} has incorrect length: "
                        f"expected {max_length + 1}, got {len(window)}"
                    )

            if verbose:
                windows_with_answers = sum(
                    1 for meta in self.window_metadata if meta["has_complete_answer"]
                )

                print(f"\n{'=' * 70}")
                print(f"Dataset Creation Summary")
                print(f"{'=' * 70}")
                print(f"  Total file segments: {len(file_segments)}")
                print(f"  Files with single window: {files_with_single_window}")
                print(f"  Files with multiple windows: {files_with_multiple_windows}")
                print(f"  Total windows created: {total_windows_created:,}")
                print(f"  Windows with complete answers: {windows_with_answers:,}")
                print(
                    f"  Windows without complete answers: {total_windows_created - windows_with_answers:,}"
                )
                print(f"  Window size: {max_length} tokens (+ 1 target)")
                print(f"{'=' * 70}\n")

        except Exception as init_error:
            print(f"\n❌ Error initializing DocumentDatasetFromTokens:")
            print(f"   {init_error}")
            traceback.print_exc()
            raise

    def _split_tokens_by_eos(
        self, tokens: list[int], eos_token_id: int, verbose: bool
    ) -> list[list[int]]:
        """
        Split concatenated token stream into individual file segments.

        EOS tokens (257) mark file boundaries and are excluded from segments.

        Args:
            tokens: Full token stream with EOS markers
            eos_token_id: Token ID representing file boundary
            verbose: Whether to print segment statistics

        Returns:
            List of token lists, one per file segment

        Example:
            Input:  [10, 20, 30, 257, 40, 50, 257, 60]
                     ^^^^^^^^^ EOS  ^^^^^^ EOS
            Output: [[10, 20, 30], [40, 50], [60]]
        """
        file_segments = []
        current_segment = []

        for token_id in tokens:
            if token_id == eos_token_id:
                # End of file boundary - save segment (excluding EOS)
                if current_segment:
                    file_segments.append(current_segment)
                current_segment = []
            else:
                # Regular token - add to current segment
                current_segment.append(token_id)

        # Handle trailing tokens (file without final EOS)
        if current_segment:
            file_segments.append(current_segment)

        if verbose and file_segments:
            segment_lengths = [len(seg) for seg in file_segments]
            print(f"\nFile Segmentation:")
            print(f"  Total segments: {len(file_segments)}")
            print(f"  Min segment length: {min(segment_lengths):,} tokens")
            print(f"  Max segment length: {max(segment_lengths):,} tokens")
            print(
                f"  Median segment length: {sorted(segment_lengths)[len(segment_lengths) // 2]:,} tokens"
            )
            print(
                f"  Average segment length: {sum(segment_lengths) / len(segment_lengths):.1f} tokens"
            )

        return file_segments

    def _create_single_padded_window(
        self,
        segment_tokens: list[int],
        max_length: int,
        pad_token_id: int,
    ) -> list[list[int]]:
        """
        Create one window from a file segment, padding or truncating as needed.

        Used for short files (≤ threshold) or when sliding windows disabled.

        Args:
            segment_tokens: Tokens from one file
            max_length: Target window length
            pad_token_id: Token ID for padding

        Returns:
            List containing one window of max_length + 1 tokens

        Logic:
            - If segment < max_length: Pad with pad_token_id
            - If segment > max_length: Truncate to max_length
            - If segment = max_length: Use as-is
            - Always add +1 token for target (next token prediction)
        """
        segment_length = len(segment_tokens)

        if segment_length > max_length:
            # Truncate long segment
            window_tokens = segment_tokens[:max_length]
        elif segment_length < max_length:
            # Pad short segment
            padding_needed = max_length - segment_length
            window_tokens = segment_tokens + [pad_token_id] * padding_needed
        else:
            # Exact length - no modification needed
            window_tokens = segment_tokens

        # Add target token (for next-token prediction)
        # If segment was padded, target is also padding
        # If segment was truncated, target is the next token from segment
        if segment_length >= max_length:
            target_token = (
                segment_tokens[max_length]
                if len(segment_tokens) > max_length
                else pad_token_id
            )
        else:
            target_token = pad_token_id

        window_with_target = window_tokens + [target_token]

        return [window_with_target]

    def _create_sliding_windows(
        self,
        segment_tokens: list[int],
        max_length: int,
        stride: int,
        pad_token_id: int,
    ) -> list[list[int]]:
        """
        Create multiple overlapping windows from a long file segment.

        Used for files > threshold length when sliding windows enabled.

        Args:
            segment_tokens: Tokens from one file (length > threshold)
            max_length: Length of each window
            stride: Number of tokens to advance between windows
            pad_token_id: Token ID for padding the final window if needed

        Returns:
            List of windows, each max_length + 1 tokens

        Example:
            segment_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 tokens
            max_length = 4
            stride = 2

            Window 1: [0, 1, 2, 3] + target [4]      # Start at 0
            Window 2: [2, 3, 4, 5] + target [6]      # Start at 2 (stride=2)
            Window 3: [4, 5, 6, 7] + target [8]      # Start at 4
            Window 4: [6, 7, 8, 9] + target [PAD]    # Start at 6, last window

        Note:
            - Windows overlap by (max_length - stride) tokens
            - Last window may be shorter and gets padded
            - Ensures all tokens in segment are covered by at least one window
        """
        windows = []
        segment_length = len(segment_tokens)

        # Create sliding windows with specified stride
        window_start_idx = 0

        while window_start_idx < segment_length:
            # Extract window tokens
            window_end_idx = window_start_idx + max_length
            window_tokens = segment_tokens[window_start_idx:window_end_idx]

            # Pad if this is the final window and it's short
            if len(window_tokens) < max_length:
                padding_needed = max_length - len(window_tokens)
                window_tokens = window_tokens + [pad_token_id] * padding_needed

            # Determine target token (next token after window)
            if window_end_idx < segment_length:
                # Window is fully within segment - target is next token
                target_token = segment_tokens[window_end_idx]
            else:
                # Window extends to/past end of segment - target is padding
                target_token = pad_token_id

            # Add window with target
            window_with_target = window_tokens + [target_token]
            windows.append(window_with_target)

            # Advance to next window start position
            window_start_idx += stride

            # Safety check: if stride is very small, prevent infinite loop
            if stride == 0:
                print("⚠ WARNING: stride is 0, breaking to prevent infinite loop")
                break

        return windows

    def _window_has_complete_answer(
        self,
        window: list[int],
        delimiter_tokens: list[int] | None,
    ) -> bool:
        """
        Check if window contains exactly 2 delimiter sequences (complete answer).

        Logic:
            - 0 delimiters: No answer in this window
            - 1 delimiter: Partial answer (split across windows)
            - 2 delimiters: Complete answer (|||answer||| fully present)
            - >2 delimiters: Multiple answers or malformed (treat as no answer)

        Args:
            window: Token sequence to check
            delimiter_tokens: Token IDs representing "|||" delimiter

        Returns:
            True if exactly 2 delimiters found (complete answer present)
            False otherwise

        Example:
            delimiter_tokens = [124, 124, 124]  # "|||" encoded as bytes

            window = [..., 124, 124, 124, 50, 124, 124, 124, ...]
                          ^^^^^^^^^^^^ delimiter 1  ^^^^^^^^^^^^ delimiter 2

            Returns: True (complete answer between delimiters)
        """
        if delimiter_tokens is None or len(delimiter_tokens) == 0:
            # No delimiter specified - cannot detect answers
            return False

        delimiter_length = len(delimiter_tokens)
        delimiter_count = 0

        # Scan window for delimiter occurrences
        for window_idx in range(len(window) - delimiter_length + 1):
            # Check if delimiter sequence matches at this position
            if window[window_idx : window_idx + delimiter_length] == delimiter_tokens:
                delimiter_count += 1

                # Early exit if we find more than 2 (malformed)
                if delimiter_count > 2:
                    return False

        # Exactly 2 delimiters = complete answer present
        return delimiter_count == 2

    def __len__(self) -> int:
        """Return the total number of training windows in the dataset."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training window and its target for next-token prediction.

        Args:
            idx: Window index (0 to len(dataset)-1)

        Returns:
            tuple: (input_ids, target_ids)
                - input_ids: Tokens for model input [max_length]
                - target_ids: Tokens for prediction target [max_length]
                  (input shifted by 1 position)

        Raises:
            IndexError: If idx out of range

        Example:
            If window = [10, 20, 30, 40, 50]
            Then:
                input_ids  = [10, 20, 30, 40]  # All except last
                target_ids = [20, 30, 40, 50]  # All except first
        """
        try:
            if idx < 0 or idx >= len(self.windows):
                raise IndexError(
                    f"Window index {idx} out of range "
                    f"(dataset has {len(self.windows)} windows)"
                )

            window = self.windows[idx]

            # Split into input and target (autoregressive next-token prediction)
            input_ids = torch.tensor(window[:-1], dtype=torch.long)
            target_ids = torch.tensor(window[1:], dtype=torch.long)

            return input_ids, target_ids

        except Exception as getitem_error:
            print(f"❌ Error retrieving window {idx}: {getitem_error}")
            traceback.print_exc()
            raise

    def has_complete_answer(self, idx: int) -> bool:
        """
        Check if window at index contains a complete answer (for weighted validation).

        Use this during training loop to decide whether to apply weighted loss.

        Args:
            idx: Window index

        Returns:
            True if window contains exactly 2 delimiters (|||answer|||)
            False otherwise

        Raises:
            IndexError: If idx out of range
        """
        if idx < 0 or idx >= len(self.window_metadata):
            raise IndexError(
                f"Metadata index {idx} out of range "
                f"(dataset has {len(self.window_metadata)} entries)"
            )

        return self.window_metadata[idx]["has_complete_answer"]

    def get_window_info(self, idx: int) -> dict:
        """
        Get detailed information about a specific window (for debugging).

        Args:
            idx: Window index

        Returns:
            dict with keys:
                - window_idx: The requested index
                - segment_idx: Which file segment this window came from
                - has_complete_answer: Whether complete answer present
                - window_length: Length of window (should be max_length + 1)
        """
        if idx < 0 or idx >= len(self.windows):
            raise IndexError(f"Window index {idx} out of range")

        return {
            "window_idx": idx,
            "segment_idx": self.window_metadata[idx]["segment_idx"],
            "has_complete_answer": self.window_metadata[idx]["has_complete_answer"],
            "window_length": len(self.windows[idx]),
        }


def evaluate_model_with_smart_weighting(
    model,
    data_loader,
    device,
    tokenizer,
    config,
    current_step,
):
    """
    Evaluate with conditional weighting based on answer presence.
    """
    model.eval()
    total_weighted_loss = 0.0
    total_standard_loss = 0.0
    weighted_batches = 0
    standard_batches = 0

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            # Check if this batch contains windows with complete answers
            batch_has_answers = [
                data_loader.dataset.has_complete_answer(
                    batch_idx * data_loader.batch_size + i
                )
                for i in range(len(input_batch))
            ]

            if any(batch_has_answers):
                # Use weighted loss for windows with complete answers
                loss = calculate_weighted_loss_delimited_target(
                    input_batch=input_batch,
                    target_batch=target_batch,
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    # ... other params ...
                )
                total_weighted_loss += loss.item()
                weighted_batches += 1
            else:
                # Use standard loss for windows without answers
                loss = calculate_loss(input_batch, target_batch, model, device)
                total_standard_loss += loss.item()
                standard_batches += 1

    # Average the losses
    avg_weighted = total_weighted_loss / weighted_batches if weighted_batches > 0 else 0
    avg_standard = total_standard_loss / standard_batches if standard_batches > 0 else 0

    model.train()

    # Return combined loss (or report separately)
    return (
        (avg_weighted + avg_standard) / 2
        if (weighted_batches + standard_batches) > 0
        else float("nan")
    )


# class DocumentDatasetFromTokens(Dataset):
#     """
#     Lightweight dataset that works directly with pre-tokenized data.

#     CRITICAL DESIGN: Respects file boundaries - windows NEVER cross from one file
#     to another. Each file ends with an EOS token (257) which marks the boundary.

#     This is essential for training where each file represents a complete,
#     semantically meaningful chunk (e.g., a question-answer pair with |||delimiters|||).

#     Design Assumption:
#         Input tokens have EOS tokens (257) placed between files to mark boundaries.
#         Windows are created within each file segment, never crossing boundaries.

#     Args:
#         tokens (list[int]): Pre-tokenized data with EOS markers between files
#         max_length (int): Maximum sequence length for model input
#         stride (int): Stride between chunks (controls overlap within files)
#         verbose (bool): Print statistics about dataset creation
#         eos_token_id (int): Token ID marking file boundaries (default: 257)

#     Returns:
#         Dataset: PyTorch Dataset yielding (input_ids, target_ids) tuples

#     Raises:
#         ValueError: If tokens list is empty or shorter than max_length
#         ValueError: If no valid windows can be created from the data
#     """

#     def __init__(
#         self,
#         tokens: list[int],
#         max_length: int,
#         stride: int,
#         verbose: bool = True,
#         eos_token_id: int = 257,
#     ):
#         """
#         Initialize dataset from pre-tokenized data with file boundary preservation.

#         Process:
#             1. Find all EOS token positions (file boundaries)
#             2. Split tokens into file segments between EOS markers
#             3. Create sliding windows WITHIN each segment (never crossing)
#             4. Store windows for __getitem__ access
#         """
#         super().__init__()

#         try:
#             # Store configuration
#             self.tokens = tokens
#             self.max_length = max_length
#             self.stride = stride
#             self.eos_token_id = eos_token_id

#             # =====================================================================
#             # SECTION 1: Input Validation
#             # =====================================================================

#             if not tokens:
#                 raise ValueError("Cannot create dataset from empty token list")

#             if not isinstance(tokens, list):
#                 raise TypeError(f"tokens must be a list, got {type(tokens).__name__}")

#             if max_length <= 0:
#                 raise ValueError(f"max_length must be positive, got {max_length}")

#             if stride <= 0:
#                 raise ValueError(f"stride must be positive, got {stride}")

#             if verbose:
#                 print(f"\nInitializing DocumentDatasetFromTokens:")
#                 print(f"  Total tokens: {len(tokens):,}")
#                 print(f"  Max length: {max_length}")
#                 print(f"  Stride: {stride}")
#                 print(f"  EOS token ID: {eos_token_id}")

#             # =====================================================================
#             # SECTION 2: Identify File Boundaries (EOS Token Positions)
#             # =====================================================================

#             # Find all EOS token positions - these mark file boundaries
#             file_boundary_positions = []

#             for token_idx, token_id in enumerate(tokens):
#                 if token_id == self.eos_token_id:
#                     file_boundary_positions.append(token_idx)

#             if verbose:
#                 print(f"\nFile boundary detection:")
#                 print(
#                     f"  Found {len(file_boundary_positions)} EOS tokens (file boundaries)"
#                 )

#                 if len(file_boundary_positions) > 0:
#                     print(
#                         f"  First boundary at position {file_boundary_positions[0]:,}"
#                     )
#                     print(
#                         f"  Last boundary at position {file_boundary_positions[-1]:,}"
#                     )
#                 else:
#                     print(
#                         f"  ⚠ WARNING: No EOS tokens found - treating entire sequence as one file"
#                     )

#             # =====================================================================
#             # SECTION 3: Extract File Segments Between Boundaries
#             # =====================================================================

#             file_segments = []
#             segment_start_position = 0

#             # Process each boundary to extract segments
#             for boundary_position in file_boundary_positions:
#                 # Extract segment from start to EOS token (EXCLUDE EOS from segment)
#                 # Rationale: EOS is a separator, not part of the file content
#                 segment_tokens = tokens[segment_start_position:boundary_position]

#                 # Only keep non-empty segments
#                 if len(segment_tokens) > 0:
#                     file_segments.append(segment_tokens)

#                 # Next segment starts after this EOS token
#                 segment_start_position = boundary_position + 1

#             # Handle any remaining tokens after the last EOS
#             if segment_start_position < len(tokens):
#                 remaining_segment = tokens[segment_start_position:]
#                 if remaining_segment:
#                     file_segments.append(remaining_segment)

#             # If no boundaries found, treat entire sequence as one segment
#             if len(file_segments) == 0 and len(tokens) > 0:
#                 if verbose:
#                     print(
#                         f"  No segments created from boundaries - using full token sequence"
#                     )
#                 file_segments.append(tokens)

#             if verbose:
#                 print(f"\nFile segments:")
#                 print(f"  Total segments: {len(file_segments)}")

#                 if file_segments:
#                     segment_lengths = [len(seg) for seg in file_segments]
#                     print(f"  Min segment length: {min(segment_lengths):,} tokens")
#                     print(f"  Max segment length: {max(segment_lengths):,} tokens")
#                     print(
#                         f"  Avg segment length: {sum(segment_lengths) / len(segment_lengths):.1f} tokens"
#                     )

#                     # Count segments that are too short
#                     too_short = sum(
#                         1 for seg_len in segment_lengths if seg_len < max_length + 1
#                     )
#                     if too_short > 0:
#                         print(
#                             f"  ⚠ {too_short} segments too short for window creation (< {max_length + 1} tokens)"
#                         )

#             # =====================================================================
#             # SECTION 4: Create Windows WITHIN Each File Segment
#             # =====================================================================

#             self.windows = []
#             self.window_segment_mapping = []  # Track which segment each window came from

#             window_creation_start = time.perf_counter()

#             total_segments_processed = 0
#             total_segments_skipped = 0

#             for segment_idx, segment_tokens in enumerate(file_segments):
#                 # Check if segment is long enough for at least one window
#                 # Need max_length + 1 because we create (input, target) pairs
#                 if len(segment_tokens) < max_length + 1:
#                     if verbose:
#                         print(
#                             f"  Segment {segment_idx}: SKIPPED "
#                             f"(only {len(segment_tokens)} tokens, need {max_length + 1})"
#                         )
#                     total_segments_skipped += 1
#                     continue

#                 # Create sliding windows WITHIN this segment
#                 segment_windows_count = 0

#                 # Slide window through segment with specified stride
#                 for window_start_idx in range(
#                     0, len(segment_tokens) - max_length, stride
#                 ):
#                     # Extract window of max_length + 1 tokens
#                     # (+1 because we need the next token as prediction target)
#                     window_end_idx = window_start_idx + max_length + 1
#                     window = segment_tokens[window_start_idx:window_end_idx]

#                     # Verify window is complete
#                     if len(window) == max_length + 1:
#                         self.windows.append(window)
#                         self.window_segment_mapping.append(segment_idx)
#                         segment_windows_count += 1
#                     else:
#                         # This should rarely happen due to our range calculation
#                         # but handle it gracefully
#                         if verbose:
#                             print(
#                                 f"    ⚠ Incomplete window at position {window_start_idx} "
#                                 f"({len(window)} tokens), skipping"
#                             )

#                 if segment_windows_count > 0:
#                     total_segments_processed += 1

#                     if verbose:
#                         coverage_pct = (
#                             (segment_windows_count * stride + max_length)
#                             / len(segment_tokens)
#                             * 100
#                         )
#                         print(
#                             f"  Segment {segment_idx}: {segment_windows_count} windows "
#                             f"from {len(segment_tokens)} tokens "
#                             f"(coverage: {coverage_pct:.1f}%)"
#                         )
#                 else:
#                     if verbose:
#                         print(
#                             f"  Segment {segment_idx}: No windows created "
#                             f"(length {len(segment_tokens)} barely meets minimum)"
#                         )

#             window_creation_time = time.perf_counter() - window_creation_start

#             # =====================================================================
#             # SECTION 5: Final Validation and Statistics
#             # =====================================================================

#             if len(self.windows) == 0:
#                 raise ValueError(
#                     f"No training windows could be created! "
#                     f"Processed {len(file_segments)} segments, but all were too short "
#                     f"or didn't yield valid windows. "
#                     f"Check that your files contain at least {max_length + 1} tokens each."
#                 )

#             if verbose:
#                 overlap_percent = (
#                     ((max_length - stride) / max_length * 100) if max_length > 0 else 0
#                 )

#                 print(f"\n{'=' * 60}")
#                 print(f"Dataset Creation Summary:")
#                 print(f"{'=' * 60}")
#                 print(f"  Input tokens: {len(tokens):,}")
#                 print(f"  File segments: {len(file_segments)}")
#                 print(f"  Segments processed: {total_segments_processed}")
#                 print(f"  Segments skipped: {total_segments_skipped}")
#                 print(f"  Total windows created: {len(self.windows):,}")
#                 print(f"  Window size: {max_length} tokens")
#                 print(f"  Stride: {stride} tokens")
#                 print(f"  Overlap: {overlap_percent:.1f}%")
#                 print(f"  Creation time: {window_creation_time:.2f} seconds")

#                 # Calculate token coverage
#                 total_segment_tokens = sum(len(seg) for seg in file_segments)
#                 effective_tokens_covered = min(
#                     total_segment_tokens, len(self.windows) * stride + max_length
#                 )
#                 coverage_pct = (
#                     (effective_tokens_covered / total_segment_tokens * 100)
#                     if total_segment_tokens > 0
#                     else 0
#                 )
#                 print(f"  Token coverage: {coverage_pct:.1f}%")
#                 print(f"{'=' * 60}")

#         except Exception as dataset_creation_error:
#             print(f"\n❌ Error creating DocumentDatasetFromTokens:")
#             print(f"   {dataset_creation_error}")
#             traceback.print_exc()
#             raise

#     def __len__(self) -> int:
#         """
#         Return the number of training windows.

#         Returns:
#             int: Number of windows in the dataset
#         """
#         return len(self.windows)

#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Get a training sample at the specified index.

#         Args:
#             idx (int): Index of the training window to retrieve

#         Returns:
#             tuple[torch.Tensor, torch.Tensor]: (input_ids, target_ids)
#                 - input_ids: Token IDs for model input [max_length]
#                 - target_ids: Token IDs for prediction target [max_length]
#                   (same as input_ids shifted by 1 position)

#         Raises:
#             IndexError: If idx is out of range

#         Note:
#             For autoregressive language modeling, the target is always
#             the input shifted by one position (predict next token).

#             Example:
#                 If window = [10, 20, 30, 40, 50]
#                 Then input_ids  = [10, 20, 30, 40]  (all except last)
#                 And  target_ids = [20, 30, 40, 50]  (all except first)
#         """
#         try:
#             # Bounds checking
#             if idx < 0 or idx >= len(self.windows):
#                 raise IndexError(
#                     f"Index {idx} out of range for dataset with {len(self.windows)} windows"
#                 )

#             # Retrieve the window
#             window = self.windows[idx]

#             # Verify window integrity
#             if len(window) != self.max_length + 1:
#                 raise RuntimeError(
#                     f"Window at index {idx} has incorrect length: "
#                     f"expected {self.max_length + 1}, got {len(window)}"
#                 )

#             # Split into input and target (next token prediction)
#             # Input is all tokens except the last
#             input_ids = torch.tensor(window[:-1], dtype=torch.long)

#             # Target is all tokens except the first (shifted by 1)
#             target_ids = torch.tensor(window[1:], dtype=torch.long)

#             return input_ids, target_ids

#         except IndexError:
#             # Re-raise IndexError with more context
#             raise
#         except Exception as getitem_error:
#             print(f"❌ Error getting item at index {idx}: {getitem_error}")
#             traceback.print_exc()
#             raise

#     def get_segment_info(self, window_idx: int) -> dict:
#         """
#         Get information about which file segment a window came from.

#         Useful for debugging or analysis to trace windows back to their source files.

#         Args:
#             window_idx (int): Window index

#         Returns:
#             dict: Information about the segment this window came from
#                 - segment_idx: Which file segment (int)
#                 - window_idx: The window index queried (int)

#         Raises:
#             IndexError: If window_idx is out of range
#         """
#         try:
#             if window_idx < 0 or window_idx >= len(self.windows):
#                 raise IndexError(
#                     f"Window index {window_idx} out of range "
#                     f"(dataset has {len(self.windows)} windows)"
#                 )

#             return {
#                 "window_idx": window_idx,
#                 "segment_idx": self.window_segment_mapping[window_idx],
#             }

#         except Exception as info_error:
#             print(f"Error getting segment info for window {window_idx}: {info_error}")
#             traceback.print_exc()
#             raise


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


# def evaluate_model(
#     model,
#     data_loader,
#     device,
#     num_batches=None,
#     tokenizer=None,
#     config=None,
#     current_step=0,
#     use_weighted_loss=False,
# ):
#     """
#     Evaluate model on data loader with optional weighted loss.

#     Args:
#         model: Model to evaluate
#         data_loader: Data loader for evaluation
#         device: Device to use
#         num_batches: Maximum number of batches to evaluate (None = all)
#         tokenizer: ByteTokenizer instance (required for weighted loss)
#         config: Training configuration dict (required for weighted loss)
#         current_step: Current global training step (for grace period logic)
#         use_weighted_loss: Whether to use delimiter-weighted loss

#     Returns:
#         float: Average loss across evaluated batches
#     """
#     try:
#         model.eval()
#         total_loss = 0.0

#         if num_batches is None:
#             num_batches = len(data_loader)
#         else:
#             num_batches = min(num_batches, len(data_loader))

#         # Determine which loss function to use
#         if use_weighted_loss:
#             # Validate required parameters for weighted loss
#             if tokenizer is None:
#                 raise ValueError(
#                     "Tokenizer required for weighted validation loss. "
#                     "Pass tokenizer parameter to evaluate_model()."
#                 )
#             if config is None:
#                 raise ValueError(
#                     "Config required for weighted validation loss. "
#                     "Pass config parameter to evaluate_model()."
#                 )

#             print(f"  Using weighted validation loss (step {current_step})")

#         with torch.no_grad():
#             for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
#                 if batch_idx >= num_batches:
#                     break

#                 # Choose loss calculation method
#                 if use_weighted_loss:
#                     # Calculate weighted loss focusing on delimited answers
#                     loss = calculate_weighted_loss_delimited_target(
#                         input_batch=input_batch,
#                         target_batch=target_batch,
#                         model=model,
#                         device=device,
#                         target_delimiter_string=config["target_delimiter_string"],
#                         answer_weight_multiplier=config["answer_weight_multiplier"],
#                         pad_token_id=tokenizer.PAD_ID,
#                         tokenizer=tokenizer,
#                         pct_validations_weighted=config["pct_validations_weighted"],
#                         steps_grace_period_before_weighting=config[
#                             "steps_grace_period_before_weighting"
#                         ],
#                         current_step=current_step,
#                         random_seed=config["validation_random_seed"],
#                     )
#                 else:
#                     # Standard uniform cross-entropy loss
#                     loss = calculate_loss(input_batch, target_batch, model, device)

#                 total_loss += loss.item()

#         model.train()

#         average_loss = total_loss / num_batches if num_batches > 0 else float("nan")

#         return average_loss

#     except Exception as evaluation_error:
#         print(f"Error during model evaluation: {evaluation_error}")
#         traceback.print_exc()
#         raise


def evaluate_model(
    model,
    data_loader,
    device,
    num_batches=None,
    tokenizer=None,
    config=None,
    current_step=0,
    use_weighted_loss=False,
):
    """
    Evaluate model on data loader with intelligent conditional weighted loss.

    HYBRID WEIGHTING STRATEGY:
        - Windows WITH complete answers (2 delimiters): Use weighted loss
        - Windows WITHOUT complete answers: Use standard loss
        - Mixed batches: Apply appropriate loss per window, then average

    This focuses validation metrics on answer prediction quality while still
    evaluating general language modeling on context/question tokens.

    Args:
        model: Model to evaluate
        data_loader: DataLoader with dataset supporting has_complete_answer()
        device: Device to use ('cuda' or 'cpu')
        num_batches: Maximum number of batches to evaluate (None = all batches)
        tokenizer: ByteTokenizer instance (required for weighted loss)
        config: Training configuration dict with weighted validation params
        current_step: Current global training step (for grace period logic)
        use_weighted_loss: Master switch for weighted validation mode

    Returns:
        float: Average loss across evaluated batches

    Note:
        The returned loss is a weighted average where windows with answers
        contribute heavily-weighted loss values, while windows without answers
        contribute standard uniform loss values. This provides a single metric
        that emphasizes answer quality while still considering overall performance.
    """
    try:
        model.eval()

        # Separate tracking for weighted and standard losses
        total_weighted_loss = 0.0  # Loss from windows with answers
        total_standard_loss = 0.0  # Loss from windows without answers
        num_weighted_windows = 0
        num_standard_windows = 0

        # Determine how many batches to evaluate
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        # =====================================================================
        # VALIDATION: Check if weighted loss is properly configured
        # =====================================================================

        if use_weighted_loss:
            # Validate required parameters for weighted loss
            if tokenizer is None:
                raise ValueError(
                    "Tokenizer required for weighted validation loss. "
                    "Pass tokenizer parameter to evaluate_model()."
                )
            if config is None:
                raise ValueError(
                    "Config required for weighted validation loss. "
                    "Pass config parameter to evaluate_model()."
                )

            # Check if dataset supports answer detection
            if not hasattr(data_loader.dataset, "has_complete_answer"):
                raise AttributeError(
                    "Dataset must have has_complete_answer() method for weighted validation. "
                    "Ensure you're using DocumentDatasetFromTokens with delimiter_tokens configured."
                )

            print(f"  Using smart weighted validation (step {current_step})")
            print(f"  → Windows with answers: weighted loss (focus on answer quality)")
            print(
                f"  → Windows without answers: standard loss (general language modeling)"
            )

        # =====================================================================
        # EVALUATION LOOP: Process batches with smart conditional weighting
        # =====================================================================

        with torch.no_grad():
            for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break

                # Get batch size (may be smaller for last batch)
                current_batch_size = input_batch.size(0)

                # =========================================================
                # STEP 1: Determine which windows in batch have answers
                # =========================================================

                # Calculate global window indices for this batch
                # This maps batch positions to dataset indices
                batch_start_idx = batch_idx * data_loader.batch_size

                window_has_answer = []
                for batch_position in range(current_batch_size):
                    global_window_idx = batch_start_idx + batch_position

                    # Check if this window has complete answer
                    try:
                        has_answer = data_loader.dataset.has_complete_answer(
                            global_window_idx
                        )
                        window_has_answer.append(has_answer)
                    except IndexError:
                        # Safety: if index out of range, treat as no answer
                        window_has_answer.append(False)

                # Count answer windows in this batch
                num_answer_windows_in_batch = sum(window_has_answer)
                num_non_answer_windows_in_batch = (
                    current_batch_size - num_answer_windows_in_batch
                )

                # =========================================================
                # STEP 2: Calculate loss based on batch composition
                # =========================================================

                if use_weighted_loss and num_answer_windows_in_batch > 0:
                    # CASE A: Batch contains windows with answers
                    # Use weighted loss (focuses on answer tokens)

                    loss = calculate_weighted_loss_delimited_target(
                        input_batch=input_batch,
                        target_batch=target_batch,
                        model=model,
                        device=device,
                        target_delimiter_string=config["target_delimiter_string"],
                        answer_weight_multiplier=config["answer_weight_multiplier"],
                        pad_token_id=tokenizer.PAD_ID,
                        tokenizer=tokenizer,
                        pct_validations_weighted=config["pct_validations_weighted"],
                        steps_grace_period_before_weighting=config[
                            "steps_grace_period_before_weighting"
                        ],
                        current_step=current_step,
                        random_seed=config["validation_random_seed"],
                    )

                    # Track as weighted loss
                    total_weighted_loss += loss.item() * num_answer_windows_in_batch
                    num_weighted_windows += num_answer_windows_in_batch

                    # If batch has mixed windows (some with, some without answers),
                    # also calculate standard loss for the non-answer windows
                    if num_non_answer_windows_in_batch > 0:
                        standard_loss = calculate_loss(
                            input_batch, target_batch, model, device
                        )
                        total_standard_loss += (
                            standard_loss.item() * num_non_answer_windows_in_batch
                        )
                        num_standard_windows += num_non_answer_windows_in_batch

                else:
                    # CASE B: Batch contains only windows without answers
                    # OR weighted loss disabled
                    # Use standard uniform cross-entropy loss

                    loss = calculate_loss(input_batch, target_batch, model, device)

                    # Track as standard loss
                    total_standard_loss += loss.item() * current_batch_size
                    num_standard_windows += current_batch_size

        # =====================================================================
        # STEP 3: Calculate final average loss
        # =====================================================================

        model.train()

        # Calculate weighted average of both loss types
        total_windows = num_weighted_windows + num_standard_windows

        if total_windows == 0:
            # Edge case: no windows processed
            return float("nan")

        if num_weighted_windows > 0 and num_standard_windows > 0:
            # Mixed evaluation: combine both loss types
            average_loss = (total_weighted_loss + total_standard_loss) / total_windows

            # Optional: print breakdown for debugging
            if use_weighted_loss:
                avg_weighted = total_weighted_loss / num_weighted_windows
                avg_standard = total_standard_loss / num_standard_windows
                print(
                    f"  Validation breakdown: "
                    f"{num_weighted_windows} windows w/ answers (loss: {avg_weighted:.4f}), "
                    f"{num_standard_windows} windows w/o answers (loss: {avg_standard:.4f})"
                )

        elif num_weighted_windows > 0:
            # Only weighted windows evaluated
            average_loss = total_weighted_loss / num_weighted_windows
            if use_weighted_loss:
                print(
                    f"  All {num_weighted_windows} validation windows had answers (weighted)"
                )

        else:
            # Only standard windows evaluated
            average_loss = total_standard_loss / num_standard_windows
            if use_weighted_loss:
                print(
                    f"  All {num_standard_windows} validation windows had no answers (standard)"
                )

        return average_loss

    except Exception as evaluation_error:
        print(f"❌ Error during model evaluation: {evaluation_error}")
        traceback.print_exc()
        raise


def train_model(
    model,
    train_loader,
    val_loader,
    config,
    device,
    output_dir,
    training_state,
    tokenizer,  # ADD THIS PARAMETER
):
    """
    Main training loop for Perseid model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to use
        output_dir: Directory to save outputs
        training_state: Dict with resume information (global_step, optimizer_state, etc.)
        tokenizer: ByteTokenizer instance (NEW - required for weighted validation)

    Returns:
        dict: Training history
    """
    try:
        print(f"\n{'=' * 60}")
        print("Starting Training")
        print(f"{'=' * 60}")

        # Print weighted validation status
        if config.get("use_weighted_validation", False):
            print(f"\n⚡ Weighted Validation ENABLED:")
            print(f"  - Delimiter: '{config['target_delimiter_string']}'")
            print(f"  - Answer weight: {config['answer_weight_multiplier']}x")
            print(
                f"  - Grace period: {config['steps_grace_period_before_weighting']} steps"
            )
            print(
                f"  - Stochastic weighting: {config['pct_validations_weighted']}% of validations"
            )
        else:
            print(f"\n📊 Standard Validation (uniform token weighting)")

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
        def get_lr(step):
            """Learning rate schedule with warmup."""
            if step < config["warmup_steps"]:
                return step / config["warmup_steps"]
            else:
                progress = (step - config["warmup_steps"]) / (
                    total_steps - config["warmup_steps"]
                )
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        # Restore scheduler state if resuming
        if training_state["scheduler_state"] is not None:
            scheduler.load_state_dict(training_state["scheduler_state"])
            print("  ✓ Restored scheduler state")

        # Training state
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "step": [],
            "weighted_validation_used": [],  # NEW: Track when weighted validation was used
        }

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

        # Training loop
        for epoch in range(start_epoch, config["num_epochs"]):
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            print(f"{'=' * 40}")

            model.train()
            epoch_loss = 0
            epoch_tokens = 0

            for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                # Calculate loss (standard training loss - no weighting here)
                loss = calculate_loss(input_batch, target_batch, model, device)
                loss = loss / config["gradient_accumulation_steps"]
                loss.backward()

                epoch_loss += loss.item() * config["gradient_accumulation_steps"]
                epoch_tokens += input_batch.numel()

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

                    # =========================================================
                    # MODIFIED SECTION: Periodic evaluation with weighted loss
                    # =========================================================
                    if global_step % config["eval_every"] == 0:
                        # Determine if we should use weighted validation
                        use_weighted_val = config.get("use_weighted_validation", False)

                        # Calculate validation loss (weighted or standard)
                        val_loss = evaluate_model(
                            model=model,
                            data_loader=val_loader,
                            device=device,
                            num_batches=config["eval_batches"],
                            tokenizer=tokenizer,  # NEW: Pass tokenizer
                            config=config,  # NEW: Pass config
                            current_step=global_step,  # NEW: Pass current step
                            use_weighted_loss=use_weighted_val,  # NEW: Enable weighted loss
                        )

                        train_loss = epoch_loss / (batch_idx + 1)
                        current_lr = scheduler.get_last_lr()[0]

                        # Record history
                        history["train_loss"].append(train_loss)
                        history["val_loss"].append(val_loss)
                        history["learning_rates"].append(current_lr)
                        history["step"].append(global_step)
                        history["weighted_validation_used"].append(
                            use_weighted_val
                        )  # NEW

                        # Print with indicator if weighted validation was used
                        weighted_indicator = "⚡" if use_weighted_val else "📊"

                        print(
                            f"{weighted_indicator} Step {global_step:5d} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens: {tokens_seen:,}"
                        )

                        # Save best model
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
                                epoch,  # Add epoch parameter
                                tokens_seen,  # Add tokens_seen parameter
                            )
                            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

                    # Periodic checkpoint
                    if global_step % config["save_every"] == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            val_loss,
                            output_dir,
                            f"step_{global_step}",
                            epoch,  # Add epoch parameter
                            tokens_seen,  # Add tokens_seen parameter
                        )

            # End of epoch evaluation
            avg_epoch_loss = epoch_loss / len(train_loader)

            # Use weighted validation for end-of-epoch evaluation
            use_weighted_val = config.get("use_weighted_validation", False)
            val_loss = evaluate_model(
                model=model,
                data_loader=val_loader,
                device=device,
                tokenizer=tokenizer,
                config=config,
                current_step=global_step,
                use_weighted_loss=use_weighted_val,
            )

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")
            if use_weighted_val:
                print(f"  (Using weighted validation ⚡)")

        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Total tokens seen: {tokens_seen:,}")

        return history

    except Exception as training_error:
        print(f"Error during training: {training_error}")
        traceback.print_exc()
        raise


def save_checkpoint(
    model, optimizer, scheduler, step, val_loss, output_dir, tag, epoch=0, tokens_seen=0
):
    """
    Save model checkpoint with all training state.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        val_loss: Current validation loss
        output_dir: Output directory
        tag: Checkpoint tag (e.g., "best", "step_1000")
        epoch: Current epoch number (NEW)
        tokens_seen: Total tokens processed (NEW)
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / f"checkpoint_{tag}.pth"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "val_loss": val_loss,
                "model_config": model.cfg,
                "epoch": epoch,  # NEW
                "tokens_seen": tokens_seen,  # NEW
            },
            checkpoint_path,
        )

    except Exception as checkpoint_save_error:
        print(f"Error saving checkpoint: {checkpoint_save_error}")
        traceback.print_exc()


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


# def main():
#     """
#     Main training pipeline for Perseid document training.
#     """
#     try:
#         print(f"\n{'=' * 60}")
#         print("Perseid Document Training Pipeline")
#         print(f"{'=' * 60}")
#         print(f"Experiment: {EXPERIMENT_NAME}")
#         print(f"Output directory: {OUTPUT_DIR}")

#         # Create output directory
#         output_dir = Path(OUTPUT_DIR)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         # Set random seeds for reproducibility
#         torch.manual_seed(42)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(42)

#         # 1. Load document
#         print(f"\n{'=' * 40}")
#         print("Step 1: Loading Document")
#         print(f"{'=' * 40}")
#         document_text = load_document(DOCUMENT_PATH)

#         # 2. Setup model and tokenizer
#         print(f"\n{'=' * 40}")
#         print("Step 2: Setting Up Model")
#         print(f"{'=' * 40}")

#         ###############################
#         # Setup ByteTokenizer
#         ###############################

#         tokenizer = setup_tokenizer()

#         # Initialize model
#         model, model_config, training_state = setup_model(
#             MODEL_SIZE,
#             MODEL_STRATEGY,
#             TRAINING_CONFIG,
#             DEVICE,
#             OUTPUT_DIR,
#             TRAINING_MODE,
#             CHECKPOINT_PATH,
#         )

#         # 3. Create data loaders
#         print(f"\n{'=' * 40}")
#         print("Step 3: Preparing Data")
#         print(f"{'=' * 40}")
#         train_loader, val_loader = create_data_loaders(
#             document_text, tokenizer, TRAINING_CONFIG, train_ratio=TRAIN_VAL_SPLIT
#         )

#         # 4. Train model
#         print(f"\n{'=' * 40}")
#         print("Step 4: Training Model")
#         print(f"{'=' * 40}")
#         history = train_model(
#             model,
#             train_loader,
#             val_loader,
#             TRAINING_CONFIG,
#             DEVICE,
#             output_dir,
#             training_state,
#         )

#         # 5. Save results
#         print(f"\n{'=' * 40}")
#         print("Step 5: Saving Results")
#         print(f"{'=' * 40}")
#         save_training_results(model, model_config, history, output_dir)

#         print(f"\n{'=' * 60}")
#         print("Training Pipeline Complete!")

#         # 5.5 Generate sample text with trained model
#         print(f"\n{'=' * 40}")
#         print("Step 5.5: Sample Generation")
#         print(f"{'=' * 40}")

#         test_prompts = [
#             "Once upon a time",
#             "The meaning of life is",
#             "In the beginning",
#         ]
#         for prompt in test_prompts:
#             # output = generate_text_simple(model, tokenizer, prompt, max_new_tokens=50)
#             output = generate_text_simple(
#                 model, tokenizer, prompt, max_new_tokens=50, device=DEVICE
#             )
#             print(f"Prompt: '{prompt}'")
#             print(f"Output: {output}\n")

#         print(f"{'=' * 60}")
#         print(f"Model and results saved to: {output_dir}")

#         return model, history

#     except Exception as e:
#         print(f"\n{'=' * 60}")
#         print("Training Pipeline Failed")
#         print(f"{'=' * 60}")
#         print(f"Error: {e}")
#         traceback.print_exc()
#         raise


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


"""
calculate_weighted_loss_delimited_target.py

Weighted validation loss calculation that focuses learning on delimited answer tokens.

This module provides a loss function that assigns heavy weights to tokens appearing
between delimiter markers (e.g., |||answer|||) while maintaining normal weights for
other content and masking padding tokens.

Usage:
    from calculate_weighted_loss_delimited_target import calculate_weighted_loss_delimited_target

    loss = calculate_weighted_loss_delimited_target(
        input_batch=input_ids,
        target_batch=target_ids,
        model=model,
        device=device,
        tokenizer=tokenizer,
        current_step=global_step
    )
"""

import sys
import traceback
from collections.abc import Sequence
import torch
import torch.nn as nn


def calculate_weighted_loss_delimited_target(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    target_delimiter_string: str = "|||",
    answer_weight_multiplier: float = 10.0,
    pad_token_id: int = 0,
    tokenizer=None,  # ByteTokenizer instance
    pct_validations_weighted: int = 100,
    steps_grace_period_before_weighting: int = 100,
    current_step: int = 0,
    random_seed: int = 42,
) -> torch.Tensor:
    """
    Calculate cross-entropy loss with heavy weighting on tokens between delimiters.

    The goal: Focus learning on the actual answer (between |||delimiters|||) rather
    than uniformly weighting all token predictions including prompt regurgitation.

    This is a "task outcome" proxy - we care most about predicting the answer correctly,
    less about perfectly mimicking formatting/prompt tokens.

    Algorithm:
        1. Run forward pass through model to get logits
        2. Calculate per-token cross-entropy loss (no reduction)
        3. For each sequence in batch:
           - Decode target tokens to text
           - Find delimiter pairs in text
           - Map delimiter positions back to token indices
           - Create weight mask with high weights for answer tokens
        4. Apply stochastic weighting decision (pct_validations_weighted)
        5. Apply grace period logic (steps_grace_period_before_weighting)
        6. Multiply per-token losses by weight mask
        7. Return weighted average loss

    Args:
        input_batch: Model input token IDs [batch_size, seq_len]
        target_batch: Ground truth target token IDs [batch_size, seq_len]
        model: Neural network model to evaluate
        device: Torch device ('cuda' or 'cpu')
        target_delimiter_string: String marking answer boundaries (default: "|||")
        answer_weight_multiplier: Weight multiplier for answer tokens (default: 10.0)
        pad_token_id: Token ID representing padding (default: 0)
        tokenizer: ByteTokenizer instance with encode/decode methods (required)
        pct_validations_weighted: Percentage of time to apply weighting 0-100 (default: 100)
        steps_grace_period_before_weighting: Steps before weighting begins (default: 100)
        current_step: Current training/validation step number (default: 0)
        random_seed: Random seed for reproducible stochastic weighting (default: 42)

    Returns:
        Scalar tensor containing weighted loss value

    Raises:
        ValueError: If tokenizer is None, or if parameters are invalid
        RuntimeError: If forward pass or loss calculation fails

    Example:
        >>> # Training data format: "What is 2+2? answer\\n|||4|||"
        >>> loss = calculate_weighted_loss_delimited_target(
        ...     input_batch=input_ids,
        ...     target_batch=target_ids,
        ...     model=model,
        ...     device=device,
        ...     target_delimiter_string="|||",
        ...     answer_weight_multiplier=10.0,
        ...     tokenizer=byte_tokenizer,
        ...     current_step=500
        ... )
        >>> loss.backward()  # Gradients focus on answer prediction

    Notes:
        - If delimiters not found: sequence gets zero weight (validation fail)
        - If only one delimiter found: sequence gets zero weight (malformed)
        - Padding tokens always get zero weight
        - Grace period uses uniform weighting (no special answer weighting)
        - Stochastic weighting uses seeded random for reproducibility
        - Multiple delimiter pairs in one sequence: all answer regions weighted
    """

    try:
        # =====================================================================
        # SECTION 1: Input Validation and Sanity Checks
        # =====================================================================

        # Check that tokenizer was provided
        if tokenizer is None:
            error_message = (
                "Tokenizer cannot be None! ByteTokenizer instance required for "
                "delimiter detection. Pass tokenizer=your_byte_tokenizer instance."
            )
            raise ValueError(error_message)

        # Validate percentage parameter is in valid range
        if not (0 <= pct_validations_weighted <= 100):
            print(
                f"Warning: pct_validations_weighted={pct_validations_weighted} "
                f"out of range [0,100]. Clamping to valid range."
            )
            pct_validations_weighted = max(0, min(100, pct_validations_weighted))

        # Ensure current_step is non-negative
        if current_step < 0:
            print(
                f"Warning: current_step={current_step} is negative. Treating as step 0."
            )
            current_step = 0

        # Check for empty batch
        batch_size, seq_len = target_batch.shape
        if batch_size == 0 or seq_len == 0:
            print("Warning: Empty batch received. Returning zero loss.")
            return torch.tensor(0.0, device=device)

        # =====================================================================
        # SECTION 2: Determine if Weighting Should Be Applied This Call
        # =====================================================================

        # Check grace period - if we're still in grace period, don't weight
        currently_in_grace_period = current_step < steps_grace_period_before_weighting

        # Stochastic weighting decision - use seeded random for reproducibility
        apply_weighting_this_call = True

        if currently_in_grace_period:
            apply_weighting_this_call = False
            # print(f"Debug: Step {current_step} in grace period, using uniform weights")

        elif pct_validations_weighted < 100:
            # Create a seeded random number generator for reproducibility
            # Seed changes with current_step so different steps get different decisions
            # but same step always gets same decision (reproducible)
            rng_for_probability = torch.Generator(device="cpu")
            rng_for_probability.manual_seed(random_seed + current_step)

            # Generate random value between 0 and 100
            random_percentage_value = (
                torch.rand(1, generator=rng_for_probability).item() * 100
            )

            # Decide if we apply weighting based on percentage threshold
            apply_weighting_this_call = (
                random_percentage_value < pct_validations_weighted
            )

            # Debug logging (can be removed in production)
            # if not apply_weighting_this_call:
            #     print(f"Debug: Step {current_step} skipping weighting (random={random_percentage_value:.1f})")

        # =====================================================================
        # SECTION 3: Forward Pass Through Model
        # =====================================================================

        try:
            # Move inputs to correct device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Get model predictions (logits)
            # Shape: [batch_size, seq_len, vocab_size]
            logits = model(input_batch)

        except Exception as forward_pass_error:
            error_message = f"Forward pass through model failed: {forward_pass_error}"
            print(f"ERROR: {error_message}")
            traceback.print_exc()
            raise RuntimeError(error_message) from forward_pass_error

        # =====================================================================
        # SECTION 4: Calculate Per-Token Cross-Entropy Loss (No Reduction)
        # =====================================================================

        try:
            # Flatten logits and targets for cross-entropy calculation
            # logits: [batch_size * seq_len, vocab_size]
            # targets: [batch_size * seq_len]
            logits_flattened = logits.view(-1, logits.size(-1))
            targets_flattened = target_batch.view(-1)

            # Calculate cross-entropy loss per token (no reduction)
            # This gives us individual loss value for each token position
            loss_function = nn.CrossEntropyLoss(reduction="none")
            per_token_losses_flat = loss_function(logits_flattened, targets_flattened)

            # Reshape back to [batch_size, seq_len] for easier manipulation
            per_token_losses = per_token_losses_flat.view(batch_size, seq_len)

        except Exception as loss_calculation_error:
            error_message = f"Loss calculation failed: {loss_calculation_error}"
            print(f"ERROR: {error_message}")
            traceback.print_exc()
            raise RuntimeError(error_message) from loss_calculation_error

        # =====================================================================
        # SECTION 5: Build Weight Mask for Each Sequence in Batch
        # =====================================================================

        # Initialize weight mask with uniform weights (1.0 for all positions)
        # Shape: [batch_size, seq_len]
        weight_mask = torch.ones_like(
            per_token_losses, dtype=torch.float32, device=device
        )

        # If we're not applying weighting this call, skip delimiter detection
        # and just mask padding tokens
        if not apply_weighting_this_call:
            # Just mask padding tokens (set weight to 0.0)
            padding_mask = target_batch == pad_token_id
            weight_mask[padding_mask] = 0.0

        else:
            # Apply delimiter-based weighting

            # Process each sequence in the batch individually
            for sequence_index in range(batch_size):
                try:
                    # Extract token IDs for this sequence
                    target_token_ids = target_batch[sequence_index].tolist()

                    # Decode tokens back to text string
                    # This is necessary to search for delimiter string
                    decoded_text = tokenizer.decode(target_token_ids)

                    # Convert delimiter string to bytes for consistent matching
                    delimiter_bytes = target_delimiter_string.encode("utf-8")
                    decoded_bytes = decoded_text.encode("utf-8")

                    # =========================================================
                    # SECTION 5A: Find All Delimiter Pairs in Decoded Text
                    # =========================================================

                    # Find all occurrences of delimiter in the byte sequence
                    delimiter_positions_list = []
                    search_start_position = 0

                    while True:
                        # Search for next delimiter occurrence
                        delimiter_position = decoded_bytes.find(
                            delimiter_bytes, search_start_position
                        )

                        # If no more delimiters found, stop searching
                        if delimiter_position == -1:
                            break

                        # Record this delimiter position
                        delimiter_positions_list.append(delimiter_position)

                        # Move search position past this delimiter
                        search_start_position = delimiter_position + len(
                            delimiter_bytes
                        )

                    # =========================================================
                    # SECTION 5B: Validate Delimiter Pairs
                    # =========================================================

                    # We need pairs of delimiters (opening and closing)
                    # If odd number of delimiters, something is malformed
                    number_of_delimiters_found = len(delimiter_positions_list)

                    if number_of_delimiters_found == 0:
                        # No delimiters found - validation fail
                        # Set entire sequence weight to zero
                        weight_mask[sequence_index, :] = 0.0
                        # print(f"Debug: Sequence {sequence_index} has no delimiters, weight=0")
                        continue

                    elif number_of_delimiters_found % 2 != 0:
                        # Odd number of delimiters - malformed (unpaired)
                        # Set entire sequence weight to zero
                        weight_mask[sequence_index, :] = 0.0
                        print(
                            f"Warning: Sequence {sequence_index} has {number_of_delimiters_found} "
                            f"delimiters (unpaired). Setting weight=0."
                        )
                        continue

                    # =========================================================
                    # SECTION 5C: Map Byte Positions to Token Indices
                    # =========================================================

                    # Process delimiter pairs
                    number_of_pairs = number_of_delimiters_found // 2

                    for pair_index in range(number_of_pairs):
                        # Get byte positions of opening and closing delimiters
                        opening_delimiter_byte_pos = delimiter_positions_list[
                            pair_index * 2
                        ]
                        closing_delimiter_byte_pos = delimiter_positions_list[
                            pair_index * 2 + 1
                        ]

                        # Calculate byte position range for answer content
                        # (content between delimiters, excluding the delimiters themselves)
                        answer_start_byte = opening_delimiter_byte_pos + len(
                            delimiter_bytes
                        )
                        answer_end_byte = closing_delimiter_byte_pos

                        # Skip if delimiters are adjacent (empty answer)
                        if answer_start_byte >= answer_end_byte:
                            continue

                        # Map byte positions to token indices
                        # Strategy: Re-encode tokens one-by-one and track cumulative byte positions
                        cumulative_byte_position = 0
                        answer_token_start_index = None
                        answer_token_end_index = None

                        for token_index, token_id in enumerate(target_token_ids):
                            # Get byte representation of this single token
                            token_bytes = tokenizer.decode([token_id]).encode("utf-8")
                            token_byte_length = len(token_bytes)

                            # Check if this token overlaps with answer region
                            token_starts_at_byte = cumulative_byte_position
                            token_ends_at_byte = (
                                cumulative_byte_position + token_byte_length
                            )

                            # Check if token starts within or after answer start
                            if (
                                answer_token_start_index is None
                                and token_ends_at_byte > answer_start_byte
                            ):
                                answer_token_start_index = token_index

                            # Check if token ends within or after answer end
                            if (
                                answer_token_start_index is not None
                                and token_starts_at_byte >= answer_end_byte
                            ):
                                answer_token_end_index = token_index
                                break

                            # Update cumulative position for next token
                            cumulative_byte_position += token_byte_length

                        # If we found answer region, apply heavy weighting
                        if answer_token_start_index is not None:
                            # If we didn't find end, weight until end of sequence
                            if answer_token_end_index is None:
                                answer_token_end_index = seq_len

                            # Apply heavy weight to answer tokens
                            weight_mask[
                                sequence_index,
                                answer_token_start_index:answer_token_end_index,
                            ] = answer_weight_multiplier

                            # Debug logging (can be removed in production)
                            # print(
                            #     f"Debug: Sequence {sequence_index} answer tokens "
                            #     f"[{answer_token_start_index}:{answer_token_end_index}] "
                            #     f"weighted {answer_weight_multiplier}x"
                            # )

                    # =========================================================
                    # SECTION 5D: Mask Padding Tokens
                    # =========================================================

                    # Set padding token weights to zero
                    # (This applies regardless of delimiter weighting)
                    padding_positions = target_batch[sequence_index] == pad_token_id
                    weight_mask[sequence_index, padding_positions] = 0.0

                except Exception as sequence_processing_error:
                    # If anything goes wrong processing this sequence,
                    # set its weight to zero and continue with other sequences
                    print(
                        f"Warning: Error processing sequence {sequence_index}: "
                        f"{sequence_processing_error}"
                    )
                    print(f"Setting sequence {sequence_index} weight to zero.")
                    weight_mask[sequence_index, :] = 0.0
                    # Don't raise - continue processing other sequences

        # =====================================================================
        # SECTION 6: Calculate Weighted Loss
        # =====================================================================

        try:
            # Apply weight mask to per-token losses
            # Element-wise multiplication
            weighted_losses = per_token_losses * weight_mask

            # Calculate total weighted loss (sum of all weighted token losses)
            total_weighted_loss = weighted_losses.sum()

            # Calculate total weight (sum of all weight mask values)
            total_weight = weight_mask.sum()

            # Calculate final normalized loss
            # Divide by total weight rather than sequence length
            # This ensures that answer tokens contribute proportionally to their weight
            if total_weight > 0:
                final_loss = total_weighted_loss / total_weight
            else:
                # Edge case: all weights are zero (all sequences failed validation)
                # Return a very small loss rather than NaN
                print(
                    "Warning: Total weight is zero (all sequences failed validation). "
                    "Returning loss=0.0"
                )
                final_loss = torch.tensor(0.0, device=device)

            return final_loss

        except Exception as weighted_loss_error:
            error_message = f"Weighted loss calculation failed: {weighted_loss_error}"
            print(f"ERROR: {error_message}")
            traceback.print_exc()
            raise RuntimeError(error_message) from weighted_loss_error

    except Exception as unexpected_error:
        # Catch-all for any unexpected errors
        error_message = (
            f"Unexpected error in calculate_weighted_loss_delimited_target: "
            f"{unexpected_error}"
        )
        print(f"ERROR: {error_message}")
        traceback.print_exc()
        raise RuntimeError(error_message) from unexpected_error


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================


def example_usage_demonstration():
    """
    Demonstrate how to use calculate_weighted_loss_delimited_target in a training loop.

    This is a conceptual example showing integration patterns.
    """
    print("\n" + "=" * 70)
    print("Example Usage of calculate_weighted_loss_delimited_target")
    print("=" * 70)

    example_code = '''
    # =========================================================================
    # Example 1: Basic Validation Loop Integration
    # =========================================================================

    # During validation phase of training
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(val_loader):

            # Calculate weighted validation loss
            val_loss = calculate_weighted_loss_delimited_target(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
                target_delimiter_string="|||",
                answer_weight_multiplier=10.0,
                pad_token_id=tokenizer.PAD_ID,
                tokenizer=tokenizer,
                pct_validations_weighted=100,
                steps_grace_period_before_weighting=200,
                current_step=global_step,
                random_seed=42
            )

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # =========================================================================
    # Example 2: Stochastic Weighting (50% of validations)
    # =========================================================================

    # Only apply delimiter weighting 50% of the time
    val_loss = calculate_weighted_loss_delimited_target(
        input_batch=input_batch,
        target_batch=target_batch,
        model=model,
        device=device,
        tokenizer=tokenizer,
        pct_validations_weighted=50,  # Only 50% of calls use weighting
        current_step=global_step,
        random_seed=42  # Reproducible randomness
    )

    # =========================================================================
    # Example 3: Grace Period (No Weighting for First 500 Steps)
    # =========================================================================

    # Allow model to learn basics before enforcing strict answer focus
    val_loss = calculate_weighted_loss_delimited_target(
        input_batch=input_batch,
        target_batch=target_batch,
        model=model,
        device=device,
        tokenizer=tokenizer,
        steps_grace_period_before_weighting=500,  # First 500 steps: uniform weighting
        current_step=global_step
    )

    # =========================================================================
    # Example 4: Custom Delimiter and Weight
    # =========================================================================

    # Use different delimiter and higher answer weighting
    val_loss = calculate_weighted_loss_delimited_target(
        input_batch=input_batch,
        target_batch=target_batch,
        model=model,
        device=device,
        tokenizer=tokenizer,
        target_delimiter_string=">>>",  # Custom delimiter
        answer_weight_multiplier=20.0,  # 20x weight on answers
        current_step=global_step
    )

    # =========================================================================
    # Example 5: Training Data Format
    # =========================================================================

    # Your training data should look like this:
    training_example = """
    What is 2+2?
    answer
    |||4|||
    """

    # Or for math expression evaluation:
    training_example_math = """
    ||expression section||

    |English|
    seven minus six

    |symbolic|
    7-6

    ||evaluation section||

    |answer|
    |||1|||
    """

    # The delimiter |||answer||| gets 10x weight during validation
    '''

    print(example_code)
    print("=" * 70 + "\n")


# =============================================================================
# This version ises a directory of chunk-toml-files, not one corpus .txt
# =============================================================================


# To:
TRAINING_DATA_DIR = "/home/oops/code/gutenberg_babble/perseids/byte_perseid/pseudo_toml_maker_training_data/toml_production_output_500"  # Directory containing .txt files
TRAINING_FILE_EXTENSION = ".toml"  # File extension to use
SHUFFLE_FILES = True  # Shuffle files before train/val split
DATA_RANDOM_SEED = 42  # Seed for reproducible file shuffling


# In main() function, replace Step 1 and Step 3:


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

        # REMOVED Step 1: Load single document
        # document_text = load_document(DOCUMENT_PATH)

        # 2. Setup model and tokenizer
        print(f"\n{'=' * 40}")
        print("Step 2: Setting Up Model")
        print(f"{'=' * 40}")

        # Setup ByteTokenizer
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

        # MODIFIED Step 3: Prepare data from directory
        print(f"\n{'=' * 40}")
        print("Step 3: Preparing Data from Directory")
        print(f"{'=' * 40}")

        train_loader, val_loader = create_data_loaders_from_directory(
            directory_path=TRAINING_DATA_DIR,
            tokenizer=tokenizer,
            config=TRAINING_CONFIG,
            train_ratio=TRAIN_VAL_SPLIT,
            file_extension=TRAINING_FILE_EXTENSION,
            shuffle_files=SHUFFLE_FILES,
            random_seed=DATA_RANDOM_SEED,
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
            tokenizer,
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
            """
[[expression section]]

[English]
one plus one

[symbolic]
1+1

[[evaluation section]]

[rpn_steps]
[('PUSH', 1), ('PUSH', 1), ('OPERATOR', '+')]

[answer]
|||
""",
            """
[[expression section]]

[English]
two times two

[symbolic]
10*2

[[evaluation section]]

[rpn_steps]
[('PUSH', 10), ('PUSH', 2), ('OPERATOR', '*')]

[answer]
|||
""",
            """
[[expression section]]

[English]
two times two

[symbolic]
2*2

[[evaluation section]]

[rpn_steps]
[('PUSH', 2), ('PUSH', 2), ('OPERATOR', '*')]

[answer]
|||
""",
            #             """
            # [[expression section]]
            # [English]
            # one plus one
            # [symbolic]
            # 1+1
            # [[evaluation section]]
            # [rpn_steps]
            # [('PUSH', 1), ('PUSH', 1), ('OPERATOR', '+')]
            # [answer]
            # |||2|||
            # """,
        ]
        for prompt in test_prompts:
            output = generate_text_simple(
                model, tokenizer, prompt, max_new_tokens=500, device=DEVICE
            )
            print(f"Prompt: '{prompt}'")
            print(f"Output: {output}\n")

        print(f"{'=' * 60}")
        print(f"Model and results saved to: {output_dir}")

        return model, history

    except Exception as main_error:
        print(f"\n{'=' * 60}")
        print("Training Pipeline Failed")
        print(f"{'=' * 60}")
        print(f"Error: {main_error}")
        traceback.print_exc()
        raise


# def main():
#     """
#     Main training pipeline for Perseid document training.
#     """
#     try:
#         print(f"\n{'=' * 60}")
#         print("Perseid Document Training Pipeline")
#         print(f"{'=' * 60}")
#         print(f"Experiment: {EXPERIMENT_NAME}")
#         print(f"Output directory: {OUTPUT_DIR}")

#         # Create output directory
#         output_dir = Path(OUTPUT_DIR)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         # Set random seeds for reproducibility
#         torch.manual_seed(42)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(42)

#         # 1. Load document
#         print(f"\n{'=' * 40}")
#         print("Step 1: Loading Document")
#         print(f"{'=' * 40}")
#         document_text = load_document(DOCUMENT_PATH)

#         # 2. Setup model and tokenizer
#         print(f"\n{'=' * 40}")
#         print("Step 2: Setting Up Model")
#         print(f"{'=' * 40}")

#         # Setup ByteTokenizer
#         tokenizer = setup_tokenizer()

#         # Initialize model
#         model, model_config, training_state = setup_model(
#             MODEL_SIZE,
#             MODEL_STRATEGY,
#             TRAINING_CONFIG,
#             DEVICE,
#             OUTPUT_DIR,
#             TRAINING_MODE,
#             CHECKPOINT_PATH,
#         )

#         # 3. Create data loaders
#         print(f"\n{'=' * 40}")
#         print("Step 3: Preparing Data")
#         print(f"{'=' * 40}")
#         train_loader, val_loader = create_data_loaders(
#             document_text, tokenizer, TRAINING_CONFIG, train_ratio=TRAIN_VAL_SPLIT
#         )

#         # 4. Train model
#         print(f"\n{'=' * 40}")
#         print("Step 4: Training Model")
#         print(f"{'=' * 40}")
#         history = train_model(
#             model,
#             train_loader,
#             val_loader,
#             TRAINING_CONFIG,
#             DEVICE,
#             output_dir,
#             training_state,
#             tokenizer,  # NEW: Pass tokenizer to training loop
#         )

#         # 5. Save results
#         print(f"\n{'=' * 40}")
#         print("Step 5: Saving Results")
#         print(f"{'=' * 40}")
#         save_training_results(model, model_config, history, output_dir)

#         print(f"\n{'=' * 60}")
#         print("Training Pipeline Complete!")

#         # 5.5 Generate sample text with trained model
#         print(f"\n{'=' * 40}")
#         print("Step 5.5: Sample Generation")
#         print(f"{'=' * 40}")

#         test_prompts = [
#             "Once upon a time",
#             "The meaning of life is",
#             "In the beginning",
#         ]
#         for prompt in test_prompts:
#             output = generate_text_simple(
#                 model, tokenizer, prompt, max_new_tokens=50, device=DEVICE
#             )
#             print(f"Prompt: '{prompt}'")
#             print(f"Output: {output}\n")

#         print(f"{'=' * 60}")
#         print(f"Model and results saved to: {output_dir}")

#         return model, history

#     except Exception as main_error:
#         print(f"\n{'=' * 60}")
#         print("Training Pipeline Failed")
#         print(f"{'=' * 60}")
#         print(f"Error: {main_error}")
#         traceback.print_exc()
#         raise


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

#         # # Download sample text data
#         # demo_file_path = "data/alice.txt"
#         # os.makedirs("data", exist_ok=True)

#         # if not os.path.exists(demo_file_path):
#         #     url = "https://www.gutenberg.org/files/11/11-0.txt"
#         #     print(f"Downloading training data from {url}")
#         #     with urllib.request.urlopen(url) as response:
#         #         text_data = response.read().decode("utf-8")
#         #     with open(demo_file_path, "w", encoding="utf-8") as file:
#         #         file.write(text_data)
#         # else:
#         #     print(f"Loading existing data from {demo_file_path}")
#         #     with open(demo_file_path, "r", encoding="utf-8") as file:
#         #         text_data = file.read()

#         # Q&A
#         user_path_or_demo_choice = input(
#             "\nEnter path to directory of .toml training files.\n"
#         )

#         # # use demo if demo is selected
#         # if user_path_or_demo_choice.lower().strip() == "demo":
#         #     file_path = demo_file_path

#         # use Q&A input path if selected
#         else:
#             training_files_dir_path = user_path_or_demo_choice

#     # use argument input path if supplied by user
#     elif len(sys.argv) == 2:
#         training_files_dir_path = sys.argv[1]
#         print(f"path argument found... {training_files_dir_path}")

#     else:
#         print("Edge case, defaulting to demo.")

#     # Document input
#     DOCUMENT_PATH = training_files_dir_path  # "./data/my_document.txt"  # Path to your text file

#     # Output configuration
#     OUTPUT_DIR = f"./models/perseid_{MODEL_SIZE}m_{Path(DOCUMENT_PATH).stem}/"
#     EXPERIMENT_NAME = (
#         f"perseid_{MODEL_SIZE}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     )

#     # Hardware settings
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     USE_BFLOAT16 = torch.cuda.is_available()  # Use bfloat16 if on GPU

#     # Run the training pipeline
#     model, history = main()


if __name__ == "__main__":
    # Test integration
    test_integration()

    print("\n" + "=" * 70)
    print("TRAINING DATA INPUT CONFIGURATION")
    print("=" * 70)

    # Get directory path from command line or user input
    if len(sys.argv) == 2:
        directory_path = sys.argv[1]
        print(f"✓ Using directory from command line argument: {directory_path}")
    else:
        print("Enter the path to your directory of .toml training files.")
        print(
            "Each .toml file should be one complete training chunk with |||answer||| delimiters."
        )
        print("\nExample directory structure:")
        print("  training_data/")
        print("  ├── chunk_001.toml")
        print("  ├── chunk_002.toml")
        print("  └── chunk_500.toml")

        directory_path = input("\nDirectory path: ").strip()

    # Validate directory exists
    directory_path = Path(directory_path)
    if not directory_path.exists():
        print(f"\n❌ ERROR: Directory not found: {directory_path}")
        sys.exit(1)

    if not directory_path.is_dir():
        print(f"\n❌ ERROR: Path is not a directory: {directory_path}")
        sys.exit(1)

    # Update global configuration
    TRAINING_DATA_DIR = directory_path

    # Output configuration (based on directory name)
    OUTPUT_DIR = f"./models/perseid_{MODEL_SIZE}m_{directory_path.stem}/"
    EXPERIMENT_NAME = (
        f"perseid_{MODEL_SIZE}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BFLOAT16 = torch.cuda.is_available()

    print(f"\n✓ Training data directory: {TRAINING_DATA_DIR}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"✓ Device: {DEVICE}")
    print("=" * 70 + "\n")

    # Run the training pipeline
    model, history = main()
