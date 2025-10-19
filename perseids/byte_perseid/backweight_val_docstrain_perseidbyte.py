"""
backweight_val_docstrain_perseidbyte.py

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
    "context_length": 1024,
    "batch_size": 11,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "num_epochs": 2,  # 7
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "eval_every": 2,
    "eval_batches": 10,
    "save_every": 500,  # 100
    "chunk_overlap": 0.1,
    # NEW: Weighted validation loss parameters
    "use_weighted_validation": True,  # Enable/disable weighted validation
    "target_delimiter_string": "|||",  # Delimiter surrounding answers
    "answer_weight_multiplier": 10.0,  # Weight multiplier for answer tokens
    "pct_validations_weighted": 100,  # Percentage of validations to weight (0-100)
    "steps_grace_period_before_weighting": 200,  # Grace period steps before weighting starts
    "validation_random_seed": 42,  # Seed for reproducible stochastic weighting
}


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
    Evaluate model on data loader with optional weighted loss.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        num_batches: Maximum number of batches to evaluate (None = all)
        tokenizer: ByteTokenizer instance (required for weighted loss)
        config: Training configuration dict (required for weighted loss)
        current_step: Current global training step (for grace period logic)
        use_weighted_loss: Whether to use delimiter-weighted loss

    Returns:
        float: Average loss across evaluated batches
    """
    try:
        model.eval()
        total_loss = 0.0

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        # Determine which loss function to use
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

            print(f"  Using weighted validation loss (step {current_step})")

        with torch.no_grad():
            for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break

                # Choose loss calculation method
                if use_weighted_loss:
                    # Calculate weighted loss focusing on delimited answers
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
                else:
                    # Standard uniform cross-entropy loss
                    loss = calculate_loss(input_batch, target_batch, model, device)

                total_loss += loss.item()

        model.train()

        average_loss = total_loss / num_batches if num_batches > 0 else float("nan")

        return average_loss

    except Exception as evaluation_error:
        print(f"Error during model evaluation: {evaluation_error}")
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
            tokenizer,  # NEW: Pass tokenizer to training loop
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
            output = generate_text_simple(
                model, tokenizer, prompt, max_new_tokens=50, device=DEVICE
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
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode("utf-8")
            with open(demo_file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            print(f"Loading existing data from {demo_file_path}")
            with open(demo_file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        # Q&A
        user_path_or_demo_choice = input(
            "\nEnter a file path to a .txt file or for a demo say 'demo'\n"
        )

        # use demo if demo is selected
        if user_path_or_demo_choice.lower().strip() == "demo":
            file_path = demo_file_path

        # use Q&A input path if selected
        else:
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

    # Run the training pipeline
    model, history = main()
