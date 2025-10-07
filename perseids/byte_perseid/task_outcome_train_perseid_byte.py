"""
pending: task_outcome_train_perseidbyte.py (still under construction!!)

TODO: still adding functions for
A. using generated training data
B. using weighted validation of the last ~16 tokens (more weight on the 'solution' to the task)

Future (not first mvp): looking for solution in longer answer


This is an interesting simpler trainer that allows re-training / continued pre-training.
Possibly should track/log what last-best validation loss was...


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

# Training settings
TRAINING_CONFIG = {
    "context_length": 1024,  # Context window for training
    "batch_size": 11,  # Batch size (increase if memory allows)
    "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
    "learning_rate": 5e-4,  # Learning rate
    "num_epochs": 7,  # Number of training epochs, default 3
    "weight_decay": 0.01,  # Weight decay for AdamW
    "warmup_steps": 100,  # Warmup steps for learning rate
    "eval_every": 2,  # Evaluate every N steps
    "eval_batches": 10,  # Number of batches for evaluation
    "save_every": 500,  # Save checkpoint every N steps
    "chunk_overlap": 0.1,  # Overlap between text chunks (0.0 to 0.5)
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


def train_model(
    model, train_loader, val_loader, config, device, output_dir, training_state
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

    Returns:
        dict: Training history
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
        history = {"train_loss": [], "val_loss": [], "learning_rates": [], "step": []}

        # global_step = 0
        # best_val_loss = float('inf')
        # tokens_seen = 0
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
        # for epoch in range(config["num_epochs"]):
        for epoch in range(start_epoch, config["num_epochs"]):
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            print(f"{'=' * 40}")

            model.train()
            epoch_loss = 0
            epoch_tokens = 0

            for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                # Calculate loss
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
                        )

            # End of epoch evaluation
            avg_epoch_loss = epoch_loss / len(train_loader)
            val_loss = evaluate_model(model, val_loader, device)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

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


def save_checkpoint(model, optimizer, scheduler, step, val_loss, output_dir, tag):
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
            },
            checkpoint_path,
        )

    except Exception as e:
        print(f"Error saving checkpoint: {e}")
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


################
# Task Outcomes
################


class TaskOutcomeDataset(Dataset):
    """
    Dataset that pairs problems with their deterministic solutions.
    Enables task-outcome based evaluation instead of token prediction.
    """

    def __init__(self, problems_file, tokenizer, max_length):
        """
        Args:
            problems_file: JSON file with format:
                [
                    {
                        "problem": "# Task: evaluate expression, What is 2+2? Answer:",
                        "answer": "4",
                        "answer_delimiters": ["```", "```"]  # optional
                    },
                    ...
                ]
            tokenizer: ByteTokenizer instance
            max_length: Maximum sequence length
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load problems and solutions
        with open(problems_file, "r") as f:
            self.problems = json.load(f)

        print(f"Loaded {len(self.problems)} task-outcome problems")

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem_data = self.problems[idx]

        # Tokenize problem prompt
        problem_tokens = self.tokenizer.encode(problem_data["problem"])

        # Truncate if needed
        if len(problem_tokens) > self.max_length:
            problem_tokens = problem_tokens[: self.max_length]

        # Convert to tensor
        input_ids = torch.tensor(problem_tokens, dtype=torch.long)

        # Return problem tokens and ground truth answer (as string)
        return {
            "input_ids": input_ids,
            "answer": problem_data["answer"],
            "problem_text": problem_data["problem"],
            "delimiters": problem_data.get("answer_delimiters", ["```", "```"]),
        }


def collate_task_outcome_batch(batch):
    """Custom collate function for variable-length problems."""
    # Find max length in batch
    max_len = max(item["input_ids"].shape[0] for item in batch)

    # Pad all sequences
    padded_inputs = []
    answers = []
    problem_texts = []
    delimiters = []

    for item in batch:
        # Pad input
        input_ids = item["input_ids"]
        padding_length = max_len - input_ids.shape[0]
        if padding_length > 0:
            padding = torch.zeros(padding_length, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])

        padded_inputs.append(input_ids)
        answers.append(item["answer"])
        problem_texts.append(item["problem_text"])
        delimiters.append(item["delimiters"])

    return {
        "input_ids": torch.stack(padded_inputs),
        "answers": answers,
        "problem_texts": problem_texts,
        "delimiters": delimiters,
    }


def train_with_task_outcomes(
    model, train_loader, config, device, output_dir, validation_suite, tokenizer
):
    """
    Training loop with outcome-based checkpointing.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Traditional training state
    global_step = 0
    best_val_loss = float("inf")  # Traditional metric

    # NEW: Task outcome state
    best_accuracy = 0.0
    best_outcome_step = 0

    # History tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "task_accuracy": [],  # NEW
        "step": [],
    }

    for epoch in range(config["num_epochs"]):
        model.train()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # UNCHANGED: Traditional gradient update
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits = model(input_batch)

            # Option A: Use weighted loss during training too
            loss = outcome_weighted_loss(
                logits, target_batch, tokenizer, weight_last_n=3, answer_weight=5.0
            )

            # Option B: Use traditional loss, only change checkpointing
            # loss = nn.functional.cross_entropy(
            #     logits.flatten(0, 1),
            #     target_batch.flatten()
            # )

            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Periodic evaluation
                if global_step % config["eval_every"] == 0:
                    # Traditional validation loss (optional, for comparison)
                    val_loss = evaluate_model(model, val_loader, device, num_batches=10)

                    # NEW: Task outcome evaluation
                    outcome_results = evaluate_task_outcomes(
                        model, tokenizer, validation_suite, device
                    )

                    current_accuracy = outcome_results["accuracy"]

                    # Log both metrics
                    history["val_loss"].append(val_loss)
                    history["task_accuracy"].append(current_accuracy)
                    history["step"].append(global_step)

                    print(
                        f"Step {global_step:5d} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Task Accuracy: {current_accuracy:.2%} "
                        f"({outcome_results['correct_count']}/{outcome_results['total_count']})"
                    )

                    # CHANGED: Checkpoint based on task accuracy
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_outcome_step = global_step

                        save_checkpoint(
                            model,
                            optimizer,
                            None,
                            global_step,
                            val_loss,
                            output_dir,
                            "best_outcome",
                        )

                        print(f"  → NEW BEST accuracy: {current_accuracy:.2%}")

                        # Optionally save failed problems for analysis
                        if outcome_results["failed_problems"]:
                            save_failed_problems(
                                outcome_results["failed_problems"],
                                output_dir,
                                global_step,
                            )

                    # Optional: Also save best perplexity for comparison
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model,
                            optimizer,
                            None,
                            global_step,
                            val_loss,
                            output_dir,
                            "best_perplexity",
                        )

    print(f"\nTraining complete!")
    print(f"Best task accuracy: {best_accuracy:.2%} at step {best_outcome_step}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return history


def save_failed_problems(failed_problems, output_dir, step):
    """Save failed problems for debugging."""
    output_path = Path(output_dir) / f"failed_problems_step_{step}.json"
    with open(output_path, "w") as f:
        json.dump(failed_problems, f, indent=2)


def extract_answer_from_generation(
    generated_text, delimiters=["```", "```"], answer_pattern=None
):
    """
    Search generated text for answer within delimiters or using regex.

    Args:
        generated_text: Full model output
        delimiters: [start_delim, end_delim] to search between
        answer_pattern: Optional regex pattern

    Returns:
        extracted_answer (str or None)
    """
    import re

    # Method 1: Delimiter-based extraction
    if delimiters and len(delimiters) == 2:
        start_delim, end_delim = delimiters

        # Find content between delimiters
        pattern = re.escape(start_delim) + r"(.*?)" + re.escape(end_delim)
        matches = re.findall(pattern, generated_text, re.DOTALL)

        if matches:
            # Return first match, stripped
            return matches[0].strip()

    # Method 2: Custom regex pattern
    if answer_pattern:
        matches = re.findall(answer_pattern, generated_text)
        if matches:
            return matches[0].strip()

    # Method 3: Fallback - look for number after "Answer:"
    answer_match = re.search(r"Answer:\s*([+-]?\d+\.?\d*)", generated_text)
    if answer_match:
        return answer_match.group(1).strip()

    # Method 4: Last resort - look for any number
    number_matches = re.findall(r"[+-]?\d+\.?\d*", generated_text)
    if number_matches:
        return number_matches[-1]  # Return last number found

    return None


def check_answer_correctness(extracted_answer, ground_truth, tolerance=1e-6):
    """
    Compare extracted answer to ground truth.

    Args:
        extracted_answer: String extracted from model output
        ground_truth: Correct answer (string or number)
        tolerance: Numerical comparison tolerance

    Returns:
        bool: True if correct
    """
    if extracted_answer is None:
        return False

    # Exact string match
    if str(extracted_answer).strip() == str(ground_truth).strip():
        return True

    # Try numerical comparison
    try:
        extracted_num = float(extracted_answer)
        ground_truth_num = float(ground_truth)
        return abs(extracted_num - ground_truth_num) < tolerance
    except (ValueError, TypeError):
        pass

    return False


def calculate_task_outcome_loss(
    model,
    batch,
    tokenizer,
    device,
    generation_max_tokens=50,
    answer_weight=10.0,
    use_token_loss=True,
):
    """
    Calculate loss based on task outcome (correct answer) instead of
    or in addition to token prediction.

    Args:
        model: The model being trained
        batch: Batch from TaskOutcomeDataset
        tokenizer: Tokenizer for decoding
        device: torch device
        generation_max_tokens: How many tokens to generate for answer
        answer_weight: How much to weight task success vs token loss
        use_token_loss: If True, combine with traditional loss

    Returns:
        loss: torch.Tensor scalar loss value
        metrics: dict with accuracy and other metrics
    """
    input_ids = batch["input_ids"].to(device)
    answers = batch["answers"]
    delimiters = batch["delimiters"]

    batch_size = input_ids.shape[0]

    # Track metrics
    correct_count = 0
    total_count = batch_size

    # Optional: Traditional token prediction loss on prompt
    token_loss = torch.tensor(0.0, device=device)
    if use_token_loss:
        # Calculate standard cross-entropy on the prompt itself
        # (This ensures model still learns language modeling)
        logits = model(input_ids)
        # Use input shifted by 1 as target
        token_loss = nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1),
            input_ids[:, 1:].flatten(),
            ignore_index=0,  # Ignore padding
        )

    # Generate answers for each problem in batch
    model.eval()
    with torch.no_grad():
        for i in range(batch_size):
            # Get problem prompt
            problem_input = input_ids[i : i + 1]

            # Generate answer
            generated_ids = problem_input.clone()

            for _ in range(generation_max_tokens):
                logits = model(generated_ids)[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Stop if we hit EOS or exceed context
                if next_token.item() == tokenizer.EOS_ID:
                    break
                if generated_ids.shape[1] > model.cfg["context_length"]:
                    generated_ids = generated_ids[:, -model.cfg["context_length"] :]

            # Decode generated text
            generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist())

            # Extract answer
            extracted = extract_answer_from_generation(
                generated_text, delimiters=delimiters[i]
            )

            # Check correctness
            is_correct = check_answer_correctness(extracted, answers[i])

            if is_correct:
                correct_count += 1

    model.train()

    # Calculate task outcome component
    # Convert accuracy to a "loss" (higher accuracy = lower loss)
    accuracy = correct_count / total_count
    task_loss = torch.tensor(1.0 - accuracy, device=device, requires_grad=False)

    # Combine losses
    if use_token_loss:
        total_loss = token_loss + (answer_weight * task_loss)
    else:
        total_loss = task_loss

    metrics = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
        "token_loss": token_loss.item() if use_token_loss else 0.0,
        "task_loss": task_loss.item(),
    }

    return total_loss, metrics


def weighted_last_n_token_loss(
    logits,
    labels,
    tokenizer,
    n_answer_tokens=5,
    answer_weight=10.0,
    answer_delimiter="```",
):
    """
    Weight the last N tokens (the answer) much more heavily in loss.

    This is a hybrid approach between pure token prediction and task outcome.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for loss calculation
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Calculate per-token losses
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
    per_token_losses = loss_fn(logits_flat, labels_flat)
    per_token_losses = per_token_losses.view(batch_size, seq_len)

    # Create weight mask
    weights = torch.ones_like(per_token_losses)

    # Find answer delimiter in each sequence and weight heavily after it
    for i in range(batch_size):
        # Decode to find delimiter position
        sequence = labels[i].cpu().tolist()
        decoded = tokenizer.decode(sequence)

        # Find last occurrence of delimiter
        delimiter_pos = decoded.rfind(answer_delimiter)

        if delimiter_pos != -1:
            # Roughly convert character position to token position
            # This is approximate - you may need more sophisticated tracking
            token_pos = int((delimiter_pos / len(decoded)) * seq_len)

            # Weight last N tokens heavily
            start_weight_pos = max(0, seq_len - n_answer_tokens)
            weights[i, start_weight_pos:] = answer_weight

    # Apply weights
    weighted_losses = per_token_losses * weights

    # Calculate final loss (only over non-padding tokens)
    mask = (labels != 0).float()
    loss = (weighted_losses * mask).sum() / mask.sum()

    return loss


def train_model_task_outcome(
    model,
    train_loader,  # Traditional training data
    task_val_loader,  # Task-outcome validation problems
    config,
    device,
    output_dir,
    training_state,
    tokenizer,
):
    """
    Training loop with task-outcome based validation.
    """
    print(f"\n{'=' * 60}")
    print("Starting Task-Outcome Based Training")
    print(f"{'=' * 60}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )

    if training_state["optimizer_state"] is not None:
        optimizer.load_state_dict(training_state["optimizer_state"])

    # Setup scheduler
    steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
    total_steps = steps_per_epoch * config["num_epochs"]

    def get_lr(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        else:
            progress = (step - config["warmup_steps"]) / (
                total_steps - config["warmup_steps"]
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    if training_state["scheduler_state"] is not None:
        scheduler.load_state_dict(training_state["scheduler_state"])

    # Training state
    history = {
        "train_loss": [],
        "val_task_accuracy": [],  # NEW: Task-based metric
        "val_task_loss": [],  # NEW: Task-based loss
        "learning_rates": [],
        "step": [],
    }

    global_step = training_state["global_step"]
    best_task_accuracy = training_state.get("best_task_accuracy", 0.0)
    tokens_seen = training_state["tokens_seen"]
    start_epoch = training_state["epoch"]

    print(f"\nStarting from epoch {start_epoch + 1}, step {global_step}")
    print(f"Best task accuracy so far: {best_task_accuracy:.2%}")

    # Training loop
    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'=' * 40}")

        model.train()
        epoch_loss = 0
        epoch_tokens = 0

        # Traditional training on general corpus
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Standard training loss
            loss = calculate_loss(input_batch, target_batch, model, device)
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            epoch_loss += loss.item() * config["gradient_accumulation_steps"]
            epoch_tokens += input_batch.numel()

            # Update weights after gradient accumulation
            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                tokens_seen += epoch_tokens
                epoch_tokens = 0

                # Periodic task-outcome evaluation
                if global_step % config["eval_every"] == 0:
                    # Evaluate on task-outcome validation set
                    task_metrics = evaluate_task_outcomes(
                        model, task_val_loader, tokenizer, device
                    )

                    train_loss = epoch_loss / (batch_idx + 1)
                    current_lr = scheduler.get_last_lr()[0]

                    history["train_loss"].append(train_loss)
                    history["val_task_accuracy"].append(task_metrics["accuracy"])
                    history["val_task_loss"].append(task_metrics["task_loss"])
                    history["learning_rates"].append(current_lr)
                    history["step"].append(global_step)

                    print(
                        f"Step {global_step:5d} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Task Accuracy: {task_metrics['accuracy']:.2%} "
                        f"({task_metrics['correct']}/{task_metrics['total']}) | "
                        f"LR: {current_lr:.2e}"
                    )

                    # Save best model based on task accuracy (not loss!)
                    if task_metrics["accuracy"] > best_task_accuracy:
                        best_task_accuracy = task_metrics["accuracy"]
                        save_checkpoint_with_task_metrics(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            task_metrics,
                            output_dir,
                            "best",
                        )
                        print(
                            f"  → Saved best model (accuracy: {task_metrics['accuracy']:.2%})"
                        )

                # Periodic checkpoint
                if global_step % config["save_every"] == 0:
                    save_checkpoint_with_task_metrics(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        task_metrics,
                        output_dir,
                        f"step_{global_step}",
                    )

        # End of epoch evaluation
        avg_epoch_loss = epoch_loss / len(train_loader)
        task_metrics = evaluate_task_outcomes(model, task_val_loader, tokenizer, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Task Accuracy: {task_metrics['accuracy']:.2%}")
        print(f"  Correct Answers: {task_metrics['correct']}/{task_metrics['total']}")

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"Best task accuracy: {best_task_accuracy:.2%}")
    print(f"Total tokens seen: {tokens_seen:,}")

    return history


def evaluate_task_outcomes(model, task_loader, tokenizer, device, max_batches=None):
    """
    Evaluate model on task-outcome dataset.

    Returns metrics dict with accuracy and counts.
    """
    model.eval()
    total_correct = 0
    total_problems = 0
    total_task_loss = 0.0

    if max_batches is None:
        max_batches = len(task_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(task_loader):
            if batch_idx >= max_batches:
                break

            # Use the task outcome loss function for evaluation
            loss, metrics = calculate_task_outcome_loss(
                model,
                batch,
                tokenizer,
                device,
                use_token_loss=False,  # Only task outcome for validation
            )

            total_correct += metrics["correct"]
            total_problems += metrics["total"]
            total_task_loss += metrics["task_loss"]

    model.train()

    num_batches = min(batch_idx + 1, max_batches)

    return {
        "accuracy": total_correct / total_problems if total_problems > 0 else 0.0,
        "correct": total_correct,
        "total": total_problems,
        "task_loss": total_task_loss / num_batches if num_batches > 0 else float("inf"),
    }


def save_checkpoint_with_task_metrics(
    model, optimizer, scheduler, step, task_metrics, output_dir, tag
):
    """Save checkpoint with task accuracy metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"checkpoint_{tag}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "task_accuracy": task_metrics["accuracy"],
            "task_loss": task_metrics["task_loss"],
            "model_config": model.cfg,
        },
        checkpoint_path,
    )


def generate_task_problems_file(output_path, num_problems=1000):
    """
    Generate task problems using your ALU expression generator.

    This assumes you have the math_expression_generator_vN.py available.
    """
    from math_expression_generator_vN import expression_generator_vN
    from alu_rpn_calculator_vN import alu_list_rpn_calculator

    problems = []

    print(f"Generating {num_problems} task problems...")

    # Generate problems in batches
    batch_size = 100
    for i in range(0, num_problems, batch_size):
        expressions = expression_generator_vN(batch_size)

        for word_expr, symbol_expr in expressions:
            # Calculate correct answer
            result = alu_list_rpn_calculator(symbol_expr)
            answer = str(result[3])  # The solution

            # Create problem variants
            problems.append(
                {
                    "problem": f"# Task: evaluate expression, What is {word_expr}? Answer:",
                    "answer": answer,
                    "answer_delimiters": ["```", "```"],
                }
            )

            problems.append(
                {
                    "problem": f"# Task: evaluate expression, {symbol_expr}= Answer:",
                    "answer": answer,
                    "answer_delimiters": ["```", "```"],
                }
            )

    # Shuffle and truncate
    import random

    random.shuffle(problems)
    problems = problems[:num_problems]

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Saved {len(problems)} problems to {output_path}")

    return problems


"""
chunk_based_data_loading.py

Utilities for loading training, validation, and test data from directories
containing pre-generated problem chunks in various formats (JSON, JSONL, TXT).

This module provides functions to load task-outcome based problems from
organized directory structures, supporting the ALU task-outcome training pipeline.
"""

import os
import json
import traceback
from pathlib import Path
from collections.abc import Iterator
import torch
from torch.utils.data import Dataset, DataLoader


def load_problems_from_chunk_file(
    chunk_file_path, expected_format="json", verbose=False
):
    """
    Load problems from a single chunk file.

    Supports multiple file formats:
    - JSON: Single list of problem dicts
    - JSONL: One problem dict per line
    - TXT: Custom text format (to be defined)

    Args:
        chunk_file_path: Path to chunk file
        expected_format: File format - 'json', 'jsonl', or 'txt'
        verbose: Print loading details

    Returns:
        list: List of problem dictionaries, each containing:
            - 'problem': Problem text prompt
            - 'answer': Correct answer string
            - 'answer_delimiters': List of [start_delim, end_delim]
            - 'problem_id': Unique identifier (optional)
            - 'metadata': Additional metadata dict (optional)

    Raises:
        FileNotFoundError: If chunk file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
        ValueError: If file format is unsupported or invalid

    Example:
        >>> problems = load_problems_from_chunk_file('train/chunk_001.json')
        >>> print(f"Loaded {len(problems)} problems")
        >>> print(problems[0]['problem'])
        '# Task: evaluate expression, What is 2+2? Answer:'
    """
    try:
        chunk_path = Path(chunk_file_path)

        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file_path}")

        if verbose:
            print(f"Loading chunk: {chunk_path.name}")

        problems_list = []

        # Determine format from file extension if not specified
        if expected_format == "auto":
            file_extension = chunk_path.suffix.lower()
            format_mapping = {
                ".json": "json",
                ".jsonl": "jsonl",
                ".txt": "txt",
            }
            expected_format = format_mapping.get(file_extension, "json")

        # Load based on format
        if expected_format == "json":
            with open(chunk_path, "r", encoding="utf-8") as json_file:
                problems_list = json.load(json_file)

                if not isinstance(problems_list, list):
                    raise ValueError(
                        f"JSON file must contain a list, got {type(problems_list)}"
                    )

        elif expected_format == "jsonl":
            with open(chunk_path, "r", encoding="utf-8") as jsonl_file:
                for line_number, line in enumerate(jsonl_file, start=1):
                    line_stripped = line.strip()

                    if not line_stripped:
                        continue  # Skip empty lines

                    try:
                        problem_dict = json.loads(line_stripped)
                        problems_list.append(problem_dict)

                    except json.JSONDecodeError as json_error:
                        print(
                            f"Warning: Skipping invalid JSON on line {line_number}: {json_error}"
                        )
                        if verbose:
                            print(f"  Line content: {line_stripped[:100]}...")
                        continue

        elif expected_format == "txt":
            # Custom text format parsing
            # This would need to be defined based on your specific text format
            # Example implementation for simple format:
            # Problem: <problem_text>
            # Answer: <answer>
            # ---

            with open(chunk_path, "r", encoding="utf-8") as txt_file:
                content = txt_file.read()

                # Split by delimiter
                problem_blocks = content.split("---")

                for block_index, block in enumerate(problem_blocks):
                    block_stripped = block.strip()

                    if not block_stripped:
                        continue

                    # Parse block
                    lines = block_stripped.split("\n")
                    problem_data = {}

                    for line in lines:
                        if line.startswith("Problem:"):
                            problem_data["problem"] = line.replace(
                                "Problem:", ""
                            ).strip()
                        elif line.startswith("Answer:"):
                            problem_data["answer"] = line.replace("Answer:", "").strip()

                    if "problem" in problem_data and "answer" in problem_data:
                        # Add default delimiters if not specified
                        if "answer_delimiters" not in problem_data:
                            problem_data["answer_delimiters"] = ["```", "```"]

                        problems_list.append(problem_data)

        else:
            raise ValueError(f"Unsupported format: {expected_format}")

        # Validate loaded problems
        for problem_index, problem in enumerate(problems_list):
            if not isinstance(problem, dict):
                raise ValueError(
                    f"Problem at index {problem_index} is not a dict: {type(problem)}"
                )

            required_keys = ["problem", "answer"]
            missing_keys = [key for key in required_keys if key not in problem]

            if missing_keys:
                raise ValueError(
                    f"Problem at index {problem_index} missing required keys: {missing_keys}"
                )

        if verbose:
            print(f"  ✓ Loaded {len(problems_list)} problems from {chunk_path.name}")

        return problems_list

    except Exception as loading_error:
        print(f"Error loading chunk file {chunk_file_path}: {loading_error}")
        traceback.print_exc()
        raise


def load_problems_from_directory(
    directory_path,
    file_pattern="*.json",
    format_type="auto",
    max_files=None,
    sort_files=True,
    verbose=True,
):
    """
    Load all problem chunks from a directory.

    This function scans a directory for chunk files matching a pattern,
    loads each file, and concatenates all problems into a single list.
    Useful for loading entire train/val/test sets from organized directories.

    Args:
        directory_path: Path to directory containing chunk files
        file_pattern: Glob pattern for matching files (e.g., '*.json', 'chunk_*.jsonl')
        format_type: File format - 'json', 'jsonl', 'txt', or 'auto' (detect from extension)
        max_files: Maximum number of files to load (None = load all)
        sort_files: Sort files alphabetically before loading
        verbose: Print loading progress

    Returns:
        dict: Dictionary containing:
            - 'problems': List of all problems from all chunks
            - 'chunk_count': Number of chunk files loaded
            - 'total_problems': Total number of problems loaded
            - 'chunk_files': List of chunk file names loaded
            - 'problems_per_chunk': Dict mapping chunk filename to problem count

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no matching files found

    Example:
        >>> # Load all training problems
        >>> train_data = load_problems_from_directory(
        ...     'data/train_chunks/',
        ...     file_pattern='*.json'
        ... )
        >>> print(f"Loaded {train_data['total_problems']} problems from {train_data['chunk_count']} chunks")

        >>> # Load only first 10 validation chunks
        >>> val_data = load_problems_from_directory(
        ...     'data/validation_chunks/',
        ...     file_pattern='val_*.jsonl',
        ...     max_files=10
        ... )
    """
    try:
        dir_path = Path(directory_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Loading problems from directory: {dir_path}")
            print(f"File pattern: {file_pattern}")
            print(f"{'=' * 60}")

        # Find all matching files
        chunk_files_list = list(dir_path.glob(file_pattern))

        if not chunk_files_list:
            raise ValueError(
                f"No files matching pattern '{file_pattern}' found in {directory_path}"
            )

        # Sort files for consistent ordering
        if sort_files:
            chunk_files_list = sorted(chunk_files_list)

        # Limit number of files if specified
        if max_files is not None:
            chunk_files_list = chunk_files_list[:max_files]

        if verbose:
            print(f"Found {len(chunk_files_list)} chunk files to load")

        # Load all problems from all chunks
        all_problems_list = []
        chunk_metadata = {}

        for chunk_file_path in chunk_files_list:
            try:
                chunk_problems = load_problems_from_chunk_file(
                    chunk_file_path, expected_format=format_type, verbose=verbose
                )

                all_problems_list.extend(chunk_problems)
                chunk_metadata[chunk_file_path.name] = len(chunk_problems)

            except Exception as chunk_load_error:
                print(
                    f"Warning: Failed to load chunk {chunk_file_path.name}: {chunk_load_error}"
                )
                if verbose:
                    traceback.print_exc()
                continue

        # Compile results
        results = {
            "problems": all_problems_list,
            "chunk_count": len(chunk_metadata),
            "total_problems": len(all_problems_list),
            "chunk_files": list(chunk_metadata.keys()),
            "problems_per_chunk": chunk_metadata,
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Loading Complete")
            print(f"{'=' * 60}")
            print(f"Total chunks loaded: {results['chunk_count']}")
            print(f"Total problems loaded: {results['total_problems']}")

            if results["total_problems"] > 0:
                avg_problems_per_chunk = (
                    results["total_problems"] / results["chunk_count"]
                )
                print(f"Average problems per chunk: {avg_problems_per_chunk:.1f}")

            print(f"{'=' * 60}\n")

        return results

    except Exception as directory_load_error:
        print(f"Error loading from directory {directory_path}: {directory_load_error}")
        traceback.print_exc()
        raise


class ChunkBasedTaskDataset(Dataset):
    """
    PyTorch Dataset for task-outcome problems loaded from chunk directories.

    This dataset loads problems from pre-generated chunk files and handles
    tokenization for training. Supports lazy loading for large datasets.

    Attributes:
        problems: List of problem dictionaries
        tokenizer: ByteTokenizer instance for encoding
        max_length: Maximum sequence length for truncation
        pad_token_id: Token ID for padding

    Example:
        >>> from byte_tokenizer import ByteTokenizer
        >>> tokenizer = ByteTokenizer()
        >>>
        >>> # Load training dataset
        >>> train_dataset = ChunkBasedTaskDataset(
        ...     chunk_directory='data/train_chunks/',
        ...     tokenizer=tokenizer,
        ...     max_length=1024
        ... )
        >>>
        >>> # Use with DataLoader
        >>> train_loader = DataLoader(
        ...     train_dataset,
        ...     batch_size=8,
        ...     shuffle=True,
        ...     collate_fn=collate_task_outcome_batch
        ... )
    """

    def __init__(
        self,
        chunk_directory,
        tokenizer,
        max_length=1024,
        file_pattern="*.json",
        format_type="auto",
        max_files=None,
        verbose=True,
    ):
        """
        Initialize chunk-based dataset.

        Args:
            chunk_directory: Directory containing chunk files
            tokenizer: Tokenizer with encode/decode methods
            max_length: Maximum sequence length
            file_pattern: Glob pattern for chunk files
            format_type: File format ('json', 'jsonl', 'txt', 'auto')
            max_files: Maximum number of chunk files to load
            verbose: Print loading progress
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.PAD_ID

        # Load all problems from directory
        loaded_data = load_problems_from_directory(
            directory_path=chunk_directory,
            file_pattern=file_pattern,
            format_type=format_type,
            max_files=max_files,
            sort_files=True,
            verbose=verbose,
        )

        self.problems = loaded_data["problems"]
        self.chunk_metadata = {
            "chunk_count": loaded_data["chunk_count"],
            "total_problems": loaded_data["total_problems"],
            "chunk_files": loaded_data["chunk_files"],
        }

        if verbose:
            print(f"Dataset initialized with {len(self.problems)} problems")

    def __len__(self):
        """Return total number of problems in dataset."""
        return len(self.problems)

    def __getitem__(self, index):
        """
        Get a single problem from dataset.

        Args:
            index: Problem index

        Returns:
            dict: Problem data with tokenized input
        """
        problem_data = self.problems[index]

        # Tokenize problem text
        problem_tokens = self.tokenizer.encode(problem_data["problem"])

        # Truncate if needed
        if len(problem_tokens) > self.max_length:
            problem_tokens = problem_tokens[: self.max_length]

        # Convert to tensor
        input_ids = torch.tensor(problem_tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "answer": problem_data["answer"],
            "problem_text": problem_data["problem"],
            "delimiters": problem_data.get("answer_delimiters", ["```", "```"]),
            "problem_id": problem_data.get("problem_id", f"problem_{index}"),
            "metadata": problem_data.get("metadata", {}),
        }


def collate_task_outcome_batch(batch_list):
    """
    Custom collate function for variable-length task-outcome problems.

    This function is used with DataLoader to batch variable-length sequences
    by padding them to the same length within each batch.

    Args:
        batch_list: List of dataset items (dicts) from __getitem__

    Returns:
        dict: Batched data with padded tensors and metadata
            - 'input_ids': Tensor of shape [batch_size, max_seq_len]
            - 'answers': List of answer strings
            - 'problem_texts': List of problem prompts
            - 'delimiters': List of delimiter pairs
            - 'problem_ids': List of problem IDs
            - 'attention_mask': Tensor indicating non-padded tokens

    Example:
        >>> train_loader = DataLoader(
        ...     train_dataset,
        ...     batch_size=8,
        ...     collate_fn=collate_task_outcome_batch
        ... )
        >>>
        >>> for batch in train_loader:
        ...     print(batch['input_ids'].shape)  # [8, max_len]
        ...     print(len(batch['answers']))     # 8
    """
    try:
        # Find maximum sequence length in this batch
        max_sequence_length = max(item["input_ids"].shape[0] for item in batch_list)

        # Initialize lists for batch data
        padded_input_ids_list = []
        attention_masks_list = []
        answers_list = []
        problem_texts_list = []
        delimiters_list = []
        problem_ids_list = []
        metadata_list = []

        # Process each item in batch
        for item in batch_list:
            input_ids = item["input_ids"]
            sequence_length = input_ids.shape[0]

            # Calculate padding needed
            padding_length = max_sequence_length - sequence_length

            # Pad input sequence
            if padding_length > 0:
                padding_tensor = torch.zeros(padding_length, dtype=torch.long)
                padded_input_ids = torch.cat([input_ids, padding_tensor])
            else:
                padded_input_ids = input_ids

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(sequence_length, dtype=torch.long)
            if padding_length > 0:
                padding_mask = torch.zeros(padding_length, dtype=torch.long)
                attention_mask = torch.cat([attention_mask, padding_mask])

            # Append to batch lists
            padded_input_ids_list.append(padded_input_ids)
            attention_masks_list.append(attention_mask)
            answers_list.append(item["answer"])
            problem_texts_list.append(item["problem_text"])
            delimiters_list.append(item["delimiters"])
            problem_ids_list.append(item["problem_id"])
            metadata_list.append(item["metadata"])

        # Stack tensors into batch
        batched_input_ids = torch.stack(padded_input_ids_list)
        batched_attention_masks = torch.stack(attention_masks_list)

        return {
            "input_ids": batched_input_ids,
            "attention_mask": batched_attention_masks,
            "answers": answers_list,
            "problem_texts": problem_texts_list,
            "delimiters": delimiters_list,
            "problem_ids": problem_ids_list,
            "metadata": metadata_list,
        }

    except Exception as collate_error:
        print(f"Error in collate function: {collate_error}")
        traceback.print_exc()
        raise


def create_chunk_based_dataloaders(
    train_dir,
    validation_dir,
    test_dir,
    tokenizer,
    config,
    train_file_pattern="*.json",
    val_file_pattern="*.json",
    test_file_pattern="*.json",
):
    """
    Create DataLoaders for training, validation, and testing from chunk directories.

    This is the main function to integrate chunk-based loading into the training pipeline.
    It creates three separate datasets and dataloaders from organized directory structures.

    Args:
        train_dir: Directory containing training chunk files
        validation_dir: Directory containing validation chunk files
        test_dir: Directory containing test/holdout chunk files
        tokenizer: ByteTokenizer instance
        config: Training configuration dict with keys:
            - 'context_length': Maximum sequence length
            - 'batch_size': Batch size for training
            - 'val_batch_size': Batch size for validation (optional, defaults to batch_size)
        train_file_pattern: Glob pattern for training files (default: '*.json')
        val_file_pattern: Glob pattern for validation files (default: '*.json')
        test_file_pattern: Glob pattern for test files (default: '*.json')

    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_info)
            - train_loader: DataLoader for training
            - val_loader: DataLoader for validation
            - test_loader: DataLoader for testing
            - dataset_info: Dict with statistics about loaded datasets

    Raises:
        FileNotFoundError: If any directory doesn't exist
        ValueError: If no valid chunk files found

    Example:
        >>> from byte_tokenizer import ByteTokenizer
        >>> tokenizer = ByteTokenizer()
        >>>
        >>> config = {
        ...     'context_length': 1024,
        ...     'batch_size': 8,
        ... }
        >>>
        >>> train_loader, val_loader, test_loader, info = create_chunk_based_dataloaders(
        ...     train_dir='data/train_chunks/',
        ...     validation_dir='data/validation_chunks/',
        ...     test_dir='data/test_chunks/',
        ...     tokenizer=tokenizer,
        ...     config=config
        ... )
        >>>
        >>> print(f"Training batches: {len(train_loader)}")
        >>> print(f"Validation batches: {len(val_loader)}")
        >>>
        >>> # Use in training loop
        >>> for batch in train_loader:
        ...     input_ids = batch['input_ids']  # [batch_size, seq_len]
        ...     answers = batch['answers']      # List of answer strings
    """
    try:
        print(f"\n{'=' * 60}")
        print("Creating Chunk-Based DataLoaders")
        print(f"{'=' * 60}")

        # Create training dataset
        print("\n[1/3] Loading Training Data")
        train_dataset = ChunkBasedTaskDataset(
            chunk_directory=train_dir,
            tokenizer=tokenizer,
            max_length=config["context_length"],
            file_pattern=train_file_pattern,
            verbose=True,
        )

        # Create validation dataset
        print("\n[2/3] Loading Validation Data")
        val_dataset = ChunkBasedTaskDataset(
            chunk_directory=validation_dir,
            tokenizer=tokenizer,
            max_length=config["context_length"],
            file_pattern=val_file_pattern,
            verbose=True,
        )

        # Create test dataset
        print("\n[3/3] Loading Test Data")
        test_dataset = ChunkBasedTaskDataset(
            chunk_directory=test_dir,
            tokenizer=tokenizer,
            max_length=config["context_length"],
            file_pattern=test_file_pattern,
            verbose=True,
        )

        # Get batch sizes
        train_batch_size = config["batch_size"]
        val_batch_size = config.get("val_batch_size", config["batch_size"])

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_task_outcome_batch,
            num_workers=0,  # Set to 0 for debugging, increase for production
            drop_last=True,  # Drop incomplete final batch
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_task_outcome_batch,
            num_workers=0,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_task_outcome_batch,
            num_workers=0,
            drop_last=False,
        )

        # Compile dataset information
        dataset_info = {
            "train": {
                "num_problems": len(train_dataset),
                "num_batches": len(train_loader),
                "chunk_count": train_dataset.chunk_metadata["chunk_count"],
                "batch_size": train_batch_size,
            },
            "validation": {
                "num_problems": len(val_dataset),
                "num_batches": len(val_loader),
                "chunk_count": val_dataset.chunk_metadata["chunk_count"],
                "batch_size": val_batch_size,
            },
            "test": {
                "num_problems": len(test_dataset),
                "num_batches": len(test_loader),
                "chunk_count": test_dataset.chunk_metadata["chunk_count"],
                "batch_size": val_batch_size,
            },
        }

        # Print summary
        print(f"\n{'=' * 60}")
        print("DataLoader Creation Complete")
        print(f"{'=' * 60}")

        print(f"\nTraining:")
        print(f"  Problems: {dataset_info['train']['num_problems']:,}")
        print(f"  Batches: {dataset_info['train']['num_batches']:,}")
        print(f"  Chunks: {dataset_info['train']['chunk_count']}")

        print(f"\nValidation:")
        print(f"  Problems: {dataset_info['validation']['num_problems']:,}")
        print(f"  Batches: {dataset_info['validation']['num_batches']:,}")
        print(f"  Chunks: {dataset_info['validation']['chunk_count']}")

        print(f"\nTest:")
        print(f"  Problems: {dataset_info['test']['num_problems']:,}")
        print(f"  Batches: {dataset_info['test']['num_batches']:,}")
        print(f"  Chunks: {dataset_info['test']['chunk_count']}")

        print(f"\n{'=' * 60}\n")

        return train_loader, val_loader, test_loader, dataset_info

    except Exception as dataloader_creation_error:
        print(f"Error creating dataloaders: {dataloader_creation_error}")
        traceback.print_exc()
        raise


"""
Document recommended directory structure for chunk-based datasets.

This function doesn't execute anything - it's documentation showing
how to organize your chunk files for this loading system.

Recommended Structure:

project_root/
├── data/
│   ├── train_chunks/
│   │   ├── train_chunk_0001.json
│   │   ├── train_chunk_0002.json
│   │   ├── train_chunk_0003.json
│   │   └── ... (many more)
│   │
│   ├── validation_chunks/
│   │   ├── validation_basic_0001.json
│   │   ├── validation_multistep_0001.json
│   │   ├── validation_negative_0001.json
│   │   └── validation_index.json  (optional metadata)
│   │
│   ├── test_chunks/
│   │   ├── holdout_comprehensive_0001.json
│   │   ├── holdout_edge_cases_0001.json
│   │   └── test_index.json  (optional metadata)
│   │
│   └── generation_logs/
│       ├── train_generation_log.json
│       ├── validation_generation_log.json
│       └── test_generation_log.json
│
└── train_perseid_byte.py


Each JSON chunk file contains:
[
    {
        "problem": "# Task: evaluate expression, What is 2+2? Answer:",
        "answer": "4",
        "answer_delimiters": ["```", "```"],
        "problem_id": "train_0001_0001",
        "metadata": {
            "chunk_file": "train_chunk_0001.json",
            "seed": 42,
            "generation_timestamp": "2025-01-15T10:30:00",
            "difficulty": "basic"
        }
    },
    {
        "problem": "# Task: evaluate expression, 7-3+2= Answer:",
        "answer": "6",
        "answer_delimiters": ["```", "```"],
        "problem_id": "train_0001_0002",
        "metadata": {...}
    },
    ...
]


Usage in training script:

    train_loader, val_loader, test_loader, info = create_chunk_based_dataloaders(
        train_dir='data/train_chunks/',
        validation_dir='data/validation_chunks/',
        test_dir='data/test_chunks/',
        tokenizer=tokenizer,
        config=TRAINING_CONFIG
    )
"""


"""
Example showing how to integrate chunk-based loading into existing pipeline.

This replaces the document-based loading in the original train_perseid_byte.py
"""

# BEFORE (original document-based approach):
"""
document_text = load_document(DOCUMENT_PATH)
train_loader, val_loader = create_data_loaders(
    document_text,
    tokenizer,
    TRAINING_CONFIG,
    train_ratio=TRAIN_VAL_SPLIT
)
"""

# AFTER (chunk-based approach):
"""
from chunk_based_data_loading import create_chunk_based_dataloaders

# Define chunk directories
TRAIN_CHUNKS_DIR = './data/train_chunks/'
VAL_CHUNKS_DIR = './data/validation_chunks/'
TEST_CHUNKS_DIR = './data/test_chunks/'

# Create dataloaders from chunks
train_loader, val_loader, test_loader, dataset_info = create_chunk_based_dataloaders(
    train_dir=TRAIN_CHUNKS_DIR,
    validation_dir=VAL_CHUNKS_DIR,
    test_dir=TEST_CHUNKS_DIR,
    tokenizer=tokenizer,
    config=TRAINING_CONFIG,
    train_file_pattern='train_*.json',
    val_file_pattern='validation_*.json',
    test_file_pattern='holdout_*.json'
)

# Rest of training pipeline remains the same
history = train_model_task_outcome(
    model,
    train_loader,
    val_loader,
    TRAINING_CONFIG,
    DEVICE,
    output_dir,
    training_state,
    tokenizer
)

# Final evaluation on held-out test set
final_test_results = evaluate_task_outcomes(
    model,
    test_loader,
    tokenizer,
    DEVICE
)
"""


"""
if __name__ == "__main__":
    # Example usage and testing
    print("Chunk-Based Data Loading Module")
    print("=" * 60)
    print("\nThis module provides utilities for loading task-outcome problems")
    print("from organized chunk directories.")
    print("\nSee example_directory_structure_documentation() for details.")
    print("\nKey functions:")
    print("  - load_problems_from_chunk_file(): Load single chunk")
    print("  - load_problems_from_directory(): Load all chunks from directory")
    print("  - ChunkBasedTaskDataset: PyTorch Dataset class")
    print("  - create_chunk_based_dataloaders(): Main integration function")
    print("\nFor integration with training pipeline, use:")
    print("  create_chunk_based_dataloaders(train_dir, val_dir, test_dir, ...)")
"""

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


"""
# for task outcomes
# In main():

# Generate task problems
generate_task_problems_file('./data/alu_train_problems.json', num_problems=5000)
generate_task_problems_file('./data/alu_val_problems.json', num_problems=1000)

# Create task-outcome dataset
task_train_dataset = TaskOutcomeDataset(
    './data/alu_train_problems.json',
    tokenizer,
    TRAINING_CONFIG["context_length"]
)

task_val_dataset = TaskOutcomeDataset(
    './data/alu_val_problems.json',
    tokenizer,
    TRAINING_CONFIG["context_length"]
)

task_train_loader = DataLoader(
    task_train_dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_task_outcome_batch
)

task_val_loader = DataLoader(
    task_val_dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    shuffle=False,
    collate_fn=collate_task_outcome_batch
)

# Use new training loop
history = train_model_task_outcome(
    model,
    train_loader,  # Traditional corpus (optional, could use task_train_loader)
    task_val_loader,  # Task-outcome validation
    TRAINING_CONFIG,
    DEVICE,
    output_dir,
    training_state,
    tokenizer
)
"""
