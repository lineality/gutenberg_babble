
"""
train_perseid_documents.py

Training module for Perseid models on text document corpus.
Handles single document input with configurable train/val split.
Trains from scratch (no pretrained weights).

Usage:
    1. Set configuration parameters at top of file
    2. Run: python train_perseid_documents.py
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
import urllib.request
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Import model architecture and configuration tools
from gemma_model import Gemma3Model
from perseid_config_tools import create_perseid_config, calculate_model_params, validate_config


# ============================================================================
# USER CONFIGURATION SECTION - MODIFY THESE SETTINGS
# ============================================================================

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

# Document input
DOCUMENT_PATH = file_path  # "./data/my_document.txt"  # Path to your text file

# Model configuration
MODEL_SIZE = 256  # Options: 256, 288, 320 (millions of parameters)
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
    "context_length": 512,  # Context window for training
    "batch_size": 1,  # Batch size (increase if memory allows)
    "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
    "learning_rate": 5e-4,  # Learning rate
    "num_epochs": 3,  # Number of training epochs
    "weight_decay": 0.01,  # Weight decay for AdamW
    "warmup_steps": 100,  # Warmup steps for learning rate
    "eval_every": 50,  # Evaluate every N steps
    "eval_batches": 10,  # Number of batches for evaluation
    "save_every": 500,  # Save checkpoint every N steps
    "chunk_overlap": 0.1,  # Overlap between text chunks (0.0 to 0.5)
}

# Output configuration
OUTPUT_DIR = f"./models/perseid_{MODEL_SIZE}m_{Path(DOCUMENT_PATH).stem}/"
EXPERIMENT_NAME = f"perseid_{MODEL_SIZE}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Hardware settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BFLOAT16 = torch.cuda.is_available()  # Use bfloat16 if on GPU

# ============================================================================
# END USER CONFIGURATION
# ============================================================================


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
                window = self.tokens[i:i + max_length + 1]  # +1 for target
                if len(window) == max_length + 1:
                    self.windows.append(window)

            if verbose:
                print(f"Created {len(self.windows):,} training windows")
                print(f"Total tokens: {len(self.tokens):,}")
                print(f"Window size: {max_length}, Stride: {stride}")
                print(f"Effective overlap: {((max_length - stride) / max_length * 100):.1f}%")

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
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"Successfully loaded with {encoding} encoding")
                print(f"Document length: {len(text):,} characters")
                return text
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode file with any supported encoding: {encodings_to_try}")

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
        print(f"\nCreating data loaders with {train_ratio:.0%} train / {(1-train_ratio):.0%} validation split")

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
            train_text,
            tokenizer,
            config["context_length"],
            stride,
            verbose=True
        )

        val_dataset = DocumentDataset(
            val_text,
            tokenizer,
            config["context_length"],
            config["context_length"],  # No overlap for validation
            verbose=True
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0
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
                print(f"WARNING: Force restart mode - existing checkpoint will be overwritten!")
                print(f"Existing checkpoint: {checkpoint_path}")
                response = input("Are you sure? Type 'yes' to continue: ")
                if response.lower() != 'yes':
                    print("Aborting to preserve existing model.")
                    sys.exit(1)
            checkpoint_exists = False  # Treat as new training

        # Load or create model configuration
        if checkpoint_exists and training_mode == "continue":
            print(f"\n{'='*60}")
            print(f"Resuming Training from Checkpoint")
            print(f"{'='*60}")
            print(f"Loading from: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Extract configuration from checkpoint
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                print("  ✓ Loaded model configuration from checkpoint")
            else:
                # Fallback: generate config if not in checkpoint
                print("  ! No config in checkpoint, generating from parameters")
                model_config = create_perseid_config(
                    target_size_millions=model_size,
                    strategy=strategy
                )

            # Override context length for current training
            model_config["context_length"] = config["context_length"]

            # Set dtype
            if USE_BFLOAT16 and device != "cpu":
                model_config["dtype"] = torch.bfloat16
            else:
                model_config["dtype"] = torch.float32

            # Initialize model with configuration
            model = Gemma3Model(model_config)

            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("  ✓ Loaded model weights")
            else:
                # Old-style checkpoint with just state dict
                model.load_state_dict(checkpoint)
                print("  ✓ Loaded model weights (legacy format)")

            # Move to device
            model = model.to(device)

            # Extract training state if available
            training_state = {
                'global_step': checkpoint.get('step', 0),
                'best_val_loss': checkpoint.get('val_loss', float('inf')),
                'optimizer_state': checkpoint.get('optimizer_state_dict', None),
                'scheduler_state': checkpoint.get('scheduler_state_dict', None),
                'epoch': checkpoint.get('epoch', 0),
                'tokens_seen': checkpoint.get('tokens_seen', 0)
            }

            print(f"\nResuming from:")
            print(f"  - Step: {training_state['global_step']:,}")
            print(f"  - Epoch: {training_state['epoch']}")
            print(f"  - Best validation loss: {training_state['best_val_loss']:.4f}")
            print(f"  - Tokens seen: {training_state['tokens_seen']:,}")

        else:
            # Starting fresh
            print(f"\n{'='*60}")
            print(f"Initializing New Model")
            print(f"{'='*60}")
            print(f"Creating Perseid-{model_size}M ({strategy} strategy)")

            # Generate Perseid configuration
            model_config = create_perseid_config(
                target_size_millions=model_size,
                strategy=strategy
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
            model = Gemma3Model(model_config)

            # Move to device
            model = model.to(device)

            # Fresh training state
            training_state = {
                'global_step': 0,
                'best_val_loss': float('inf'),
                'optimizer_state': None,
                'scheduler_state': None,
                'epoch': 0,
                'tokens_seen': 0
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
        loss = nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten()
        )

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


def train_model(model, train_loader, val_loader, config, device, output_dir, training_state):
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
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95)
        )

        # Restore optimizer state if resuming
        if training_state['optimizer_state'] is not None:
            optimizer.load_state_dict(training_state['optimizer_state'])
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
                progress = (step - config["warmup_steps"]) / (total_steps - config["warmup_steps"])
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        # Restore scheduler state if resuming
        if training_state['scheduler_state'] is not None:
            scheduler.load_state_dict(training_state['scheduler_state'])
            print("  ✓ Restored scheduler state")

        # Training state
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "step": []
        }

        # global_step = 0
        # best_val_loss = float('inf')
        # tokens_seen = 0
        global_step = training_state['global_step']
        best_val_loss = training_state['best_val_loss']
        tokens_seen = training_state['tokens_seen']
        start_epoch = training_state['epoch']

        print(f"\nStarting from epoch {start_epoch + 1}, step {global_step}")

        print(f"Total training steps: {total_steps:,}")
        print(f"Warmup steps: {config['warmup_steps']:,}")
        print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")

        # Training loop
        # for epoch in range(config["num_epochs"]):
        for epoch in range(start_epoch, config["num_epochs"]):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            print(f"{'='*40}")

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
                            model, val_loader, device,
                            num_batches=config["eval_batches"]
                        )

                        train_loss = epoch_loss / (batch_idx + 1)
                        current_lr = scheduler.get_last_lr()[0]

                        history["train_loss"].append(train_loss)
                        history["val_loss"].append(val_loss)
                        history["learning_rates"].append(current_lr)
                        history["step"].append(global_step)

                        print(f"Step {global_step:5d} | "
                              f"Train Loss: {train_loss:.4f} | "
                              f"Val Loss: {val_loss:.4f} | "
                              f"LR: {current_lr:.2e} | "
                              f"Tokens: {tokens_seen:,}")

                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(
                                model, optimizer, scheduler,
                                global_step, best_val_loss,
                                output_dir, "best"
                            )
                            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

                    # Periodic checkpoint
                    if global_step % config["save_every"] == 0:
                        save_checkpoint(
                            model, optimizer, scheduler,
                            global_step, val_loss,
                            output_dir, f"step_{global_step}"
                        )

            # End of epoch evaluation
            avg_epoch_loss = epoch_loss / len(train_loader)
            val_loss = evaluate_model(model, val_loader, device)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
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

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step,
            'val_loss': val_loss,
            'model_config': model.cfg,
        }, checkpoint_path)

    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        traceback.print_exc()


def save_training_results(model, model_config, history, output_dir):
    """
    Save final model and training artifacts.

    Args:
        model: Trained model
        model_config: Model configuration
        history: Training history
        output_dir: Output directory
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving training results to {output_dir}")

        # Save final model weights
        model_path = output_dir / "perseid_model_final.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Model weights saved to {model_path}")

        # Save configuration
        config_path = output_dir / "model_config.json"
        config_save = {k: v for k, v in model_config.items() if k != "dtype"}
        config_save["dtype"] = str(model_config["dtype"])
        with open(config_path, 'w') as f:
            json.dump(config_save, f, indent=2)
        print(f"  ✓ Configuration saved to {config_path}")

        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  ✓ Training history saved to {history_path}")

        # Plot and save training curves
        if len(history["train_loss"]) > 0:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(history["step"], history["train_loss"], label="Train Loss")
            plt.plot(history["step"], history["val_loss"], label="Val Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Progress")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 3, 2)
            val_perplexity = [torch.exp(torch.tensor(loss)).item() for loss in history["val_loss"]]
            plt.plot(history["step"], val_perplexity)
            plt.xlabel("Step")
            plt.ylabel("Perplexity")
            plt.title("Validation Perplexity")
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 3, 3)
            plt.plot(history["step"], history["learning_rates"])
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / "training_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Training curves saved to {plot_path}")

        # Save training configuration
        train_config_path = output_dir / "training_config.json"
        with open(train_config_path, 'w') as f:
            json.dump(TRAINING_CONFIG, f, indent=2)
        print(f"  ✓ Training config saved to {train_config_path}")

        print(f"\nAll outputs saved to: {output_dir}")

    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()


def main():
    """
    Main training pipeline for Perseid document training.
    """
    try:
        print(f"\n{'='*60}")
        print("Perseid Document Training Pipeline")
        print(f"{'='*60}")
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
        print(f"\n{'='*40}")
        print("Step 1: Loading Document")
        print(f"{'='*40}")
        document_text = load_document(DOCUMENT_PATH)

        # 2. Setup model and tokenizer
        print(f"\n{'='*40}")
        print("Step 2: Setting Up Model")
        print(f"{'='*40}")

        # For MVP, we'll use a simple tokenizer
        # In production, this would use the Gemma tokenizer
        print("\nNote: Using placeholder tokenizer for MVP")
        print("Production version would use GemmaTokenizer")

        # # Placeholder tokenizer class for MVP
        # class SimpleTokenizer:
        #     def encode(self, text):
        #         # Simple character-level tokenizer for MVP
        #         return [ord(c) % 256 for c in text]

        #     def decode(self, ids):
        #         return ''.join([chr(i) for i in ids])

        # tokenizer = SimpleTokenizer()
        #

        class GemmaTokenizer:
            """Tokenizer for Gemma 3 model"""
            def __init__(self, tokenizer_file_path: str):
                tok_file = Path(tokenizer_file_path)
                self._tok = Tokenizer.from_file(str(tok_file))
                self.eos_token = "<end_of_turn>"
                self.pad_token = "<end_of_turn>"

            def encode(self, text: str) -> list[int]:
                return self._tok.encode(text).ids

            def decode(self, ids: list[int]) -> str:
                return self._tok.decode(ids, skip_special_tokens=False)

        ###############################
        # Download and setup tokenizer
        ###############################

        repo_id = "google/gemma-3-270m"
        local_dir = Path(repo_id).parts[-1]

        from huggingface_hub import hf_hub_download

        # Download tokenizer
        tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file_path):
            tokenizer_file_path = hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=local_dir
            )

        tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)

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
        print(f"\n{'='*40}")
        print("Step 3: Preparing Data")
        print(f"{'='*40}")
        train_loader, val_loader = create_data_loaders(
            document_text,
            tokenizer,
            TRAINING_CONFIG,
            train_ratio=TRAIN_VAL_SPLIT
        )

        # 4. Train model
        print(f"\n{'='*40}")
        print("Step 4: Training Model")
        print(f"{'='*40}")
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
        print(f"\n{'='*40}")
        print("Step 5: Saving Results")
        print(f"{'='*40}")
        save_training_results(
            model,
            model_config,
            history,
            output_dir
        )

        print(f"\n{'='*60}")
        print("Training Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Model and results saved to: {output_dir}")

        return model, history

    except Exception as e:
        print(f"\n{'='*60}")
        print("Training Pipeline Failed")
        print(f"{'='*60}")
        print(f"Error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the training pipeline
    model, history = main()
