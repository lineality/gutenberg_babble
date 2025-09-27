# train_classifier_perseid_byte.py
"""
Training script for PerseidByte classifier.
Supports two-stage training: frozen backbone → fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import seaborn as sns

# Import our modules
from byte_tokenizer import ByteTokenizer
from perseid_model import PERSEID_BYTE_CONFIG_BASE
from perseidbyte288_classification_config import (
    PerseidByteClassifier,
    ClassificationConfig,
)
from perseid_byte_data_utils import load_imdb_dataset


def evaluate_model(
    model: PerseidByteClassifier, data_loader: DataLoader, device: str, num_classes: int
) -> dict:
    """
    Evaluate model and compute metrics.

    Returns dict with loss, accuracy, precision, recall, f1
    """
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits, labels)

            # Get predictions
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())  # Convert to Python list
            all_labels.extend(labels.cpu().numpy().tolist())  # Convert to Python list
            total_loss += loss.item()
            num_batches += 1

    # Calculate metrics with zero_division handling
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    model.train()

    return {
        "loss": total_loss / num_batches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def train_epoch(
    model: PerseidByteClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def plot_training_history(history: dict, save_path: Path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss plot
    axes[0, 0].plot(history["train_loss"], label="Train", marker="o")
    axes[0, 0].plot(history["val_loss"], label="Val", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(history["train_acc"], label="Train", marker="o")
    axes[0, 1].plot(history["val_acc"], label="Val", marker="s")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy Curves")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score plot
    axes[1, 0].plot(history["train_f1"], label="Train", marker="o")
    axes[1, 0].plot(history["val_f1"], label="Val", marker="s")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("F1 Score Curves")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate plot (if available)
    if "learning_rates" in history:
        axes[1, 1].plot(history["learning_rates"], marker="o")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=150)
    plt.close()


def safe_torch_save(obj, path):
    """Save checkpoint with only Python/PyTorch native types."""
    # Ensure all numpy arrays are converted to Python types or tensors
    if isinstance(obj, dict):
        clean_obj = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                clean_obj[k] = v.tolist()  # Convert numpy arrays to lists
            elif isinstance(v, (np.integer, np.floating)):
                clean_obj[k] = v.item()  # Convert numpy scalars to Python types
            else:
                clean_obj[k] = v
        torch.save(clean_obj, path)
    else:
        torch.save(obj, path)


def safe_torch_load(path, device):
    """Load checkpoint with weights_only=False for compatibility."""
    return torch.load(path, map_location=device, weights_only=False)


def main():
    """Main training pipeline."""

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Paths
    DATA_PATH = "data/IMDB Dataset.csv"
    OUTPUT_DIR = Path("models/perseid_classifier_imdb")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    # Classification config
    cls_config = ClassificationConfig(
        max_seq_length=1024,
        chunking_strategy="A",
        chunk_overlap=0.1,
        classifier_hidden_dim=256,
        dropout_rate=0.1,
        pooling_method="mean",
        training_mode="freeze_finetune",
        stage1_epochs=5,
        stage2_epochs=3,
        stage1_lr=1e-3,
        stage2_backbone_lr=1e-5,
        stage2_head_lr=1e-4,
        batch_size=8,
        train_val_split=0.8,
    )

    # Validate config
    is_valid, errors, warnings = cls_config.validate()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    # ============================================================
    # DATA LOADING
    # ============================================================

    print("\n📊 Loading data...")
    tokenizer = ByteTokenizer()

    train_loader, val_loader = load_imdb_dataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_seq_length=cls_config.max_seq_length,
        chunking_strategy=cls_config.chunking_strategy,
        chunk_overlap=cls_config.chunk_overlap,
        train_val_split=cls_config.train_val_split,
        batch_size=cls_config.batch_size,
        sample_size=2000,  # Increased sample size for better balance
    )

    # Print class distribution
    print("\n📊 Checking class distribution...")
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch["labels"].numpy().tolist())
    
    val_labels = []
    for batch in val_loader:
        val_labels.extend(batch["labels"].numpy().tolist())
    
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    
    print(f"  Train: {train_neg} negative, {train_pos} positive ({train_pos/(train_pos+train_neg)*100:.1f}% positive)")
    print(f"  Val: {val_neg} negative, {val_pos} positive ({val_pos/(val_pos+val_neg)*100:.1f}% positive)")

    # ============================================================
    # MODEL SETUP
    # ============================================================

    print("\n🤖 Initializing model...")

    backbone_config = PERSEID_BYTE_CONFIG_BASE.copy()
    if device == "cpu":
        backbone_config["dtype"] = torch.float32

    model = PerseidByteClassifier(
        backbone_config=backbone_config,
        num_classes=2,
        classification_config=cls_config,
        device=device,
    )

    # Load pretrained weights if available
    pretrained_path = Path("models/perseid_256m_alice/perseid_model_final.pth")
    if pretrained_path.exists():
        print(f"📦 Loading pretrained weights from {pretrained_path}")
        try:
            state_dict = safe_torch_load(pretrained_path, device)
            model.backbone.load_state_dict(state_dict, strict=False)
            print("  ✓ Pretrained weights loaded")
        except Exception as e:
            print(f"  ⚠️ Could not load pretrained weights: {e}")
    else:
        print("  ⚠️ No pretrained weights found, training from scratch")

    # Print model info
    info = model.get_model_info()
    print(f"\n📈 Model Statistics:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Classifier parameters: {info['classifier_parameters']:,}")

    # ============================================================
    # TRAINING
    # ============================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "learning_rates": [],
    }

    best_val_acc = 0.0

    # Stage 1: Frozen backbone training
    if cls_config.training_mode in ["freeze_finetune", "frozen_only"]:
        print(f"\n{'=' * 60}")
        print("STAGE 1: FROZEN BACKBONE TRAINING")
        print(f"{'=' * 60}")

        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cls_config.stage1_lr,
            weight_decay=0.01,
        )

        for epoch in range(cls_config.stage1_epochs):
            print(f"\n📅 Stage 1 - Epoch {epoch + 1}/{cls_config.stage1_epochs}")

            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

            # Evaluate
            train_metrics = evaluate_model(model, train_loader, device, num_classes=2)
            val_metrics = evaluate_model(model, val_loader, device, num_classes=2)

            # Save history (convert all to Python floats)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_metrics["loss"]))
            history["train_acc"].append(float(train_metrics["accuracy"]))
            history["val_acc"].append(float(val_metrics["accuracy"]))
            history["train_f1"].append(float(train_metrics["f1"]))
            history["val_f1"].append(float(val_metrics["f1"]))
            history["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))

            print(f"\n📊 Stage 1 - Epoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": cls_config.__dict__,
                    "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                               for k, v in val_metrics.items() if k not in ["predictions", "labels"]},
                    "epoch": epoch,
                }
                safe_torch_save(checkpoint, OUTPUT_DIR / "best_model_stage1.pth")
                print(f"  💾 Saved best model (Val Acc: {best_val_acc:.4f})")

    # Stage 2: Fine-tuning
    if cls_config.training_mode == "freeze_finetune":
        print(f"\n{'=' * 60}")
        print("STAGE 2: END-TO-END FINE-TUNING")
        print(f"{'=' * 60}")

        model.unfreeze_backbone()

        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cls_config.stage2_backbone_lr},
                {"params": model.classifier_head.parameters(), "lr": cls_config.stage2_head_lr},
            ],
            weight_decay=0.01,
        )

        for epoch in range(cls_config.stage2_epochs):
            print(f"\n📅 Stage 2 - Epoch {epoch + 1}/{cls_config.stage2_epochs}")

            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

            # Evaluate
            train_metrics = evaluate_model(model, train_loader, device, num_classes=2)
            val_metrics = evaluate_model(model, val_loader, device, num_classes=2)

            # Save history
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_metrics["loss"]))
            history["train_acc"].append(float(train_metrics["accuracy"]))
            history["val_acc"].append(float(val_metrics["accuracy"]))
            history["train_f1"].append(float(train_metrics["f1"]))
            history["val_f1"].append(float(val_metrics["f1"]))
            history["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))

            print(f"\n📊 Stage 2 - Epoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": cls_config.__dict__,
                    "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                               for k, v in val_metrics.items() if k not in ["predictions", "labels"]},
                    "epoch": epoch + cls_config.stage1_epochs,
                }
                safe_torch_save(checkpoint, OUTPUT_DIR / "best_model_final.pth")
                print(f"  💾 Saved best model (Val Acc: {best_val_acc:.4f})")

    # ============================================================
    # FINAL EVALUATION & VISUALIZATION
    # ============================================================

    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION")
    print(f"{'=' * 60}")

    # Load best model
    best_model_path = (OUTPUT_DIR / "best_model_final.pth" 
                      if (OUTPUT_DIR / "best_model_final.pth").exists() 
                      else OUTPUT_DIR / "best_model_stage1.pth")
    
    if best_model_path.exists():
        checkpoint = safe_torch_load(best_model_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"📦 Loaded best model from {best_model_path}")
    else:
        print("⚠️ No saved model found, using current model state")

    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, device, num_classes=2)

    print(f"\n🏆 Best Model Performance:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  F1 Score: {final_metrics['f1']:.4f}")

    # Plot training history
    plot_training_history(history, OUTPUT_DIR)
    print(f"\n📊 Training curves saved to {OUTPUT_DIR / 'training_curves.png'}")

    # Plot confusion matrix
    plot_confusion_matrix(
        final_metrics["labels"],
        final_metrics["predictions"],
        class_names=["Negative", "Positive"],
        save_path=OUTPUT_DIR,
    )
    print(f"📊 Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")

    # Save training history
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
