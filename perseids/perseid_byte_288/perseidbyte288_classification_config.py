# # perseidbyte288_classification_config.py

# perseid_byte_classifier.py
"""
Complete classification system for PerseidByte - all-in-one file.
We'll split it into modules after it works.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
from perseid_model import PerseidByteModel, PERSEID_BYTE_CONFIG_BASE


@dataclass
class ClassificationConfig:
    """User-configurable classification settings."""

    # Sequence Processing
    max_seq_length: int = 1024
    chunking_strategy: Literal["A", "B", "A&B"] = "A"
    chunk_overlap: float = 0.1

    # Model Architecture
    classifier_hidden_dim: int = 256
    dropout_rate: float = 0.1
    pooling_method: Literal["mean", "last_token"] = "mean"

    # Training Strategy
    training_mode: Literal["freeze_finetune", "end_to_end", "frozen_only"] = (
        "freeze_finetune"
    )
    stage1_epochs: int = 8
    stage2_epochs: int = 5
    stage1_lr: float = 1e-3
    stage2_backbone_lr: float = 1e-5
    stage2_head_lr: float = 1e-4

    # Data Processing
    batch_size: int = 16
    train_val_split: float = 0.8

    def validate(self):
        """Validate configuration parameters."""
        errors = []
        warnings = []

        if not (512 <= self.max_seq_length <= 32768):
            errors.append(
                f"max_seq_length {self.max_seq_length} not in range [512, 32768]"
            )

        if not (0.0 <= self.chunk_overlap <= 0.5):
            errors.append(f"chunk_overlap {self.chunk_overlap} not in range [0.0, 0.5]")

        if self.batch_size > 32:
            warnings.append(
                f"batch_size {self.batch_size} is large, may cause memory issues"
            )

        return len(errors) == 0, errors, warnings


class PerseidByteClassifier(nn.Module):
    """
    Classification model built on PerseidByte backbone.
    """

    def __init__(
        self,
        backbone_config: dict,
        num_classes: int,
        classification_config: ClassificationConfig,
        device: str = "cpu",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.config = classification_config
        self.backbone_config = backbone_config
        self.device = device

        # Initialize backbone transformer
        self.backbone = PerseidByteModel(backbone_config)

        # Get embedding dimension and dtype from backbone config
        emb_dim = backbone_config["emb_dim"]
        hidden_dim = classification_config.classifier_hidden_dim
        dropout = classification_config.dropout_rate
        dtype = backbone_config.get("dtype", torch.float32)

        # Classification head with matching dtype
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes, dtype=dtype),
        )

        # Move model to device
        self.to(device)

        # Track training stage
        self.training_stage = 1

    def freeze_backbone(self):
        """Freeze backbone parameters for Stage 1 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.training_stage = 1
        print("🔒 Backbone frozen - Stage 1 training (classifier only)")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for Stage 2 training."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.training_stage = 2
        print("🔓 Backbone unfrozen - Stage 2 training (end-to-end)")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass - we'll add chunking strategies after basic works.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Ensure input is on correct device
        input_ids = input_ids.to(self.device)

        # Truncate if too long
        if input_ids.size(1) > self.config.max_seq_length:
            input_ids = input_ids[:, : self.config.max_seq_length]

        # Get embeddings from backbone
        embeddings = self.backbone.get_embeddings(
            input_ids, pooling_method=self.config.pooling_method
        )

        # Ensure embeddings have correct dtype for classifier
        embeddings = embeddings.to(self.classifier_head[1].weight.dtype)

        # Classify
        logits = self.classifier_head(embeddings)

        return logits

    def get_model_info(self) -> dict:
        """Get model information for logging/saving."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": sum(p.numel() for p in self.backbone.parameters()),
            "classifier_parameters": sum(
                p.numel() for p in self.classifier_head.parameters()
            ),
            "training_stage": self.training_stage,
            "num_classes": self.num_classes,
        }


# Test the implementation
if __name__ == "__main__":
    print("🧪 Testing PerseidByte Classification Framework")
    print("=" * 60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test configuration
    config = ClassificationConfig(
        max_seq_length=512, chunking_strategy="A", batch_size=4
    )

    is_valid, errors, warnings = config.validate()
    print(f"\nConfig validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    for warning in warnings:
        print(f"  ⚠️  {warning}")

    # Test model initialization
    print(f"\nInitializing classifier...")

    # Modify backbone config for testing (ensure dtype is set)
    test_backbone_config = PERSEID_BYTE_CONFIG_BASE.copy()
    if device == "cpu":
        test_backbone_config["dtype"] = torch.float32

    model = PerseidByteClassifier(
        backbone_config=test_backbone_config,
        num_classes=2,  # Binary classification
        classification_config=config,
        device=device,
    )

    # Test model info
    info = model.get_model_info()
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")

    # Test freeze/unfreeze
    print(f"\nTesting training stages...")
    model.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 trainable: {frozen_trainable:,}")

    model.unfreeze_backbone()
    unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 trainable: {unfrozen_trainable:,}")

    # Test forward pass with proper device handling
    print(f"\nTesting forward pass...")
    batch_size = 2
    seq_len = 256

    # Create input on CPU first, model will move it to correct device
    input_ids = torch.randint(0, 259, (batch_size, seq_len))

    # Test forward
    with torch.no_grad():
        output = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")

    print(f"\n✅ Classification framework test complete!")

# """
# Configuration system for PerseidByte classification tasks.
# """

# # import torch
# from dataclasses import dataclass
# from typing import Literal  # , Optional
# # from pathlib import Path


# @dataclass
# class ClassificationConfig:
#     """User-configurable classification settings."""

#     # === Sequence Processing ===
#     max_seq_length: int = 1024  # Max tokens per sequence
#     chunking_strategy: Literal["A", "B", "A&B"] = "A"  # Chunking approach
#     chunk_overlap: float = 0.1  # Overlap between chunks (0.0-0.5)

#     # === Model Architecture ===
#     classifier_hidden_dim: int = 256  # Hidden layer size in classifier
#     dropout_rate: float = 0.1  # Dropout rate (0.0-0.5)
#     pooling_method: Literal["mean", "last_token"] = "mean"  # Embedding pooling

#     # === Training Strategy ===
#     training_mode: Literal["freeze_finetune", "end_to_end", "frozen_only"] = (
#         "freeze_finetune"
#     )
#     stage1_epochs: int = 8  # Epochs for frozen backbone training
#     stage2_epochs: int = 5  # Epochs for end-to-end fine-tuning
#     stage1_lr: float = 1e-3  # Learning rate for stage 1
#     stage2_backbone_lr: float = 1e-5  # Backbone LR for stage 2
#     stage2_head_lr: float = 1e-4  # Classifier head LR for stage 2

#     # === Data Processing ===
#     batch_size: int = 16  # Batch size (hardware dependent)
#     train_val_split: float = 0.8  # Train/validation split ratio
#     augmentation: bool = False  # Data augmentation (future)

#     # === Evaluation ===
#     eval_every: int = 100  # Evaluate every N steps
#     early_stopping_patience: int = 5  # Early stopping patience

#     def validate(self):
#         """Validate configuration parameters."""
#         errors = []
#         warnings = []

#         # Validate ranges
#         if not (512 <= self.max_seq_length <= 32768):
#             errors.append(
#                 f"max_seq_length {self.max_seq_length} not in range [512, 32768]"
#             )

#         if not (0.0 <= self.chunk_overlap <= 0.5):
#             errors.append(f"chunk_overlap {self.chunk_overlap} not in range [0.0, 0.5]")

#         if not (0.6 <= self.train_val_split <= 0.9):
#             warnings.append(
#                 f"train_val_split {self.train_val_split} outside recommended range [0.6, 0.9]"
#             )

#         if self.batch_size > 32:
#             warnings.append(
#                 f"batch_size {self.batch_size} is large, may cause memory issues"
#             )

#         # Strategy-specific validations
#         if self.chunking_strategy == "B" and self.batch_size > 8:
#             warnings.append("Strategy B with large batch_size may cause memory issues")

#         if self.training_mode == "frozen_only" and self.stage2_epochs > 0:
#             warnings.append("frozen_only mode ignores stage2_epochs")

#         return len(errors) == 0, errors, warnings
