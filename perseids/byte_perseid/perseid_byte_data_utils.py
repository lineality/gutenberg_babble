# perseid_byte_data_utils.py
"""
Data utilities for PerseidByte classification including chunking strategies.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Literal, Optional
import pandas as pd
from pathlib import Path
from byte_tokenizer import ByteTokenizer


class ClassificationDataset(Dataset):
    """
    Dataset for text classification with configurable chunking strategies.

    Supports:
    - Strategy A: Split documents into chunks with same label
    - Strategy B: Hierarchical aggregation of chunks
    - Strategy A&B: Hybrid approach using both strategies
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: ByteTokenizer,
        max_seq_length: int = 1024,
        chunking_strategy: Literal["A", "B", "A&B"] = "A",
        chunk_overlap: float = 0.1,
        stage: Literal["train", "eval"] = "train",
    ):
        """
        Args:
            texts: List of text documents
            labels: List of integer labels
            tokenizer: ByteTokenizer instance
            max_seq_length: Maximum sequence length per chunk
            chunking_strategy: How to handle long documents
            chunk_overlap: Overlap ratio between chunks (0.0 to 0.5)
            stage: "train" or "eval" mode (affects strategy A&B behavior)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chunking_strategy = chunking_strategy
        self.chunk_overlap = chunk_overlap
        self.stage = stage

        # Process all documents
        self.samples = []
        self._process_documents(texts, labels)

        print(f"📊 Dataset created:")
        print(f"   Strategy: {chunking_strategy}")
        print(f"   Original documents: {len(texts)}")
        print(f"   Total samples: {len(self.samples)}")

    def _process_documents(self, texts: List[str], labels: List[int]):
        """Process documents according to chunking strategy."""

        for text, label in zip(texts, labels):
            # Tokenize full document
            tokens = self.tokenizer.encode(text)

            if len(tokens) <= self.max_seq_length:
                # Document fits in one chunk
                self.samples.append(
                    {
                        "tokens": tokens,
                        "label": label,
                        "strategy": "single",
                        "doc_id": len(self.samples),
                    }
                )
            else:
                # Document needs chunking
                if self.chunking_strategy == "A":
                    self._add_chunks(tokens, label)

                elif self.chunking_strategy == "B":
                    self._add_hierarchical(tokens, label)

                elif self.chunking_strategy == "A&B":
                    # In training: alternate between strategies
                    # In eval: use hierarchical for consistency
                    if self.stage == "train":
                        # Add both chunk samples AND hierarchical sample
                        self._add_chunks(tokens, label)
                        self._add_hierarchical(tokens, label)
                    else:
                        # Eval: use hierarchical only for consistency
                        self._add_hierarchical(tokens, label)

    def _add_chunks(self, tokens: List[int], label: int):
        """Strategy A: Split into overlapping chunks with same label."""
        stride = int(self.max_seq_length * (1 - self.chunk_overlap))

        for i in range(0, len(tokens), stride):
            chunk = tokens[i : i + self.max_seq_length]

            # Pad last chunk if needed
            if len(chunk) < self.max_seq_length:
                chunk = chunk + [self.tokenizer.PAD_ID] * (
                    self.max_seq_length - len(chunk)
                )

            self.samples.append(
                {
                    "tokens": chunk,
                    "label": label,
                    "strategy": "chunk",
                    "doc_id": len(self.samples),
                }
            )

            # Stop if we've covered the document
            if i + self.max_seq_length >= len(tokens):
                break

    def _add_hierarchical(self, tokens: List[int], label: int):
        """Strategy B: Store full document for hierarchical processing."""
        # For hierarchical, we store all tokens but will process in chunks during forward pass
        self.samples.append(
            {
                "tokens": tokens,  # Full token list
                "label": label,
                "strategy": "hierarchical",
                "doc_id": len(self.samples),
            }
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return a single sample."""
        sample = self.samples[idx]

        # For hierarchical samples, return full tokens (will be chunked in collate_fn)
        # For chunk samples, return as-is
        tokens = sample["tokens"]

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "strategy": sample["strategy"],
            "doc_id": sample["doc_id"],
        }


def create_classification_collate_fn(max_seq_length: int, tokenizer: ByteTokenizer):
    """
    Create a collate function that handles different chunking strategies.
    """

    def collate_fn(batch):
        """Custom collate for handling hierarchical samples."""

        # Separate samples by strategy
        regular_samples = []
        hierarchical_samples = []

        for sample in batch:
            if sample["strategy"] == "hierarchical":
                hierarchical_samples.append(sample)
            else:
                regular_samples.append(sample)

        all_inputs = []
        all_labels = []
        all_strategies = []

        # Process regular samples (single/chunk)
        if regular_samples:
            for sample in regular_samples:
                tokens = sample["tokens"]

                # Truncate or pad as needed
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                elif len(tokens) < max_seq_length:
                    padding = torch.full(
                        (max_seq_length - len(tokens),),
                        tokenizer.PAD_ID,
                        dtype=torch.long,
                    )
                    tokens = torch.cat([tokens, padding])

                all_inputs.append(tokens)
                all_labels.append(sample["label"])
                all_strategies.append(sample["strategy"])

        # Process hierarchical samples
        if hierarchical_samples:
            for sample in hierarchical_samples:
                tokens = sample["tokens"]

                # Split into chunks for hierarchical processing
                chunks = []
                stride = int(max_seq_length * 0.9)  # 10% overlap for hierarchical

                for i in range(0, len(tokens), stride):
                    chunk = tokens[i : i + max_seq_length]

                    if len(chunk) < max_seq_length:
                        padding = torch.full(
                            (max_seq_length - len(chunk),),
                            tokenizer.PAD_ID,
                            dtype=torch.long,
                        )
                        chunk = torch.cat([chunk, padding])

                    chunks.append(chunk)

                    if i + max_seq_length >= len(tokens):
                        break

                # Add all chunks for this document
                # (model will aggregate in forward pass)
                all_inputs.extend(chunks)
                all_labels.extend([sample["label"]] * len(chunks))
                all_strategies.extend(["hierarchical"] * len(chunks))

        if not all_inputs:
            raise ValueError("Empty batch!")

        # Stack all inputs
        input_ids = torch.stack(all_inputs)
        labels = (
            torch.stack(all_labels)
            if isinstance(all_labels[0], torch.Tensor)
            else torch.tensor(all_labels)
        )

        return {"input_ids": input_ids, "labels": labels, "strategies": all_strategies}

    return collate_fn


def load_imdb_dataset(
    data_path: str,
    tokenizer: ByteTokenizer,
    max_seq_length: int = 1024,
    chunking_strategy: Literal["A", "B", "A&B"] = "A",
    chunk_overlap: float = 0.1,
    train_val_split: float = 0.8,
    batch_size: int = 16,
    sample_size: Optional[int] = None,  # For testing with subset
) -> Tuple[DataLoader, DataLoader]:
    """
    Load IMDB dataset and create train/val data loaders.

    Args:
        data_path: Path to IMDB CSV file
        tokenizer: ByteTokenizer instance
        max_seq_length: Maximum sequence length
        chunking_strategy: How to handle long documents
        chunk_overlap: Overlap between chunks
        train_val_split: Ratio for train/val split
        batch_size: Batch size for data loaders
        sample_size: Optional subset size for testing

    Returns:
        train_loader, val_loader
    """
    print(f"\n📂 Loading IMDB dataset from {data_path}")

    # Load CSV
    df = pd.read_csv(data_path)

    # Sample if requested (for testing)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"   Using {len(df)} samples for testing")

    print(f"   Total reviews: {len(df)}")

    # Convert sentiment to labels (positive=1, negative=0)
    label_map = {"positive": 1, "negative": 0}
    df["label"] = df["sentiment"].map(label_map)

    # Split into train/val
    split_idx = int(len(df) * train_val_split)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    print(f"   Train: {len(train_df)} reviews")
    print(f"   Val: {len(val_df)} reviews")

    # Create datasets
    train_dataset = ClassificationDataset(
        texts=train_df["review"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        chunking_strategy=chunking_strategy,
        chunk_overlap=chunk_overlap,
        stage="train",
    )

    val_dataset = ClassificationDataset(
        texts=val_df["review"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        chunking_strategy=chunking_strategy,
        chunk_overlap=chunk_overlap,
        stage="eval",
    )

    # Create collate function
    collate_fn = create_classification_collate_fn(max_seq_length, tokenizer)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"\n✅ Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    return train_loader, val_loader


# Test the data utilities
if __name__ == "__main__":
    print("🧪 Testing Data Utilities")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = ByteTokenizer()

    # Test with sample data
    test_texts = [
        "This is a short text.",
        "This is a longer text that will need to be chunked because it contains many words and exceeds the maximum sequence length that we have configured for our model. "
        * 10,
        "Another short one.",
    ]
    test_labels = [1, 0, 1]

    print("\nTesting different chunking strategies:")

    for strategy in ["A", "B", "A&B"]:
        print(f"\n📊 Strategy {strategy}:")
        dataset = ClassificationDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizer,
            max_seq_length=128,
            chunking_strategy=strategy,
            chunk_overlap=0.1,
        )

        print(f"   Dataset size: {len(dataset)}")

        # Check first sample
        sample = dataset[0]
        print(f"   First sample shape: {sample['tokens'].shape}")
        print(f"   First sample strategy: {sample['strategy']}")

    # Test IMDB loading (requires downloaded CSV)
    imdb_path = Path("data/IMDB Dataset.csv")
    if imdb_path.exists():
        print(f"\n📂 Testing IMDB dataset loading...")
        train_loader, val_loader = load_imdb_dataset(
            data_path=str(imdb_path),
            tokenizer=tokenizer,
            max_seq_length=512,
            chunking_strategy="A",
            batch_size=4,
            sample_size=100,  # Just test with 100 samples
        )
