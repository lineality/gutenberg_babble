# checkpoint_to_final_convert.py
from pathlib import Path
import torch
import json
import traceback

# Import model architecture and configuration tools
from byte_tokenizer import ByteTokenizer

from datetime import datetime

# from generate_text_tools_perseid import generate_text_simple
from perseid_model import PerseidByteModel


def convert_checkpoint_to_final_model(checkpoint_path, output_path=None):
    """
    Extract model weights from a full checkpoint file and save as inference-only model.

    This reduces file size significantly by removing optimizer state, scheduler state,
    and training metadata that are only needed for continuing training.

    Args:
        checkpoint_path (str or Path): Path to checkpoint file (checkpoint_best.pth or checkpoint_latest.pth)
        output_path (str or Path, optional): Where to save the extracted model.
                                            If None, saves as 'model_weights_only.pth' in same directory.

    Returns:
        Path: Path to the saved model weights file

    Side Effects:
        - Creates a new .pth file containing only model weights
        - Prints file size comparison

    Example:
        >>> convert_checkpoint_to_final_model('models/my_model/checkpoint_best.pth')
        ✓ Loaded checkpoint from: models/my_model/checkpoint_best.pth
        ✓ Checkpoint size: 2,456.3 MB
        ✓ Model weights extracted
        ✓ Saved inference model to: models/my_model/model_weights_only.pth
        ✓ Inference model size: 1,152.4 MB
        ✓ Space saved: 1,303.9 MB (53.1%)
    """
    try:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Generate timestamp for unique identification
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :21
        ]  # Include microseconds

        # Determine output path
        if output_path is None:
            output_path = (
                checkpoint_path.parent / f"model_weights_only_{timestamp_str}.pth"
            )
        else:
            output_path = Path(output_path)

        print(f"\n{'=' * 60}")
        print("Converting Checkpoint to Inference Model")
        print(f"{'=' * 60}")

        # Get original file size
        original_size_bytes = checkpoint_path.stat().st_size
        original_size_mb = original_size_bytes / (1024**2)

        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
        print(f"✓ Checkpoint size: {original_size_mb:,.1f} MB")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract just the model state dict
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
            print(f"✓ Model weights extracted")
        else:
            # Handle old-style checkpoints that are just the state dict
            model_state_dict = checkpoint
            print(
                f"✓ Legacy checkpoint format detected, using entire checkpoint as model weights"
            )

        # Save only the model weights
        torch.save(model_state_dict, output_path)

        # Get new file size
        new_size_bytes = output_path.stat().st_size
        new_size_mb = new_size_bytes / (1024**2)

        # Calculate space saved
        space_saved_mb = original_size_mb - new_size_mb
        percent_saved = (space_saved_mb / original_size_mb) * 100

        print(f"✓ Saved inference model to: {output_path}")
        print(f"✓ Inference model size: {new_size_mb:,.1f} MB")
        print(f"✓ Space saved: {space_saved_mb:,.1f} MB ({percent_saved:.1f}%)")
        print(f"\n{'=' * 60}")

        return output_path

    except Exception as conversion_error:
        print(f"❌ Error converting checkpoint: {conversion_error}")
        traceback.print_exc()
        raise


def batch_convert_checkpoints_in_directory(model_directory):
    """
    Convert all checkpoint files in a directory to inference-only models.

    Finds all checkpoint_*.pth files and creates corresponding *_weights_only.pth files.

    Args:
        model_directory (str or Path): Directory containing checkpoint files

    Returns:
        list[Path]: List of paths to created inference model files

    Side Effects:
        - Creates multiple new .pth files (one per checkpoint)
        - Prints progress for each conversion

    Example:
        >>> batch_convert_checkpoints_in_directory('models/perseid_288m_corpus/')
        Found 3 checkpoint files to convert

        [1/3] Converting checkpoint_best.pth...
        ✓ Saved: checkpoint_best_weights_only.pth (1,152.4 MB saved)

        [2/3] Converting checkpoint_latest.pth...
        ✓ Saved: checkpoint_latest_weights_only.pth (1,148.2 MB saved)

        [3/3] Converting checkpoint_epoch_30.pth...
        ✓ Saved: checkpoint_epoch_30_weights_only.pth (1,151.8 MB saved)

        Total space saved: 3,452.4 MB
    """
    try:
        model_directory = Path(model_directory)

        if not model_directory.exists():
            raise FileNotFoundError(f"Directory not found: {model_directory}")

        # Find all checkpoint files
        checkpoint_files = list(model_directory.glob("checkpoint_*.pth"))

        if not checkpoint_files:
            print(f"No checkpoint files found in {model_directory}")
            return []

        print(f"\n{'=' * 60}")
        print(f"Batch Converting Checkpoints")
        print(f"{'=' * 60}")
        print(f"Found {len(checkpoint_files)} checkpoint files to convert\n")

        converted_files = []
        total_space_saved_mb = 0.0

        for idx, checkpoint_path in enumerate(checkpoint_files, 1):
            print(
                f"[{idx}/{len(checkpoint_files)}] Converting {checkpoint_path.name}..."
            )

            # Create output filename
            output_filename = checkpoint_path.stem + "_weights_only.pth"
            output_path = checkpoint_path.parent / output_filename

            # Get sizes before conversion
            original_size_mb = checkpoint_path.stat().st_size / (1024**2)

            # Convert
            try:
                convert_checkpoint_to_final_model(checkpoint_path, output_path)

                new_size_mb = output_path.stat().st_size / (1024**2)
                space_saved_mb = original_size_mb - new_size_mb
                total_space_saved_mb += space_saved_mb

                converted_files.append(output_path)
                print(f"✓ Saved: {output_path.name} ({space_saved_mb:,.1f} MB saved)\n")

            except Exception as file_error:
                print(f"⚠️  Failed to convert {checkpoint_path.name}: {file_error}\n")
                continue

        print(f"{'=' * 60}")
        print(f"Batch Conversion Complete")
        print(f"{'=' * 60}")
        print(f"Converted {len(converted_files)} files")
        print(f"Total space saved: {total_space_saved_mb:,.1f} MB")

        return converted_files

    except Exception as batch_error:
        print(f"❌ Error in batch conversion: {batch_error}")
        traceback.print_exc()
        raise


def load_model_for_inference(model_weights_path, model_config_path=None):
    """
    Load a model from weights-only file for inference (text generation).

    This is the counterpart to convert_checkpoint_to_final_model() - it loads
    the extracted weights back into a model for use.

    Args:
        model_weights_path (str or Path): Path to weights-only .pth file
        model_config_path (str or Path, optional): Path to model_config.json file.
                                                   If None, looks in same directory as weights.

    Returns:
        tuple: (model, model_config) ready for inference

    Raises:
        FileNotFoundError: If weights or config file not found

    Example:
        >>> model, config = load_model_for_inference('models/my_model/model_weights_only.pth')
        >>> # Now use model for text generation
        >>> output = generate_text_simple(model, tokenizer, "Once upon a time", max_new_tokens=50)
    """
    try:
        model_weights_path = Path(model_weights_path)

        if not model_weights_path.exists():
            raise FileNotFoundError(
                f"Model weights file not found: {model_weights_path}"
            )

        print(f"\n{'=' * 60}")
        print("Loading Model for Inference")
        print(f"{'=' * 60}")
        print(f"Loading weights from: {model_weights_path}")

        # Find model configuration
        if model_config_path is None:
            # Look in same directory as weights
            model_config_path = model_weights_path.parent / "model_config.json"
        else:
            model_config_path = Path(model_config_path)

        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Model configuration not found: {model_config_path}\n"
                f"Need model_config.json to reconstruct model architecture."
            )

        print(f"Loading config from: {model_config_path}")

        # Load configuration
        with open(model_config_path, "r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)

        # Handle dtype string conversion
        if "dtype" in model_config:
            dtype_str = model_config["dtype"]
            if "bfloat16" in dtype_str:
                model_config["dtype"] = torch.bfloat16
            elif "float16" in dtype_str:
                model_config["dtype"] = torch.float16
            else:
                model_config["dtype"] = torch.float32

        print(f"✓ Model configuration loaded")

        # Initialize model with configuration
        model = PerseidByteModel(model_config)

        # Load weights
        state_dict = torch.load(model_weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

        print(f"✓ Model weights loaded")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model ready for inference")
        print(f"  Parameters: {total_params:,}")
        print(f"  Context length: {model_config['context_length']}")
        print(f"{'=' * 60}\n")

        return model, model_config

    except Exception as load_error:
        print(f"❌ Error loading model: {load_error}")
        traceback.print_exc()
        raise


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


if __name__ == "__main__":
    """
    If run with path to checkpoint file argument
    save a timestamped runnable file
    and test generative inference on it
    """
    import sys
    import os
    from pathlib import Path

    # use argument input path if supplied by user
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        print(f"path argument found... {file_path}")

        # if "--convert-checkpoints" in sys.argv:
        file_path = Path(file_path)
        parent_directory = file_path.parent
        print(f"saving to {parent_directory}")

        # Convert your best checkpoint to inference-only format
        new_model_path = convert_checkpoint_to_final_model(
            checkpoint_path=file_path,
        )

        # try/test inference
        #
        # Load the converted model
        model, config = load_model_for_inference(new_model_path)

        # Use for inference
        tokenizer = ByteTokenizer()
        output = generate_text_simple(
            model, tokenizer, "This is about ", max_new_tokens=100
        )
        print(output)
