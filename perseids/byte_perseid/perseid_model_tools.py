# perseio_model_tools.py
import torch
import numpy as np

import torch
from safetensors.torch import save_file, load_file
from perseid_model import PerseidByteModel, PERSEID_BYTE_CONFIG_BASE


def convert_pth_to_safetensors(pth_path, output_path):
    """
    Convert a .pth checkpoint to safetensors format.
    """
    # Load the PyTorch checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu")

    # If checkpoint is a dict with 'model' key (common pattern)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    # If checkpoint is a dict with 'state_dict' key
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    # If checkpoint is already a state_dict
    elif isinstance(checkpoint, dict):
        # Check if it looks like a state_dict (keys are layer names)
        first_key = next(iter(checkpoint.keys()))
        if isinstance(checkpoint[first_key], torch.Tensor):
            state_dict = checkpoint
        else:
            print("Checkpoint structure:")
            for key in checkpoint.keys():
                print(f"  {key}: {type(checkpoint[key])}")
            raise ValueError("Cannot identify state_dict in checkpoint")
    else:
        state_dict = checkpoint

    # Convert to safetensors
    # Ensure all tensors are contiguous and on CPU
    state_dict_safe = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            state_dict_safe[key] = tensor.cpu().contiguous()

    # Save as safetensors
    save_file(state_dict_safe, output_path)
    print(f"Saved to {output_path}")

    # Report file sizes
    import os

    pth_size = os.path.getsize(pth_path) / (1024 * 1024)
    safe_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"Original .pth size: {pth_size:.2f} MB")
    print(f"Safetensors size: {safe_size:.2f} MB")
    print(f"Size ratio: {safe_size / pth_size:.2f}x")

    return state_dict_safe


if __name__ == "__main__":
    import glob

    # Example path with wildcard
    pattern = "./models/perseid_*/perseid_model_final.pth"

    # Find all matching paths
    matching_paths = glob.glob(pattern)

    print("Found...")
    print(matching_paths)
    print("default to using first option")

    print(f"""
    Here are optional models, which you the chosen one?
    {matching_paths}
    """)
    for indx, valu in enumerate(matching_paths):
        print(f"index-> {indx}, model-> {valu}")

    print("Enter the Index...")
    try:
        pick_this_index = int(input())
    except Exception as e:
        raise e

    CHECKPOINT_PATH1 = matching_paths[pick_this_index]

    # Convert your model
    convert_pth_to_safetensors(CHECKPOINT_PATH1, "perseid_model_final.safetensors")

    # To load it back:
    from safetensors.torch import load_file

    # Load safetensors
    state_dict = load_file("perseid_model_final.safetensors")

    # Load into model
    model = PerseidByteModel(PERSEID_BYTE_CONFIG_BASE)
    model.load_state_dict(state_dict)

    # import glob

    # Example path with wildcard
    pattern = "./models/perseid_*/perseid_model_final.pth"

    # Find all matching paths
    matching_paths = glob.glob(pattern)

    print("Found...")
    print(matching_paths)
    print("default to using first option")

    print(f"""
    Here are optional models, which you the chosen one?
    {matching_paths}
    """)
    for indx, valu in enumerate(matching_paths):
        print(f"index-> {indx}, model-> {valu}")

    print("Enter the Index...")
    try:
        pick_this_index = int(input())
    except Exception as e:
        raise e

    CHECKPOINT_PATH2 = matching_paths[pick_this_index]

    # Load your model
    checkpoint = torch.load(CHECKPOINT_PATH2, map_location="cpu")

    # Calculate actual parameter memory
    total_params = 0
    total_bytes = 0

    for key, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            num_params = tensor.numel()
            bytes_size = num_params * tensor.element_size()
            total_params += num_params
            total_bytes += bytes_size
            print(f"{key}: {num_params:,} params, {bytes_size / 1024 / 1024:.2f} MB")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Uncompressed size: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Compressed .pth size: ~143 MB")
    print(f"Compression ratio: {total_bytes / 1024 / 1024 / 143:.2f}x")
