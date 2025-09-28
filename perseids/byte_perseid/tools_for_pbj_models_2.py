
# perseid_model_tools.py
import torch
import numpy as np
from safetensors.torch import save_file, load_file
from perseid_model import PerseidByteModel, PERSEID_BYTE_CONFIG_BASE
import copy


def detect_model_config(state_dict):
    """
    Detect the model configuration from a state_dict.
    """
    config = copy.deepcopy(PERSEID_BYTE_CONFIG_BASE)
    
    # Detect vocabulary size from embedding layer
    if "tok_emb.weight" in state_dict:
        vocab_size, emb_dim = state_dict["tok_emb.weight"].shape
        config["vocab_size"] = vocab_size
        config["emb_dim"] = emb_dim
    
    # Detect number of layers
    layer_numbers = []
    for key in state_dict.keys():
        if "blocks." in key:
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_numbers.append(int(parts[1]))
    
    if layer_numbers:
        config["n_layers"] = max(layer_numbers) + 1
    
    # Detect hidden_dim from FFN
    if "blocks.0.ff.fc1.weight" in state_dict:
        hidden_dim, _ = state_dict["blocks.0.ff.fc1.weight"].shape
        config["hidden_dim"] = hidden_dim
    
    # Detect attention dimensions
    if "blocks.0.att.W_query.weight" in state_dict:
        total_q_dim, _ = state_dict["blocks.0.att.W_query.weight"].shape
        
        # Detect head_dim from q_norm if available
        if "blocks.0.att.q_norm.scale" in state_dict:
            head_dim = state_dict["blocks.0.att.q_norm.scale"].shape[0]
            config["head_dim"] = head_dim
            config["n_heads"] = total_q_dim // head_dim
        else:
            # Guess based on common configurations
            if total_q_dim == 640:
                config["n_heads"] = 4
                config["head_dim"] = 160
            elif total_q_dim == 1024:
                config["n_heads"] = 4  
                config["head_dim"] = 256
    
    # Detect n_kv_groups from K projection size
    if "blocks.0.att.W_key.weight" in state_dict:
        total_k_dim, _ = state_dict["blocks.0.att.W_key.weight"].shape
        if "head_dim" in config:
            config["n_kv_groups"] = total_k_dim // config["head_dim"]
        else:
            config["n_kv_groups"] = 1  # Default
    
    # Build layer_types based on n_layers
    layer_types = []
    for i in range(config["n_layers"]):
        # Default pattern: full attention every 6 layers
        if (i + 1) % 6 == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("sliding_attention")
    
    # Ensure last layer is full_attention
    if layer_types and layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    
    config["layer_types"] = layer_types
    
    print("\nDetected configuration:")
    print(f"  vocab_size: {config['vocab_size']}")
    print(f"  emb_dim: {config['emb_dim']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  n_heads: {config['n_heads']}")
    print(f"  head_dim: {config['head_dim']}")
    print(f"  n_kv_groups: {config['n_kv_groups']}")
    print(f"  hidden_dim: {config['hidden_dim']}")
    
    return config


def analyze_checkpoint(checkpoint_path):
    """
    Analyze a checkpoint file and report its structure and size.
    """
    print(f"\nAnalyzing: {checkpoint_path}")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Determine the state_dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("Found state_dict under 'model' key")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Found state_dict under 'state_dict' key")
        else:
            # Check if it's already a state_dict
            first_key = next(iter(checkpoint.keys()))
            if isinstance(checkpoint[first_key], torch.Tensor):
                state_dict = checkpoint
                print("Checkpoint is directly a state_dict")
            else:
                print("Checkpoint structure:")
                for key in list(checkpoint.keys())[:10]:
                    print(f"  {key}: {type(checkpoint[key])}")
                return None
    else:
        state_dict = checkpoint
    
    # Calculate actual parameter memory
    total_params = 0
    total_bytes = 0
    
    print("\nLayer-wise parameter breakdown:")
    print("-" * 60)
    
    # Group parameters by layer
    layer_params = {}
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            num_params = tensor.numel()
            bytes_size = num_params * tensor.element_size()
            total_params += num_params
            total_bytes += bytes_size
            
            # Group by layer
            if "blocks." in key:
                layer_num = key.split(".")[1]
                if layer_num not in layer_params:
                    layer_params[layer_num] = {"params": 0, "bytes": 0}
                layer_params[layer_num]["params"] += num_params
                layer_params[layer_num]["bytes"] += bytes_size
            elif "tok_emb" in key:
                print(f"Embedding: {num_params:,} params, {bytes_size/1024/1024:.2f} MB")
            elif "final_norm" in key:
                print(f"Final Norm: {num_params:,} params, {bytes_size/1024/1024:.2f} MB")
            elif "out_head" in key:
                print(f"Output Head: {num_params:,} params, {bytes_size/1024/1024:.2f} MB")
    
    # Print layer summaries
    if layer_params:
        print(f"\nFound {len(layer_params)} transformer layers:")
        for i in range(min(3, len(layer_params))):
            layer_data = layer_params.get(str(i), {"params": 0, "bytes": 0})
            print(f"  Layer {i}: {layer_data['params']:,} params, {layer_data['bytes']/1024/1024:.2f} MB")
        if len(layer_params) > 3:
            print(f"  ... ({len(layer_params) - 3} more layers)")
    
    import os
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    print("\nSummary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Theoretical size (uncompressed): {total_bytes/1024/1024:.2f} MB")
    print(f"  Actual file size: {file_size:.2f} MB")
    print(f"  Compression ratio: {total_bytes/1024/1024/file_size:.2f}x")
    
    # Detect configuration
    detected_config = detect_model_config(state_dict)
    
    return state_dict, detected_config


def convert_pth_to_safetensors(pth_path, output_path):
    """
    Convert a .pth checkpoint to safetensors format with auto-config detection.
    """
    # Analyze the checkpoint first
    state_dict, detected_config = analyze_checkpoint(pth_path)
    
    if state_dict is None:
        raise ValueError("Could not extract state_dict from checkpoint")
    
    # Convert to safetensors
    state_dict_safe = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            state_dict_safe[key] = tensor.cpu().contiguous()
    
    # Save as safetensors
    save_file(state_dict_safe, output_path)
    print(f"\nSaved to {output_path}")
    
    # Report file sizes
    import os
    pth_size = os.path.getsize(pth_path) / (1024 * 1024)
    safe_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original .pth size: {pth_size:.2f} MB")
    print(f"Safetensors size: {safe_size:.2f} MB")
    print(f"Size difference: {(1 - safe_size/pth_size)*100:.1f}% smaller")
    
    # Test loading with detected config
    print("\nTesting model loading with detected config...")
    try:
        model = PerseidByteModel(detected_config)
        model.load_state_dict(state_dict_safe, strict=True)
        print("✓ Model loaded successfully with detected configuration!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nYou may need to manually adjust the configuration.")
    
    return state_dict_safe, detected_config


if __name__ == "__main__":
    import glob
    
    # Find all matching models
    pattern = "./models/perseid_*/perseid_model_final.pth"
    matching_paths = glob.glob(pattern)
    
    if not matching_paths:
        print("No models found. Looking for individual file...")
        # Try looking for a single file
        if os.path.exists("perseid_model_final.pth"):
            matching_paths = ["perseid_model_final.pth"]
        else:
            print("No model files found!")
            exit(1)
    
    print("Found models:")
    for idx, path in enumerate(matching_paths):
        print(f"  [{idx}] {path}")
    
    if len(matching_paths) == 1:
        pick_this_index = 0
        print(f"Using: {matching_paths[0]}")
    else:
        print("\nEnter the index of the model to convert: ", end="")
        try:
            pick_this_index = int(input())
        except Exception as e:
            print(f"Invalid input: {e}")
            exit(1)
    
    checkpoint_path = matching_paths[pick_this_index]
    output_path = checkpoint_path.replace(".pth", ".safetensors")
    
    # Convert and analyze
    state_dict_safe, detected_config = convert_pth_to_safetensors(
        checkpoint_path, 
        output_path
    )
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("\nTo use this model in your code:")
    print("-" * 60)
    print("from safetensors.torch import load_file")
    print("from perseid_model import PerseidByteModel")
    print("")
    print("# Load the configuration detected from your model")
    print("config = {")
    for key in ["vocab_size", "emb_dim", "n_layers", "n_heads", "head_dim", "n_kv_groups", "hidden_dim"]:
        print(f"    '{key}': {detected_config[key]},")
    print("    # ... other config values")
    print("}")
    print("")
    print("# Load model")
    print("model = PerseidByteModel(config)")
    print(f"state_dict = load_file('{output_path}')")
    print("model.load_state_dict(state_dict)")
