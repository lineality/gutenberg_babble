"""
perseid_config_tools.py

Configuration utilities for creating Perseid model variants based on a Gemma architecture.
Provides functions to calculate parameter counts, validate configurations, and
generate optimal configurations for target model sizes.
"""

import traceback
from typing import Optional


def calculate_model_params(config: dict) -> dict[str, int]:
    """
    Calculate detailed parameter counts for a given model configuration.

    Args:
        config (dict): Model configuration dictionary containing:
            - vocab_size: Size of vocabulary
            - emb_dim: Embedding dimension
            - n_layers: Number of transformer layers
            - n_heads: Number of attention heads
            - n_kv_groups: Number of key-value groups for GQA
            - head_dim: Dimension per attention head
            - hidden_dim: Hidden dimension for feedforward layers

    Returns:
        dict: Detailed parameter counts with keys:
            - embedding: Token embedding parameters
            - attention_per_layer: Attention parameters per layer
            - ffn_per_layer: Feedforward parameters per layer
            - norm_per_layer: Normalization parameters per layer
            - total_per_layer: Total parameters per layer
            - final_norm: Final normalization parameters
            - output_head: Output projection parameters
            - total: Total model parameters
            - total_millions: Total in millions
    """
    try:
        # Extract configuration values
        vocab_size = config["vocab_size"]
        emb_dim = config["emb_dim"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        n_kv_groups = config.get(
            "n_kv_groups", n_heads
        )  # Default to MHA if not specified
        head_dim = config.get("head_dim", emb_dim // n_heads)
        hidden_dim = config["hidden_dim"]

        # Calculate embedding parameters
        embedding_params = vocab_size * emb_dim

        # Calculate per-layer attention parameters
        # Q projection: emb_dim -> (n_heads * head_dim)
        q_params = emb_dim * (n_heads * head_dim)

        # K projection: emb_dim -> (n_kv_groups * head_dim)
        k_params = emb_dim * (n_kv_groups * head_dim)

        # V projection: emb_dim -> (n_kv_groups * head_dim)
        v_params = emb_dim * (n_kv_groups * head_dim)

        # Output projection: (n_heads * head_dim) -> emb_dim
        out_proj_params = (n_heads * head_dim) * emb_dim

        # QK normalization (if used)
        qk_norm_params = 0
        if config.get("qk_norm", False):
            # Q norm and K norm, each has head_dim parameters
            qk_norm_params = 2 * head_dim

        attention_params_per_layer = (
            q_params + k_params + v_params + out_proj_params + qk_norm_params
        )

        # Calculate per-layer feedforward parameters
        # For reference: Gemma uses gated FFN with three projections
        # gate_proj: emb_dim -> hidden_dim
        gate_params = emb_dim * hidden_dim

        # up_proj: emb_dim -> hidden_dim
        up_params = emb_dim * hidden_dim

        # down_proj: hidden_dim -> emb_dim
        down_params = hidden_dim * emb_dim

        ffn_params_per_layer = gate_params + up_params + down_params

        # Calculate per-layer normalization parameters
        #  For reference: Gemma uses multiple RMSNorm layers per transformer block
        # - input_layernorm
        # - post_attention_layernorm
        # - pre_feedforward_layernorm
        # - post_feedforward_layernorm
        norm_params_per_layer = 4 * emb_dim

        # Total parameters per layer
        total_params_per_layer = (
            attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
        )

        # Calculate final normalization parameters
        final_norm_params = emb_dim

        # Calculate output head parameters (might be tied with embeddings)
        output_head_params = vocab_size * emb_dim

        # Calculate total parameters
        total_params = (
            embedding_params
            + (n_layers * total_params_per_layer)
            + final_norm_params
            + output_head_params
        )

        # Account for weight tying if applicable
        if config.get("tie_embeddings", True):
            # Subtract output head params as they're shared with embeddings
            total_params -= output_head_params
            output_head_params = 0  # Shared with embeddings

        return {
            "embedding": embedding_params,
            "attention_per_layer": attention_params_per_layer,
            "ffn_per_layer": ffn_params_per_layer,
            "norm_per_layer": norm_params_per_layer,
            "total_per_layer": total_params_per_layer,
            "final_norm": final_norm_params,
            "output_head": output_head_params,
            "total": total_params,
            "total_millions": total_params / 1_000_000,
            "total_billions": total_params / 1_000_000_000,
        }
    except Exception as e:
        print(f"Error calculating model parameters: {e}")
        traceback.print_exc()
        raise


def validate_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate that a model configuration satisfies all architectural constraints.

    Args:
        config (dict): Model configuration dictionary

    Returns:
        tuple: (is_valid, list_of_issues)
            - is_valid (bool): True if configuration is valid
            - list_of_issues (list): List of validation issues found
    """
    issues = []

    try:
        # Extract key configuration values
        emb_dim = config.get("emb_dim")
        n_heads = config.get("n_heads")
        n_kv_groups = config.get("n_kv_groups", n_heads)
        head_dim = config.get("head_dim")
        hidden_dim = config.get("hidden_dim")
        n_layers = config.get("n_layers")
        vocab_size = config.get("vocab_size")

        # Check required fields exist
        required_fields = ["emb_dim", "n_heads", "n_layers", "hidden_dim", "vocab_size"]
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")

        if issues:  # Early return if missing required fields
            return False, issues

        # Validate embedding dimension is divisible by number of heads
        if head_dim is None:
            if emb_dim % n_heads != 0:
                issues.append(
                    f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads}) "
                    f"when head_dim is not specified"
                )
            computed_head_dim = emb_dim // n_heads
        else:
            computed_head_dim = head_dim
            if n_heads * head_dim != emb_dim and not config.get(
                "allow_dim_mismatch", False
            ):
                issues.append(
                    f"n_heads ({n_heads}) * head_dim ({head_dim}) = {n_heads * head_dim} "
                    f"doesn't match emb_dim ({emb_dim})"
                )

        # Validate head dimension is even (required for RoPE)
        if computed_head_dim % 2 != 0:
            issues.append(
                f"head_dim ({computed_head_dim}) must be even for RoPE implementation"
            )

        # Validate n_heads is divisible by n_kv_groups (for GQA)
        if n_heads % n_kv_groups != 0:
            issues.append(
                f"n_heads ({n_heads}) must be divisible by n_kv_groups ({n_kv_groups})"
            )

        # Validate hidden dimension for 8-bit quantization friendliness
        if hidden_dim % 128 != 0:
            issues.append(
                f"hidden_dim ({hidden_dim}) should be divisible by 128 for "
                f"optimal 8-bit quantization performance"
            )

        # Validate embedding dimension for 8-bit quantization friendliness
        if emb_dim % 64 != 0:
            issues.append(
                f"emb_dim ({emb_dim}) should be divisible by 64 for "
                f"optimal quantization performance"
            )

        # Validate layer_types if specified
        if "layer_types" in config:
            layer_types = config["layer_types"]
            if len(layer_types) != n_layers:
                issues.append(
                    f"layer_types length ({len(layer_types)}) must match "
                    f"n_layers ({n_layers})"
                )

            # Check valid attention types
            valid_types = {"sliding_attention", "full_attention"}
            invalid_types = set(layer_types) - valid_types
            if invalid_types:
                issues.append(
                    f"Invalid attention types found: {invalid_types}. "
                    f"Valid types are: {valid_types}"
                )

            # Check for at least one full attention layer (recommended)
            if "full_attention" not in layer_types:
                issues.append(
                    "Warning: No full_attention layers found. At least one is recommended "
                    "for capturing long-range dependencies"
                )

        # Validate reasonable ranges
        if n_layers < 1 or n_layers > 100:
            issues.append(f"n_layers ({n_layers}) seems unreasonable (expected 1-100)")

        if emb_dim < 64 or emb_dim > 8192:
            issues.append(f"emb_dim ({emb_dim}) seems unreasonable (expected 64-8192)")

        # Update vocab_size validation for byte tokenizer
        if (
            vocab_size < 256 or vocab_size > 100_000
        ):  # ← CHANGED: Lower bound for byte tokenizer
            issues.append(
                f"vocab_size ({vocab_size}) seems unreasonable (expected 256-100000 for byte tokenizer)"
            )

        # Add specific validation for byte tokenizer
        if vocab_size == 259:
            print("  ✓ Detected ByteTokenizer configuration (vocab_size=259)")

        is_valid = len(issues) == 0
        return is_valid, issues

    except Exception as e:
        issues.append(f"Validation error: {e}")
        traceback.print_exc()
        return False, issues


def create_perseid_config(
    target_size_millions: int,
    base_config: Optional[dict] = None,
    strategy: str = "balanced",
) -> dict:
    """
    Create a Perseid configuration targeting a specific parameter count.

    Args:
        target_size_millions (int): Target size in millions (256, 288, 320)
        base_config (dict, optional): Base configuration to modify
        strategy (str): Sizing strategy - "balanced", "deep", or "wide"
            - balanced: Balance between depth and width
            - deep: Favor more layers with smaller dimensions
            - wide: Favor fewer layers with larger dimensions

    Returns:
        dict: New Perseid configuration optimized for target size
    """
    try:
        if base_config is None:
            # Use the PERSEID_BYTE_CONFIG_BASE as default base
            from perseid_model import PERSEID_BYTE_CONFIG_BASE

            base_config = PERSEID_BYTE_CONFIG_BASE.copy()

        # Create a copy to avoid modifying the original
        new_config = base_config.copy()

        # Define configuration presets for different strategies
        if target_size_millions == 256:
            if strategy == "balanced":
                new_config.update(
                    {
                        "emb_dim": 576,
                        "hidden_dim": 1536,
                        "n_layers": 16,
                        "n_heads": 3,
                        "head_dim": 192,
                    }
                )
            elif strategy == "deep":
                new_config.update(
                    {
                        "emb_dim": 512,
                        "hidden_dim": 1536,
                        "n_layers": 18,
                        "n_heads": 4,
                        "head_dim": 128,
                    }
                )
            elif strategy == "wide":
                new_config.update(
                    {
                        "emb_dim": 640,
                        "hidden_dim": 1664,
                        "n_layers": 14,
                        "n_heads": 4,
                        "head_dim": 160,
                    }
                )

        elif target_size_millions == 288:
            if strategy == "balanced":
                new_config.update(
                    {
                        "emb_dim": 640,
                        "hidden_dim": 1792,
                        "n_layers": 16,
                        "n_heads": 4,
                        "head_dim": 160,
                    }
                )
            elif strategy == "deep":
                new_config.update(
                    {
                        "emb_dim": 576,
                        "hidden_dim": 1664,
                        "n_layers": 18,
                        "n_heads": 3,
                        "head_dim": 192,
                    }
                )
            elif strategy == "wide":
                new_config.update(
                    {
                        "emb_dim": 704,
                        "hidden_dim": 1920,
                        "n_layers": 14,
                        "n_heads": 4,
                        "head_dim": 176,
                    }
                )

        elif target_size_millions == 320:
            if strategy == "balanced":
                new_config.update(
                    {
                        "emb_dim": 704,
                        "hidden_dim": 1920,
                        "n_layers": 16,
                        "n_heads": 4,
                        "head_dim": 176,
                    }
                )
            elif strategy == "deep":
                new_config.update(
                    {
                        "emb_dim": 640,
                        "hidden_dim": 1792,
                        "n_layers": 20,
                        "n_heads": 4,
                        "head_dim": 160,
                    }
                )
            elif strategy == "wide":
                new_config.update(
                    {
                        "emb_dim": 768,
                        "hidden_dim": 2048,
                        "n_layers": 14,
                        "n_heads": 4,
                        "head_dim": 192,
                    }
                )
        else:
            raise ValueError(
                f"Unsupported target size: {target_size_millions}M. "
                f"Supported sizes are: 256, 288, 320"
            )

        # Update layer_types to match new n_layers
        n_layers = new_config["n_layers"]

        # Maintain similar distribution of attention types as original
        # Original has full_attention every 6 layers
        layer_types = []
        for i in range(n_layers):
            if (i + 1) % 6 == 0:  # Every 6th layer is full attention
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")

        # Ensure last layer is full_attention for better performance
        if layer_types[-1] != "full_attention":
            layer_types[-1] = "full_attention"

        new_config["layer_types"] = layer_types

        # Validate the configuration
        is_valid, issues = validate_config(new_config)

        if not is_valid:
            print(
                f"Configuration validation issues for {target_size_millions}M {strategy}:"
            )
            for issue in issues:
                print(f"  - {issue}")
            print("Attempting automatic fixes...")

            # Attempt to fix common issues
            # Fix head dimension divisibility
            if new_config["emb_dim"] % new_config["n_heads"] != 0:
                # Adjust emb_dim to be divisible by n_heads
                new_emb_dim = (
                    new_config["emb_dim"] // new_config["n_heads"]
                ) * new_config["n_heads"]
                print(
                    f"  Adjusting emb_dim from {new_config['emb_dim']} to {new_emb_dim}"
                )
                new_config["emb_dim"] = new_emb_dim
                new_config["head_dim"] = new_emb_dim // new_config["n_heads"]

            # Re-validate after fixes
            is_valid, issues = validate_config(new_config)
            if not is_valid:
                print("Warning: Configuration still has issues after automatic fixes:")
                for issue in issues:
                    print(f"  - {issue}")

        # Calculate and display parameter count
        param_info = calculate_model_params(new_config)
        actual_millions = param_info["total_millions"]

        print(f"\nPerseid-{target_size_millions}M ({strategy} strategy):")
        print(f"  Target: {target_size_millions}M parameters")
        print(f"  Actual: {actual_millions:.2f}M parameters")
        print(
            f"  Difference: {actual_millions - target_size_millions:.2f}M "
            f"({100 * (actual_millions - target_size_millions) / target_size_millions:.1f}%)"
        )
        print(f"  Configuration:")
        print(f"    - emb_dim: {new_config['emb_dim']}")
        print(f"    - hidden_dim: {new_config['hidden_dim']}")
        print(f"    - n_layers: {new_config['n_layers']}")
        print(f"    - n_heads: {new_config['n_heads']}")
        print(
            f"    - head_dim: {new_config.get('head_dim', new_config['emb_dim'] // new_config['n_heads'])}"
        )

        return new_config

    except Exception as e:
        print(f"Error creating Perseid configuration: {e}")
        traceback.print_exc()
        raise


def compare_configurations(configs: dict[str, dict]) -> None:
    """
    Compare multiple model configurations side by side.

    Args:
        configs (dict): Dictionary mapping configuration names to configurations
    """
    try:
        print("\n" + "=" * 80)
        print("Configuration Comparison")
        print("=" * 80)

        # Collect parameter information for all configs
        all_params = {}
        for name, config in configs.items():
            all_params[name] = calculate_model_params(config)

        # Display comparison table
        print(
            f"\n{'Model':<20} {'Total Params':<15} {'Layers':<8} {'Emb Dim':<10} {'Hidden Dim':<12}"
        )
        print("-" * 65)

        for name, config in configs.items():
            params = all_params[name]
            print(
                f"{name:<20} "
                f"{params['total_millions']:.1f}M{'':<10} "
                f"{config['n_layers']:<8} "
                f"{config['emb_dim']:<10} "
                f"{config['hidden_dim']:<12}"
            )

        # Display detailed breakdown for each
        print("\nDetailed Parameter Breakdown:")
        print("-" * 65)

        for name, params in all_params.items():
            print(f"\n{name}:")
            print(f"  Embedding:     {params['embedding']:>12,} params")
            print(f"  Per Layer:")
            print(f"    - Attention: {params['attention_per_layer']:>12,} params")
            print(f"    - FFN:       {params['ffn_per_layer']:>12,} params")
            print(f"    - Norms:     {params['norm_per_layer']:>12,} params")
            print(f"  Total/Layer:   {params['total_per_layer']:>12,} params")
            print(f"  Final Norm:    {params['final_norm']:>12,} params")
            print(f"  Output Head:   {params['output_head']:>12,} params")
            print(f"  " + "-" * 40)
            print(
                f"  TOTAL:         {params['total']:>12,} params ({params['total_millions']:.2f}M)"
            )

    except Exception as e:
        print(f"Error comparing configurations: {e}")
        traceback.print_exc()


def fine_tune_config_for_target(
    base_config: dict, target_millions: float, tolerance: float = 1.0
) -> dict:
    """
    Fine-tune a configuration to hit an exact parameter target.

    Args:
        base_config (dict): Starting configuration
        target_millions (float): Target parameter count in millions
        tolerance (float): Acceptable deviation in millions

    Returns:
        dict: Fine-tuned configuration
    """
    try:
        config = base_config.copy()
        current_params = calculate_model_params(config)
        current_millions = current_params["total_millions"]

        print(f"\nFine-tuning configuration for {target_millions}M parameters...")
        print(f"Starting from {current_millions:.2f}M")

        # Strategy: Adjust hidden_dim as it has the most impact
        iterations = 0
        max_iterations = 20

        while (
            abs(current_millions - target_millions) > tolerance
            and iterations < max_iterations
        ):
            iterations += 1

            # Calculate adjustment needed
            param_diff_millions = target_millions - current_millions

            # Estimate impact of hidden_dim change
            # Each unit change in hidden_dim affects roughly 3 * emb_dim * n_layers parameters
            params_per_hidden_unit = 3 * config["emb_dim"] * config["n_layers"]
            hidden_dim_adjustment = int(
                param_diff_millions * 1_000_000 / params_per_hidden_unit
            )

            # Make adjustment (but keep it reasonable)
            hidden_dim_adjustment = max(-256, min(256, hidden_dim_adjustment))

            if hidden_dim_adjustment == 0:
                # If adjustment is too small, make minimum change
                hidden_dim_adjustment = 64 if param_diff_millions > 0 else -64

            new_hidden_dim = config["hidden_dim"] + hidden_dim_adjustment

            # Ensure 8-bit quantization friendliness
            new_hidden_dim = round(new_hidden_dim / 128) * 128
            new_hidden_dim = max(128, new_hidden_dim)  # Minimum reasonable size

            config["hidden_dim"] = new_hidden_dim

            # Recalculate
            current_params = calculate_model_params(config)
            current_millions = current_params["total_millions"]

            print(
                f"  Iteration {iterations}: hidden_dim={new_hidden_dim}, params={current_millions:.2f}M"
            )

        if abs(current_millions - target_millions) <= tolerance:
            print(f"Successfully fine-tuned to {current_millions:.2f}M parameters")
        else:
            print(
                f"Reached maximum iterations. Final: {current_millions:.2f}M parameters"
            )

        return config

    except Exception as e:
        print(f"Error fine-tuning configuration: {e}")
        traceback.print_exc()
        raise


# Example usage function
def main():
    """
    Example usage of the Perseid configuration utilities.
    """
    try:
        print("Creating Perseid Model Configurations")
        print("=" * 80)

        # Create configurations for different sizes and strategies
        configs = {}

        # Create 256M variants
        configs["Perseid-256M-balanced"] = create_perseid_config(
            256, strategy="balanced"
        )
        configs["Perseid-256M-deep"] = create_perseid_config(256, strategy="deep")
        configs["Perseid-256M-wide"] = create_perseid_config(256, strategy="wide")

        # Create 288M variant
        configs["Perseid-288M-balanced"] = create_perseid_config(
            288, strategy="balanced"
        )

        # Create 320M variant (if desired)
        configs["Perseid-320M-balanced"] = create_perseid_config(
            320, strategy="balanced"
        )

        # Compare all configurations
        compare_configurations(configs)

        # Fine-tune a configuration for exact target
        print("\n" + "=" * 80)
        print("Fine-tuning for exact targets")
        print("=" * 80)

        fine_tuned = fine_tune_config_for_target(
            configs["Perseid-256M-balanced"], target_millions=256.0, tolerance=0.5
        )

        # Validate final configuration
        is_valid, issues = validate_config(fine_tuned)
        if is_valid:
            print("\n✓ Fine-tuned configuration is valid")
        else:
            print("\n✗ Fine-tuned configuration has issues:")
            for issue in issues:
                print(f"  - {issue}")

        return configs

    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run example usage
    perseid_configs = main()
