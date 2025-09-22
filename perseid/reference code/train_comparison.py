
#!/usr/bin/env python
"""
Perseid vs. Gemma Comparison Training Script
File: train_comparison.py

This script trains and compares Gemma baseline with Perseid variants
to evaluate architectural modifications on fine-tuning performance.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

# Import models and utilities
from gemma_model import Gemma3Model, GEMMA3_CONFIG_270M, load_weights_into_gemma
from perseid_model import PerseidModel
from train_gemma_module import (
    GemmaTokenizer,
    create_gemma_dataloader,
    calc_loss_batch,
    calc_loss_loader,
    generate_text_simple,
)


class ComparisonExperiment:
    """
    Manages comparison experiments between Gemma and Perseid model variants.

    This class handles:
    - Model initialization with pretrained weights
    - Parallel training with identical conditions
    - Metric tracking and comparison
    - Result visualization and reporting
    """

    def __init__(self, base_config: Dict, experiment_name: str = "perseid_vs_gemma"):
        """
        Initialize comparison experiment.

        Args:
            base_config: Base model configuration (Gemma 270M config)
            experiment_name: Name for this experiment run
        """
        self.base_config = base_config
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"experiments/{experiment_name}_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("Warning: Using CPU - training will be slow")

        # Storage for experiment results
        self.models = {}
        self.training_histories = {}
        self.evaluation_metrics = {}

        print(f"Experiment initialized: {experiment_name}")
        print(f"Results will be saved to: {self.results_dir}")

    def initialize_models(self, dropout_strategies: List[str], pretrained_weights_path: Optional[str] = None):
        """
        Initialize Gemma baseline and Perseid variants with same pretrained weights.

        Args:
            dropout_strategies: List of dropout strategies to test
            pretrained_weights_path: Path to pretrained Gemma weights
        """
        try:
            print("\n" + "="*50)
            print("Initializing models for comparison")
            print("="*50)

            # Load pretrained weights if provided
            pretrained_weights = None
            if pretrained_weights_path:
                print(f"Loading pretrained weights from {pretrained_weights_path}")
                from safetensors.torch import load_file
                pretrained_weights = load_file(pretrained_weights_path)

            # Initialize Gemma baseline (original architecture)
            print("\n1. Initializing Gemma baseline model...")
            gemma_model = Gemma3Model(self.base_config)
            if pretrained_weights:
                load_weights_into_gemma(gemma_model, self.base_config, pretrained_weights)
            gemma_model = gemma_model.to(self.device)
            self.models["gemma_baseline"] = gemma_model
            print(f"   Gemma baseline: {sum(p.numel() for p in gemma_model.parameters()):,} parameters")

            # Initialize Perseid variants with different dropout strategies
            for strategy in dropout_strategies:
                print(f"\n2. Initializing Perseid variant: {strategy}")
                perseid_model = PerseidModel(self.base_config, dropout_strategy=strategy)

                # Load same pretrained weights into Perseid
                if pretrained_weights:
                    self._load_weights_into_perseid(perseid_model, pretrained_weights)

                perseid_model = perseid_model.to(self.device)
                model_key = f"perseid_{strategy}"
                self.models[model_key] = perseid_model
                print(f"   {perseid_model.model_variant}: {sum(p.numel() for p in perseid_model.parameters()):,} parameters")

            print(f"\nTotal models initialized: {len(self.models)}")

        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            traceback.print_exc()
            raise

    def _load_weights_into_perseid(self, perseid_model: PerseidModel, weights_dict: Dict):
        """
        Load Gemma pretrained weights into Perseid model.

        Dropout layers have no weights, so we only load the base architecture weights.

        Args:
            perseid_model: Perseid model instance
            weights_dict: Dictionary of pretrained weights
        """
        try:
            # Use the same loading function as Gemma since base architecture is identical
            load_weights_into_gemma(perseid_model, self.base_config, weights_dict)
            print(f"   Loaded pretrained weights into {perseid_model.model_variant}")

        except Exception as e:
            print(f"Error loading weights into Perseid: {str(e)}")
            traceback.print_exc()
            raise

    def run_comparison_training(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        eval_interval: int = 50,
        gradient_accumulation_steps: int = 4
    ):
        """
        Run parallel training on all model variants with identical conditions.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for all models
            eval_interval: Steps between evaluations
            gradient_accumulation_steps: Gradient accumulation for memory efficiency
        """
        try:
            print("\n" + "="*50)
            print("Starting comparison training")
            print("="*50)
            print(f"Epochs: {num_epochs}, LR: {learning_rate}, Eval interval: {eval_interval}")

            # Initialize optimizers for each model
            optimizers = {}
            for model_name, model in self.models.items():
                optimizers[model_name] = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=0.01,
                    betas=(0.9, 0.95)
                )
                self.training_histories[model_name] = {
                    "train_losses": [],
                    "val_losses": [],
                    "steps": [],
                    "epoch_times": []
                }

            # Training loop
            global_step = 0
            for epoch in range(num_epochs):
                epoch_start = time.time()
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 30)

                # Train each model on same batches
                for batch_idx, (input_batch, target_batch) in enumerate(train_loader):

                    # Train each model variant
                    for model_name, model in self.models.items():
                        model.train()
                        optimizer = optimizers[model_name]

                        # Calculate loss
                        loss = calc_loss_batch(input_batch, target_batch, model, self.device)
                        loss = loss / gradient_accumulation_steps
                        loss.backward()

                        # Update weights after accumulation
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()

                    # Evaluation at intervals
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        global_step += 1

                        if global_step % eval_interval == 0:
                            self._evaluate_all_models(
                                train_loader, val_loader, global_step, epoch
                            )

                # End of epoch timing
                epoch_time = time.time() - epoch_start
                for model_name in self.models:
                    self.training_histories[model_name]["epoch_times"].append(epoch_time)

                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

            print("\n" + "="*50)
            print("Training completed!")
            print("="*50)

        except Exception as e:
            print(f"Error during comparison training: {str(e)}")
            traceback.print_exc()
            raise

    def _evaluate_all_models(self, train_loader, val_loader, step: int, epoch: int):
        """
        Evaluate all models and record metrics.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            step: Current global step
            epoch: Current epoch
        """
        print(f"\nEvaluation at step {step} (epoch {epoch + 1}):")

        for model_name, model in self.models.items():
            model.eval()

            with torch.no_grad():
                # Calculate losses
                train_loss = calc_loss_loader(train_loader, model, self.device, num_batches=10)
                val_loss = calc_loss_loader(val_loader, model, self.device, num_batches=10)

                # Store metrics
                self.training_histories[model_name]["train_losses"].append(train_loss)
                self.training_histories[model_name]["val_losses"].append(val_loss)
                self.training_histories[model_name]["steps"].append(step)

                # Calculate perplexity
                val_perplexity = torch.exp(torch.tensor(val_loss)).item()

                print(f"  {model_name:30s} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                      f"Perplexity: {val_perplexity:.2f}")

    def analyze_results(self):
        """
        Analyze and compare training results across all model variants.

        Generates comparison metrics and identifies best performing variant.
        """
        print("\n" + "="*50)
        print("Results Analysis")
        print("="*50)

        analysis = {}

        for model_name, history in self.training_histories.items():
            if len(history["val_losses"]) > 0:
                final_val_loss = history["val_losses"][-1]
                best_val_loss = min(history["val_losses"])
                convergence_speed = self._calculate_convergence_speed(history["val_losses"])
                overfitting_gap = history["train_losses"][-1] - history["val_losses"][-1]

                analysis[model_name] = {
                    "final_val_loss": final_val_loss,
                    "best_val_loss": best_val_loss,
                    "final_perplexity": torch.exp(torch.tensor(final_val_loss)).item(),
                    "convergence_speed": convergence_speed,
                    "overfitting_gap": overfitting_gap,
                    "total_training_time": sum(history["epoch_times"])
                }

                print(f"\n{model_name}:")
                print(f"  Final validation loss: {final_val_loss:.4f}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                print(f"  Final perplexity: {analysis[model_name]['final_perplexity']:.2f}")
                print(f"  Convergence speed: {convergence_speed:.4f}")
                print(f"  Overfitting gap: {overfitting_gap:.4f}")
                print(f"  Total training time: {analysis[model_name]['total_training_time']:.2f}s")

        # Identify best model
        if analysis:
            best_model = min(analysis.keys(), key=lambda x: analysis[x]["best_val_loss"])
            print(f"\n🏆 Best performing model: {best_model}")
            print(f"   with validation loss: {analysis[best_model]['best_val_loss']:.4f}")

        self.evaluation_metrics = analysis

        # Save analysis to file
        analysis_path = self.results_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to {analysis_path}")

    def _calculate_convergence_speed(self, losses: List[float]) -> float:
        """
        Calculate convergence speed as rate of improvement.

        Args:
            losses: List of loss values

        Returns:
            Convergence speed metric
        """
        if len(losses) < 2:
            return 0.0

        # Calculate average rate of improvement
        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(improvement)

        return np.mean(improvements) if improvements else 0.0

    def plot_comparison(self):
        """
        Generate comparison plots for all model variants.

        Creates visualizations for loss curves, perplexity, and convergence.
        """
        try:
            print("\nGenerating comparison plots...")

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Perseid vs. Gemma Comparison - {self.experiment_name}", fontsize=16)

            colors = plt.cm.Set1(np.linspace(0, 1, len(self.models)))

            for idx, (model_name, history) in enumerate(self.training_histories.items()):
                if len(history["steps"]) == 0:
                    continue

                color = colors[idx]

                # Training loss
                axes[0, 0].plot(history["steps"], history["train_losses"],
                              label=model_name, color=color, linestyle='-', alpha=0.7)

                # Validation loss
                axes[0, 1].plot(history["steps"], history["val_losses"],
                              label=model_name, color=color, linestyle='-')

                # Perplexity
                perplexities = [torch.exp(torch.tensor(loss)).item()
                               for loss in history["val_losses"]]
                axes[1, 0].plot(history["steps"], perplexities,
                              label=model_name, color=color, linestyle='-')

                # Overfitting gap
                if len(history["train_losses"]) == len(history["val_losses"]):
                    gaps = [val - train for train, val in
                           zip(history["train_losses"], history["val_losses"])]
                    axes[1, 1].plot(history["steps"], gaps,
                                  label=model_name, color=color, linestyle='-')

            # Configure subplots
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Steps")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].set_title("Validation Loss")
            axes[0, 1].set_xlabel("Steps")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].set_title("Validation Perplexity")
            axes[1, 0].set_xlabel("Steps")
            axes[1, 0].set_ylabel("Perplexity")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].set_title("Generalization Gap (Val - Train)")
            axes[1, 1].set_xlabel("Steps")
            axes[1, 1].set_ylabel("Loss Difference")
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = self.results_dir / "comparison_plots.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved to {plot_path}")
            plt.show()

        except Exception as e:
            print(f"Error generating plots: {str(e)}")
            traceback.print_exc()

    def save_results(self):
        """
        Save all experimental results and model checkpoints.

        Saves training histories, metrics, and best model weights.
        """
        try:
            print("\nSaving experimental results...")

            # Save training histories
            histories_path = self.results_dir / "training_histories.json"
            with open(histories_path, "w") as f:
                json.dump(self.training_histories, f, indent=2, default=str)
            print(f"Training histories saved to {histories_path}")

            # Save best model checkpoints
            checkpoints_dir = self.results_dir / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)

            for model_name, model in self.models.items():
                checkpoint_path = checkpoints_dir / f"{model_name}_final.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': self.base_config,
                    'model_variant': model_name,
                    'training_history': self.training_histories.get(model_name, {}),
                    'final_metrics': self.evaluation_metrics.get(model_name, {})
                }, checkpoint_path)
                print(f"  {model_name} checkpoint saved")

            # Save experiment summary
            summary = {
                "experiment_name": self.experiment_name,
                "timestamp": self.timestamp,
                "base_config": self.base_config,
                "models_tested": list(self.models.keys()),
                "evaluation_metrics": self.evaluation_metrics,
                "best_model": min(self.evaluation_metrics.keys(),
                                 key=lambda x: self.evaluation_metrics[x]["best_val_loss"])
                                 if self.evaluation_metrics else None
            }

            summary_path = self.results_dir / "experiment_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Experiment summary saved to {summary_path}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            traceback.print_exc()

    def generate_comparison_samples(self, tokenizer, prompts: List[str], max_tokens: int = 50):
        """
        Generate text samples from all models for qualitative comparison.

        Args:
            tokenizer: Tokenizer instance
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
        """
        print("\n" + "="*50)
        print("Generating comparison samples")
        print("="*50)

        samples = {}

        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            print("-" * 30)
            samples[prompt] = {}

            for model_name, model in self.models.items():
                model.eval()

                # Generate text
                with torch.no_grad():
                    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
                    generated_ids = generate_text_simple(
                        model, input_ids, max_tokens,
                        eos_token_id=tokenizer.encode(tokenizer.eos_token)[-1] if tokenizer.eos_token else None
                    )
                    generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist())

                samples[prompt][model_name] = generated_text
                print(f"\n{model_name}:")
                print(f"  {generated_text}")

        # Save samples
        samples_path = self.results_dir / "generation_samples.json"
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"\nGeneration samples saved to {samples_path}")

        return samples


def main():
    """
    Main function to run Perseid vs. Gemma comparison experiment.

    This function orchestrates the complete experimental pipeline:
    1. Setup and configuration
    2. Data preparation
    3. Model initialization
    4. Training comparison
    5. Results analysis and visualization
    """
    try:
        print("\n" + "="*70)
        print(" PERSEID vs. GEMMA ARCHITECTURE COMPARISON EXPERIMENT")
        print("="*70)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Configuration
        GEMMA3_CONFIG_270M = {
            "vocab_size": 262_144,
            "context_length": 512,  # Reduced for memory efficiency
            "emb_dim": 640,
            "n_heads": 4,
            "n_layers": 18,
            "hidden_dim": 2048,
            "head_dim": 256,
            "qk_norm": True,
            "n_kv_groups": 1,
            "rope_local_base": 10_000.0,
            "rope_base": 1_000_000.0,
            "sliding_window": 512,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "dtype": torch.bfloat16,
            "query_pre_attn_scalar": 256,
        }

        # Training settings
        TRAINING_CONFIG = {
            "num_epochs": 3,
            "batch_size": 1,  # Small for memory constraints
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "eval_interval": 20,  # Evaluate every N steps
            "context_length": 512,
            "max_tokens_generate": 50
        }

        # Dropout strategies to test
        DROPOUT_STRATEGIES = [
            "baseline",      # No dropout (Gemma equivalent)
            "conservative",  # Light dropout (0.05)
            "moderate",      # Traditional dropout (0.1)
            "attention_only" # Dropout only on attention
        ]

        print("\nExperiment Configuration:")
        print(f"  Models to compare: Gemma baseline + {len(DROPOUT_STRATEGIES)} Perseid variants")
        print(f"  Training epochs: {TRAINING_CONFIG['num_epochs']}")
        print(f"  Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
        print(f"  Context length: {TRAINING_CONFIG['context_length']}")

        # Initialize experiment
        experiment = ComparisonExperiment(
            base_config=GEMMA3_CONFIG_270M,
            experiment_name="perseid_dropout_comparison"
        )

        # Setup tokenizer
        print("\n" + "="*50)
        print("Setting up tokenizer")
        print("="*50)

        from huggingface_hub import hf_hub_download

        repo_id = "google/gemma-3-270m"
        local_dir = Path(repo_id).parts[-1]

        tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file_path):
            print("Downloading Gemma tokenizer...")
            tokenizer_file_path = hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=local_dir
            )

        tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)
        print("Tokenizer loaded successfully")

        # Load training data
        print("\n" + "="*50)
        print("Loading training data")
        print("="*50)

        import urllib.request

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

        # Clean text data (remove Gutenberg headers/footers)
        start_marker = "*** START OF"
        end_marker = "*** END OF"

        start_idx = text_data.find(start_marker)
        if start_idx != -1:
            start_idx = text_data.find("\n", start_idx) + 1
            text_data = text_data[start_idx:]

        end_idx = text_data.find(end_marker)
        if end_idx != -1:
            text_data = text_data[:end_idx]

        print(f"Loaded {len(text_data):,} characters")

        # Tokenize and check data size
        all_tokens = tokenizer.encode(text_data)
        print(f"Total tokens: {len(all_tokens):,}")

        # Create data loaders
        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))

        train_loader = create_gemma_dataloader(
            text_data[:split_idx],
            tokenizer=tokenizer,
            batch_size=TRAINING_CONFIG["batch_size"],
            max_length=TRAINING_CONFIG["context_length"],
            stride=TRAINING_CONFIG["context_length"],
            drop_last=True,
            shuffle=True
        )

        val_loader = create_gemma_dataloader(
            text_data[split_idx:],
            tokenizer=tokenizer,
            batch_size=TRAINING_CONFIG["batch_size"],
            max_length=TRAINING_CONFIG["context_length"],
            stride=TRAINING_CONFIG["context_length"],
            drop_last=False,
            shuffle=False
        )

        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        # Download pretrained weights
        print("\n" + "="*50)
        print("Downloading pretrained weights")
        print("="*50)

        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        print(f"Weights downloaded to {weights_file}")

        # Initialize models
        experiment.initialize_models(
            dropout_strategies=DROPOUT_STRATEGIES,
            pretrained_weights_path=weights_file
        )

        # Run comparison training
        experiment.run_comparison_training(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=TRAINING_CONFIG["num_epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            eval_interval=TRAINING_CONFIG["eval_interval"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"]
        )

        # Analyze results
        experiment.analyze_results()

        # Generate comparison plots
        experiment.plot_comparison()

        # Generate text samples for qualitative comparison
        test_prompts = [
            "Alice was beginning to get very tired",
            "The Queen of Hearts",
            "Down the rabbit hole",
            "Once upon a time"
        ]

        experiment.generate_comparison_samples(
            tokenizer=tokenizer,
            prompts=test_prompts,
            max_tokens=TRAINING_CONFIG["max_tokens_generate"]
        )

        # Save all results
        experiment.save_results()

        print("\n" + "="*70)
        print(" EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Results saved to: {experiment.results_dir}")

        # Print final summary
        if experiment.evaluation_metrics:
            print("\nFinal Performance Summary:")
            print("-" * 40)

            # Sort models by validation loss
            sorted_models = sorted(
                experiment.evaluation_metrics.items(),
                key=lambda x: x[1]["best_val_loss"]
            )

            for rank, (model_name, metrics) in enumerate(sorted_models, 1):
                print(f"{rank}. {model_name:30s} - Val Loss: {metrics['best_val_loss']:.4f}, "
                      f"Perplexity: {metrics['final_perplexity']:.2f}")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

