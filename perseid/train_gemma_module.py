# gemini_train.py
# Training script for Gemma 3 270M model
# Adapted from gpt_train.py for Gemma architecture

import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import urllib.request
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import json

# Import Gemma model architecture (assuming it's in a separate file)
# You'll need to save the Gemma model code from the notebook into gemma_model.py
from gemma_model import Gemma3Model, GEMMA3_CONFIG_270M, load_weights_into_gemma


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


def apply_chat_template(user_text, use_instruct=False):
    """Apply chat template for instruct model"""
    if use_instruct:
        return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
    return user_text


def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs with batch dimension"""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs back to text"""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def create_gemma_dataloader(text_data, tokenizer, batch_size, max_length, stride, drop_last=True, shuffle=True):
    """Create dataloader for Gemma training"""
    # Tokenize all text
    token_ids = tokenizer.encode(text_data)
    
    # Create sliding window inputs
    input_ids = []
    target_ids = []
    
    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1:i + max_length + 1]
        
        if len(input_chunk) == max_length:
            input_ids.append(input_chunk)
            target_ids.append(target_chunk)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    target_ids = torch.tensor(target_ids)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(input_ids, target_ids)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    
    return dataloader


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Forward pass
    logits = model(input_batch)
    
    # Calculate loss
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over dataloader"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on train and validation sets"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, eos_token_id=None):
    """Simple text generation for Gemma"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for last token
            logits = model(idx)[:, -1, :]
            
            # Get next token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            
            # Append to sequence
            idx = torch.cat((idx, next_token), dim=1)
            
            # Truncate if exceeding context length
            if idx.shape[1] > model.cfg["context_length"]:
                idx = idx[:, -model.cfg["context_length"]:]
    
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, use_instruct=False):
    """Generate and print a sample during training"""
    model.eval()
    
    # Apply chat template if using instruct model
    prompt = apply_chat_template(start_context, use_instruct)
    
    # Encode prompt
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    
    # Get EOS token ID
    eos_token_id = tokenizer.encode(tokenizer.eos_token)[-1] if tokenizer.eos_token else None
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            eos_token_id=eos_token_id
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, use_instruct=False,
                       gradient_accumulation_steps=1):
    """Training loop for Gemma model with gradient accumulation"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Calculate loss (with gradient accumulation)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Track tokens
                tokens_seen += input_batch.numel() * gradient_accumulation_steps
                
                # Evaluation
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # Generate sample after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context, use_instruct
        )
    
    return train_losses, val_losses, track_tokens_seen


def main(config, settings):
    """Main training function"""
    torch.manual_seed(123)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    ###############################
    # Download and setup tokenizer
    ###############################
    
    repo_id = "google/gemma-3-270m-it" if settings["use_instruct"] else "google/gemma-3-270m"
    local_dir = Path(repo_id).parts[-1]
    
    # Download tokenizer
    tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_file_path):
        tokenizer_file_path = hf_hub_download(
            repo_id=repo_id, 
            filename="tokenizer.json", 
            local_dir=local_dir
        )
    
    tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)
    
    ###############################
    # Download training data
    ###############################
    
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    
    ###############################
    # Initialize model
    ###############################
    
    # Adjust config for training
    train_config = config.copy()
    train_config["context_length"] = settings["context_length"]  # Use smaller context for training
    
    model = Gemma3Model(train_config)
    
    # Load pretrained weights if specified
    if settings["use_pretrained"]:
        print("Loading pretrained weights...")
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
        load_weights_into_gemma(model, train_config, weights_dict)
        del weights_dict  # Free memory
        print("Pretrained weights loaded")
    
    # Move model to device
    model = model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 2 / 1e9:.2f} GB (bfloat16)")
    
    ###############################
    # Setup optimizer
    ###############################
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=(0.9, 0.95)  # Common for modern LLMs
    )
    
    ###############################
    # Create data loaders
    ###############################
    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    train_loader = create_gemma_dataloader(
        text_data[:split_idx],
        tokenizer=tokenizer,
        batch_size=settings["batch_size"],
        max_length=settings["context_length"],
        stride=settings["context_length"],
        drop_last=True,
        shuffle=True
    )
    
    val_loader = create_gemma_dataloader(
        text_data[split_idx:],
        tokenizer=tokenizer,
        batch_size=settings["batch_size"],
        max_length=settings["context_length"],
        stride=settings["context_length"],
        drop_last=False,
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    ###############################
    # Train model
    ###############################
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        use_instruct=settings["use_instruct"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"]
    )
    
    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    
    # Model configuration
    GEMMA3_CONFIG_270M = {
        "vocab_size": 262_144,
        "context_length": 32_768,  # Will be overridden by training settings
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
    
    # Training settings optimized for 12-13GB VRAM
    TRAINING_SETTINGS = {
        # Model settings
        "use_pretrained": True,  # Start from pretrained weights for fine-tuning
        "use_instruct": False,   # Use instruct variant
        
        # Memory-conscious settings
        "context_length": 512,   # Reduced from 32k to fit in memory
        "batch_size": 1,         # Small batch size due to memory constraints
        "gradient_accumulation_steps": 4,  # Simulate larger batch size
        
        # Training hyperparameters
        "learning_rate": 5e-5,   # Lower LR for fine-tuning
        "num_epochs": 3,         # Fewer epochs for fine-tuning
        "weight_decay": 0.01,    # Less regularization than GPT-2
    }
    
    # Run training
    train_losses, val_losses, tokens_seen, model = main(
        GEMMA3_CONFIG_270M,
        TRAINING_SETTINGS
    )
    
    # Plot results
    epochs_tensor = torch.linspace(0, TRAINING_SETTINGS["num_epochs"], len(train_losses))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_tensor, train_losses, label="Training Loss")
    plt.plot(epochs_tensor, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Gemma 3 270M Training Progress")
    plt.savefig("gemma_training_loss.pdf")
    plt.show()
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), "gemma3_270m_finetuned.pth")
    print("Model saved to gemma3_270m_finetuned.pth")

