"""
Training script for MAMBA Password Model

Features:
- Mixed Precision Training (AMP) for 2x memory savings
- Warmup + Cosine Annealing scheduler
- Gradient accumulation for larger effective batch sizes
- Early stopping to prevent overfitting
- OneCycleLR scheduler option
- Gradient checkpointing for memory efficiency

Usage:
    python train.py --config config.yaml --data passwords.txt --epochs 100
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Optional
import math

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MambaPasswordModel, MLPEncoder, create_mamba_model, create_mlp_encoder
from models.password_dataset import load_password_file, create_dataloader, PasswordDataset
from utils import PasswordTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train MAMBA Password Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--data', type=str, required=True, help='Password data file')
    parser.add_argument('--output', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--max_samples', type=int, default=100000, help='Max samples to load')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'onecycle'], help='LR scheduler type')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load configuration from YAML file"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    """Get torch device"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing Learning Rate Scheduler.

    Linear warmup for the first warmup_steps, then cosine annealing.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement
    for 'patience' consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric (loss or accuracy)
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  Improvement: {self.best_score:.4f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered! Best epoch: {self.best_epoch}")

        return self.early_stop

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class Trainer:
    """
    Training manager for MAMBA password model.

    Features:
    - Mixed Precision Training (AMP)
    - Warmup + Cosine scheduler or OneCycleLR
    - Gradient accumulation
    - Gradient clipping
    - Early stopping
    - Gradient checkpointing
    """

    def __init__(
        self,
        model: MambaPasswordModel,
        mlp_encoder: MLPEncoder,
        tokenizer: PasswordTokenizer,
        device: torch.device,
        lr: float = 0.001,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        scheduler_type: str = 'cosine',
        gradient_checkpointing: bool = False
    ):
        self.model = model.to(device)
        self.mlp_encoder = mlp_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scheduler_type = scheduler_type
        self.total_steps = total_steps

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
            print("Gradient checkpointing enabled")

        # Optimizers with weight decay
        self.model_optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.mlp_optimizer = optim.AdamW(
            mlp_encoder.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate schedulers
        if scheduler_type == 'onecycle':
            # OneCycleLR: aggressive learning rate scheduling
            self.model_scheduler = optim.lr_scheduler.OneCycleLR(
                self.model_optimizer,
                max_lr=lr * 10,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0
            )
            self.mlp_scheduler = optim.lr_scheduler.OneCycleLR(
                self.mlp_optimizer,
                max_lr=lr * 10,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0
            )
            print(f"Using OneCycleLR scheduler (max_lr={lr * 10:.2e})")
        else:
            # Default: Warmup + Cosine Annealing
            self.model_scheduler = WarmupCosineScheduler(
                self.model_optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=lr * 0.01
            )
            self.mlp_scheduler = WarmupCosineScheduler(
                self.mlp_optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=lr * 0.01
            )
            print(f"Using Warmup + Cosine Annealing scheduler (warmup={warmup_steps})")

        # AMP GradScaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Log AMP status
        if self.use_amp:
            print(f"Mixed Precision Training (AMP) enabled")
        else:
            print(f"Mixed Precision Training (AMP) disabled")

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        # Check if model supports gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            # Manual implementation for custom models
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with AMP support"""
        self.model.train()
        self.mlp_encoder.train()

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Handle batch with or without features
            if len(batch) == 3:
                input_ids, labels, features = batch
            else:
                input_ids, labels = batch
                features = torch.zeros(input_ids.size(0), 64)

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            features = features.to(self.device)

            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    latent = self.mlp_encoder(features)
                    loss = self.model.compute_loss(input_ids, latent, labels)
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                # Optimizer step every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.model_optimizer)
                    self.scaler.unscale_(self.mlp_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.mlp_encoder.parameters(), max_norm=1.0)

                    # Optimizer step
                    self.scaler.step(self.model_optimizer)
                    self.scaler.step(self.mlp_optimizer)
                    self.scaler.update()

                    # Zero gradients
                    self.model_optimizer.zero_grad()
                    self.mlp_optimizer.zero_grad()

                    # Update schedulers
                    self.model_scheduler.step()
                    self.mlp_scheduler.step()

                    total_loss += accumulated_loss
                    accumulated_loss = 0.0
                    num_batches += 1
                    self.global_step += 1
            else:
                # Standard precision training
                latent = self.mlp_encoder(features)
                loss = self.model.compute_loss(input_ids, latent, labels)
                loss = loss / self.gradient_accumulation_steps

                loss.backward()
                accumulated_loss += loss.item()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.mlp_encoder.parameters(), max_norm=1.0)

                    # Optimizer step
                    self.model_optimizer.step()
                    self.mlp_optimizer.step()

                    # Zero gradients
                    self.model_optimizer.zero_grad()
                    self.mlp_optimizer.zero_grad()

                    # Update schedulers
                    self.model_scheduler.step()
                    self.mlp_scheduler.step()

                    total_loss += accumulated_loss
                    accumulated_loss = 0.0
                    num_batches += 1
                    self.global_step += 1

            # Update progress bar
            current_lr = self.model_optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': accumulated_loss if accumulated_loss > 0 else total_loss / max(num_batches, 1),
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.epoch += 1

        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set with AMP support"""
        self.model.eval()
        self.mlp_encoder.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if len(batch) == 3:
                input_ids, labels, features = batch
            else:
                input_ids, labels = batch
                features = torch.zeros(input_ids.size(0), 64)

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            features = features.to(self.device)

            if self.use_amp:
                with autocast():
                    latent = self.mlp_encoder(features)
                    loss = self.model.compute_loss(input_ids, latent, labels)
            else:
                latent = self.mlp_encoder(features)
                loss = self.model.compute_loss(input_ids, latent, labels)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, path: str, loss: float):
        """Save training checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'mlp_state_dict': self.mlp_encoder.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),
            'mlp_optimizer': self.mlp_optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'use_amp': self.use_amp,
            'scheduler_type': self.scheduler_type,
        }

        # Save scheduler state (WarmupCosineScheduler uses current_step)
        if self.scheduler_type == 'cosine':
            checkpoint['model_scheduler_step'] = self.model_scheduler.current_step
            checkpoint['mlp_scheduler_step'] = self.mlp_scheduler.current_step
        else:
            checkpoint['model_scheduler_state'] = self.model_scheduler.state_dict()
            checkpoint['mlp_scheduler_state'] = self.mlp_scheduler.state_dict()

        if self.use_amp:
            checkpoint['scaler'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.mlp_encoder.load_state_dict(checkpoint['mlp_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
        self.mlp_optimizer.load_state_dict(checkpoint['mlp_optimizer'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        # Restore scheduler state
        if self.scheduler_type == 'cosine':
            if 'model_scheduler_step' in checkpoint:
                self.model_scheduler.current_step = checkpoint['model_scheduler_step']
                self.mlp_scheduler.current_step = checkpoint['mlp_scheduler_step']
        else:
            if 'model_scheduler_state' in checkpoint:
                self.model_scheduler.load_state_dict(checkpoint['model_scheduler_state'])
                self.mlp_scheduler.load_state_dict(checkpoint['mlp_scheduler_state'])

        if self.use_amp and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        print(f"Loaded checkpoint from epoch {self.epoch}")


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Get device
    device = get_device(args.device if args.device != 'auto' else config['training'].get('device', 'auto'))
    print(f"Using device: {device}")

    # Check AMP availability
    use_amp = args.amp and device.type == 'cuda'
    if args.amp and device.type != 'cuda':
        print("Warning: AMP requested but CUDA not available. Falling back to FP32.")

    # Load password data
    print(f"Loading password data from {args.data}...")
    passwords, frequencies = load_password_file(args.data, max_samples=args.max_samples)
    print(f"Loaded {len(passwords)} passwords")

    # Split into train/val
    split_idx = int(len(passwords) * 0.9)
    train_passwords = passwords[:split_idx]
    val_passwords = passwords[split_idx:]

    # Create tokenizer
    tokenizer = PasswordTokenizer()

    # Create dataloaders
    train_loader = create_dataloader(
        train_passwords,
        tokenizer,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = create_dataloader(
        val_passwords,
        tokenizer,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Create models
    model = create_mamba_model(config)
    mlp_encoder = create_mlp_encoder(config)

    print(f"MAMBA Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"MLP Encoder parameters: {sum(p.numel() for p in mlp_encoder.parameters()):,}")

    # Calculate total training steps
    total_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps

    # Create trainer
    trainer = Trainer(
        model=model,
        mlp_encoder=mlp_encoder,
        tokenizer=tokenizer,
        device=device,
        lr=config['training']['learning_rate'],
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        use_amp=use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scheduler_type=args.scheduler,
        gradient_checkpointing=args.gradient_checkpointing
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode='min',
        verbose=True
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Training loop
    num_epochs = config['training']['epochs']

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        current_lr = trainer.model_optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_loss < trainer.best_loss:
            trainer.best_loss = val_loss
            trainer.save_checkpoint(
                os.path.join(args.output, 'best_model.pt'),
                val_loss
            )

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                os.path.join(args.output, f'checkpoint_epoch_{epoch + 1}.pt'),
                val_loss
            )

        # Early stopping check
        if early_stopping(val_loss, epoch + 1):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save final model
    trainer.save_checkpoint(
        os.path.join(args.output, 'final_model.pt'),
        val_loss
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_loss:.4f}")


if __name__ == '__main__':
    main()
