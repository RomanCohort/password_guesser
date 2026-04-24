"""
Knowledge Distillation for Password Guessing Models

Features:
- Teacher-student distillation
- Temperature scaling for soft labels
- Intermediate layer matching
- Attention transfer
- Progressive distillation

Usage:
    from optimization.distillation import KnowledgeDistillation, DistillationTrainer

    # Create distillation trainer
    distiller = DistillationTrainer(
        teacher=large_model,
        student=small_model,
        temperature=4.0
    )

    # Train student with distillation
    distiller.train(train_loader, epochs=10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import copy


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for distillation loss vs hard label loss
    beta: float = 0.0   # Weight for intermediate layer matching
    gamma: float = 0.0  # Weight for attention transfer
    teacher_device: str = 'cuda'
    student_device: str = 'cuda'


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    Combines:
    - KL divergence between teacher and student soft logits
    - Cross entropy with hard labels
    - Intermediate layer matching (optional)
    - Attention transfer (optional)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        ignore_index: int = 0
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab]
            teacher_logits: Teacher model logits [batch, seq_len, vocab]
            labels: Ground truth labels [batch, seq_len]

        Returns:
            Total loss and dictionary of individual losses
        """
        # Soften logits with temperature
        T = self.temperature

        # KL divergence loss (distillation)
        # Only compute for non-padding positions
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs.view(-1, student_logits.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='none'
        )

        # Mask padding
        mask = (labels != self.ignore_index).view(-1)
        kl_loss = kl_loss.view(-1)[mask].mean() * (T ** 2)

        # Hard label loss (cross entropy)
        ce_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        return total_loss, {
            'kl_loss': kl_loss.item(),
            'ce_loss': ce_loss.item(),
            'total_loss': total_loss.item()
        }


class IntermediateLoss(nn.Module):
    """
    Loss for matching intermediate layer representations.

    Encourages student to learn similar representations as teacher
    at corresponding layers.
    """

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        normalize: bool = True
    ):
        super().__init__()
        self.normalize = normalize

        # Projection to align dimensions if different
        if teacher_dim != student_dim:
            self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intermediate layer matching loss.

        Args:
            teacher_features: Teacher intermediate features
            student_features: Student intermediate features

        Returns:
            MSE loss between projected features
        """
        # Project student features to match teacher dimension
        student_proj = self.projection(student_features)

        if self.normalize:
            teacher_features = F.normalize(teacher_features, dim=-1)
            student_proj = F.normalize(student_proj, dim=-1)

        return F.mse_loss(teacher_features, student_proj)


class AttentionTransferLoss(nn.Module):
    """
    Loss for transferring attention patterns.

    Helps student learn similar attention patterns as teacher.
    Useful for transformer-based architectures.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        teacher_attns: List[torch.Tensor],
        student_attns: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention transfer loss.

        Args:
            teacher_attns: List of teacher attention weights
            student_attns: List of student attention weights

        Returns:
            Sum of MSE losses for each attention layer
        """
        total_loss = 0.0

        for t_attn, s_attn in zip(teacher_attns, student_attns):
            # Normalize attention weights
            t_attn = F.normalize(t_attn.view(t_attn.size(0), -1), dim=-1)
            s_attn = F.normalize(s_attn.view(s_attn.size(0), -1), dim=-1)

            total_loss += F.mse_loss(t_attn, s_attn)

        return total_loss / len(teacher_attns)


class KnowledgeDistillation:
    """
    Main knowledge distillation class.

    Manages teacher-student training with multiple distillation strategies.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None
    ):
        self.teacher = teacher
        self.student = student
        self.config = config or DistillationConfig()

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Initialize losses
        self.distill_loss = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha
        )

        # Track intermediate layers for matching
        self.teacher_hooks = []
        self.student_hooks = []
        self.teacher_intermediates = {}
        self.student_intermediates = {}

    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        self.teacher_intermediates.clear()
        self.student_intermediates.clear()

        def make_hook(storage, name):
            def hook(module, input, output):
                storage[name] = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
            return hook

        # Register on teacher
        for name, module in self.teacher.named_modules():
            if isinstance(module, nn.Linear) and 'layers' in name:
                self.teacher_hooks.append(
                    module.register_forward_hook(make_hook(self.teacher_intermediates, name))
                )

        # Register on student
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Linear) and 'layers' in name:
                self.student_hooks.append(
                    module.register_forward_hook(make_hook(self.student_intermediates, name))
                )

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.teacher_hooks + self.student_hooks:
            hook.remove()
        self.teacher_hooks.clear()
        self.student_hooks.clear()

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total distillation loss"""
        return self.distill_loss(student_logits, teacher_logits, labels)

    @torch.no_grad()
    def get_teacher_logits(self, *args, **kwargs) -> torch.Tensor:
        """Get teacher model logits"""
        self.teacher.eval()
        return self.teacher(*args, **kwargs)


class DistillationTrainer:
    """
    Trainer for knowledge distillation.

    Handles the training loop for distilling knowledge from
    a large teacher model to a smaller student model.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001
    ):
        self.teacher = teacher.to(config.teacher_device if config else 'cuda')
        self.student = student.to(config.student_device if config else 'cuda')
        self.config = config or DistillationConfig()

        self.distiller = KnowledgeDistillation(teacher, student, config)

        self.optimizer = optimizer or torch.optim.AdamW(
            student.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = None

    def train_epoch(
        self,
        dataloader,
        epoch: int,
        mlp_encoder: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch with distillation.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            mlp_encoder: Optional MLP encoder for features

        Returns:
            Dictionary of average losses
        """
        self.student.train()
        self.teacher.eval()

        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        num_batches = 0

        for batch in dataloader:
            # Parse batch
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    input_ids, labels, features = batch
                else:
                    input_ids, labels = batch
                    features = torch.zeros(input_ids.size(0), 64)
            else:
                continue

            input_ids = input_ids.to(self.config.student_device)
            labels = labels.to(self.config.student_device)
            features = features.to(self.config.student_device)

            # Get latent if using MLP encoder
            if mlp_encoder is not None:
                latent = mlp_encoder(features)
            else:
                latent = features

            # Teacher forward
            with torch.no_grad():
                teacher_input = input_ids.to(self.config.teacher_device)
                teacher_latent = latent.to(self.config.teacher_device)
                teacher_logits = self.teacher(teacher_input, teacher_latent)
                teacher_logits = teacher_logits.to(self.config.student_device)

            # Student forward
            student_logits = self.student(input_ids, latent)

            # Compute loss
            loss, loss_dict = self.distiller.compute_loss(
                student_logits, teacher_logits, labels
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss_dict['total_loss']
            total_kl += loss_dict['kl_loss']
            total_ce += loss_dict['ce_loss']
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'kl_loss': total_kl / num_batches,
            'ce_loss': total_ce / num_batches
        }

    def train(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        mlp_encoder: Optional[nn.Module] = None,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop for distillation.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            epochs: Number of training epochs
            mlp_encoder: Optional MLP encoder
            save_path: Path to save best student model
            verbose: Print progress

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'kl_loss': [],
            'ce_loss': []
        }

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, mlp_encoder)
            history['train_loss'].append(train_metrics['loss'])
            history['kl_loss'].append(train_metrics['kl_loss'])
            history['ce_loss'].append(train_metrics['ce_loss'])

            # Validate
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, mlp_encoder)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        torch.save(self.student.state_dict(), save_path)
                        if verbose:
                            print(f"Saved best student model to {save_path}")

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {train_metrics['loss']:.4f}"
                if val_loader is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                print(msg)

        return history

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        mlp_encoder: Optional[nn.Module] = None
    ) -> float:
        """Evaluate student model"""
        self.student.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    input_ids, labels, features = batch
                else:
                    input_ids, labels = batch
                    features = torch.zeros(input_ids.size(0), 64)
            else:
                continue

            input_ids = input_ids.to(self.config.student_device)
            labels = labels.to(self.config.student_device)
            features = features.to(self.config.student_device)

            if mlp_encoder is not None:
                latent = mlp_encoder(features)
            else:
                latent = features

            logits = self.student(input_ids, latent)

            # Cross entropy only for evaluation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


class ProgressiveDistillation:
    """
    Progressive knowledge distillation.

    Gradually distills knowledge through multiple stages,
    with intermediate checkpoints and curriculum learning.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student_configs: List[Dict[str, Any]],
        base_config: Optional[DistillationConfig] = None
    ):
        """
        Args:
            teacher: Teacher model
            student_configs: List of student configs for each stage
                            (e.g., decreasing sizes)
            base_config: Base distillation config
        """
        self.teacher = teacher
        self.student_configs = student_configs
        self.config = base_config or DistillationConfig()
        self.stages = []

    def add_stage(
        self,
        student: nn.Module,
        temperature: float,
        alpha: float
    ):
        """Add a distillation stage"""
        config = DistillationConfig(
            temperature=temperature,
            alpha=alpha,
            teacher_device=self.config.teacher_device,
            student_device=self.config.student_device
        )
        self.stages.append({
            'student': student,
            'config': config
        })

    def train_progressive(
        self,
        train_loader,
        val_loader=None,
        epochs_per_stage: int = 10,
        save_dir: Optional[str] = None
    ) -> List[Dict[str, List[float]]]:
        """
        Train through all stages progressively.

        Each stage uses the previous student as a new teacher.
        """
        all_history = []
        current_teacher = self.teacher

        for i, stage in enumerate(self.stages):
            print(f"\n=== Stage {i + 1}/{len(self.stages)} ===")

            trainer = DistillationTrainer(
                teacher=current_teacher,
                student=stage['student'],
                config=stage['config']
            )

            save_path = None
            if save_dir:
                save_path = f"{save_dir}/student_stage_{i + 1}.pt"

            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_stage,
                save_path=save_path
            )

            all_history.append(history)

            # Use this student as teacher for next stage
            current_teacher = copy.deepcopy(stage['student'])

        return all_history


def create_student_model(
    teacher: nn.Module,
    compression_ratio: float = 0.5,
    config: Optional[DistillationConfig] = None
) -> nn.Module:
    """
    Create a smaller student model based on teacher architecture.

    Args:
        teacher: Teacher model
        compression_ratio: Target size ratio (student/teacher)
        config: Distillation config

    Returns:
        Student model with reduced dimensions
    """
    # Copy teacher architecture
    student = copy.deepcopy(teacher)

    # Scale down dimensions
    for name, module in student.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features

            new_in = int(in_features * compression_ratio)
            new_out = int(out_features * compression_ratio)

            # Create smaller linear layer
            new_module = nn.Linear(new_in, new_out, bias=module.bias is not None)

            # Initialize weights
            nn.init.xavier_uniform_(new_module.weight)
            if module.bias is not None:
                nn.init.zeros_(new_module.bias)

            # Replace module (simplified - would need proper parent tracking)
            # This is a placeholder for proper implementation

    return student
