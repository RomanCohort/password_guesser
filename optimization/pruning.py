"""
Model Pruning for Password Guessing Models

Features:
- Magnitude-based pruning
- Structured pruning (entire neurons/channels)
- Unstructured pruning (individual weights)
- Iterative pruning with retraining
- Lottery ticket hypothesis support

Usage:
    from optimization.pruning import prune_model, PruningScheduler

    # Magnitude pruning
    pruned = prune_model(model, amount=0.3, method='magnitude')

    # Structured pruning
    pruned = prune_model(model, amount=0.3, method='structured')

    # Iterative pruning with retraining
    scheduler = PruningScheduler(model, train_loader, epochs_per_round=5)
    scheduler.run(total_rounds=10, target_sparsity=0.8)
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import copy


@dataclass
class PruningConfig:
    """Configuration for model pruning"""
    method: str = 'magnitude'  # 'magnitude', 'structured', 'random', 'gradient'
    amount: float = 0.3  # Fraction of weights to prune
    prune_layers: List[str] = field(default_factory=list)  # Specific layers to prune (empty = all)
    retrain: bool = True  # Whether to retrain after pruning
    iterative: bool = False  # Iterative pruning
    reinit: bool = False  # Reinitialize remaining weights (lottery ticket)


class MagnitudePruning:
    """
    Magnitude-based weight pruning.

    Removes weights with smallest absolute values.
    Simple but effective for many architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.3,
        prune_layers: Optional[List[str]] = None
    ):
        self.model = model
        self.amount = amount
        self.prune_layers = prune_layers
        self.pruned_modules = []

    def apply(self) -> nn.Module:
        """Apply magnitude pruning to the model"""
        self.pruned_modules = []

        for name, module in self.model.named_modules():
            # Skip if not in specified layers
            if self.prune_layers and name not in self.prune_layers:
                continue

            if isinstance(module, nn.Linear):
                # Prune weights
                prune.l1_unstructured(module, name='weight', amount=self.amount)
                self.pruned_modules.append((name, module))

                # Optionally prune biases
                if module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=self.amount)

            elif isinstance(module, nn.Conv1d):
                prune.l1_unstructured(module, name='weight', amount=self.amount)
                self.pruned_modules.append((name, module))

        return self.model

    def remove_masks(self):
        """Remove pruning masks and make pruning permanent"""
        for name, module in self.pruned_modules:
            prune.remove(module, 'weight')
            if module.bias is not None and hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')

    def get_sparsity(self) -> float:
        """Get current model sparsity"""
        total_params = 0
        zero_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total_params += mask.numel()
                    zero_params += (mask == 0).sum().item()
                else:
                    weight = module.weight
                    total_params += weight.numel()
                    zero_params += (weight == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class StructuredPruning:
    """
    Structured pruning for entire neurons/channels.

    Removes entire rows/columns of weight matrices,
    resulting in actual model size reduction.
    """

    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.3
    ):
        self.model = model
        self.amount = amount

    def compute_importance(self, module: nn.Linear) -> torch.Tensor:
        """
        Compute importance score for each neuron.

        Uses L2 norm of input weights.
        """
        weight = module.weight.data
        # L2 norm across input dimension for each output neuron
        importance = torch.norm(weight, p=2, dim=1)
        return importance

    def apply_to_layer(
        self,
        module: nn.Linear,
        input_size: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Apply structured pruning to a single layer.

        Returns new input and output sizes.
        """
        if not isinstance(module, nn.Linear):
            return module.in_features, module.out_features

        # Compute importance scores
        importance = self.compute_importance(module)

        # Determine number of neurons to keep
        num_neurons = importance.size(0)
        num_keep = max(1, int(num_neurons * (1 - self.amount)))

        # Get indices of most important neurons
        _, indices = torch.topk(importance, num_keep)
        indices = torch.sort(indices)[0]

        # Prune output neurons
        new_weight = module.weight.data[indices]
        if module.bias is not None:
            new_bias = module.bias.data[indices]
            module.bias = nn.Parameter(new_bias)

        module.weight = nn.Parameter(new_weight)
        module.out_features = num_keep

        return module.in_features, num_keep

    def apply(self) -> nn.Module:
        """Apply structured pruning to the model"""
        # This is a simplified version
        # Full implementation would need to track layer dependencies
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.apply_to_layer(module)

        return self.model


class GradientPruning:
    """
    Gradient-based pruning.

    Removes weights with smallest gradient magnitudes,
    which contribute least to the loss reduction.
    """

    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.3
    ):
        self.model = model
        self.amount = amount
        self.gradient_sums = {}
        self.hooks = []

    def register_hooks(self):
        """Register backward hooks to accumulate gradients"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                def make_hook(layer_name):
                    def hook(module, grad_input, grad_output):
                        if module.weight.grad is not None:
                            if layer_name not in self.gradient_sums:
                                self.gradient_sums[layer_name] = torch.zeros_like(module.weight)
                            self.gradient_sums[layer_name] += module.weight.grad.abs()
                    return hook
                self.hooks.append(module.register_backward_hook(make_hook(name)))

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def apply(self) -> nn.Module:
        """Apply gradient-based pruning"""
        if not self.gradient_sums:
            raise ValueError("No gradients accumulated. Run backward pass first.")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.gradient_sums:
                # Use accumulated gradients as importance
                importance = self.gradient_sums[name]
                threshold = torch.quantile(importance, self.amount)

                # Create mask
                mask = (importance > threshold).float()
                module.weight.data *= mask

        return self.model

    def reset(self):
        """Reset accumulated gradients"""
        self.gradient_sums.clear()


class IterativePruning:
    """
    Iterative pruning with retraining.

    Gradually increases sparsity over multiple rounds,
    allowing the model to adapt to each pruning step.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.8,
        num_rounds: int = 10,
        retrain_epochs: int = 5
    ):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.num_rounds = num_rounds
        self.retrain_epochs = retrain_epochs

        # Store initial weights for lottery ticket hypothesis
        self.initial_weights = copy.deepcopy(model.state_dict())

    def get_sparsity_schedule(self) -> List[float]:
        """Generate sparsity schedule for each round"""
        return np.linspace(
            self.initial_sparsity,
            self.final_sparsity,
            self.num_rounds
        ).tolist()

    def prune_round(
        self,
        target_sparsity: float,
        train_loader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable
    ) -> float:
        """
        Execute one round of pruning and retraining.

        Returns achieved sparsity.
        """
        # Apply pruning
        pruner = MagnitudePruning(self.model, amount=target_sparsity)
        pruner.apply()

        current_sparsity = pruner.get_sparsity()

        # Retrain
        self.model.train()
        for epoch in range(self.retrain_epochs):
            total_loss = 0.0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch[:2]
                    features = batch[2] if len(batch) > 2 else torch.zeros(input_ids.size(0), 64)
                else:
                    continue

                optimizer.zero_grad()
                logits = self.model(input_ids, features)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return current_sparsity

    def run(
        self,
        train_loader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        reinit: bool = False
    ) -> List[float]:
        """
        Run full iterative pruning schedule.

        Args:
            train_loader: Training data
            optimizer: Optimizer instance
            loss_fn: Loss function
            reinit: Whether to reinitialize remaining weights

        Returns:
            List of sparsity after each round
        """
        schedule = self.get_sparsity_schedule()
        sparsities = []

        for round_idx, target_sparsity in enumerate(schedule):
            print(f"\nRound {round_idx + 1}/{self.num_rounds} - Target sparsity: {target_sparsity:.2%}")

            current_sparsity = self.prune_round(
                target_sparsity,
                train_loader,
                optimizer,
                loss_fn
            )
            sparsities.append(current_sparsity)

            print(f"  Achieved sparsity: {current_sparsity:.2%}")

            # Lottery ticket hypothesis: reinitialize remaining weights
            if reinit:
                self._reinit_remaining_weights()

        return sparsities

    def _reinit_remaining_weights(self):
        """Reinitialize weights that weren't pruned (Lottery Ticket Hypothesis)"""
        current_state = self.model.state_dict()

        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                # Get mask (where current weights are non-zero)
                mask = (param.data != 0).float()

                # Reinitialize from initial weights
                param.data = self.initial_weights[name] * mask


class PruningScheduler:
    """
    High-level pruning scheduler.

    Manages the entire pruning workflow including
    evaluation, retraining, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.best_model = None
        self.best_score = float('-inf')
        self.history = []

    def evaluate(self) -> float:
        """Evaluate model performance"""
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.train_loader:
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch[:2]
                    features = batch[2] if len(batch) > 2 else torch.zeros(input_ids.size(0), 64)
                else:
                    continue

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                features = features.to(self.device)

                logits = self.model(input_ids, features)
                predictions = logits.argmax(dim=-1)

                correct = (predictions == labels).float().sum()
                total_correct += correct.item()
                total_samples += labels.numel()

        return total_correct / total_samples if total_samples > 0 else 0.0

    def run(
        self,
        total_rounds: int = 10,
        target_sparsity: float = 0.8,
        retrain_epochs: int = 5,
        lr: float = 0.0001,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run pruning schedule.

        Args:
            total_rounds: Number of pruning rounds
            target_sparsity: Final target sparsity
            retrain_epochs: Epochs to retrain after each round
            lr: Learning rate for retraining
            save_path: Path to save best model

        Returns:
            Dictionary with pruning history and results
        """
        iterative_pruner = IterativePruning(
            self.model,
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            num_rounds=total_rounds,
            retrain_epochs=retrain_epochs
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        def loss_fn(logits, labels):
            return nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )

        sparsities = iterative_pruner.run(
            self.train_loader,
            optimizer,
            loss_fn
        )

        # Final evaluation
        final_accuracy = self.evaluate()
        final_sparsity = sparsities[-1] if sparsities else 0.0

        results = {
            'sparsities': sparsities,
            'final_accuracy': final_accuracy,
            'final_sparsity': final_sparsity,
            'total_rounds': total_rounds
        }

        if save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'results': results
            }, save_path)
            print(f"Saved pruned model to {save_path}")

        return results


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    method: str = 'magnitude',
    **kwargs
) -> nn.Module:
    """
    Convenience function to prune a model.

    Args:
        model: Model to prune
        amount: Fraction of weights to prune
        method: Pruning method ('magnitude', 'structured', 'random', 'gradient')
        **kwargs: Additional arguments for specific methods

    Returns:
        Pruned model
    """
    if method == 'magnitude':
        pruner = MagnitudePruning(model, amount, kwargs.get('prune_layers'))
        return pruner.apply()

    elif method == 'structured':
        pruner = StructuredPruning(model, amount)
        return pruner.apply()

    elif method == 'random':
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.random_unstructured(module, name='weight', amount=amount)
        return model

    elif method == 'gradient':
        pruner = GradientPruning(model, amount)
        return pruner.apply()

    else:
        raise ValueError(f"Unknown pruning method: {method}")


def get_model_sparsity(model: nn.Module) -> float:
    """Get current model sparsity"""
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()

    return zero_params / total_params if total_params > 0 else 0.0


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module):
    """Print model summary with pruning info"""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    sparsity = get_model_sparsity(model)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Effective parameters: {int(total * (1 - sparsity)):,}")
