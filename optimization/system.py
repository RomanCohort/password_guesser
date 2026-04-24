"""
System-level Optimizations for Password Guessing

Features:
- Incremental training from checkpoints
- Evaluation cache to avoid redundant scoring
- Model registry for managing multiple model versions
- Configuration management
- Performance monitoring

Usage:
    from optimization.system import EvaluationCache, IncrementalTrainer

    # Evaluation cache
    cache = EvaluationCache()
    score = cache.get_or_compute(model, "password123", latent, score_fn)

    # Incremental training
    trainer = IncrementalTrainer(model, checkpoint_dir="checkpoints/")
    trainer.train(train_loader, new_epochs=5)
"""

import os
import json
import hashlib
import time
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
import copy


class EvaluationCache:
    """
    Cache for password evaluation scores.

    Avoids redundant computation by caching (password, model_hash) -> score.
    Uses LRU eviction when cache exceeds max_size.
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(password: str, model_hash: str) -> str:
        """Create cache key from password and model hash"""
        return hashlib.md5(f"{password}:{model_hash}".encode()).hexdigest()

    @staticmethod
    def get_model_hash(model: nn.Module) -> str:
        """Compute hash of model parameters"""
        h = hashlib.md5()
        for param in model.parameters():
            h.update(param.data.numpy().tobytes())
        return h.hexdigest()[:16]

    def get(self, password: str, model_hash: str) -> Optional[float]:
        """Get cached score if available"""
        key = self._make_key(password, model_hash)
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, password: str, model_hash: str, score: float):
        """Cache a score"""
        key = self._make_key(password, model_hash)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = score

        # Evict oldest entries if over limit
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def get_or_compute(
        self,
        model: nn.Module,
        password: str,
        latent: torch.Tensor,
        score_fn: Callable
    ) -> float:
        """
        Get cached score or compute and cache it.

        Args:
            model: Model instance
            password: Password string
            latent: Latent tensor for conditioning
            score_fn: Function(model, password, latent) -> score

        Returns:
            Password score
        """
        model_hash = self.get_model_hash(model)
        cached = self.get(password, model_hash)

        if cached is not None:
            return cached

        score = score_fn(model, password, latent)
        self.put(password, model_hash, score)
        return score

    def batch_get_or_compute(
        self,
        model: nn.Module,
        passwords: List[str],
        latent: torch.Tensor,
        score_fn: Callable
    ) -> List[float]:
        """
        Batch score passwords with caching.

        Args:
            model: Model instance
            passwords: List of password strings
            latent: Latent tensor
            score_fn: Scoring function

        Returns:
            List of scores
        """
        model_hash = self.get_model_hash(model)
        results = []
        uncached = []
        uncached_indices = []

        for i, pwd in enumerate(passwords):
            cached = self.get(pwd, model_hash)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached.append(pwd)
                uncached_indices.append(i)

        # Compute uncached scores
        if uncached:
            for idx, pwd in zip(uncached_indices, uncached):
                score = score_fn(model, pwd, latent)
                self.put(pwd, model_hash, score)
                results[idx] = score

        return results

    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0.0
        }

    def save(self, path: str):
        """Save cache to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'cache': dict(self._cache),
            'hits': self._hits,
            'misses': self._misses
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load cache from disk"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self._cache = OrderedDict(data['cache'])
            self._hits = data.get('hits', 0)
            self._misses = data.get('misses', 0)


@dataclass
class CheckpointInfo:
    """Information about a training checkpoint"""
    path: str
    epoch: int
    global_step: int
    train_loss: float
    val_loss: float
    timestamp: float
    is_best: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    Manages training checkpoints with versioning.

    Features:
    - Automatic checkpoint saving
    - Best model tracking
    - Checkpoint pruning (keep only N most recent)
    - Incremental training support
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 5,
        save_best: bool = True
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        self.save_best = save_best
        self.checkpoints: List[CheckpointInfo] = []
        self.best_checkpoint: Optional[CheckpointInfo] = None
        self.best_val_loss = float('inf')

        os.makedirs(checkpoint_dir, exist_ok=True)
        self._load_index()

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        global_step: int,
        train_loss: float,
        val_loss: float,
        metadata: Optional[Dict] = None
    ) -> CheckpointInfo:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            global_step: Current global step
            train_loss: Training loss
            val_loss: Validation loss
            metadata: Additional metadata

        Returns:
            CheckpointInfo for the saved checkpoint
        """
        is_best = val_loss < self.best_val_loss

        # Determine save path
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            self.best_val_loss = val_loss
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

        # Build checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            if hasattr(scheduler, 'state_dict'):
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            elif hasattr(scheduler, 'current_step'):
                checkpoint_data['scheduler_step'] = scheduler.current_step

        # Save
        torch.save(checkpoint_data, path)

        # Record checkpoint info
        info = CheckpointInfo(
            path=path,
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            timestamp=time.time(),
            is_best=is_best,
            metadata=metadata or {}
        )

        if is_best:
            self.best_checkpoint = info

        self.checkpoints.append(info)
        self._prune_old_checkpoints()
        self._save_index()

        return info

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> CheckpointInfo:
        """
        Load the latest checkpoint.

        Returns:
            CheckpointInfo for the loaded checkpoint
        """
        if not self.checkpoints:
            raise FileNotFoundError("No checkpoints found")

        # Find latest non-best checkpoint
        latest = max(self.checkpoints, key=lambda c: c.timestamp)
        return self._load_checkpoint(latest, model, optimizer, scheduler, device)

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> CheckpointInfo:
        """Load the best checkpoint"""
        if self.best_checkpoint is None:
            raise FileNotFoundError("No best checkpoint found")

        return self._load_checkpoint(
            self.best_checkpoint, model, optimizer, scheduler, device
        )

    def load_specific(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """Load a specific checkpoint file"""
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def _load_checkpoint(
        self,
        info: CheckpointInfo,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        device: str
    ) -> CheckpointInfo:
        """Internal method to load a checkpoint"""
        checkpoint = torch.load(info.path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler:
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif 'scheduler_step' in checkpoint and hasattr(scheduler, 'current_step'):
                scheduler.current_step = checkpoint['scheduler_step']

        return info

    def _prune_old_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        # Keep best + max_keep most recent
        non_best = [c for c in self.checkpoints if not c.is_best]

        if len(non_best) > self.max_keep:
            to_remove = non_best[:-self.max_keep]
            for checkpoint in to_remove:
                if os.path.exists(checkpoint.path) and 'best' not in checkpoint.path:
                    os.remove(checkpoint.path)
                self.checkpoints.remove(checkpoint)

    def _save_index(self):
        """Save checkpoint index"""
        index_path = os.path.join(self.checkpoint_dir, 'checkpoint_index.json')
        data = {
            'best_val_loss': self.best_val_loss,
            'checkpoints': [asdict(c) for c in self.checkpoints]
        }
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_index(self):
        """Load checkpoint index"""
        index_path = os.path.join(self.checkpoint_dir, 'checkpoint_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                data = json.load(f)

            self.best_val_loss = data.get('best_val_loss', float('inf'))

            for c_data in data.get('checkpoints', []):
                info = CheckpointInfo(**c_data)
                self.checkpoints.append(info)
                if info.is_best:
                    self.best_checkpoint = info


class IncrementalTrainer:
    """
    Incremental training support.

    Allows continuing training from a checkpoint with
    potentially different hyperparameters or additional data.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str = 'checkpoints',
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

    def has_checkpoint(self) -> bool:
        """Check if there's an existing checkpoint"""
        return len(self.checkpoint_manager.checkpoints) > 0

    def resume(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Resume from the latest or best checkpoint.

        Returns:
            Dictionary with checkpoint info
        """
        if load_best:
            info = self.checkpoint_manager.load_best(
                self.model, optimizer, device=self.device
            )
        else:
            info = self.checkpoint_manager.load_latest(
                self.model, optimizer, device=self.device
            )

        return {
            'epoch': info.epoch,
            'global_step': info.global_step,
            'val_loss': info.val_loss
        }

    def save(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        global_step: int,
        train_loss: float,
        val_loss: float
    ):
        """Save checkpoint"""
        self.checkpoint_manager.save(
            self.model, optimizer, scheduler,
            epoch, global_step, train_loss, val_loss
        )


class PerformanceMonitor:
    """
    Monitor training and inference performance.

    Tracks metrics like throughput, latency, memory usage,
    and training progress.
    """

    def __init__(self):
        self.metrics: Dict[str, List[Tuple[float, float]]] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        """Start timing an operation"""
        self._start_times[name] = time.perf_counter()

    def end(self, name: str, value: Optional[float] = None):
        """End timing and record"""
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((elapsed, value or elapsed))
            del self._start_times[name]

    def record(self, name: str, value: float):
        """Record a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((time.time(), value))

    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}

        values = [v[1] for v in self.metrics[name]]
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'total': sum(values)
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summaries for all metrics"""
        return {name: self.get_summary(name) for name in self.metrics}

    def print_summary(self):
        """Print performance summary"""
        for name, summary in self.get_all_summaries().items():
            print(f"{name}:")
            for key, value in summary.items():
                print(f"  {key}: {value:.4f}")
