"""
Distributed training utilities.

Provides helpers for initializing ``torch.distributed``, wrapping models
with :class:`~torch.nn.parallel.DistributedDataParallel`, and performing
collective operations such as tensor reduction and barrier synchronisation.
"""

import os
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = 'nccl', init_method: str = 'env://') -> bool:
    """Initialize the distributed training environment.

    Inspects the ``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, and
    ``MASTER_PORT`` environment variables.  If ``RANK`` and ``WORLD_SIZE``
    are both set, the process group is initialised.

    Args:
        backend: Communication backend (``'nccl'`` for GPU, ``'gloo'`` for CPU).
        init_method: URL specifying how to initialise the process group.

    Returns:
        ``True`` if distributed training was successfully initialised,
        ``False`` otherwise (e.g. running in single-process mode).
    """
    rank = os.environ.get('RANK')
    world_size = os.environ.get('WORLD_SIZE')

    if rank is None or world_size is None:
        logger.info(
            "RANK or WORLD_SIZE not set – running in single-process mode."
        )
        return False

    try:
        rank = int(rank)
        world_size = int(world_size)

        # Set default master address/port if not provided
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        logger.info(
            "Distributed training initialised: rank=%d, world_size=%d, backend=%s",
            rank,
            world_size,
            backend,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to initialise distributed training: %s", exc)
        return False


def cleanup_distributed():
    """Destroy the distributed process group if it is initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")


class DistributedTrainer:
    """Wrapper that manages distributed training setup and model wrapping.

    On construction the trainer attempts to initialise the distributed
    process group.  If the environment variables are not set (single-process
    mode) the trainer degrades gracefully: ``is_distributed`` remains
    ``False`` and all methods behave as thin pass-throughs.

    Args:
        model: The :class:`~torch.nn.Module` to train.
        device: The target device.
        backend: Communication backend (default ``'nccl'``).
        find_unused_parameters: Forwarded to :class:`DDP`.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        backend: str = 'nccl',
        find_unused_parameters: bool = False,
    ):
        self.model = model
        self.device = device
        self.rank = 0
        self.world_size = 1
        self.is_distributed = False
        self._ddp_model: Optional[DDP] = None
        self._find_unused = find_unused_parameters
        self._backend = backend
        self._setup(backend)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self, backend: str):
        """Attempt to initialise distributed training.

        After this call ``is_distributed``, ``rank``, and ``world_size``
        reflect the actual environment.
        """
        self.is_distributed = setup_distributed(backend=backend)

        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Move model to device
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(self) -> nn.Module:
        """Wrap the model with :class:`DDP` when running distributed.

        In single-process mode the unwrapped model is returned.

        Returns:
            The (possibly DDP-wrapped) model.
        """
        if self.is_distributed:
            self._ddp_model = DDP(
                self.model,
                device_ids=[self.device] if self.device.type == 'cuda' else None,
                output_device=self.device if self.device.type == 'cuda' else None,
                find_unused_parameters=self._find_unused,
            )
            return self._ddp_model
        return self.model

    # ------------------------------------------------------------------
    # Sampler helpers
    # ------------------------------------------------------------------

    def create_sampler(
        self, dataset, shuffle: bool = True
    ) -> Optional[DistributedSampler]:
        """Create a :class:`DistributedSampler` for the given dataset.

        Returns ``None`` in single-process mode.

        Args:
            dataset: A :class:`~torch.utils.data.Dataset`.
            shuffle: Whether to shuffle indices across epochs.

        Returns:
            A :class:`DistributedSampler` or ``None``.
        """
        if self.is_distributed:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
        return None

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    def is_main_process(self) -> bool:
        """Return ``True`` if this is the rank-0 process."""
        return self.rank == 0

    def reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Average-reduce a tensor across all processes.

        In single-process mode the tensor is returned unchanged.

        Args:
            tensor: A tensor residing on any device.

        Returns:
            The averaged tensor.
        """
        if not self.is_distributed:
            return tensor

        # Move to CUDA for NCCL backend if needed
        if self._backend == 'nccl' and tensor.device.type != 'cuda':
            tensor = tensor.cuda()

        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def barrier(self):
        """Synchronise all processes.  No-op in single-process mode."""
        if self.is_distributed:
            dist.barrier()

    def save_on_main(self, state: dict, path: str):
        """Save *state* to *path* only from the main process.

        Args:
            state: Dictionary of serialisable objects (typically model and
                optimizer state dicts).
            path: Destination file path.
        """
        if self.is_main_process():
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            torch.save(state, path)
            logger.info("Checkpoint saved to %s", path)
