"""
Training monitoring and logging utilities.

Provides :class:`TrainingMonitor` as a unified interface that dispatches
metric logging to optional backends such as `Weights & Biases`_ (WandB)
and `TensorBoard`_.

.. _Weights & Biases: https://wandb.ai/
.. _TensorBoard: https://www.tensorflow.org/tensorboard
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Central training monitor that aggregates and dispatches metrics.

    Supports optional WandB and TensorBoard backends.  All metrics are
    also stored in memory and can be queried or persisted to disk.

    Args:
        log_dir: Root directory for log files.
        project_name: Name used by external loggers (e.g. WandB project).
    """

    def __init__(
        self,
        log_dir: str = 'logs',
        project_name: str = 'password_guesser',
    ):
        self.log_dir = log_dir
        self.project_name = project_name
        self.metrics: Dict[str, List[tuple]] = defaultdict(list)  # name -> [(step, value)]
        self.current_epoch = 0
        self.start_time = time.time()
        self._writers: List[Any] = []

        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Backend registration
    # ------------------------------------------------------------------

    def add_wandb(self, config: Optional[dict] = None, **kwargs):
        """Register a :class:`WandBLogger` as a backend.

        If the ``wandb`` package is not installed the call is silently
        ignored.

        Args:
            config: Optional run configuration dict forwarded to ``wandb.init``.
            **kwargs: Additional keyword arguments for :class:`WandBLogger`.
        """
        try:
            wb = WandBLogger(
                project=self.project_name,
                config=config,
                enabled=True,
                **kwargs,
            )
            wb.init(config=config)
            self._writers.append(wb)
            logger.info("WandB logger registered.")
        except Exception as exc:
            logger.warning("Could not initialise WandB logger: %s", exc)

    def add_tensorboard(self, log_dir: Optional[str] = None):
        """Register a :class:`TensorBoardLogger` as a backend.

        If ``torch.utils.tensorboard`` is not available the call is silently
        ignored.

        Args:
            log_dir: Optional override for the TensorBoard log directory.
        """
        try:
            tb = TensorBoardLogger(
                log_dir=log_dir or os.path.join(self.log_dir, 'tensorboard')
            )
            tb.init()
            self._writers.append(tb)
            logger.info("TensorBoard logger registered.")
        except Exception as exc:
            logger.warning("Could not initialise TensorBoard logger: %s", exc)

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single scalar metric.

        Args:
            name: Metric name (e.g. ``'train/loss'``).
            value: Scalar value.
            step: Global step number.  Auto-incremented if ``None``.
        """
        if step is None:
            step = len(self.metrics[name])

        self.metrics[name].append((step, value))

        for writer in self._writers:
            try:
                if isinstance(writer, TensorBoardLogger):
                    writer.log_scalar(name, value, step)
                elif isinstance(writer, WandBLogger):
                    writer.log({name: value}, step=step)
            except Exception as exc:
                logger.debug("Writer error for %s: %s", type(writer).__name__, exc)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar metrics at once.

        Args:
            metrics: Mapping of metric names to scalar values.
            step: Global step number.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        extra: Optional[dict] = None,
    ):
        """Log standard epoch-level metrics.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for the epoch.
            val_loss: Validation loss for the epoch.
            lr: Current learning rate.
            extra: Optional additional metrics to log.
        """
        self.current_epoch = epoch
        step = epoch

        self.log_metric('epoch/train_loss', train_loss, step=step)
        self.log_metric('epoch/val_loss', val_loss, step=step)
        self.log_metric('epoch/lr', lr, step=step)

        if extra:
            for name, value in extra.items():
                self.log_metric(f'epoch/{name}', value, step=step)

        # Persist epoch summary to disk
        epoch_summary = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
            'elapsed_seconds': time.time() - self.start_time,
        }
        if extra:
            epoch_summary['extra'] = extra

        summary_path = os.path.join(self.log_dir, 'epoch_log.jsonl')
        try:
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(epoch_summary, ensure_ascii=False) + '\n')
        except Exception as exc:
            logger.debug("Failed to write epoch log: %s", exc)

    def log_generation_sample(self, epoch: int, passwords: List[str]):
        """Log a sample of generated passwords for inspection.

        Args:
            epoch: Current epoch number.
            passwords: List of generated password strings.
        """
        self.log_metric('generation/count', len(passwords), step=epoch)

        sample_path = os.path.join(self.log_dir, f'generated_epoch_{epoch}.txt')
        try:
            with open(sample_path, 'w', encoding='utf-8') as f:
                for pw in passwords:
                    f.write(pw + '\n')
        except Exception as exc:
            logger.debug("Failed to write generation sample: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self):
        """Flush all backends and release resources."""
        for writer in self._writers:
            try:
                writer.finish() if hasattr(writer, 'finish') else writer.close()
            except Exception as exc:
                logger.debug("Error closing writer %s: %s", type(writer).__name__, exc)

        # Persist in-memory metrics to disk
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        try:
            serialisable = {
                name: [(s, v) for s, v in entries]
                for name, entries in self.metrics.items()
            }
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(serialisable, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("Failed to save metrics: %s", exc)

        logger.info(
            "Training monitor finished. Epoch=%d, elapsed=%.1fs",
            self.current_epoch,
            time.time() - self.start_time,
        )


class WandBLogger:
    """Weights & Biases logger wrapper.

    Args:
        project: WandB project name.
        config: Optional run configuration.
        enabled: If ``False`` all calls are no-ops.
    """

    def __init__(
        self,
        project: str,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.project = project
        self._wandb = None

        if enabled:
            try:
                import wandb  # noqa: F401
                self._wandb = wandb
            except ImportError:
                logger.warning(
                    "wandb package not installed.  WandB logging disabled."
                )
                self.enabled = False

    def init(self, config: Optional[dict] = None):
        """Initialise a WandB run.

        Args:
            config: Optional run configuration forwarded to ``wandb.init``.
        """
        if not self.enabled or self._wandb is None:
            return

        self._wandb.init(
            project=self.project,
            config=config or {},
            reinit=True,
        )

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number.
        """
        if not self.enabled or self._wandb is None:
            return

        kwargs = {'step': step} if step is not None else {}
        self._wandb.log(metrics, **kwargs)

    def finish(self):
        """End the WandB run."""
        if not self.enabled or self._wandb is None:
            return

        self._wandb.finish()


class TensorBoardLogger:
    """TensorBoard logger wrapper.

    Args:
        log_dir: Directory where TensorBoard event files are written.
    """

    def __init__(self, log_dir: str = 'runs/password_guesser'):
        self.log_dir = log_dir
        self._writer = None

    def init(self):
        """Create the TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning(
                "torch.utils.tensorboard not available. "
                "TensorBoard logging disabled."
            )

    def log_scalar(self, tag: str, value: float, step: int):
        """Write a scalar value.

        Args:
            tag: Metric tag.
            value: Scalar value.
            step: Global step.
        """
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict, step: int):
        """Write multiple scalar values.

        Args:
            metrics: Mapping of tags to scalar values.
            step: Global step.
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_text(self, tag: str, text: str, step: int):
        """Write a text entry.

        Args:
            tag: Text tag.
            text: Text content.
            step: Global step.
        """
        if self._writer is not None:
            self._writer.add_text(tag, text, step)

    def close(self):
        """Flush and close the SummaryWriter."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
