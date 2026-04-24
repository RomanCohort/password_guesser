"""
Performance profiling and monitoring utilities.

Provides a function-level timing decorator, GPU memory/utilisation helpers,
and system-level resource monitoring (CPU, RAM, disk).
"""

import time
import functools
import os
from typing import Optional, Dict, Any, Callable


def profile(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
) -> Callable:
    """Decorator that measures and logs wall-clock execution time.

    Can be used with or without arguments::

        @profile
        def train_one_epoch(...): ...

        @profile(name='custom_name')
        def train_one_epoch(...): ...

    Args:
        func: The function to wrap (when used without parentheses).
        name: Optional custom name used in the log output.

    Returns:
        The wrapped function (when used as a decorator).
    """
    def decorator(fn: Callable) -> Callable:
        tag = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                # Use print to avoid dependency on logging configuration
                print(
                    f"[profile] {tag} took {elapsed:.4f}s",
                    flush=True,
                )
        return wrapper

    if func is not None:
        # Used as @profile without parentheses
        return decorator(func)
    # Used as @profile(...) with parentheses
    return decorator


class GPUProfiler:
    """GPU performance and memory monitoring.

    All methods are safe to call even when CUDA is not available; they
    return empty dicts in that case.
    """

    @staticmethod
    def get_gpu_stats() -> Dict[str, Any]:
        """Get comprehensive GPU utilisation and memory statistics.

        Returns a dictionary with per-GPU information including memory
        allocated/reserved, utilisation percentage, and device name.

        Returns:
            Dictionary keyed by GPU index, or empty dict if CUDA unavailable.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return {}

            stats = {}
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                allocated = torch.cuda.memory_allocated(idx)
                reserved = torch.cuda.memory_reserved(idx)
                total = props.total_memory

                stats[f'gpu_{idx}'] = {
                    'name': props.name,
                    'total_memory_mb': total / (1024 ** 2),
                    'allocated_memory_mb': allocated / (1024 ** 2),
                    'reserved_memory_mb': reserved / (1024 ** 2),
                    'memory_utilization_pct': (allocated / total * 100) if total > 0 else 0.0,
                    'cuda_capability': f"{props.major}.{props.minor}",
                }

            return stats
        except ImportError:
            return {}

    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get GPU memory usage in megabytes.

        Returns:
            Dictionary with ``'allocated_mb'`` and ``'reserved_mb'`` keys,
            or empty dict if CUDA is unavailable.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return {}

            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return {
                'allocated_mb': allocated / (1024 ** 2),
                'reserved_mb': reserved / (1024 ** 2),
            }
        except ImportError:
            return {}


class SystemProfiler:
    """System-level resource monitoring (CPU, memory, disk)."""

    @staticmethod
    def cpu_usage() -> float:
        """Get current CPU utilisation as a percentage.

        Uses :func:`os.cpu_count` and ``/proc/stat`` on Linux or
        falls back to :func:`psutil.cpu_percent` when available.

        Returns:
            CPU usage percentage (0.0 - 100.0).
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            pass

        # Fallback: read /proc/stat (Linux only)
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            values = [int(v) for v in line.split()[1:]]
            idle = values[3]
            total = sum(values)
            if total == 0:
                return 0.0
            return (1.0 - idle / total) * 100.0
        except (FileNotFoundError, OSError, ValueError):
            return 0.0

    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get system memory usage.

        Returns:
            Dictionary with ``'total_mb'``, ``'used_mb'``, ``'available_mb'``,
            and ``'percent'`` keys.
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total_mb': mem.total / (1024 ** 2),
                'used_mb': mem.used / (1024 ** 2),
                'available_mb': mem.available / (1024 ** 2),
                'percent': mem.percent,
            }
        except ImportError:
            pass

        # Fallback: read /proc/meminfo (Linux only)
        try:
            info = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(':')
                    if key in ('MemTotal', 'MemAvailable', 'MemFree'):
                        info[key] = int(parts[1]) * 1024  # Convert kB to bytes

            total = info.get('MemTotal', 0)
            available = info.get('MemAvailable', info.get('MemFree', 0))
            used = total - available
            return {
                'total_mb': total / (1024 ** 2),
                'used_mb': used / (1024 ** 2),
                'available_mb': available / (1024 ** 2),
                'percent': (used / total * 100) if total > 0 else 0.0,
            }
        except (FileNotFoundError, OSError, ValueError):
            return {
                'total_mb': 0.0,
                'used_mb': 0.0,
                'available_mb': 0.0,
                'percent': 0.0,
            }

    @staticmethod
    def disk_usage(path: str = '.') -> Dict[str, float]:
        """Get disk usage for the filesystem containing *path*.

        Args:
            path: Any path on the target filesystem.

        Returns:
            Dictionary with ``'total_mb'``, ``'used_mb'``, ``'free_mb'``,
            and ``'percent'`` keys.
        """
        try:
            usage = os.statvfs(path)
            total = usage.f_blocks * usage.f_frsize
            free = usage.f_bavail * usage.f_frsize
            used = total - free
            return {
                'total_mb': total / (1024 ** 2),
                'used_mb': used / (1024 ** 2),
                'free_mb': free / (1024 ** 2),
                'percent': (used / total * 100) if total > 0 else 0.0,
            }
        except (AttributeError, OSError):
            # Windows or unsupported path
            try:
                import shutil
                total, used, free = shutil.disk_usage(path)
                return {
                    'total_mb': total / (1024 ** 2),
                    'used_mb': used / (1024 ** 2),
                    'free_mb': free / (1024 ** 2),
                    'percent': (used / total * 100) if total > 0 else 0.0,
                }
            except Exception:
                return {
                    'total_mb': 0.0,
                    'used_mb': 0.0,
                    'free_mb': 0.0,
                    'percent': 0.0,
                }

    @staticmethod
    def get_all_stats() -> Dict[str, Any]:
        """Collect all available system and GPU statistics.

        Returns:
            Nested dictionary with ``'cpu'``, ``'memory'``, ``'disk'``,
            and ``'gpu'`` sections.
        """
        return {
            'cpu': {
                'usage_percent': SystemProfiler.cpu_usage(),
                'count': os.cpu_count() or 0,
            },
            'memory': SystemProfiler.memory_usage(),
            'disk': SystemProfiler.disk_usage(),
            'gpu': GPUProfiler.get_gpu_stats(),
        }
