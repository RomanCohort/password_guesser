"""
Async Task Management for Web API

Provides background task execution with status tracking.
"""

import asyncio
import uuid
import time
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import Future


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Information about an async task"""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Task duration in seconds"""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def wait_time(self) -> float:
        """Time waited before starting"""
        if self.started_at is None:
            return time.time() - self.created_at
        return self.started_at - self.created_at

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result if self.status == TaskStatus.COMPLETED else None,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class TaskManager:
    """Async task manager for background processing"""

    def __init__(
        self,
        max_concurrent: int = 4,
        cleanup_interval: float = 300,
        max_age: float = 3600
    ):
        self.max_concurrent = max_concurrent
        self.cleanup_interval = cleanup_interval
        self.max_age = max_age

        self.tasks: Dict[str, TaskInfo] = {}
        self._running: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

    def _generate_id(self) -> str:
        """Generate a unique task ID"""
        return str(uuid.uuid4())[:8]

    async def submit(
        self,
        func: Callable[..., Awaitable[Any]],
        **kwargs
    ) -> str:
        """
        Submit an async function for background execution.

        Args:
            func: Async function to execute
            **kwargs: Arguments to pass to the function

        Returns:
            Task ID for tracking
        """
        task_id = self._generate_id()

        async with self._lock:
            self.tasks[task_id] = TaskInfo(
                task_id=task_id,
                metadata={"kwargs": kwargs}
            )

        # Create background task
        task = asyncio.create_task(
            self._execute(task_id, func, kwargs)
        )
        self._running[task_id] = task

        return task_id

    def submit_sync(
        self,
        func: Callable[..., Any],
        **kwargs
    ) -> str:
        """
        Submit a synchronous function for background execution.

        Args:
            func: Sync function to execute
            **kwargs: Arguments to pass to the function

        Returns:
            Task ID for tracking
        """
        async def wrapper(**kw):
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**kw))

        return asyncio.create_task(self.submit(wrapper, **kwargs))

    async def _execute(
        self,
        task_id: str,
        func: Callable[..., Awaitable[Any]],
        kwargs: dict
    ):
        """Execute a task with proper status tracking"""
        async with self._semaphore:
            async with self._lock:
                if task_id not in self.tasks:
                    return
                self.tasks[task_id].status = TaskStatus.RUNNING
                self.tasks[task_id].started_at = time.time()

            try:
                result = await func(**kwargs)

                async with self._lock:
                    self.tasks[task_id].result = result
                    self.tasks[task_id].status = TaskStatus.COMPLETED
                    self.tasks[task_id].progress = 1.0
                    self.tasks[task_id].completed_at = time.time()

            except asyncio.CancelledError:
                async with self._lock:
                    self.tasks[task_id].status = TaskStatus.CANCELLED
                    self.tasks[task_id].completed_at = time.time()
                raise

            except Exception as e:
                async with self._lock:
                    self.tasks[task_id].status = TaskStatus.FAILED
                    self.tasks[task_id].error = str(e)
                    self.tasks[task_id].completed_at = time.time()

            finally:
                self._running.pop(task_id, None)

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID"""
        return self.tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        async with self._lock:
            if task_id not in self.tasks:
                return False

            task_info = self.tasks[task_id]
            if task_info.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False

            # Cancel running asyncio task
            if task_id in self._running:
                self._running[task_id].cancel()

            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = time.time()
            return True

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> list:
        """List tasks, optionally filtered by status"""
        tasks = []
        for task_info in self.tasks.values():
            if status is None or task_info.status == status:
                tasks.append(task_info)
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)[:limit]

    async def cleanup(self):
        """Remove old completed/failed tasks"""
        now = time.time()
        async with self._lock:
            to_remove = [
                task_id for task_id, task in self.tasks.items()
                if task.completed_at and (now - task.completed_at) > self.max_age
            ]
            for task_id in to_remove:
                del self.tasks[task_id]

    async def get_stats(self) -> dict:
        """Get task queue statistics"""
        counts = {status: 0 for status in TaskStatus}
        for task in self.tasks.values():
            counts[task.status] += 1

        return {
            "total_tasks": len(self.tasks),
            "running_tasks": len(self._running),
            "max_concurrent": self.max_concurrent,
            "counts": {s.value: c for s, c in counts.items()},
        }


class ProgressTracker:
    """Helper for tracking task progress"""

    def __init__(self, task_manager: TaskManager, task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id

    async def update(self, progress: float, message: str = None):
        """Update task progress (0.0 to 1.0)"""
        task = await self.task_manager.get_task(self.task_id)
        if task:
            task.progress = max(0.0, min(1.0, progress))
            if message:
                task.metadata["message"] = message

    async def complete(self, result: Any = None):
        """Mark task as complete with result"""
        task = await self.task_manager.get_task(self.task_id)
        if task:
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = time.time()

    async def fail(self, error: str):
        """Mark task as failed with error message"""
        task = await self.task_manager.get_task(self.task_id)
        if task:
            task.error = error
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
