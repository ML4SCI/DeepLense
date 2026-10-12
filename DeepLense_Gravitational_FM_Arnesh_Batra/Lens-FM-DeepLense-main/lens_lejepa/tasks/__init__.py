"""Task registry: ``build_task("classification", config)``."""

from __future__ import annotations

from ..config import FinetuneConfig
from .base import Task
from .classification import ClassificationTask
from .regression import RegressionTask
from .super_resolution import SuperResolutionTask

TASKS: dict[str, type[Task]] = {}


def register(task_cls: type[Task]) -> type[Task]:
    """Register a :class:`Task` subclass under its ``name`` (usable as a decorator)."""
    if not task_cls.name:
        raise ValueError("A task needs a non-empty name.")
    if task_cls.name in TASKS:
        raise ValueError(f"Task {task_cls.name!r} is already registered.")
    TASKS[task_cls.name] = task_cls
    return task_cls


for _cls in (ClassificationTask, RegressionTask, SuperResolutionTask):
    register(_cls)


def build_task(config: FinetuneConfig) -> Task:
    if config.task not in TASKS:
        raise ValueError(f"Unknown task {config.task!r}; available: {sorted(TASKS)}.")
    return TASKS[config.task](config)


__all__ = ["TASKS", "ClassificationTask", "RegressionTask", "SuperResolutionTask", "Task", "build_task", "register"]
