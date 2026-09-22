"""Task registry.

A `Task` yields (state, label_index) and owns one Question shared by every
item. An `MCQTask` yields (state, question, label_index) for sets where every
item carries its own question and options.
"""
from __future__ import annotations

import bench.tasks.ai2d  # noqa: F401,E402
import bench.tasks.banking  # noqa: F401,E402
import bench.tasks.newsgroups  # noqa: F401,E402
import bench.tasks.pets  # noqa: F401,E402
import bench.tasks.pope  # noqa: F401,E402
import bench.tasks.prompt_injection  # noqa: F401,E402
from bench.tasks.base import (  # noqa: F401
    MCQ_TASKS,
    TASKS,
    MCQTask,
    Task,
    get_mcq_task,
    get_task,
)
