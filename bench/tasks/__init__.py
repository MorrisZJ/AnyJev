"""Task registry. Each task yields (state, label_index) and owns its Question."""
from __future__ import annotations

import bench.tasks.banking  # noqa: F401,E402
import bench.tasks.newsgroups  # noqa: F401,E402
import bench.tasks.prompt_injection  # noqa: F401,E402
from bench.tasks.base import TASKS, Task, get_task  # noqa: F401
