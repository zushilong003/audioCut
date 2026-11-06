"""src.config 包 — 集中放置项目配置和 .env."""
from __future__ import annotations

# 让 `import config` 可以直接访问 config/config.py 里的所有常量
from .config import *  # noqa: F401,F403
from . import config as _config  # noqa: F401
