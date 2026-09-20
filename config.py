"""上游 `core.config` 的替身：这一路只用到三个值，没必要拖进整份配置。

上游 `core/config.py` 有 100 多行（视频模型、分辨率、并发、ROOT 路径…），
逐分镜评审真正读的只有 `vertex_project()` / `vertex_location()` /
`planner_gemini_model()`。把整份配置搬过来等于把 scheme_c 的目录约定
（`ROOT / "bench"`、`ROOT.parent / "CG_Generation"`）也一起继承，
而这个模块要能在任何目录下跑。
"""

from __future__ import annotations

import os

#: 评审默认用它。上游 `.env` 的 planner 模型也是这个值。
#: 逐 shot 评估文档实测：3.7-flash 轮间噪声最小（板块极差中位 0.90），
#: 3.8-flash 更严但噪声是它的 2~3 倍。**换模型就是换尺子**，跨轮对比必须锁死。
DEFAULT_JUDGE_MODEL = "gemini-3.7-flash"


class ConfigError(RuntimeError):
    """缺必需的环境变量。消息要能直接说出该 export 什么。"""


def vertex_project() -> str:
    value = (
        os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.getenv("VERTEX_PROJECT", "").strip()
    )
    if not value:
        raise ConfigError(
            "缺少 GCP project：请设置 GOOGLE_CLOUD_PROJECT 或 VERTEX_PROJECT"
        )
    return value


def vertex_location() -> str:
    return os.getenv("GOOGLE_CLOUD_LOCATION", "").strip() or "global"


def planner_gemini_model() -> str:
    return os.getenv("PLANNER_GEMINI_MODEL", "").strip() or DEFAULT_JUDGE_MODEL
