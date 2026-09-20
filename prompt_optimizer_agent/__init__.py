"""Optional post-evaluation prompt optimization helpers.

The core evaluator does not require Google ADK.  ``root_agent`` is loaded only
when the optional ``optimizer`` extra is installed, while the deterministic
``optimize_video_prompts`` function remains available for local inspection.
"""

from __future__ import annotations

from typing import Any

from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

__all__ = ["optimize_video_prompts", "root_agent"]


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        try:
            from shot_eval.prompt_optimizer_agent.agent import root_agent
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("google.adk"):
                raise ModuleNotFoundError(
                    "Prompt Optimization Agent requires the optional dependency. "
                    "Install with: pip install 'shot-eval[optimizer]'"
                ) from exc
            raise
        return root_agent
    raise AttributeError(name)
