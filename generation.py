"""Generation-model metadata: normalize, resolve, infer.

The *generation* metadata describes how the video was *produced* (video model,
planner LLM, prompt LLM, granularity …).  This is orthogonal to the *judge*
metadata that describes how the video was *evaluated* (judge model, prompt
version, beat source …).

Only stdlib — this file must work with the system ``python3``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# -- normalisation ----------------------------------------------------------

#: The canonical keys that a ``generation`` dict must contain.
GENERATION_KEYS = (
    "video_provider",
    "video_model",
    "prompt_model",
    "planner_model",
    "granularity",
    "plan_kind",
)


def normalize_meta(meta: dict[str, Any] | None) -> dict[str, str]:
    """Turn the ``meta`` block of ``video_prompts.json`` into a canonical
    *generation* dict with exactly :data:`GENERATION_KEYS`.

    Unknown / missing values are represented as ``"unknown"``; empty strings
    are normalised to ``"unknown"`` as well.
    """
    if not meta or not isinstance(meta, dict):
        return {k: "unknown" for k in GENERATION_KEYS}

    def _s(val: Any) -> str:
        v = str(val).strip() if val is not None else ""
        return v if v else "unknown"

    return {
        "video_provider": _s(meta.get("video_provider") or meta.get("provider")),
        "video_model": _s(meta.get("video_model")),
        "prompt_model": _s(meta.get("llm_model")),
        "planner_model": _s(meta.get("planner_model")),
        "granularity": _s(meta.get("granularity")),
        "plan_kind": _s(meta.get("plan_kind")),
    }


# -- bundle loading ---------------------------------------------------------

def load_bundle(path: str | Path) -> dict[str, Any] | None:
    """Load ``video_prompts.json`` from a run directory or file path.

    Returns the parsed dict, or *None* on any I/O or parse error so that
    callers never crash for old / missing files.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "video_prompts.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_generation(run_path: str | Path) -> dict[str, str]:
    """Return the canonical generation dict for a run directory or file.

    Falls back to all-``"unknown"`` if the bundle is missing or corrupt.
    """
    bundle = load_bundle(run_path)
    if bundle is None:
        return normalize_meta(None)
    return normalize_meta(bundle.get("meta"))


# -- inference from task clip path ------------------------------------------

def infer_run_root_from_clip(clip_path: str) -> Path | None:
    """Guess the run root from a task's clip path.

    Evaluation tasks record ``clip`` as an absolute path like
    ``/…/runs/淀粉/videos/A01-01.mp4``.  The run root is the *parent of
    ``videos/``*.  Returns *None* when the heuristic does not apply.
    """
    if not clip_path:
        return None
    p = Path(clip_path)
    if p.parent.name == "videos":
        return p.parent.parent
    return None


def infer_generation(result: dict[str, Any], run_root: str | Path | None = None) -> dict[str, str]:
    """Best-effort generation metadata for an evaluation result.

    Priority:
    1. ``result["generation"]`` if it already exists (explicit wins).
    2. ``run_root / video_prompts.json`` if *run_root* is provided.
    3. Inferred from the first task's clip path.
    4. All ``"unknown"`` if nothing works.
    """
    # 1 — explicit
    explicit = result.get("generation")
    if isinstance(explicit, dict) and explicit:
        # Normalise in case the stored dict is partial or stale.
        merged = {k: "unknown" for k in GENERATION_KEYS}
        for k in GENERATION_KEYS:
            v = str(explicit.get(k, "")).strip()
            if v:
                merged[k] = v
        return merged

    # 2 — from run_root
    if run_root:
        gen = resolve_generation(run_root)
        if gen.get("video_model", "unknown") != "unknown":
            return gen

    # 3 — infer from first task clip
    tasks = result.get("tasks") or []
    if tasks:
        clip = (tasks[0] if isinstance(tasks[0], dict) else {}).get("clip", "")
        root = infer_run_root_from_clip(clip)
        if root:
            gen = resolve_generation(root)
            if gen.get("video_model", "unknown") != "unknown":
                return gen

    # 4 — give up
    return {k: "unknown" for k in GENERATION_KEYS}


def video_model_label(gen: dict[str, str]) -> str:
    """Human-friendly label for the video model.

    Shows ``video_model``; falls back to ``video_provider``, then ``"未知"``.
    """
    model = gen.get("video_model", "unknown")
    if model and model != "unknown":
        return model
    provider = gen.get("video_provider", "unknown")
    if provider and provider != "unknown":
        return provider
    return "未知"
