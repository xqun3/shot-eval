"""ADK-discoverable Prompt Optimization Agent.

Exports ``root_agent`` — a google.adk.agents.Agent (LlmAgent) that wraps
the deterministic ``optimize_video_prompts`` function via FunctionTool.

The agent is instructed to make exactly one tool call with user-provided
paths/policy, then summarize the returned artifacts. It never regenerates
video, bypasses tool safety, or makes extra tool calls.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

# ---------------------------------------------------------------------------
# Tool function — docstring + primitive params → FunctionTool schema
# ---------------------------------------------------------------------------

def _optimize_video_prompts_tool(
    video_prompts_path: str,
    eval_json_path: str,
    output_dir: str = "",
    min_confidence: float = 0.8,
    allow_single_run: bool = False,
    dry_run: bool = False,
) -> dict:
    """Apply grounded prompt optimizations from an evaluation report.

    Reads the source video_prompts.json bundle and a completed evaluation JSON,
    matches optimization patches to bundle entries by scene_id/unit_id, applies
    eligible patches deterministically, rebuilds affected video prompts via the
    repository source-of-truth assembler, and writes outputs to a separate
    directory (never overwriting the source bundle).

    Args:
        video_prompts_path: Absolute path to the source video_prompts.json.
        eval_json_path: Absolute path to a completed evaluation JSON with
            optimization.scenes[].
        output_dir: Output directory. Default: run/optimizations/<eval-stem>/.
        min_confidence: Minimum patch confidence for auto-apply (default 0.8).
        allow_single_run: Allow auto-apply with repeats<2 (default False).
        dry_run: If True, compute the plan but write nothing to disk.

    Returns:
        A dict with summary, scene_results, output paths, and any errors.
    """
    result = optimize_video_prompts(
        video_prompts_path=video_prompts_path,
        eval_json_path=eval_json_path,
        output_dir=output_dir,
        min_confidence=min_confidence,
        allow_single_run=allow_single_run,
        dry_run=dry_run,
    )
    if result.get("error"):
        return {"status": "error", "error": result["error"]}
    plan = result.get("regeneration_plan") or {}
    categories: dict[str, int] = {}
    for entry in plan.get("entries") or []:
        category = str(entry.get("category") or "unknown")
        categories[category] = categories.get(category, 0) + 1
    return {
        "status": "ok",
        "summary": result.get("summary") or {},
        "output_dir": result.get("output_dir"),
        "files_written": result.get("files_written") or [],
        "regeneration_categories": categories,
        "dry_run": bool(result.get("dry_run")),
        "note": "No video was generated; old videos are invalidated for changed prompts.",
    }


optimize_tool = FunctionTool(func=_optimize_video_prompts_tool)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.environ.get("SHOT_OPTIMIZER_AGENT_MODEL", "gemini-3.1-pro-preview")

_INSTRUCTION = """\
You are the Prompt Optimization Agent. Your sole task is to apply grounded,
deterministic prompt optimizations from a post-evaluation analysis.

WORKFLOW:
1. The user provides two paths: a video_prompts.json bundle and an evaluation
   JSON with optimization data. They may also provide policy overrides.
2. You MUST call the optimize_video_prompts tool exactly ONCE with these paths
   and any policy parameters the user specified. Do not invent paths or values.
3. After the tool returns, summarize the results:
   - How many scenes had patches applied vs review-only vs skipped.
   - Any lint warnings or global rule candidates.
   - The output directory and files written.
   - Remind the user that old videos are invalidated and regeneration is needed.

SAFETY RULES:
- NEVER regenerate video or trigger any paid generation.
- NEVER bypass the tool — all optimization logic is in the tool function.
- NEVER modify source files — outputs go to a separate directory.
- NEVER auto-edit based on linter warnings — they are advisory only.
- NEVER trust rejected_patches — only valid_patches are considered.
- Do NOT evaluate duration or inspect audio.
- Do NOT mutate clip_prompt_system.md.
- Make exactly ONE tool call per invocation. No more, no less.
"""

root_agent = Agent(
    name="prompt_optimizer_agent",
    model=_DEFAULT_MODEL,
    description="Post-evaluation prompt optimization: applies grounded patches to video prompts.",
    instruction=_INSTRUCTION,
    tools=[optimize_tool],
)
