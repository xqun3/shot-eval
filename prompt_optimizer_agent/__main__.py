"""CLI entry point: ``python -m shot_eval.prompt_optimizer_agent``

Normal mode runs via ADK Runner + InMemorySessionService.
--dry-run mode runs the deterministic core directly without ADK/network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m shot_eval.prompt_optimizer_agent",
        description="Post-evaluation prompt optimization via ADK agent or deterministic core.",
    )
    p.add_argument("--run", default="", metavar="DIR",
                   help="Run directory containing video_prompts.json")
    p.add_argument("--eval", required=True, metavar="PATH",
                   help="Evaluation JSON with optimization.scenes[]")
    p.add_argument("--out", default="", metavar="DIR",
                   help="Output directory (default: run/optimizations/<eval-stem>/)")
    p.add_argument("--min-confidence", type=float, default=0.8,
                   help="Minimum patch confidence for auto-apply (default 0.8)")
    p.add_argument("--allow-single-run", action="store_true",
                   help="Allow auto-apply even with repeats < 2")
    p.add_argument("--agent-model", default="", metavar="MODEL",
                   help="Override ADK agent model (default: SHOT_OPTIMIZER_AGENT_MODEL or gemini-3.1-pro-preview)")
    p.add_argument("--project", default="", metavar="ID",
                   help="GCP project for Vertex AI")
    p.add_argument("--location", default="", metavar="LOC",
                   help="Vertex AI location")
    p.add_argument("--dry-run", action="store_true",
                   help="Deterministic preview only — no ADK model call, writes nothing")
    return p


def _resolve_vp_path(args: argparse.Namespace) -> str:
    """Resolve video_prompts.json from --run or --eval sibling."""
    if args.run:
        run_dir = Path(args.run).expanduser().resolve()
        vp = run_dir / "video_prompts.json"
        if vp.exists():
            return str(vp)
        if run_dir.suffix == ".json" and run_dir.exists():
            return str(run_dir)

    # Try to find video_prompts.json near the eval file
    eval_path = Path(args.eval).expanduser().resolve()
    # Walk up from eval dir looking for video_prompts.json
    for parent in [eval_path.parent, eval_path.parent.parent, eval_path.parent.parent.parent]:
        vp = parent / "video_prompts.json"
        if vp.exists():
            return str(vp)

    return ""


async def _run_adk(
    video_prompts_path: str,
    eval_json_path: str,
    output_dir: str,
    min_confidence: float,
    allow_single_run: bool,
    model: str,
) -> dict:
    """Run via ADK Runner + InMemorySessionService."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from shot_eval.prompt_optimizer_agent.agent import root_agent

    # Override model if requested
    if model:
        root_agent.model = model

    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="prompt_optimizer", session_service=session_service)

    session = await session_service.create_session(
        app_name="prompt_optimizer", user_id="cli_user"
    )

    # Build user message
    msg_parts = [
        f"Optimize prompts using:",
        f"  video_prompts: {video_prompts_path}",
        f"  evaluation: {eval_json_path}",
    ]
    if output_dir:
        msg_parts.append(f"  output_dir: {output_dir}")
    msg_parts.append(f"  min_confidence: {min_confidence}")
    if allow_single_run:
        msg_parts.append(f"  allow_single_run: true")

    user_content = types.Content(
        role="user",
        parts=[types.Part(text="\n".join(msg_parts))],
    )

    # Collect events and enforce the one-tool-call contract in code.
    final_text = ""
    tool_call_count = 0
    tool_result: dict | None = None
    async for event in runner.run_async(
        user_id="cli_user",
        session_id=session.id,
        new_message=user_content,
    ):
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            function_call = getattr(part, "function_call", None)
            if function_call and function_call.name == "_optimize_video_prompts_tool":
                tool_call_count += 1
            function_response = getattr(part, "function_response", None)
            if function_response and function_response.name == "_optimize_video_prompts_tool":
                response = function_response.response
                tool_result = dict(response) if isinstance(response, dict) else {"response": response}
            text = getattr(part, "text", None)
            if text:
                final_text += text

    if tool_call_count != 1:
        raise RuntimeError(
            f"ADK contract violation: expected exactly one optimizer tool call, got {tool_call_count}"
        )
    if tool_result is None:
        raise RuntimeError("ADK optimizer tool returned no function response")
    if tool_result.get("status") != "ok":
        raise RuntimeError(f"optimizer tool failed: {tool_result.get('error', tool_result)}")
    return {
        "agent_response": final_text,
        "tool_result": tool_result,
        "tool_call_count": tool_call_count,
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project
    if args.location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    if args.project or os.environ.get("GOOGLE_CLOUD_PROJECT"):
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    if args.agent_model:
        os.environ["SHOT_OPTIMIZER_AGENT_MODEL"] = args.agent_model

    eval_path = str(Path(args.eval).expanduser().resolve())
    vp_path = _resolve_vp_path(args)

    if not vp_path:
        print("Error: cannot find video_prompts.json. Use --run to specify.", file=sys.stderr)
        return 2

    output_dir = args.out

    if args.dry_run:
        # Deterministic mode — no ADK, no network
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        print(f"[dry-run] video_prompts: {vp_path}")
        print(f"[dry-run] evaluation: {eval_path}")
        print(f"[dry-run] min_confidence: {args.min_confidence}")
        print(f"[dry-run] allow_single_run: {args.allow_single_run}")
        print()

        result = optimize_video_prompts(
            video_prompts_path=vp_path,
            eval_json_path=eval_path,
            output_dir=output_dir,
            min_confidence=args.min_confidence,
            allow_single_run=args.allow_single_run,
            dry_run=True,
        )

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            return 1

        summary = result.get("summary", {})
        print("=== Dry-Run Plan ===")
        print(f"  Total scenes: {summary.get('total_scenes', 0)}")
        print(f"  Would apply: {summary.get('applied', 0)}")
        print(f"  Review-only: {summary.get('review_only', 0)}")
        print(f"  Skipped: {summary.get('skipped', 0)}")
        print(f"  Plan-only mode: {summary.get('plan_only', False)}")
        print(f"  Lint warnings: {summary.get('lint_warning_count', 0)}")
        print(f"  Global candidates: {summary.get('global_candidates', 0)}")
        print()

        # Print regeneration plan summary
        regen = result.get("regeneration_plan", {})
        entries = regen.get("entries", [])
        if entries:
            print("=== Regeneration Plan ===")
            for e in entries:
                print(f"  {e.get('unit_id', '?'):10s} [{e.get('category', '?')}]"
                      f"  priority={e.get('priority', '?')}")
            print()

        print("[dry-run] No files written.")
        return 0

    # Normal mode: ADK Runner
    print(f"video_prompts: {vp_path}")
    print(f"evaluation: {eval_path}")
    print(f"Running via ADK Runner...")
    print()

    try:
        result = asyncio.run(_run_adk(
            video_prompts_path=vp_path,
            eval_json_path=eval_path,
            output_dir=output_dir,
            min_confidence=args.min_confidence,
            allow_single_run=args.allow_single_run,
            model=args.agent_model,
        ))
        print("=== Agent Response ===")
        print(result.get("agent_response", "(no response)"))
        tool_result = result.get("tool_result") or {}
        print()
        print(f"ADK tool calls: {result.get('tool_call_count', 0)}")
        print(f"Output directory: {tool_result.get('output_dir', '')}")
        for path in tool_result.get("files_written") or []:
            print(f"  {path}")
        return 0
    except Exception as exc:
        print(f"ADK Runner failed: {exc}", file=sys.stderr)
        print(
            "No deterministic fallback was executed. Fix ADK/Vertex credentials, "
            "or use --dry-run for a no-write deterministic preview.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
