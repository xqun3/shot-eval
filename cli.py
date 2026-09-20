"""命令行入口。开关与上游 `scenebench` 对齐，能对上的都保持同名同义。"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from shot_eval import adapters, bench, judge, prompts
from shot_eval.generation import resolve_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shot_eval",
        description="逐 shot 评估：让模型看每一段视频，判它和分镜说好的是不是一回事。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例
  # 先拿 2 段、1 轮、不评渲染模式跑通（4 次模型调用）
  shot_eval runs/carbs_shot --limit 2 --repeats 1 --render off

  # 正式一轮：21 段 × 3 轮 × 2 次请求 = 126 次调用
  shot_eval runs/carbs_shot --repeats 3 --out runs/carbs_shot/eval

  # 换个口径：用口播原文而不是 visual_beat 当对齐参照
  shot_eval runs/carbs_shot --beat script

注意
  * `--repeats 1` 只能看现象。temperature=0 也不是确定性的，
    要下结论至少 3 轮，看中位数和轮间极差。
  * 换 `--judge-model` 就是换尺子，跨轮对比前先确认两轮用的是同一个。
""",
    )
    parser.add_argument(
        "run",
        help="run 目录（含 video_prompts.json 与 videos/），或直接给 video_prompts.json",
    )
    parser.add_argument("--beat", default="visual_beat", choices=judge.BEAT_SOURCES,
                        help="beat_alignment 拿什么当参照物（默认 visual_beat）")
    parser.add_argument("--style", default="style_lock", choices=judge.STYLE_SOURCES,
                        help="style_adherence 拿什么当参照物；off = 本轮不评这一维度")
    parser.add_argument("--render", default="blind", choices=judge.RENDER_SOURCES,
                        help="渲染模式盲判；off 能省掉一半调用")
    parser.add_argument("--design-source", default="generated", choices=adapters.DESIGN_SOURCES,
                        help="design_fidelity 对照哪份 scene_design："
                             "generated=送进视频模型的那份（默认）、plan=分镜方案里的那份")
    parser.add_argument("--judge-model", default="", choices=judge.JUDGE_MODELS,
                        help="评审模型。空 = 默认 gemini-3.7-flash（轮间噪声最小）")
    parser.add_argument("--repeats", type=int, default=1, help="跑几轮（默认 1；下结论要 ≥3）")
    parser.add_argument("--concurrency", type=int, default=judge.MAX_CONCURRENCY,
                        help=f"并发片段数（默认 {judge.MAX_CONCURRENCY}）")
    parser.add_argument("--scenes", nargs="*", default=[], metavar="ID",
                        help="只评这些片段，如 A01-01 A03-02")
    parser.add_argument("--limit", type=int, default=0, help="只评前 N 段")
    parser.add_argument("--no-deductions", action="store_true", help="不打印扣分项明细")
    parser.add_argument("--out", default="", metavar="DIR", help="把完整结果 JSON 落到这个目录")
    parser.add_argument("--no-html", action="store_true",
                        help="只落 JSON，不生成可视化 HTML 报告")
    parser.add_argument("--expect", action="append", default=[], metavar="片段:维度",
                        type=bench.parse_expect,
                        help="判据回归：这些片段的这些维度必须被扣到，可重复")
    parser.add_argument("--min-hit-rate", type=float, default=0.5,
                        help="--expect 的命中率门槛（默认 0.5）")
    parser.add_argument("--project", default="", metavar="ID",
                        help="GCP project；等价于 GOOGLE_CLOUD_PROJECT，命令行优先")
    parser.add_argument("--location", default="", metavar="LOC",
                        help="Vertex location；等价于 GOOGLE_CLOUD_LOCATION（默认 global）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印装载结果和第一条评审 prompt，不发任何请求")

    # --- optimization flags ---
    parser.add_argument("--optimize-prompt", action="store_true",
                        help="评审完成后进行链路归因和 prompt 优化分析（每片段 +1 调用）")
    parser.add_argument("--optimization-model", default="", metavar="MODEL",
                        help="优化分析用的模型（默认与 --judge-model 相同）")
    return parser


async def _run(args: argparse.Namespace) -> int:
    tasks = bench.pick(
        adapters.tasks_from_run(args.run, design_source=args.design_source),
        args.scenes,
        args.limit or None,
    )
    if not tasks:
        print("没有匹配到任何片段", file=sys.stderr)
        return 2

    print(f"判据 {prompts.version_tag()}（{prompts.path()}）")
    print(f"{len(tasks)} 个片段 · design={args.design_source} "
          f"beat={args.beat} style={args.style} render={args.render}")
    for task in tasks:
        item = task.item
        print(f"  {item.scene_id:10s} {item.duration:5.1f}s  {item.render_mode or '?':9s} "
              f"{task.clip.name}")

    if args.dry_run:
        print()
        # --- generation metadata ---
        gen = resolve_generation(args.run)
        print("生成模型元数据 (generation metadata):")
        for k, v in gen.items():
            print(f"  {k}: {v}")
        print()
        print("=" * 72)
        print(f"第一条评审 prompt（{tasks[0].item.scene_id}）")
        print("=" * 72)
        print(judge.scene_prompt(tasks[0].style_lock, tasks[0].item, args.beat, args.style))
        if args.optimize_prompt and tasks[0].item.provenance:
            from shot_eval.optimize import build_optimization_prompt
            print()
            print("=" * 72)
            print("优化 prompt 示例")
            print("=" * 72)
            print(build_optimization_prompt(tasks[0].item.provenance, [[{"dimension": "example", "severity": "major", "evidence": "..."}]]))
        return 0

    calls = len(tasks) * args.repeats * (1 if args.render == "off" else 2)
    if args.optimize_prompt:
        calls += len(tasks)  # +1/scene for optimization
    print(f"本轮将发出约 {calls} 次模型调用")
    print()

    result = await bench.run_batch(
        tasks,
        repeats=args.repeats,
        beat_source=args.beat,
        style_source=args.style,
        render_source=args.render,
        judge_model=args.judge_model or None,
        concurrency=args.concurrency,
    )
    result["design_source"] = args.design_source
    result["generation"] = resolve_generation(args.run)

    # --- Optimization pass ---
    if args.optimize_prompt:
        from shot_eval.optimize import optimize_scenes
        requested_opt_model = args.optimization_model or args.judge_model or None
        resolved_opt_model = judge.judge_model(requested_opt_model)
        result["config"]["prompt_optimization"] = True
        result["config"]["optimization_model"] = resolved_opt_model
        print(f"  优化分析 · {len(tasks)} 个片段 · {resolved_opt_model} …", flush=True)
        try:
            opt_results = await optimize_scenes(
                result["scenes"], tasks,
                model=args.judge_model or None,
                optimization_model=resolved_opt_model,
            )
            result["optimization"] = {
                "model": resolved_opt_model,
                "config": {"enabled": True},
                "scenes": opt_results,
                "errors": [r.get("error") for r in opt_results if r.get("error")],
            }
        except Exception as exc:
            result["optimization"] = {
                "model": resolved_opt_model,
                "config": {"enabled": True},
                "scenes": [],
                "errors": [str(exc)],
            }
        print(f"    优化分析完成", flush=True)

    bench.print_report(result, show_deductions=not args.no_deductions)

    # Print optimization summary if available
    if args.optimize_prompt and result.get("optimization"):
        _print_optimization_summary(result["optimization"])

    if args.out:
        out_dir = Path(args.out).expanduser()
        saved = bench.save(result, out_dir, name=tasks[0].source)
        print()
        print(f"完整结果（含每一轮的原始判决）→ {saved}")
        if not args.no_html:
            from shot_eval import report
            html = report.write_report(
                result, saved.with_suffix(".html"), title=f"逐 shot 评估 · {tasks[0].source}"
            )
            print(f"可视化报告（视频 + 评分 + 判词）→ {html}")

    return 0 if bench.check_expectations(result, args.expect, args.min_hit_rate) else 1


def _print_optimization_summary(opt: dict) -> None:
    """Print a terminal summary of optimization results."""
    scenes = opt.get("scenes") or []
    if not scenes:
        return
    print()
    print("链路归因 + prompt 优化")
    for s in scenes:
        sid = s.get("scene_id", "?")
        status = s.get("status", "?")
        if status in ("unassessable", "skip"):
            print(f"  {sid}: {status} — {s.get('reason', '')}")
            continue
        if status == "error":
            print(f"  {sid}: ✗ {s.get('error', '')[:60]}")
            continue
        findings = s.get("findings") or []
        valid = s.get("valid_patches") or []
        rejected = s.get("rejected_patches") or []
        actions = s.get("non_prompt_actions") or []
        print(f"  {sid}: {len(findings)} findings, "
              f"{len(valid)} valid patches, "
              f"{len(rejected)} rejected patches, "
              f"{len(actions)} non-prompt actions")
        for f in findings:
            grounded = "✓" if f.get("quote_grounded") else "✗"
            print(f"    [{f.get('stage')}/{f.get('severity')}] {grounded} {f.get('problem', '')[:60]}")
        for p in valid:
            target = p.get("target", "?")
            op = p.get("op", "?")
            anchor = p.get("anchor", "")[:30]
            content = p.get("content", "")[:30]
            print(f"    patch {target}.{op}: {anchor!r} → {content!r}")
        for p in rejected:
            print(f"    ⚠ rejected: {p.get('target')}.{p.get('op')}: {p.get('rejection_reason', '')[:50]}")
        for a in actions:
            print(f"    action: {a.get('action')}: {a.get('reason', '')[:50]}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project
    if args.location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
