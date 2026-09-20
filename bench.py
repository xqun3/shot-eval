"""跑批、聚合、出报告、判据回归。

本模块只保留与数据来源无关的批处理、聚合、报告和判据回归能力：
`SceneTask` / `run_batch` / `_aggregate` / `print_report` / `check_expectations`。
上游的三个装载器（`tasks_from_case` / `tasks_from_project` / `tasks_from_dir`）
全部丢掉——它们认的是 scheme_c 的 `state.json` 布局，我们的数据不长那样，
装载这件事交给 `adapters.py`。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shot_eval import judge, prompts
from shot_eval.models import SCENE_DIMENSIONS, Provenance, SceneSlice
from shot_eval.scoring import score_result

# --- 任务 ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SceneTask:
    """一个待评审的分镜片段：素材 + 它「本来该演什么」。

    `style_lock` 单独带着而不是指回某个用例对象：一批里可以混不同的片子，
    而 `judge.judge_scene` 这一路真正需要的也只有这一个字段。
    """

    source: str
    #: 片段是怎么来的：`shot` = 生成时直接落盘的单段 mp4；`slice` = 从成片切的
    origin: str
    item: SceneSlice
    clip: Path
    style_lock: str
    provenance: Provenance | None = None

    @property
    def key(self) -> str:
        return f"{self.source}/{self.item.scene_id}"


def pick(tasks: Sequence[SceneTask], only: Sequence[str], limit: int | None) -> list[SceneTask]:
    """按 scene_id 过滤 + 截断。`only` 同时匹配 `key` 和裸 `scene_id`。"""
    picked = list(tasks)
    if only:
        wanted = set(only)
        picked = [t for t in picked if t.item.scene_id in wanted or t.key in wanted]
    return picked[:limit] if limit else picked


# --- 跑批 ---------------------------------------------------------------------


def _spread(values: list[float]) -> float:
    """极差 —— 这就是这套测量在该维度上的噪声下限。"""
    return round(max(values) - min(values), 2) if len(values) > 1 else 0.0


def _stat(values: list[float]) -> dict[str, float]:
    return {
        "median": round(statistics.median(values), 2),
        "spread": _spread(values),
        "values": [round(v, 2) for v in values],
    }


async def run_batch(
    tasks: Sequence[SceneTask],
    *,
    repeats: int = 1,
    beat_source: str = "visual_beat",
    style_source: str = "style_lock",
    render_source: str = "blind",
    judge_model: str | None = None,
    concurrency: int | None = None,
    on_round=None,
) -> dict[str, Any]:
    """跑 `repeats` 轮，每轮评审全部片段，按片段聚合。

    重复跑的意义：`temperature=0` 也不是确定性的，单轮的「这一段扣了 1.5 分」
    可能下一轮就不扣了。跑一轮只能看现象，跑三轮才知道那个现象是不是稳定的。
    """
    started = time.time()
    scenes = [(t.style_lock, t.item, t.clip) for t in tasks]
    result: dict[str, Any] = {
        "started_at": started,
        "config": {
            "beat_source": beat_source,
            "style_source": style_source,
            "render_source": render_source,
        },
        # 判据是口径的一部分：改了 prompts.yaml 再跑，两批数据就不可比
        "prompt_version_tag": prompts.version_tag(),
        "prompts_path": str(prompts.path()),
        "judge_model": judge.judge_model(judge_model),
        "thinking_level": judge.thinking_level() or "（模型默认）",
        "repeats": repeats,
        "tasks": [
            {"key": t.key, "source": t.source, "origin": t.origin,
             "scene_id": t.item.scene_id, "clip": str(t.clip),
             "duration": t.item.duration,
             "render_declared": t.item.render_mode}
            for t in tasks
        ],
    }

    rounds: list[list[dict[str, Any]]] = []
    for i in range(repeats):
        print(f"  第 {i + 1}/{repeats} 轮 · {len(tasks)} 个片段 …", flush=True)
        done = {"n": 0}

        def tick(_verdict, total=len(tasks), state=done):
            state["n"] += 1
            print(f"    {state['n']}/{total}", end="\r", flush=True)

        verdicts = await judge.judge_scenes(
            scenes,
            on_done=tick,
            beat_source=beat_source,
            style_source=style_source,
            render_source=render_source,
            model=judge_model,
            concurrency=concurrency,
        )
        print(f"    {len(tasks)}/{len(tasks)} 完成", flush=True)
        rounds.append(verdicts)
        if on_round:
            on_round(i, verdicts)

    result["scenes"] = [_aggregate(t, [r[n] for r in rounds]) for n, t in enumerate(tasks)]
    result["elapsed"] = round(time.time() - started, 1)
    return result


def _aggregate(task: SceneTask, verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [v for v in verdicts if "error" not in v]
    entry: dict[str, Any] = {
        "key": task.key,
        "source": task.source,
        "origin": task.origin,
        "scene_id": task.item.scene_id,
        "render_declared": task.item.render_mode,
        "context": task.item.as_context(),
        "rounds": verdicts,
        "errors": [str(v["error"]) for v in verdicts if "error" in v],
        "dims": {},
        # 「这一维度有没有被扣到」的命中轮次 —— 判据回归看的就是它，
        # 分数看不出区别：4.0 可能是一条 major，也可能是三条 minor。
        "hits": {},
    }
    # 「参照物缺失判 0」在这张表里必须和「真扣到 0 分」分开显示：参照物缺失时
    # 全批判 0，照直印出来就是一片 0.0，看着像片子崩了。
    entry["unreferenced"] = sorted(
        {d for v in ok for d in v.get("unreferenced") or []}
    ) if ok else []
    for dim in SCENE_DIMENSIONS:
        values = [float(v[dim]) for v in ok if v.get(dim) is not None]
        if values:
            entry["dims"][dim] = _stat(values)
        entry["hits"][dim] = sum(
            1 for v in ok if any(d.get("dimension") == dim for d in v.get("deductions") or [])
        )
    # 渲染模式不是打分维度，是一个是/否 —— 单独记命中率
    matches = [v["render_match"] for v in ok if v.get("render_match") is not None]
    entry["render_match"] = {
        "judged": len(matches),
        "matched": sum(1 for m in matches if m),
        "observed": [v.get("render_verdict") for v in ok if v.get("render_verdict")],
    } if matches else None
    entry["judged"] = len(ok)
    entry["deduction_count"] = [len(v.get("deductions") or []) for v in ok]
    return entry


# --- 输出 ---------------------------------------------------------------------


def _w(text: str) -> int:
    """终端显示宽度。中文占两格，按 len() 对齐会错位。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _w(text))


DIM_LABEL = {
    "beat_alignment": "画面还原",
    "design_fidelity": "分镜还原",
    "style_adherence": "风格",
    "science_accuracy": "科学",
    "clarity": "清晰",
    "instructional_value": "教育",
}


def print_report(result: dict[str, Any], show_deductions: bool = True) -> None:
    scenes = result["scenes"]
    conf = result["config"]
    print()
    print(f"评审模型 {result['judge_model']} · thinking={result['thinking_level']}")
    print(f"判据 {result['prompt_version_tag']} · "
          f"beat={conf['beat_source']} style={conf['style_source']} render={conf['render_source']}")
    print(f"{len(scenes)} 个片段 × {result['repeats']} 轮 · 用时 {result['elapsed']}s")

    # 总分诊断
    ov = score_result(result)
    if ov["score"] is None:
        print(f"Overall — · coverage {ov['covered_weight']}/{ov['max_weight']}（无有效结论）")
    else:
        print(f"Overall {ov['score']} ({ov['grade']}) · "
              f"coverage {ov['covered_weight']}/{ov['max_weight']}"
              + (f" · {ov['severity']['critical']} critical" if ov["severity"]["critical"] else ""))

    dims = [d for d in SCENE_DIMENSIONS if any(d in s["dims"] for s in scenes)]
    key_w = max([_w(s["key"]) for s in scenes] + [8])

    print()
    print("逐片段（中位分；带 ± 的是轮间极差）")
    print("  " + _pad("片段", key_w) + "  " + "  ".join(_pad(DIM_LABEL[d], 10) for d in dims) + "  扣分项")
    for scene in scenes:
        cells = []
        for dim in dims:
            stat = scene["dims"].get(dim)
            if not stat:
                cells.append(_pad("—", 10))
                continue
            if dim in scene["unreferenced"]:
                cells.append(_pad("无参照", 10))
                continue
            text = f"{stat['median']:.1f}"
            if stat["spread"]:
                text += f"±{stat['spread']:.1f}"
            cells.append(_pad(text, 10))
        counts = scene["deduction_count"]
        tail = "/".join(str(c) for c in counts) if counts else "—"
        if scene["errors"]:
            tail += f"  ✗ {scene['errors'][0][:40]}"
        print("  " + _pad(scene["key"], key_w) + "  " + "  ".join(cells) + "  " + tail)

    print()
    print("按维度汇总")
    for dim in dims:
        # 无参照的片段不进统计：它们的 0 分是「没东西可比」，混进中位数会把
        # 这一维度的整体水平拉成一个假数字。
        rated = [s for s in scenes if dim in s["dims"] and dim not in s["unreferenced"]]
        skipped = sum(1 for s in scenes if dim in s["unreferenced"])
        line = f"  {_pad(DIM_LABEL[dim], 10)}"
        if not rated:
            print(line + f" 全部无参照（{skipped} 段），本维度无结论")
            continue
        values = [s["dims"][dim]["median"] for s in rated]
        spreads = [s["dims"][dim]["spread"] for s in rated]
        hit = sum(s["hits"].get(dim, 0) for s in rated)
        total = sum(s["judged"] for s in rated)
        worst = min(rated, key=lambda s: s["dims"][dim]["median"])
        print(line + f" 均值 {statistics.mean(values):.2f}"
              f" · 中位 {statistics.median(values):.2f}"
              f" · 最低 {worst['dims'][dim]['median']:.1f}（{worst['key']}）"
              f" · 扣到 {hit}/{total} 段次"
              + (f" · 轮间极差中位 {statistics.median(spreads):.2f}" if result["repeats"] > 1 else "")
              + (f" · {skipped} 段无参照未计入" if skipped else ""))

    rendered = [s for s in scenes if s.get("render_match")]
    if rendered:
        judged = sum(s["render_match"]["judged"] for s in rendered)
        matched = sum(s["render_match"]["matched"] for s in rendered)
        bad = [s["key"] for s in rendered if s["render_match"]["matched"] < s["render_match"]["judged"]]
        print(f"  {_pad('渲染模式', 10)} 盲判吻合 {matched}/{judged} 段次"
              + (f" · 不吻合：{', '.join(bad[:6])}" if bad else ""))

    print()
    print("  「扣到 x/y 段次」= 这一维度产出过扣分项的片段轮次数。分数看不出区别的地方它能：")
    print("  一个维度全批一条都扣不到，说明那条判据是哑的，不是片子完美。")
    print("  「无参照」= 分镜里根本没有可比对的东西，计分时按 0 记，")
    print("  但那不是画面的问题，所以不进上面的中位数。")

    if not show_deductions:
        return
    print()
    print("扣分项明细")
    repeats = result["repeats"]
    empty = True
    for scene in scenes:
        rounds = [v for v in scene["rounds"] if "error" not in v]
        if not any(v.get("deductions") for v in rounds):
            continue
        empty = False
        print(f"  {scene['key']}")
        for i, verdict in enumerate(rounds, 1):
            items = verdict.get("deductions") or []
            # 多轮时逐轮列全：同一个问题在不同轮次可能定级不同，
            # 只给第 1 轮会让人以为严重程度是确定的。
            if repeats > 1:
                print(f"    · 第 {i}/{len(rounds)} 轮"
                      + ("" if items else "：无扣分项"))
            for d in items:
                indent = "      " if repeats > 1 else "    "
                print(f"{indent}[{d.get('dimension')}/{d.get('severity')} "
                      f"−{d.get('points')}] {d.get('timestamp')} {d.get('evidence')}")
                print(f"{indent}    期望：{d.get('expected')}")
    if empty:
        print("  （本批没有任何扣分项）")


def save(result: dict[str, Any], out_dir: Path, name: str = "") -> Path:
    """整份结果落盘。`rounds` 原样保留——聚合过的数字复核不了，原始判决可以。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(result["started_at"]))
    path = out_dir / f"shots-{name + '-' if name else ''}{stamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --- 判据回归 -----------------------------------------------------------------


def parse_expect(raw: str) -> tuple[str, str]:
    """`<source>/<scene_id>:<dimension>` 或 `<scene_id>:<dimension>`。"""
    target, _, dim = raw.rpartition(":")
    if not target or dim not in SCENE_DIMENSIONS:
        raise argparse.ArgumentTypeError(
            f"--expect 格式应为 <片段>:<维度>，维度取值 {SCENE_DIMENSIONS}，收到 {raw!r}"
        )
    return target, dim


def check_expectations(
    result: dict[str, Any], expects: Iterable[tuple[str, str]], min_rate: float
) -> bool:
    """判据回归：指定片段的指定维度必须被扣到，命中率低于门槛就算没过。

    要求的是「扣到了」而不是「分数低于某值」：改判据时想确认的是那条规则会不会
    触发，而分数是各条扣分项汇总后的结果，一个 4.0 分说不清是哪条判据在起作用。
    """
    expects = list(expects)
    if not expects:
        return True
    print()
    print(f"判据回归（命中率门槛 {min_rate:.0%}）")
    passed = True
    for target, dim in expects:
        matched = [s for s in result["scenes"] if s["key"] == target or s["scene_id"] == target]
        if not matched:
            print(f"  ✗ {target}:{dim} —— 这一批里没有这个片段")
            passed = False
            continue
        for scene in matched:
            hits, judged = scene["hits"].get(dim, 0), scene["judged"]
            rate = hits / judged if judged else 0.0
            ok = judged > 0 and rate >= min_rate
            passed = passed and ok
            print(f"  {'✓' if ok else '✗'} {scene['key']}:{dim} —— 扣到 {hits}/{judged}")
    return passed
