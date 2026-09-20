"""评估结果服务：自动发现所有跑批产出，在浏览器里看视频、评分和判词。

为什么在单文件报告之外还要一个服务：

1. **视频要能拖进度条。** `file://` 下浏览器直接读本地文件没问题，但一旦想把
   报告放到别的机器上看就得有人做 HTTP Range——不支持 Range 的静态服务器
   （包括 `python -m http.server`）会让 `<video>` 只能从头播，跳转直接失效。
2. **多个口径要能横着比。** 换 `--beat` / `--design-source` 就是换尺子，
   结论只在同一口径内可比。首页把各轮的中位分并排列出来，
   「哪个口径真的能区分片子」这个问题才看得见答案。
3. **不用每次重新生成文件。** JSON 是唯一事实来源，页面按请求现渲染，
   跑完新一轮刷新即可。

只用标准库：服务端读的是 JSON，不碰 pydantic / google-genai，
所以系统 python3 直接能跑，不需要 uv 建环境。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import socket
import threading
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from shot_eval import report
from shot_eval.generation import infer_generation, video_model_label

#: 一次读多少字节。视频是流式发的，不整份读进内存——
#: 21 段加起来 20 多 MB 还好，成片动辄几百 MB 就不是了。
CHUNK = 256 * 1024

#: 内联 SVG 图标。做成文件就得给这个「只用标准库」的模块配一个静态资源目录，
#: 为一个 16px 的方块不值得。
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#1b1814"/>'
    '<path d="M13 10.5 22 16l-9 5.5z" fill="hsl(32 68% 58%)"/>'
    '<rect x="8" y="10.5" width="2.6" height="11" rx="1.3" fill="hsl(152 40% 48%)"/>'
    "</svg>"
).encode("utf-8")


@dataclass(frozen=True, slots=True)
class Run:
    """一份跑批产出。"""

    rid: str            # URL 里的标识，来自相对路径
    path: Path          # shots-*.json
    root: Path          # 它所属的 run 目录（videos/ 的父目录）

    @property
    def label(self) -> str:
        return self.path.stem.replace("shots-", "")


def _run_root(result_path: Path) -> Path:
    """从结果文件回推 run 目录 —— 也就是 `videos/` 的上一级。

    不靠目录名猜（`eval/` 可能有子目录，比如 `eval/beat-script/`），
    直接从结果里记的片段路径回推：那是跑批当时真实用过的文件。
    """
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        clip = (data.get("tasks") or [{}])[0].get("clip")
        if clip:
            return Path(clip).parent.parent
    except Exception:
        pass
    # 回退：eval/ 或 eval/<子目录>/ 的上级
    for parent in result_path.parents:
        if (parent / "videos").is_dir():
            return parent
    return result_path.parent


def discover(roots: list[Path]) -> list[Run]:
    """找出所有 `shots-*.json`。新跑的一轮刷新首页就能看到。"""
    runs: list[Run] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("shots-*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rid = urllib.parse.quote(str(path.relative_to(root)), safe="")
            runs.append(Run(rid=rid, path=resolved, root=_run_root(resolved)))
    # 新的排前面
    runs.sort(key=lambda r: r.path.stat().st_mtime, reverse=True)
    return runs


class Store:
    """按 mtime 缓存解析结果。377 KB 的 JSON 每次请求重解一遍没必要。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[Path, tuple[float, dict[str, Any]]] = {}

    def load(self, path: Path) -> dict[str, Any]:
        stamp = path.stat().st_mtime
        with self._lock:
            hit = self._cache.get(path)
            if hit and hit[0] == stamp:
                return hit[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._cache[path] = (stamp, data)
        return data


# --- 首页 ---------------------------------------------------------------------

INDEX_CSS = """
:root{
  --ink:hsl(35 20% 92%);--ink-2:hsl(35 12% 72%);--ink-3:hsl(35 9% 52%);
  --bg:hsl(30 8% 9%);--surface:hsl(30 7% 12%);--surface-2:hsl(30 7% 15%);
  --line:hsl(30 6% 21%);--line-soft:hsl(30 6% 17%);--accent:hsl(32 68% 58%);
  --sans:"Inter","Noto Sans SC",-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{
  margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(hsl(30 10% 14%) .5px,transparent .5px);background-size:22px 22px;
}
.wrap{max-width:1180px;margin:0 auto;padding:56px 28px 90px}
h1{font-size:26px;font-weight:600;letter-spacing:-.02em;margin:0}
.lede{color:var(--ink-3);margin:8px 0 0;max-width:62ch}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);
   font-weight:500;margin:44px 0 14px}
a{color:inherit;text-decoration:none}
table{width:100%;border-collapse:collapse}
th{font-size:11.5px;color:var(--ink-3);font-weight:500;text-align:left;
   padding:0 12px 8px 0;white-space:nowrap}
td{padding:12px 12px 12px 0;border-top:1px solid var(--line-soft);vertical-align:top}
tr.run{transition:background .12s ease}
tr.run:hover{background:var(--surface)}
tr.run:hover .open{color:var(--accent)}
.name{font-family:var(--mono);font-size:13.5px}
.sub{color:var(--ink-3);font-size:11.5px;margin-top:3px}
.chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.chip{font-size:10.5px;padding:1px 7px;border:1px solid var(--line);border-radius:999px;color:var(--ink-3)}
.chip b{color:var(--ink-2);font-weight:500}
.num{font-variant-numeric:tabular-nums;font-size:15px;font-weight:600;letter-spacing:-.01em}
.hit{font-size:11px;color:var(--ink-3);font-variant-numeric:tabular-nums}
.hit.mute{color:hsl(6 40% 45%)}
.open{color:var(--ink-3);font-size:12px;white-space:nowrap}
.empty{color:var(--ink-3);padding:28px 0}
.note{color:var(--ink-3);font-size:12.5px;margin-top:18px;max-width:78ch}
code{font-family:var(--mono);font-size:12px;color:var(--ink-2)}
"""


def _score_color(v: float | None) -> str:
    if v is None:
        return "var(--ink-3)"
    return f"hsl({max(0.0, min(1.0, v / 5)) * 148 + 4:.0f} 45% 52%)"


def index_page(runs: list[Run], store: Store) -> bytes:
    rows = []
    for run in runs:
        try:
            result = store.load(run.path)
        except Exception as error:
            rows.append(f'<tr class="run"><td colspan="10">{html.escape(run.path.name)}'
                        f' —— 读不出来：{html.escape(str(error))}</td></tr>')
            continue
        brief = report.summarize(result, run_root=run.root)
        cfg = result.get("config") or {}
        gen = brief.get("generation") or {}
        vm_label = video_model_label(gen)
        vm_chip = (
            f'<span class="chip">视频模型 <b>{html.escape(vm_label)}</b></span>'
        )
        vp = gen.get("video_provider", "unknown")
        vm = gen.get("video_model", "unknown")
        provider_chip = ""
        if (vp and vp != "unknown" and vm and vm != "unknown" and vp != vm):
            provider_chip = (
                f'<span class="chip">Provider <b>{html.escape(vp)}</b></span>'
            )
        chips = vm_chip + provider_chip + "".join(
            f'<span class="chip">{html.escape(k)} <b>{html.escape(str(v))}</b></span>'
            for k, v in [
                ("beat", cfg.get("beat_source")),
                ("style", cfg.get("style_source")),
                ("render", cfg.get("render_source")),
                ("design", result.get("design_source") or "generated"),
                ("模型", result.get("judge_model")),
                ("轮次", f"{result.get('repeats')} 轮"),
            ] if v
        )
        cells = []
        for d in report.DIMENSIONS:
            st = brief["dims"][d["key"]]
            hits, judged = st["hits"], st["judged"]
            shown = "—" if st["mean"] is None else f'{st["mean"]:.2f}'
            med = "—" if st["median"] is None else f'{st["median"]:.2f}'
            # 「一条都没扣到」要显眼：那说明这条判据在这批素材上是哑的。
            mute = " mute" if judged and hits == 0 else ""
            cells.append(
                f'<td><div class="num" style="color:{_score_color(st["mean"])}">{shown}</div>'
                f'<div class="hit{mute}">中位 {med} · 扣到 {hits}/{judged}</div></td>'
            )
        rd = brief["render"]
        rd_txt = (f'{rd["matched"]}/{rd["judged"]}' if rd else "—")
        ov = brief.get("overall") or {}
        ov_score = ov.get("score")
        ov_grade = ov.get("grade", "")
        ov_cov = ov.get("covered_weight", 0)
        ov_max = ov.get("max_weight", 72)
        ov_crit = (ov.get("severity") or {}).get("critical", 0)
        ov_txt = f'{ov_score} {ov_grade}' if ov_score is not None else "—"
        ov_sub = f'{ov_cov}/{ov_max}' if ov_score is not None else ""
        ov_crit_html = (f' · <span style="color:hsl(6 55% 60%)">{ov_crit} crit</span>'
                        if ov_crit else "")
        opt = brief.get("optimization") or {}
        opt_problems = sum((opt.get("stage_problems") or {}).values()) if opt else 0
        opt_text = (f' · 优化 {opt_problems} 问题/{opt.get("valid_patches", 0)} patch'
                    if opt else "")
        rows.append(
            f'<tr class="run" onclick="location.href=\'/report/{run.rid}\'">'
            f'<td><div class="name">{html.escape(run.label)}</div>'
            f'<div class="sub">{brief["scene_count"]} 段 · {brief["deductions"]} 条扣分'
            + (f' · <span style="color:hsl(6 55% 60%)">{brief["errors"]} 轮失败</span>'
               if brief["errors"] else "")
            + opt_text
            + f' · {result.get("elapsed")}s</div>'
            f'<div class="chips">{chips}</div></td>'
            f'<td><div class="num">{html.escape(ov_txt)}</div>'
            f'<div class="hit">{html.escape(ov_sub)}{ov_crit_html}</div></td>'
            + "".join(cells)
            + f'<td><div class="num">{rd_txt}</div><div class="hit">渲染吻合</div></td>'
            f'<td class="open">打开 →</td></tr>'
        )

    heads = "".join(f"<th>{html.escape(d['label'])}</th>" for d in report.DIMENSIONS)
    body = (
        f'<table><thead><tr><th>跑批</th><th>Overall</th>{heads}<th>渲染</th><th></th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows else
        '<p class="empty">没有找到任何 <code>shots-*.json</code>。'
        '跑一轮评估（带 <code>--out</code>）之后刷新本页。</p>'
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>逐 shot 评估 · 结果总览</title>
<meta name="description" content="AI 生成视频逐 shot 评估的全部跑批结果，按口径横向对照。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{INDEX_CSS}</style></head>
<body><div class="wrap">
<h1>逐 shot 评估</h1>
<p class="lede">每一行是一次跑批。换 <code>beat</code> / <code>style</code> /
<code>design</code> 就是换尺子，<b>只有同口径的两行才能直接比分数</b>。</p>
<p style="margin:14px 0 0"><a href="/compare" class="compare-link" style="display:inline-block;padding:6px 18px;border:1px solid var(--accent);border-radius:6px;color:var(--accent);font-size:13px;text-decoration:none">对比两份报告 Compare ↔</a></p>
<h2>全部结果</h2>
{body}
<p class="note">「扣到 x/y」是这一维度产出过扣分项的片段轮次数。
某一维度中位 5.00 且 <span style="color:hsl(6 40% 45%)">扣到 0/y</span> 时，
结论应当是「这条判据在这批素材上没有区分度」，而不是「画面无可挑剔」。</p>
</div></body></html>""".encode("utf-8")


# --- 对比 ---------------------------------------------------------------------

#: 需要比对的口径字段。全部相等才算「同尺度」。
COMPAT_FIELDS = (
    "judge_model", "prompt_version_tag",
    "beat_source", "style_source", "render_source", "design_source",
)


def _compat_value(result: dict[str, Any], field: str) -> str:
    """从结果字典里取口径字段，config 里的字段也能取到。"""
    if field in result:
        return str(result[field])
    cfg = result.get("config") or {}
    return str(cfg.get(field, ""))


def _compat_warning(left: dict[str, Any], right: dict[str, Any]) -> str:
    """检查两份报告的口径兼容性，返回 HTML 警告；完全兼容时返回空字符串。"""
    mismatches: list[str] = []
    for field in COMPAT_FIELDS:
        lv = _compat_value(left, field)
        rv = _compat_value(right, field)
        if lv != rv:
            mismatches.append(field)
    if not mismatches:
        return ""
    if mismatches == ["judge_model"]:
        return (
            '<div class="compat-warn" style="background:hsl(40 60% 16%);border:1px solid hsl(40 50% 30%);'
            'border-radius:8px;padding:12px 18px;margin:16px 0;color:hsl(40 80% 70%);font-size:13px">'
            '⚠️ <b>Judge 模型不同</b>：左侧用 <code>{}</code>，右侧用 <code>{}</code>。'
            '不同模型的评分标尺可能不同，分差可能来自模型而非视频。'
            '</div>'.format(
                html.escape(_compat_value(left, "judge_model")),
                html.escape(_compat_value(right, "judge_model")),
            )
        )
    return (
        '<div class="compat-warn" style="background:hsl(6 50% 14%);border:1px solid hsl(6 40% 30%);'
        'border-radius:8px;padding:12px 18px;margin:16px 0;color:hsl(6 65% 72%);font-size:13px">'
        '⚠️ <b>口径不同，不可直接比分数</b>：{} 不一致。'
        '仍可做并排诊断，但分差不能归因于视频质量。'
        '</div>'.format(html.escape(", ".join(mismatches)))
    )


def build_paired_data(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    left_rid: str,
    right_rid: str,
) -> list[dict[str, Any]]:
    """把两份结果的场景按 key 配对，返回确定性排列的配对列表。

    配对策略：
    1. 使用 scene["key"] 做稳定匹配（回退 source+scene_id）。
    2. 结果按左侧原始顺序排列，右侧独有的追加到末尾。
    3. 每一项标注 left/right 各自的数据或 None。
    """
    def _scene_key(scene: dict[str, Any]) -> str:
        k = scene.get("key")
        if k:
            return str(k)
        return f'{scene.get("source", "")}/{scene.get("scene_id", "")}'

    def _slim_scene(scene: dict[str, Any], rid: str,
                    tasks_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """只保留对比需要的字段，不内联 video_prompt/provenance。"""
        task = tasks_map.get(_scene_key(scene), {})
        clip = task.get("clip", "")
        filename = Path(clip).name if clip else ""
        context = scene.get("context") or {}
        beats = context.get("beats") or []
        scripts = [
            {
                "start": beat.get("start"),
                "duration": beat.get("duration"),
                "script": beat.get("script"),
                "visual_beat": beat.get("visual_beat"),
            }
            for beat in beats if isinstance(beat, dict)
        ]
        return {
            "key": _scene_key(scene),
            "scene_id": scene.get("scene_id"),
            "source": scene.get("source"),
            "video_url": f"/media/{rid}/{filename}" if filename else "",
            "duration": task.get("duration"),
            # 用户要求展示：每个shot的原始口播与实际送入视频模型的完整Prompt。
            # 不带 provenance/video_prompt 以外的规划上下文，控制对比页载荷。
            "scripts": scripts,
            "video_prompt": str(context.get("prompt") or ""),
            "dims": scene.get("dims") or {},
            "unreferenced": scene.get("unreferenced") or [],
            "deduction_count": scene.get("deduction_count") or [],
            "hits": scene.get("hits") or {},
            "judged": scene.get("judged", 0),
            "errors": scene.get("errors") or [],
            "render_match": scene.get("render_match"),
            "render_declared": scene.get("render_declared"),
            "rounds": _slim_rounds(scene.get("rounds") or []),
        }

    def _slim_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """只保留扣分和分数，不内联整段判词。"""
        out = []
        for r in rounds:
            slim: dict[str, Any] = {}
            # 完整保留实际评审内容；故意排除 scene prompt/provenance/video_prompt。
            for dk in ("deductions", "error", "observed", "visible_text", "issues", "unreferenced"):
                if dk in r:
                    slim[dk] = r[dk]
            for flag_key, _ in report.FLAGS:
                if flag_key in r:
                    slim[flag_key] = r[flag_key]
            for d in report.DIMENSIONS:
                k = d["key"]
                if k in r:
                    slim[k] = r[k]
                reason_key = f"{k}_reason"
                if reason_key in r:
                    slim[reason_key] = r[reason_key]
            for rk in ("render_verdict", "render_match", "render_confidence", "render_reason", "render_declared"):
                if rk in r:
                    slim[rk] = r[rk]
            out.append(slim)
        return out

    left_scenes = left_result.get("scenes") or []
    right_scenes = right_result.get("scenes") or []

    left_tasks = {t["key"]: t for t in left_result.get("tasks") or []}
    right_tasks = {t["key"]: t for t in right_result.get("tasks") or []}

    left_by_key = {}
    left_order = []
    for s in left_scenes:
        k = _scene_key(s)
        left_by_key[k] = s
        left_order.append(k)

    right_by_key = {}
    right_order = []
    for s in right_scenes:
        k = _scene_key(s)
        right_by_key[k] = s
        right_order.append(k)

    pairs = []
    seen = set()
    # Left original order first
    for k in left_order:
        seen.add(k)
        ld = _slim_scene(left_by_key[k], left_rid, left_tasks)
        rd = (_slim_scene(right_by_key[k], right_rid, right_tasks)
              if k in right_by_key else None)
        pairs.append({"key": k, "left": ld, "right": rd})
    # Right-only scenes appended
    for k in right_order:
        if k not in seen:
            seen.add(k)
            rd = _slim_scene(right_by_key[k], right_rid, right_tasks)
            pairs.append({"key": k, "left": None, "right": rd})

    return pairs


def _meta_block(result: dict[str, Any], run_root: str | Path | None = None) -> dict[str, Any]:
    """提取报告元数据，用于对比页显示。"""
    cfg = result.get("config") or {}
    gen = infer_generation(result, run_root=run_root)
    return {
        "judge_model": result.get("judge_model", ""),
        "prompt_version_tag": result.get("prompt_version_tag", ""),
        "design_source": result.get("design_source", ""),
        "beat_source": cfg.get("beat_source", ""),
        "style_source": cfg.get("style_source", ""),
        "render_source": cfg.get("render_source", ""),
        "repeats": result.get("repeats"),
        "elapsed": result.get("elapsed"),
        "generation": gen,
    }


def _gen_chips_html(gen: dict[str, str]) -> str:
    """Render generation-model chips for compare summary cards."""
    parts: list[str] = []
    vm = gen.get("video_model", "unknown")
    vp = gen.get("video_provider", "unknown")
    vm_label = vm if vm and vm != "unknown" else (vp if vp and vp != "unknown" else "未知")
    parts.append(f'视频模型=<code>{html.escape(vm_label)}</code>')
    if vp and vp != "unknown" and vm and vm != "unknown" and vp != vm:
        parts.append(f'Provider=<code>{html.escape(vp)}</code>')
    pm = gen.get("prompt_model", "unknown")
    if pm and pm != "unknown":
        parts.append(f'Prompt LLM=<code>{html.escape(pm)}</code>')
    plm = gen.get("planner_model", "unknown")
    if plm and plm != "unknown":
        parts.append(f'Planner=<code>{html.escape(plm)}</code>')
    gr = gen.get("granularity", "unknown")
    if gr and gr != "unknown":
        parts.append(f'粒度=<code>{html.escape(gr)}</code>')
    return (
        f'<div style="margin-top:4px;font-size:11px;color:var(--ink-2)">'
        f'{" · ".join(parts)}</div>'
    )


def _gen_diff_notice(left_meta: dict[str, Any], right_meta: dict[str, Any]) -> str:
    """If the generation video models differ, return an informational notice.

    This is NOT a compatibility warning — comparing different generation models
    is a valid experiment.
    """
    lg = left_meta.get("generation") or {}
    rg = right_meta.get("generation") or {}
    lvm = video_model_label(lg)
    rvm = video_model_label(rg)
    if lvm == rvm:
        return ""
    return (
        '<div style="background:hsl(210 30% 14%);border:1px solid hsl(210 30% 28%);'
        'border-radius:8px;padding:10px 16px;margin:10px 0;color:hsl(210 60% 72%);font-size:13px">'
        'ℹ️ <b>视频生成模型不同</b>：左侧 <code>{}</code>，右侧 <code>{}</code>。'
        '这是生成产物的差异，不是评审尺度的差异——对比不同生成模型正是本功能的目的。'
        '</div>'.format(html.escape(lvm), html.escape(rvm))
    )


def compare_page(
    runs: list[Run],
    store: Store,
    left_rid: str | None,
    right_rid: str | None,
) -> bytes:
    """渲染对比页。任何时候都返回合法 HTML，不抛异常。"""

    # -- 选择控件 --
    def _opt(rid: str | None, run: Run, result: dict[str, Any]) -> str:
        model = result.get("judge_model", "?")
        cfg = result.get("config") or {}
        beat = cfg.get("beat_source", "")
        # 用路径区分同名报告（不同 root 下可能有同名文件）
        label = f"{run.label} [{html.escape(model)}]"
        if beat:
            label += f" beat={html.escape(beat)}"
        label += f" ({html.escape(str(run.path.parent.relative_to(run.path.parent.parent.parent)))})"
        sel = " selected" if run.rid == rid else ""
        return f'<option value="{html.escape(run.rid)}"{sel}>{label}</option>'

    options_left = ['<option value="">— 选择左侧报告 —</option>']
    options_right = ['<option value="">— 选择右侧报告 —</option>']
    run_map: dict[str, Run] = {}
    for run in runs:
        try:
            result = store.load(run.path)
        except Exception:
            continue
        run_map[run.rid] = run
        options_left.append(_opt(left_rid, run, result))
        options_right.append(_opt(right_rid, run, result))

    selector_html = f"""
    <form method="get" action="/compare" style="display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:18px 0">
      <label style="flex:1;min-width:280px">
        <span style="font-size:12px;color:var(--ink-3)">左侧 (Left)</span><br>
        <select name="left" style="width:100%;padding:6px 10px;background:var(--surface-2);color:var(--ink);
          border:1px solid var(--line);border-radius:6px;font-size:13px;font-family:var(--mono)">
          {"".join(options_left)}
        </select>
      </label>
      <label style="flex:1;min-width:280px">
        <span style="font-size:12px;color:var(--ink-3)">右侧 (Right)</span><br>
        <select name="right" style="width:100%;padding:6px 10px;background:var(--surface-2);color:var(--ink);
          border:1px solid var(--line);border-radius:6px;font-size:13px;font-family:var(--mono)">
          {"".join(options_right)}
        </select>
      </label>
      <button type="submit" style="padding:8px 22px;background:var(--accent);color:#000;
        border:none;border-radius:6px;font-weight:600;font-size:13px;cursor:pointer">
        对比 Compare
      </button>
    </form>
    """

    # -- 没有选择或无效 rid --
    if not left_rid or not right_rid:
        return _compare_shell(selector_html, "<p style='color:var(--ink-3)'>请选择两份报告。</p>")
    left_run = run_map.get(left_rid)
    right_run = run_map.get(right_rid)
    if not left_run:
        return _compare_shell(
            selector_html,
            f"<p style='color:hsl(6 55% 60%)'>左侧报告 ID 无效：<code>{html.escape(left_rid)}</code></p>"
        )
    if not right_run:
        return _compare_shell(
            selector_html,
            f"<p style='color:hsl(6 55% 60%)'>右侧报告 ID 无效：<code>{html.escape(right_rid)}</code></p>"
        )

    # -- 加载数据 --
    left_result = store.load(left_run.path)
    right_result = store.load(right_run.path)
    left_brief = report.summarize(left_result, run_root=left_run.root)
    right_brief = report.summarize(right_result, run_root=right_run.root)
    left_meta = _meta_block(left_result, run_root=left_run.root)
    right_meta = _meta_block(right_result, run_root=right_run.root)

    compat_html = _compat_warning(left_result, right_result)
    gen_diff_html = _gen_diff_notice(left_meta, right_meta)

    paired = build_paired_data(left_result, right_result, left_rid, right_rid)

    # -- 汇总卡片 --
    def _summary_card(label: str, brief: dict[str, Any], meta: dict[str, Any]) -> str:
        ov = brief.get("overall") or {}
        ov_score = ov.get("score")
        ov_grade = ov.get("grade", "")
        ov_txt = f'{ov_score} {ov_grade}' if ov_score is not None else "—"
        cov = ov.get("covered_weight", 0)
        maxw = ov.get("max_weight", 72)
        crit = (ov.get("severity") or {}).get("critical", 0)
        dims_html = ""
        for d in report.DIMENSIONS:
            st = brief["dims"][d["key"]]
            mean_s = f'{st["mean"]:.2f}' if st["mean"] is not None else "—"
            med_s = f'{st["median"]:.2f}' if st["median"] is not None else "—"
            dims_html += (
                f'<div style="padding:6px 0;border-bottom:1px solid var(--line-soft)">'
                f'<span style="font-size:12px;color:var(--ink-3)">{html.escape(d["label"])}</span> '
                f'<span class="mono" style="font-size:14px;font-weight:600;color:{_score_color(st["mean"])}">'
                f'{mean_s}</span>'
                f'<span style="font-size:11px;color:var(--ink-3)"> 中位 {med_s} · 扣到 {st["hits"]}/{st["judged"]}</span>'
                f'</div>'
            )
        rd = brief["render"]
        rd_txt = f'{rd["matched"]}/{rd["judged"]}' if rd else "—"
        return (
            f'<div style="flex:1;min-width:300px;background:var(--surface);border:1px solid var(--line);'
            f'border-radius:10px;padding:16px 20px">'
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);'
            f'margin-bottom:6px">{html.escape(label)}</div>'
            f'<div style="font-size:20px;font-weight:600;font-variant-numeric:tabular-nums">{html.escape(ov_txt)}</div>'
            f'<div style="font-size:11px;color:var(--ink-3)">'
            f'覆盖 {cov}/{maxw}'
            + (f' · <span style="color:hsl(6 55% 60%)">{crit} critical</span>' if crit else "")
            + f' · {brief["scene_count"]} 场景 · {brief["deductions"]} 扣分'
            + (f' · <span style="color:hsl(6 55% 60%)">{brief["errors"]} 失败</span>'
               if brief["errors"] else "")
            + f'</div>'
            f'<div style="margin-top:6px;font-size:11px;color:var(--ink-3)">'
            f'model=<code>{html.escape(meta.get("judge_model", ""))}</code> '
            f'beat=<code>{html.escape(meta.get("beat_source", ""))}</code> '
            f'render=<code>{html.escape(meta.get("render_source", ""))}</code></div>'
            + _gen_chips_html(meta.get("generation") or {}) +
            f'<div style="padding:6px 0;font-size:12px;color:var(--ink-3)">渲染吻合 {rd_txt}</div>'
            f'</div>'
        )

    summary_html = (
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin:16px 0">'
        f'{_summary_card("左侧 LEFT", left_brief, left_meta)}'
        f'{_summary_card("右侧 RIGHT", right_brief, right_meta)}'
        f'</div>'
    )

    # -- 配对场景表 --
    pair_rows = []
    for i, p in enumerate(paired):
        k = html.escape(p["key"])
        ls = p.get("left")
        rs = p.get("right")
        lclass = "missing" if ls is None else ""
        rclass = "missing" if rs is None else ""

        def _dim_cells(side: dict[str, Any] | None) -> str:
            if side is None:
                return '<td colspan="6" style="color:var(--ink-3);text-align:center">— 缺失 —</td>'
            cells = ""
            for d in report.DIMENSIONS:
                dk = d["key"]
                dd = side["dims"].get(dk) or {}
                med = dd.get("median")
                med_s = f"{med:.2f}" if med is not None else "—"
                unreferenced = dk in (side.get("unreferenced") or [])
                color = "var(--ink-3)" if unreferenced else _score_color(med)
                cells += f'<td style="font-variant-numeric:tabular-nums;font-size:13px;color:{color}">{med_s}</td>'
            return cells

        def _delta_cell(l_side: dict[str, Any] | None, r_side: dict[str, Any] | None, dk: str) -> str:
            if l_side is None or r_side is None:
                return ""
            lm = (l_side["dims"].get(dk) or {}).get("median")
            rm = (r_side["dims"].get(dk) or {}).get("median")
            if lm is not None and rm is not None:
                d = lm - rm
                color = "var(--good)" if d > 0.01 else ("var(--bad)" if d < -0.01 else "var(--ink-3)")
                return f'<span style="color:{color};font-size:11px">{d:+.2f}</span>'
            return ""

        # 每一行是一个配对场景
        pair_rows.append(
            f'<tr class="pair-row" data-idx="{i}" style="cursor:pointer;border-bottom:1px solid var(--line-soft)"'
            f' onclick="selectPair({i})">'
            f'<td style="font-family:var(--mono);font-size:12px;padding:8px 6px;white-space:nowrap">{k}</td>'
            + _dim_cells(ls) + _dim_cells(rs) +
            f'</tr>'
        )

    dim_heads_l = "".join(f'<th style="font-size:10px;padding:4px 6px;color:var(--ink-3)">{html.escape(d["short"])}↙</th>' for d in report.DIMENSIONS)
    dim_heads_r = "".join(f'<th style="font-size:10px;padding:4px 6px;color:var(--ink-3)">{html.escape(d["short"])}↘</th>' for d in report.DIMENSIONS)

    pairs_table = (
        f'<div style="overflow-x:auto;margin:12px 0">'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px">'
        f'<thead><tr><th style="text-align:left;padding:4px 6px;font-size:11px;color:var(--ink-3)">Scene</th>'
        f'{dim_heads_l}{dim_heads_r}</tr></thead>'
        f'<tbody>{"".join(pair_rows)}</tbody></table></div>'
    )

    # -- 所有配对详情默认展开；上方配对表仅用于快速定位。 --
    detail_panel = (
        '<div style="margin:26px 0 10px"><h2 style="margin:0 0 6px">全部逐 shot 对比</h2>'
        '<p style="margin:0;color:var(--ink-3);font-size:12px">所有视频、维度、逐轮评估内容和扣分均已展开；点击上表行可定位。</p></div>'
        '<div id="all-pair-details" style="display:grid;gap:24px"></div>'
    )

    # -- 用 JSON 嵌入配对数据供 JS 使用 --
    pairs_json = json.dumps(paired, ensure_ascii=False).replace("</", "<\\/")

    js = r"""
<script>
var PAIRS = __PAIRS__;
var DIMS = __DIMS__;
var SEV_LABEL = __SEV_LABEL__;

function esc(s){
  if(s==null)return '';
  var d=document.createElement('div');d.appendChild(document.createTextNode(String(s)));return d.innerHTML;
}
function fmtTime(t){
  if(t==null||t===''||isNaN(t))return '—';
  var s=parseFloat(t);var m=Math.floor(s/60);var ss=(s%60).toFixed(1);
  return m+':'+(ss<10?'0':'')+ss;
}
function parseSeek(t){
  if(typeof t==='number')return t;
  var raw=String(t||'');
  var m=raw.match(/(\d{1,2}):(\d{2}(?:\.\d+)?)/);
  if(m)return Number(m[1])*60+Number(m[2]);
  var s=raw.match(/(\d+(?:\.\d+)?)\s*s/i);
  return s?Number(s[1]):null;
}
function parseSeek(t){
  if(typeof t==='number')return t;
  var raw=String(t||'');
  var m=raw.match(/(\d{1,2}):(\d{2}(?:\.\d+)?)/);
  if(m)return Number(m[1])*60+Number(m[2]);
  var s=raw.match(/(\d+(?:\.\d+)?)\s*s/i);
  return s?Number(s[1]):null;
}
function selectPair(idx){
  document.querySelectorAll('.pair-row').forEach(function(r){
    r.style.background=r.getAttribute('data-idx')===String(idx)?'var(--surface-2)':'';
  });
  var el=document.getElementById('pair-detail-'+idx);
  if(el)el.scrollIntoView({behavior:'smooth',block:'start'});
}
function renderNarrationAndPrompt(s){
  var h='<details open style="margin:10px 0"><summary style="cursor:pointer;font-size:12px;color:var(--accent)">原始口播（评估输入）</summary>';
  var scripts=s.scripts||[];
  if(!scripts.length){h+='<div style="padding:6px 0;color:var(--ink-3);font-size:12px">该报告没有保存本 shot 的口播文本。</div>';}
  for(var i=0;i<scripts.length;i++){
    var b=scripts[i]||{};
    var range=(b.start!=null&&b.duration!=null)?fmtTime(b.start)+'–'+fmtTime(Number(b.start)+Number(b.duration))+'：':'';
    h+='<div style="padding:5px 0;font-size:12px;line-height:1.55"><span style="font-family:var(--mono);color:var(--ink-3)">'+esc(range)+'</span>'+esc(b.script||'—')+'</div>';
  }
  h+='</details>';
  h+='<details style="margin:8px 0"><summary style="cursor:pointer;font-size:12px;color:var(--accent)">送入视频模型的完整 Prompt</summary>';
  h+='<pre style="margin:7px 0 0;padding:10px;background:var(--surface-2);border:1px solid var(--line-soft);border-radius:6px;white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:11px;line-height:1.5;color:var(--ink-2);max-height:460px;overflow:auto">'+esc(s.video_prompt||'该报告没有保存完整视频 Prompt。')+'</pre>';
  h+='</details>';
  return h;
}
function renderSide(s,label){
  if(!s) return '<div style="flex:1;min-width:320px;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:20px;text-align:center;color:var(--ink-3)"><p>此侧无该场景</p><p style="font-size:12px">Missing on '+esc(label)+' side</p></div>';
  var h='<div style="flex:1;min-width:320px;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px;overflow:hidden">';
  h+='<div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);margin-bottom:6px">'+esc(label)+'</div>';
  if(s.video_url)h+='<video controls preload="metadata" playsinline style="width:100%;border-radius:8px;background:#000;border:1px solid var(--line);max-height:260px" id="video-'+esc(label)+'-'+esc(s.key)+'" src="'+esc(s.video_url)+'"></video>';
  h+='<div style="font-family:var(--mono);font-size:13px;margin:8px 0">'+esc(s.scene_id)+' <span style="color:var(--ink-3);font-size:11px">'+esc(s.source)+'</span></div>';
  h+=renderNarrationAndPrompt(s);
  h+='<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line-soft);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:8px 0">';
  for(var i=0;i<DIMS.length;i++){
    var d=DIMS[i], dd=s.dims[d.key]||{}, med=dd.median, ms=med!=null?med.toFixed(2):'—';
    var unref=(s.unreferenced||[]).indexOf(d.key)>=0, c=unref?'var(--ink-3)':scoreColor(med);
    h+='<div style="background:var(--surface);padding:6px 8px"><div style="font-size:10px;color:var(--ink-3)">'+esc(d.label)+'</div><div style="font-size:15px;font-weight:600;color:'+c+';font-variant-numeric:tabular-nums">'+ms+'</div></div>';
  }
  h+='</div>';
  var rounds=s.rounds||[];
  for(var ri=0;ri<rounds.length;ri++){
    var r=rounds[ri], deds=r.deductions||[];
    h+='<div style="font-size:11px;color:var(--ink-3);margin:12px 0 4px;border-bottom:1px solid var(--line-soft);padding-bottom:4px">第 '+(ri+1)+' 轮评估内容</div>';
    if(r.error){h+='<div style="color:hsl(6 55% 60%);font-size:12px;margin:4px 0">错误: '+esc(r.error)+'</div>';continue;}
    if(r.observed)h+='<p style="margin:5px 0;font-size:12px;line-height:1.55"><span style="color:var(--ink-3)">观察：</span>'+esc(r.observed)+'</p>';
    var reasons=[];
    for(var xi=0;xi<DIMS.length;xi++){
      var xd=DIMS[xi], reason=r[xd.key+'_reason'];
      if(reason)reasons.push('<div style="font-size:11px;margin:2px 0"><span style="color:var(--ink-3)">'+esc(xd.short)+'：</span>'+esc(reason)+'</div>');
    }
    if(reasons.length)h+='<div style="padding:6px 8px;background:var(--surface-2);border-radius:5px">'+reasons.join('')+'</div>';
    var flags=[];
    if(r.has_subtitle_or_caption)flags.push('字幕');
    if(r.has_watermark_or_logo)flags.push('水印/Logo');
    if(r.has_brand_name)flags.push('品牌');
    if(flags.length)h+='<div style="font-size:11px;color:var(--bad);margin-top:5px">合规命中：'+esc(flags.join('、'))+'</div>';
    if((r.visible_text||[]).length)h+='<div style="font-size:11px;color:var(--ink-3);margin-top:3px">画面文字：'+esc((r.visible_text||[]).join(' · '))+'</div>';
    if((r.issues||[]).length)h+='<div style="font-size:11px;color:var(--warn);margin-top:3px">其他问题：'+esc((r.issues||[]).join('；'))+'</div>';
    if(r.render_verdict)h+='<div style="font-size:11px;color:var(--ink-3);margin-top:3px">Render：'+esc(r.render_verdict)+' · '+(r.render_match?'吻合':'不符')+(r.render_reason?' · '+esc(r.render_reason):'')+'</div>';
    if(!deds.length)h+='<div style="font-size:11px;color:var(--good);margin:5px 0">本轮无扣分项</div>';
    for(var di=0;di<deds.length;di++){
      var dd=deds[di], sevLabel=SEV_LABEL[dd.severity]||dd.severity||'?', sevClass=dd.severity||'minor';
      h+='<div style="border-left:2px solid '+(sevClass==='critical'?'var(--bad)':sevClass==='major'?'var(--warn)':'var(--accent-dim)')+';padding:0 0 0 10px;margin:6px 0">';
      h+='<div style="display:flex;gap:6px;align-items:baseline;flex-wrap:wrap">';
      h+='<span style="font-size:11px;padding:1px 6px;border-radius:4px;font-weight:500;'+(sevClass==='critical'?'background:hsl(6 45% 20%);color:hsl(6 70% 72%)':sevClass==='major'?'background:hsl(40 45% 18%);color:hsl(40 75% 68%)':'background:var(--surface-2);color:var(--ink-2)')+'">'+esc(sevLabel)+'</span>';
      if(dd.dimension)h+='<span style="font-size:12px;color:var(--ink-2)">'+esc(dd.dimension)+'</span>';
      if(dd.points!=null)h+='<span style="font-family:var(--mono);font-size:12px;color:var(--bad)">−'+esc(dd.points)+'</span>';
      if(dd.timestamp!=null)h+='<button style="font-family:var(--mono);font-size:12px;color:var(--accent);background:none;border:none;border-bottom:1px dashed var(--accent);cursor:pointer;padding:0" onclick="seekVideo(\''+esc(label)+'\',\''+esc(s.key)+'\','+JSON.stringify(dd.timestamp)+')">'+esc(dd.timestamp)+'</button>';
      h+='</div>';
      if(dd.evidence)h+='<p style="margin:4px 0 0;font-size:13px">'+esc(dd.evidence)+'</p>';
      if(dd.expected)h+='<p style="margin:2px 0 0;font-size:12px;color:var(--ink-3)">应为: '+esc(dd.expected)+'</p>';
      h+='</div>';
    }
  }
  h+='</div>';
  return h;
}
function scoreColor(v){
  if(v==null)return 'var(--ink-3)';
  return 'hsl('+(Math.max(0,Math.min(1,v/5))*148+4).toFixed(0)+' 45% 52%)';
}
function seekVideo(label,key,t){
  var seconds=parseSeek(t), vid=document.getElementById('video-'+label+'-'+key);
  if(vid&&seconds!=null){vid.currentTime=seconds;vid.play();}
}
function renderPairDetail(p,idx){
  var h='<article id="pair-detail-'+idx+'" style="scroll-margin-top:20px;padding:16px;border:1px solid var(--line);border-radius:10px;background:var(--surface)">';
  h+='<div style="font-family:var(--mono);font-size:14px;font-weight:600;margin:0 0 10px">'+esc(p.key)+'</div>';
  h+='<div style="display:flex;gap:16px;flex-wrap:wrap;margin:12px 0">'+renderSide(p.left,'LEFT')+renderSide(p.right,'RIGHT')+'</div>';
  if(p.left&&p.right){
    h+='<div style="background:var(--surface-2);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:8px 0"><div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);margin-bottom:6px">维度差异 (Left − Right)</div><div style="display:flex;gap:12px;flex-wrap:wrap">';
    for(var i=0;i<DIMS.length;i++){
      var d=DIMS[i],lm=(p.left.dims[d.key]||{}).median,rm=(p.right.dims[d.key]||{}).median;
      if(lm!=null&&rm!=null){var delta=lm-rm,dc=delta>0.01?'var(--good)':delta<-0.01?'var(--bad)':'var(--ink-3)';h+='<div style="font-size:12px"><span style="color:var(--ink-3)">'+esc(d.short)+'</span> <span style="color:'+dc+';font-weight:600;font-variant-numeric:tabular-nums">'+(delta>0?'+':'')+delta.toFixed(2)+'</span></div>';}
    }
    h+='</div></div>';
  }
  return h+'</article>';
}
function renderAllPairs(){
  var target=document.getElementById('all-pair-details');
  if(target)target.innerHTML=PAIRS.map(function(p,idx){return renderPairDetail(p,idx);}).join('');
}
renderAllPairs();
</script>
""".replace("__PAIRS__", pairs_json).replace(
        "__DIMS__", json.dumps(report.DIMENSIONS, ensure_ascii=False).replace("</", "<\\/")
    ).replace(
        "__SEV_LABEL__", json.dumps(report.SEVERITY_LABEL, ensure_ascii=False).replace("</", "<\\/")
    )

    body_html = selector_html + compat_html + gen_diff_html + summary_html + pairs_table + detail_panel + js
    return _compare_shell("", body_html)


def _compare_shell(selector: str, body: str) -> bytes:
    """对比页的外层 HTML 壳。"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>逐 shot 评估 · 对比</title>
<style>{INDEX_CSS}</style></head>
<body><div class="wrap">
<h1><a href="/" style="color:inherit;text-decoration:none" title="返回总览">← 逐 shot 评估</a>
<span style="color:var(--ink-3);font-weight:400;margin-left:8px">对比</span></h1>
{selector}
{body}
</div></body></html>""".encode("utf-8")


# --- 服务 ---------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "shot_eval"
    runs: list[Run] = []
    roots: list[Path] = []
    store = Store()

    # 默认的 log_message 把每个 Range 请求都打一行，拖一次进度条刷几十行。
    # 只压掉 206，其余照常——真出错时还是要看得见。
    def log_message(self, fmt, *args):  # noqa: D102
        if any(str(a) == "206" for a in args):
            return
        super().log_message(fmt, *args)

    def _find(self, rid: str) -> Run | None:
        for run in self.runs:
            if run.rid == rid:
                return run
        # 没命中就重扫一次：跑批是在服务起来之后完成的，很常见
        type(self).runs = discover(self.roots)
        return next((r for r in self.runs if r.rid == rid), None)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                type(self).runs = discover(self.roots)
                self._send(index_page(self.runs, self.store), "text/html; charset=utf-8")
            elif path == "/compare":
                self._compare(parsed.query)
            elif path.startswith("/report/"):
                self._report(path[len("/report/"):])
            elif path.startswith("/media/"):
                self._media(path[len("/media/"):])
            elif path.startswith("/raw/"):
                self._raw(path[len("/raw/"):])
            elif path == "/favicon.ico":
                self._send(FAVICON, "image/svg+xml")
            else:
                self._fail(HTTPStatus.NOT_FOUND, "没有这个地址")
        except (BrokenPipeError, ConnectionResetError):
            pass  # 浏览器拖进度条时会直接掐掉上一个 Range 请求

    # -- 各路由 --
    def _compare(self, query: str) -> None:
        """GET /compare?left=<rid>&right=<rid> —— 对比两份报告。"""
        type(self).runs = discover(self.roots)
        qs = urllib.parse.parse_qs(query, keep_blank_values=True)
        raw_left = (qs.get("left") or [None])[0] or None
        raw_right = (qs.get("right") or [None])[0] or None
        # 下拉框的 value 已是 Run.rid（内部 % 编码）；浏览器表单提交后
        # parse_qs 正好恢复该字符串。手写 ?left=eval%2F... 则会得到未编码路径，
        # 为兼容两种入口，仅当 direct rid 不存在时回退 quote 一次。
        known = {run.rid for run in self.runs}
        def normalize(raw: str | None) -> str | None:
            if not raw:
                return None
            if raw in known:
                return raw
            encoded = urllib.parse.quote(raw, safe="")
            return encoded if encoded in known else raw
        left_rid = normalize(raw_left)
        right_rid = normalize(raw_right)
        page = compare_page(self.runs, self.store, left_rid, right_rid)
        self._send(page, "text/html; charset=utf-8", cache=False)

    def _report(self, rid: str) -> None:
        run = self._find(rid)
        if not run:
            return self._fail(HTTPStatus.NOT_FOUND, "没有这份结果")
        result = self.store.load(run.path)
        payload = report.build_payload(result, video_base=f"/media/{rid}", run_root=run.root)
        page = report.render_html(payload, f"逐 shot 评估 · {run.label}")
        # 服务端多一个回首页的入口——单文件报告里没有这个概念
        page = page.replace(
            '<h1>逐 shot 评估<span id="runname"></span></h1>',
            '<h1><a href="/" style="color:inherit;text-decoration:none" '
            'title="返回总览">← 逐 shot 评估</a><span id="runname"></span></h1>',
        )
        self._send(page.encode("utf-8"), "text/html; charset=utf-8", cache=False)

    def _raw(self, rid: str) -> None:
        run = self._find(rid)
        if not run:
            return self._fail(HTTPStatus.NOT_FOUND, "没有这份结果")
        self._send(run.path.read_bytes(), "application/json; charset=utf-8")

    def _media(self, tail: str) -> None:
        rid, _, name = tail.partition("/")
        run = self._find(rid)
        if not run:
            return self._fail(HTTPStatus.NOT_FOUND, "没有这份结果")
        # 只允许取 videos/ 下的一个文件名，不接受任何路径成分
        if not re.fullmatch(r"[\w.\-]+\.(mp4|mov|webm)", name):
            return self._fail(HTTPStatus.BAD_REQUEST, "文件名不合法")
        target = (run.root / "videos" / name).resolve()
        if not target.is_file() or run.root.resolve() not in target.parents:
            return self._fail(HTTPStatus.NOT_FOUND, "没有这个片段")
        self._stream(target)

    # -- 发送 --
    def _send(self, body: bytes, ctype: str, cache: bool = True) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, status: HTTPStatus, message: str) -> None:
        body = f"<!DOCTYPE html><meta charset=utf-8><body style='font:14px sans-serif;" \
               f"background:#17150f;color:#ddd;padding:48px'>{html.escape(message)}" \
               f"<p><a href='/' style='color:#d99'>返回总览</a></p>".encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, target: Path) -> None:
        """带 Range 的视频响应。

        没有 Range 就没有拖动：浏览器拿不到 206 就只能整条下完再播，
        而评估的核心动作恰恰是「点一个扣分时间点跳过去看」。
        """
        size = target.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        rng = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
        if match:
            lo, hi = match.group(1), match.group(2)
            if lo:
                start = int(lo)
                end = int(hi) if hi else size - 1
            elif hi:                       # bytes=-500 表示最后 500 字节
                start = max(0, size - int(hi))
            if start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        remaining = end - start + 1
        with target.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shot_eval.serve",
        description="起一个本地服务，浏览所有逐 shot 评估结果。",
    )
    parser.add_argument("roots", nargs="*", default=["runs"],
                        help="扫描哪些目录下的 shots-*.json（默认 runs）")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1",
                        help="默认只监听本机；要给同事看就用 0.0.0.0")
    args = parser.parse_args(argv)

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    Handler.roots = roots
    Handler.runs = discover(roots)

    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    shown = args.host if args.host != "0.0.0.0" else socket.gethostname()
    print(f"扫描目录：{', '.join(str(r) for r in roots)}")
    print(f"发现 {len(Handler.runs)} 份结果")
    for run in Handler.runs:
        print(f"  {run.label}  ←  {run.path}")
    print()
    print(f"→ http://{shown}:{args.port}/")
    print("Ctrl-C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
