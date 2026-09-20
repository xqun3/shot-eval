"""把评估结果渲染成一个**自包含的 HTML 报告**。

为什么是单文件、数据内嵌、不起服务：
评估产出要能随手发给别人、能塞进 `runs/` 里跟素材一起归档、半年后还打得开。
起服务的方案在这三点上都输——而且 `fetch()` 在 `file://` 下被同源策略挡掉，
数据只能内嵌（`<script type="application/json">`）才能双击直接看。
视频不内嵌：21 段 mp4 有 20 多 MB，base64 之后翻 1.33 倍，
浏览器还得把整串解码进内存才能播。按相对路径引用，`<video>` 在 file:// 下能直接放。

所以报告必须和 `videos/` 保持相对位置。默认写到 `runs/<run>/eval/report.html`，
引用 `../videos/<unit>.mp4`。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

#: 维度的中文名 + 权重。权重取自 `docs/scene-eval.md` 的整片计分表，
#: 这里用来算加权诊断总分和排序用的综合——
#: 21 段里「先看哪一段」需要一个次序，纯按某一维排会漏掉别的维度崩了的段。
DIMENSIONS: list[dict[str, Any]] = [
    {"key": "science_accuracy", "label": "科学准确", "short": "科学", "weight": 20},
    {"key": "beat_alignment", "label": "画面还原", "short": "还原", "weight": 12},
    {"key": "clarity", "label": "清晰易懂", "short": "清晰", "weight": 10},
    {"key": "design_fidelity", "label": "分镜还原", "short": "分镜", "weight": 8},
    {"key": "instructional_value", "label": "教育价值", "short": "教育", "weight": 8},
    {"key": "style_adherence", "label": "风格一致", "short": "风格", "weight": 4},
]

try:  # 出报告不该要求装 pydantic —— 服务端只读 JSON，用系统 python 就能跑
    from shot_eval.models import SCENE_DIMENSIONS

    assert {d["key"] for d in DIMENSIONS} == set(SCENE_DIMENSIONS)
except ModuleNotFoundError:  # pragma: no cover
    pass

from shot_eval.scoring import score_result
from shot_eval.generation import infer_generation, video_model_label

SEVERITY_LABEL = {"minor": "轻微", "major": "严重", "critical": "致命"}

FLAGS = [
    ("has_subtitle_or_caption", "字幕"),
    ("has_watermark_or_logo", "水印 / Logo"),
    ("has_brand_name", "品牌名"),
]


def _video_href(clip: str, html_path: Path | None, video_base: str | None) -> str:
    """片段 mp4 的引用地址。

    两种部署形态用两套地址，所以这里是参数而不是常量：
    * 单文件报告 —— 相对路径，报告和 `videos/` 一起搬走才有效。
    * 服务 —— `/media/<run>/<file>`，由服务端做 Range 流式返回。
    """
    target = Path(clip)
    if video_base is not None:
        return f"{video_base.rstrip('/')}/{target.name}"
    if html_path is None:
        return target.as_uri()
    try:
        rel = os.path.relpath(target, html_path.parent)
    except ValueError:
        return target.as_uri()
    # Windows 反斜杠在 href 里不合法；这台机器是 linux，但报告可能被拷走
    return rel.replace(os.sep, "/")


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 2) if values else None


def summarize_optimization(result: dict[str, Any]) -> dict[str, Any] | None:
    """把逐片段归因压成视频级优化总览；没有归因数据时返回 None。"""
    optimization = result.get("optimization") or {}
    scenes = optimization.get("scenes") or []
    if not scenes:
        return None
    stage_problems = {stage: 0 for stage in ("script_to_plan", "plan_to_prompt", "prompt_to_video")}
    stage_unassessable = dict.fromkeys(stage_problems, 0)
    statuses: dict[str, int] = {}
    valid = rejected = actions = ungrounded = 0
    for scene in scenes:
        status = str(scene.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        valid += len(scene.get("valid_patches") or [])
        rejected += len(scene.get("rejected_patches") or [])
        actions += len(scene.get("non_prompt_actions") or [])
        for finding in scene.get("findings") or []:
            stage = finding.get("stage")
            if stage not in stage_problems:
                continue
            if finding.get("status") == "problem":
                stage_problems[stage] += 1
            elif finding.get("status") == "unassessable":
                stage_unassessable[stage] += 1
            if finding.get("quote_grounded") is False:
                ungrounded += 1
    return {
        "model": optimization.get("model"),
        "scene_count": len(scenes),
        "statuses": statuses,
        "stage_problems": stage_problems,
        "stage_unassessable": stage_unassessable,
        "valid_patches": valid,
        "rejected_patches": rejected,
        "non_prompt_actions": actions,
        "ungrounded_findings": ungrounded,
        "errors": len(optimization.get("errors") or []),
    }


def summarize(result: dict[str, Any], *, run_root: str | Path | None = None) -> dict[str, Any]:
    """一份结果的缩略：每个维度的跨片段中位分 + 扣分命中 + 渲染吻合率。

    服务首页要拿它横向对比多个口径，报告页的汇总条也是同一套数——
    两处各算一遍迟早会因为「无参照要不要计入」这类细节而对不上。

    *run_root* is optional; when provided it is used by
    :func:`infer_generation` so the service can pass ``run.root`` reliably.
    """
    scenes = result.get("scenes") or []
    dims = {}
    for d in DIMENSIONS:
        key = d["key"]
        rated = [s for s in scenes
                 if s.get("dims", {}).get(key) and key not in (s.get("unreferenced") or [])]
        values = [s["dims"][key]["median"] for s in rated]
        dims[key] = {
            "mean": _mean(values),
            "median": _median(values),
            "lowest": min(values, default=None),
            "hits": sum(s.get("hits", {}).get(key, 0) for s in rated),
            "judged": sum(s.get("judged", 0) for s in rated),
            "unreferenced": len(scenes) - len(rated),
        }
    rm = [s["render_match"] for s in scenes if s.get("render_match")]
    return {
        "dims": dims,
        "scene_count": len(scenes),
        "deductions": sum(sum(s.get("deduction_count") or []) for s in scenes),
        "errors": sum(len(s.get("errors") or []) for s in scenes),
        "render": {
            "judged": sum(r["judged"] for r in rm),
            "matched": sum(r["matched"] for r in rm),
        } if rm else None,
        "overall": score_result(result),
        "optimization": summarize_optimization(result),
        "generation": infer_generation(result, run_root=run_root),
    }


def build_payload(
    result: dict[str, Any],
    *,
    html_path: Path | None = None,
    video_base: str | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """把跑批结果收敛成前端要的形状。

    这里只做**重排和补齐**，不做任何再计算：分数、命中数、极差全部沿用
    `bench._aggregate` 算好的值。报告里出现一个跑批时不存在的数字，
    就意味着报告和 JSON 会在某天对不上。
    """
    clips = {t["key"]: t for t in result.get("tasks") or []}
    scenes = []
    for scene in result.get("scenes") or []:
        task = clips.get(scene["key"], {})
        scenes.append({
            **scene,
            "video": _video_href(task.get("clip", ""), html_path, video_base),
            "duration": task.get("duration"),
        })
    gen = infer_generation(result, run_root=run_root)
    return {
        "meta": {
            "judge_model": result.get("judge_model"),
            "thinking_level": result.get("thinking_level"),
            "prompt_version_tag": result.get("prompt_version_tag"),
            "prompts_path": result.get("prompts_path"),
            "repeats": result.get("repeats"),
            "elapsed": result.get("elapsed"),
            "design_source": result.get("design_source"),
            "config": result.get("config") or {},
        },
        "generation": gen,
        "dimensions": DIMENSIONS,
        "severity_label": SEVERITY_LABEL,
        "flags": FLAGS,
        "scenes": scenes,
        "optimization": result.get("optimization"),
        "optimization_summary": summarize_optimization(result),
        "overall": score_result(result),
    }


def render_html(payload: dict[str, Any], title: str) -> str:
    """把 payload 塞进模板。

    `</script>` 出现在数据里会提前关掉标签——评审文本是模型生成的，
    谁也不能保证它不会引用一段 HTML。
    """
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__TITLE__", title).replace("__DATA__", blob)


def write_report(result: dict[str, Any], out_path: Path, title: str = "") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(result, html_path=out_path)
    out_path.write_text(
        render_html(payload, title or "逐 shot 评估报告"), encoding="utf-8"
    )
    return out_path


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<meta name="description" content="AI 生成视频的逐 shot 评估报告：六维评分、扣分项明细、渲染模式盲判与全部参照材料。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --ink:hsl(35 20% 92%);
  --ink-2:hsl(35 12% 72%);
  --ink-3:hsl(35 9% 52%);
  --bg:hsl(30 8% 9%);
  --surface:hsl(30 7% 12%);
  --surface-2:hsl(30 7% 15%);
  --line:hsl(30 6% 21%);
  --line-soft:hsl(30 6% 17%);
  --accent:hsl(32 68% 58%);
  --accent-dim:hsl(32 40% 30%);
  --good:hsl(152 40% 48%);
  --warn:hsl(40 70% 55%);
  --bad:hsl(6 62% 56%);
  --radius:10px;
  --stick:118px;            /* 顶栏+汇总条高度，载入后由 JS 量准 */
  --sans:"Inter","Noto Sans SC",-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font-family:var(--sans);font-size:14px;line-height:1.6;
  -webkit-font-smoothing:antialiased;
  /* 极淡的纸纹，避免大片死平的深色 */
  background-image:radial-gradient(hsl(30 10% 14%) 0.5px,transparent 0.5px);
  background-size:22px 22px;
}
h1,h2,h3{margin:0;font-weight:600;letter-spacing:-0.01em}
a{color:var(--accent)}
button{font:inherit;color:inherit;background:none;border:none;cursor:pointer}
code,.mono{font-family:var(--mono);font-size:0.92em;font-variant-numeric:tabular-nums}

/* ---------- 顶栏 ---------- */
.topbar{
  position:sticky;top:0;z-index:30;
  display:flex;gap:20px;align-items:baseline;flex-wrap:wrap;
  padding:14px 24px;border-bottom:1px solid var(--line);
  background:hsl(30 8% 9% / .88);backdrop-filter:blur(12px);
}
.topbar h1{font-size:15px;letter-spacing:.01em}
.topbar h1 span{color:var(--ink-3);font-weight:400;margin-left:8px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.chip{
  padding:3px 9px;border:1px solid var(--line);border-radius:999px;
  font-size:11.5px;color:var(--ink-2);white-space:nowrap;
}
.chip b{color:var(--ink);font-weight:500}

/* ---------- 汇总条 ---------- */
.summary{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
  gap:1px;background:var(--line-soft);
  border-bottom:1px solid var(--line);
}
.sumcell{
  background:var(--surface);padding:12px 16px;text-align:left;
  transition:background .15s ease;position:relative;
}
.sumcell:hover{background:var(--surface-2)}
.sumcell.active{background:var(--surface-2)}
.sumcell.active::after{
  content:"";position:absolute;left:0;right:0;bottom:0;height:2px;background:var(--accent);
}
.sumcell .lab{font-size:11.5px;color:var(--ink-3);display:flex;gap:6px;align-items:center}
.sumcell .val{font-size:22px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.sumcell .sub{font-size:11px;color:var(--ink-3)}
.wt{font-size:10px;color:var(--ink-3);border:1px solid var(--line);border-radius:4px;padding:0 4px}

/* ---------- 主体 ---------- */
.layout{display:grid;grid-template-columns:minmax(260px,318px) 1fr;min-height:calc(100vh - var(--stick))}
@media (max-width:980px){.layout{grid-template-columns:1fr}.rail{max-height:340px}}

.rail{
  border-right:1px solid var(--line);overflow-y:auto;
  position:sticky;top:var(--stick);max-height:calc(100vh - var(--stick));
}
.railhead{
  display:flex;align-items:center;gap:8px;padding:10px 16px;
  border-bottom:1px solid var(--line-soft);color:var(--ink-3);font-size:11.5px;
  position:sticky;top:0;background:var(--surface);z-index:2;
}
.railhead select{
  margin-left:auto;background:var(--surface-2);color:var(--ink-2);
  border:1px solid var(--line);border-radius:6px;padding:3px 6px;font-size:11.5px;
}
.row{
  display:block;width:100%;text-align:left;padding:10px 16px;
  border-bottom:1px solid var(--line-soft);border-left:2px solid transparent;
  transition:background .12s ease,border-color .12s ease;
}
.row:hover{background:var(--surface)}
.row.on{background:var(--surface-2);border-left-color:var(--accent)}
.row.dim{opacity:.32}
.rowtop{display:flex;align-items:baseline;gap:8px}
.rowid{font-family:var(--mono);font-size:13px;font-weight:500}
.rowmeta{margin-left:auto;font-size:11px;color:var(--ink-3)}
.bars{display:flex;gap:3px;margin-top:7px}
.bar{height:4px;flex:1;border-radius:2px;background:var(--line)}
.bar i{display:block;height:100%;border-radius:2px}
.rowtags{display:flex;gap:5px;margin-top:6px;flex-wrap:wrap}
.tag{font-size:10.5px;padding:1px 6px;border-radius:4px;background:var(--surface-2);color:var(--ink-3);border:1px solid var(--line)}
.tag.hot{color:var(--bad);border-color:hsl(6 40% 30%)}
.tag.cg{color:var(--ink-2)}

/* ---------- 详情 ---------- */
.detail{padding:0 0 80px}
.vidwrap{
  position:sticky;top:var(--stick);z-index:10;background:var(--bg);
  padding:20px 28px 14px;border-bottom:1px solid var(--line);
}
.vidgrid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:24px;align-items:start}
@media (max-width:1180px){.vidgrid{grid-template-columns:1fr}}
video{
  width:100%;border-radius:var(--radius);background:#000;display:block;
  border:1px solid var(--line);
}
.vidside h2{font-size:19px;font-family:var(--mono);letter-spacing:-.01em}
.vidside .ttl{color:var(--ink-2);font-size:13px;margin-top:2px}
.scorerow{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line-soft);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;margin-top:14px}
.score{background:var(--surface);padding:9px 11px}
.score .k{font-size:11px;color:var(--ink-3)}
.score .v{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.score .sp{font-size:11px;color:var(--ink-3);margin-left:3px;font-weight:400}
.score.unref .v{font-size:13px;color:var(--ink-3);font-weight:500}
.dots{display:flex;gap:3px;margin-top:4px}
.dot{width:5px;height:5px;border-radius:50%}

section{padding:22px 28px;border-bottom:1px solid var(--line-soft)}
section > h3{
  font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);
  margin-bottom:12px;font-weight:500;
}
.note{color:var(--ink-3);font-size:12.5px;margin-top:10px}

/* 扣分项 */
.ded{border-left:2px solid var(--line);padding:0 0 0 14px;margin-bottom:16px}
.ded.critical{border-left-color:var(--bad)}
.ded.major{border-left-color:var(--warn)}
.ded.minor{border-left-color:var(--accent-dim)}
.dedhead{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.sev{font-size:11px;padding:1px 7px;border-radius:4px;font-weight:500}
.sev.critical{background:hsl(6 45% 20%);color:hsl(6 70% 72%)}
.sev.major{background:hsl(40 45% 18%);color:hsl(40 75% 68%)}
.sev.minor{background:var(--surface-2);color:var(--ink-2)}
.dimname{font-size:12px;color:var(--ink-2)}
.pts{font-family:var(--mono);font-size:12px;color:var(--bad)}
.seek{
  font-family:var(--mono);font-size:12px;color:var(--accent);
  border-bottom:1px dashed var(--accent-dim);padding-bottom:1px;
}
.seek:hover{border-bottom-style:solid}
.seek.bad{color:var(--warn);border-bottom-color:hsl(40 40% 32%)}
.seek[disabled]{color:var(--ink-3);border:none;cursor:default}
.ded p{margin:4px 0 0}
.ded .exp{color:var(--ink-3);font-size:13px}
.roundlabel{font-size:11.5px;color:var(--ink-3);margin:14px 0 8px;display:flex;align-items:center;gap:8px}
.roundlabel::after{content:"";flex:1;height:1px;background:var(--line-soft)}

/* 参照材料 */
details{border-top:1px solid var(--line-soft)}
details:first-of-type{border-top:none}
summary{
  padding:10px 0;cursor:pointer;font-size:13px;color:var(--ink-2);
  display:flex;align-items:center;gap:8px;list-style:none;
}
summary::-webkit-details-marker{display:none}
summary::before{content:"+";font-family:var(--mono);color:var(--ink-3);width:12px}
details[open] > summary::before{content:"−"}
summary:hover{color:var(--ink)}
.body{
  padding:2px 0 16px 20px;white-space:pre-wrap;word-break:break-word;
  font-size:13px;color:var(--ink-2);line-height:1.75;
}
.body.mono{font-family:var(--mono);font-size:12px;line-height:1.7}

.pills{display:flex;gap:7px;flex-wrap:wrap}
.pill{font-size:11.5px;padding:3px 10px;border-radius:999px;border:1px solid var(--line);color:var(--ink-3)}
.pill.bad{border-color:hsl(6 40% 32%);color:hsl(6 65% 70%);background:hsl(6 40% 14%)}
.pill.ok{color:var(--ink-3)}

table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:7px 12px 7px 0;text-align:left;vertical-align:top;border-bottom:1px solid var(--line-soft)}
th{font-size:11.5px;color:var(--ink-3);font-weight:500}
tbody tr:last-child td{border-bottom:none}
.obs{font-size:13.5px;line-height:1.7}
.reasons{display:grid;gap:8px;margin-top:10px}
.reason{display:grid;grid-template-columns:76px 1fr;gap:12px;font-size:13px}
.reason b{color:var(--ink-3);font-weight:500;font-size:12px;padding-top:1px}
.empty{color:var(--ink-3);font-size:13px}
kbd{
  font-family:var(--mono);font-size:11px;border:1px solid var(--line);
  border-bottom-width:2px;border-radius:4px;padding:1px 5px;color:var(--ink-3);
}
</style>
</head>
<body>
<script id="eval-data" type="application/json">__DATA__</script>

<header class="topbar">
  <h1>逐 shot 评估<span id="runname"></span></h1>
  <div class="chips" id="metachips"></div>
</header>

<div class="summary" id="summary"></div>

<div class="layout">
  <nav class="rail" id="rail" aria-label="片段列表">
    <div class="railhead">
      <span id="railcount"></span>
      <select id="sort" aria-label="排序">
        <option value="order">按顺序</option>
        <option value="worst">最差优先</option>
        <option value="deductions">扣分项最多</option>
      </select>
    </div>
    <div id="rows"></div>
  </nav>
  <main class="detail" id="detail"></main>
</div>

<script>
const DATA = JSON.parse(document.getElementById('eval-data').textContent);
const DIMS = DATA.dimensions, SCENES = DATA.scenes, META = DATA.meta;
const SEV = DATA.severity_label;
const TOTAL_W = DIMS.reduce((a, d) => a + d.weight, 0);
const OVERALL = DATA.overall || {};
const OPT_SUMMARY = DATA.optimization_summary || null;
const GEN = DATA.generation || {};

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* 0→红 5→绿 连续色标。分档着色会在 3.4/3.5 这种噪声区间里制造假的「等级差」。*/
function scoreColor(v){
  if (v == null) return 'var(--ink-3)';
  const h = Math.max(0, Math.min(1, v / 5)) * 148 + 4;   // 4°红 → 152°绿
  return `hsl(${h} 45% 52%)`;
}
function median(xs){
  if (!xs.length) return null;
  const s = [...xs].sort((a,b)=>a-b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
}
function mean(xs){
  return xs.length ? xs.reduce((a,b)=>a+b, 0) / xs.length : null;
}
/* 排序用的加权综合。无参照的维度退出计算并按比例重算分母——
   拿 0 分进平均会把「没东西可对照」说成「画面很差」。*/
function composite(scene){
  let sum = 0, w = 0;
  for (const d of DIMS){
    const stat = scene.dims[d.key];
    if (!stat || scene.unreferenced.includes(d.key)) continue;
    sum += stat.median * d.weight; w += d.weight;
  }
  return w ? sum / w : null;
}
function parseTs(ts){
  const m = /(\d{1,2}):(\d{2}(?:\.\d+)?)/.exec(ts || '');
  if (m) return +m[1] * 60 + parseFloat(m[2]);
  const s = /(\d+(?:\.\d+)?)\s*s/i.exec(ts || '');
  return s ? parseFloat(s[1]) : null;
}

/* ---------- 顶部 ---------- */
document.getElementById('runname').textContent =
  SCENES.length ? ' · ' + SCENES[0].source : '';
document.getElementById('metachips').innerHTML = [
  ['视频模型', GEN.video_model && GEN.video_model !== 'unknown' ? GEN.video_model : (GEN.video_provider && GEN.video_provider !== 'unknown' ? GEN.video_provider : '未知')],
  GEN.video_provider && GEN.video_provider !== 'unknown' && GEN.video_model && GEN.video_model !== 'unknown' && GEN.video_provider !== GEN.video_model ? ['视频 Provider', GEN.video_provider] : null,
  GEN.prompt_model && GEN.prompt_model !== 'unknown' ? ['Prompt LLM', GEN.prompt_model] : null,
  GEN.planner_model && GEN.planner_model !== 'unknown' ? ['Planner LLM', GEN.planner_model] : null,
  GEN.granularity && GEN.granularity !== 'unknown' ? ['粒度', GEN.granularity] : null,
  ['评审模型', META.judge_model],
  ['判据', META.prompt_version_tag],
  ['轮次', META.repeats + ' 轮'],
  ['对齐参照', META.config.beat_source],
  ['风格参照', META.config.style_source],
  ['渲染盲判', META.config.render_source],
  ['分镜口径', META.design_source || 'generated'],
  ['用时', META.elapsed + 's'],
].filter(Boolean).map(([k, v]) => `<span class="chip">${esc(k)} <b>${esc(v)}</b></span>`).join('');

/* ---------- 汇总条 ---------- */
let filterDim = null;
function renderSummary(){
  const overallCell = OVERALL.score != null ? `<div class="sumcell" style="border-right:2px solid var(--line)">
    <div class="lab">Overall <span class="wt">${OVERALL.covered_weight}/${OVERALL.max_weight}</span></div>
    <div class="val" style="color:${scoreColor(OVERALL.score / 20)};font-size:28px">${OVERALL.score} <span style="font-size:18px;color:var(--ink-2)">${OVERALL.grade}</span></div>
    <div class="sub">${(OVERALL.severity||{}).critical||0} critical</div>
  </div>` : '';
  const optCell = OPT_SUMMARY ? `<div class="sumcell" style="border-right:2px solid var(--line)">
    <div class="lab">优化归因 <span class="wt">${OPT_SUMMARY.scene_count} 段</span></div>
    <div class="val" style="font-size:24px">${Object.values(OPT_SUMMARY.stage_problems||{}).reduce((a,b)=>a+b,0)} 问题</div>
    <div class="sub">稿→分 ${OPT_SUMMARY.stage_problems.script_to_plan||0} · 分→P ${OPT_SUMMARY.stage_problems.plan_to_prompt||0} · P→片 ${OPT_SUMMARY.stage_problems.prompt_to_video||0}<br>${OPT_SUMMARY.valid_patches} patch · ${OPT_SUMMARY.non_prompt_actions} 非Prompt</div>
  </div>` : '';
  const cells = DIMS.map(d => {
    const rated = SCENES.filter(s => s.dims[d.key] && !s.unreferenced.includes(d.key));
    const values = rated.map(s => s.dims[d.key].median);
    const avg = mean(values), med = median(values);
    const hits = rated.reduce((a, s) => a + (s.hits[d.key] || 0), 0);
    const judged = rated.reduce((a, s) => a + s.judged, 0);
    const skipped = SCENES.length - rated.length;
    return `<button class="sumcell${filterDim === d.key ? ' active' : ''}" data-dim="${d.key}"
              title="主值为跨片段算术均值；点击只看这一维度被扣到的片段">
      <div class="lab">${esc(d.label)}均值 <span class="wt">权重 ${d.weight}</span></div>
      <div class="val" style="color:${scoreColor(avg)}">${avg == null ? '—' : avg.toFixed(2)}</div>
      <div class="sub">中位 ${med == null ? '—' : med.toFixed(2)} · 扣到 ${hits}/${judged} 段次${skipped ? ` · ${skipped} 段无参照` : ''}</div>
    </button>`;
  });
  const rm = SCENES.filter(s => s.render_match);
  if (rm.length){
    const judged = rm.reduce((a, s) => a + s.render_match.judged, 0);
    const matched = rm.reduce((a, s) => a + s.render_match.matched, 0);
    const pct = judged ? Math.round(matched / judged * 100) : 0;
    cells.push(`<div class="sumcell">
      <div class="lab">渲染模式盲判</div>
      <div class="val" style="color:${pct === 100 ? 'var(--good)' : 'var(--warn)'}">${pct}%</div>
      <div class="sub">吻合 ${matched}/${judged} 段次</div>
    </div>`);
  }
  document.getElementById('summary').innerHTML = overallCell + optCell + cells.join('');
  document.querySelectorAll('.sumcell[data-dim]').forEach(el =>
    el.onclick = () => { filterDim = filterDim === el.dataset.dim ? null : el.dataset.dim; renderSummary(); renderRail(); });
}

/* ---------- 左栏 ---------- */
let current = 0;
function visible(){
  let list = SCENES.map((s, i) => ({ s, i }));
  if (filterDim) list = list.filter(({s}) => (s.hits[filterDim] || 0) > 0);
  const mode = document.getElementById('sort').value;
  if (mode === 'worst') list.sort((a, b) => (composite(a.s) ?? 9) - (composite(b.s) ?? 9));
  if (mode === 'deductions') list.sort((a, b) =>
    b.s.deduction_count.reduce((x,y)=>x+y,0) - a.s.deduction_count.reduce((x,y)=>x+y,0));
  return list;
}
function renderRail(){
  const list = visible();
  document.getElementById('railcount').textContent =
    filterDim ? `${list.length} / ${SCENES.length} 段（${DIMS.find(d=>d.key===filterDim).label}被扣到）`
              : `${SCENES.length} 个片段`;
  document.getElementById('rows').innerHTML = list.map(({ s, i }) => {
    const bars = DIMS.map(d => {
      const st = s.dims[d.key], un = s.unreferenced.includes(d.key);
      const pct = st && !un ? st.median / 5 * 100 : 0;
      return `<span class="bar" title="${esc(d.label)} ${un ? '无参照' : (st ? st.median : '—')}">
                <i style="width:${pct}%;background:${un ? 'var(--line)' : scoreColor(st && st.median)}"></i></span>`;
    }).join('');
    const totalDed = s.deduction_count.reduce((a,b)=>a+b,0);
    const mismatch = s.render_match && s.render_match.matched < s.render_match.judged;
    return `<button class="row${i === current ? ' on' : ''}" data-i="${i}">
      <div class="rowtop">
        <span class="rowid">${esc(s.scene_id)}</span>
        <span class="rowmeta">${s.duration ? s.duration.toFixed(0) + 's' : ''}</span>
      </div>
      <div class="bars">${bars}</div>
      <div class="rowtags">
        <span class="tag cg">${esc(s.render_declared || '?')}</span>
        ${mismatch ? '<span class="tag hot">渲染不符</span>' : ''}
        <span class="tag">${totalDed} 条扣分</span>
        ${s.errors.length ? '<span class="tag hot">评审失败</span>' : ''}
      </div>
    </button>`;
  }).join('');
  document.querySelectorAll('.row').forEach(el =>
    el.onclick = () => select(+el.dataset.i));
}

/* ---------- 右栏 ---------- */
function select(i){
  current = i;
  renderRail();
  renderDetail(SCENES[i]);
  document.querySelector('.row.on')?.scrollIntoView({ block: 'nearest' });
}

function seekBtn(ts, dur){
  const t = parseTs(ts);
  if (t == null) return `<span class="seek" disabled>${esc(ts || '全段')}</span>`;
  // 模型偶尔给出超出片段长度的时间点（3 秒的片段报 `01:05-03:00`，实测 3.1-pro
  // 上不少见）。照着跳过去只会停在片尾，看上去像播放器坏了——所以夹住并标出来，
  // 把「模型报错了时间」和「播放器不好使」区分开。
  if (dur && t > dur + 0.5){
    const safe = Math.max(0, dur - 0.5);
    return `<button class="seek bad" data-seek="${safe}"
      title="模型给的时间点超出片段长度（${dur.toFixed(1)}s），已跳到接近片尾处">${esc(ts)} ⚠</button>`;
  }
  return `<button class="seek" data-seek="${t}">${esc(ts)}</button>`;
}

function renderDetail(s){
  const ctx = s.context || {};
  const rounds = s.rounds.filter(r => !r.error);
  const beats = ctx.beats || [];

  const scores = DIMS.map(d => {
    const st = s.dims[d.key], un = s.unreferenced.includes(d.key);
    if (!st) return `<div class="score unref"><div class="k">${esc(d.label)}</div><div class="v">未评</div></div>`;
    if (un) return `<div class="score unref"><div class="k">${esc(d.label)}</div><div class="v">无参照</div>
      <div class="sub" style="font-size:11px;color:var(--ink-3)">分镜里没有可比对的东西</div></div>`;
    const dots = st.values.map((v, n) =>
      `<span class="dot" style="background:${scoreColor(v)}" title="第 ${n + 1} 轮 ${v}"></span>`).join('');
    return `<div class="score">
      <div class="k">${esc(d.label)}</div>
      <div class="v" style="color:${scoreColor(st.median)}">${st.median.toFixed(1)}${
        st.spread ? `<span class="sp">±${st.spread.toFixed(1)}</span>` : ''}</div>
      <div class="dots">${dots}</div>
    </div>`;
  }).join('');

  // 渲染模式：声明 vs 每轮盲判
  const rv = rounds.filter(r => r.render_verdict);
  const renderBlock = rv.length ? `
    <section>
      <h3>渲染模式盲判</h3>
      <table><thead><tr><th style="width:70px">轮次</th><th style="width:110px">盲判</th>
        <th style="width:80px">置信度</th><th>依据</th></tr></thead><tbody>
      ${rv.map((r, n) => `<tr>
        <td class="mono">${n + 1}</td>
        <td><span class="pill ${r.render_match ? 'ok' : 'bad'}">${esc(r.render_verdict)}</span></td>
        <td class="mono">${(r.render_confidence ?? 0).toFixed(2)}</td>
        <td>${esc(r.render_reason)}</td></tr>`).join('')}
      </tbody></table>
      <p class="note">声明的是 <code>${esc(s.render_declared || '未声明')}</code>。
        盲判是<b>独立的第二次请求</b>，不给任何分镜信息——给了声明值模型会被锚死。</p>
    </section>` : '';

  // 扣分项
  const dedBlock = rounds.some(r => (r.deductions || []).length) ? rounds.map((r, n) => {
    const items = r.deductions || [];
    const head = rounds.length > 1 ? `<div class="roundlabel">第 ${n + 1} / ${rounds.length} 轮${
      items.length ? '' : ' · 无扣分项'}</div>` : '';
    return head + items.map(d => {
      const dim = DIMS.find(x => x.key === d.dimension);
      return `<div class="ded ${esc(d.severity)}">
        <div class="dedhead">
          <span class="sev ${esc(d.severity)}">${esc(SEV[d.severity] || d.severity)}</span>
          <span class="dimname">${esc(dim ? dim.label : d.dimension)}</span>
          <span class="pts">−${d.points}</span>
          ${seekBtn(d.timestamp, s.duration)}
        </div>
        <p>${esc(d.evidence)}</p>
        <p class="exp">期望：${esc(d.expected)}</p>
      </div>`;
    }).join('');
  }).join('') : '<p class="empty">这一段三轮都没有挑出扣分项。</p>';

  // 每轮观察 + 合规
  const obsBlock = rounds.map((r, n) => {
    const flags = DATA.flags.map(([k, label]) =>
      `<span class="pill ${r[k] ? 'bad' : 'ok'}">${r[k] ? '⚠ 出现' : '无'}${esc(label)}</span>`).join('');
    const texts = (r.visible_text || []).length
      ? `<p class="note">画面文字：${(r.visible_text).map(t => `<code>${esc(t)}</code>`).join(' ')}</p>` : '';
    const issues = (r.issues || []).length
      ? `<p class="note">其他问题：${(r.issues).map(esc).join('；')}</p>` : '';
    return `<details ${n === 0 ? 'open' : ''}>
      <summary>第 ${n + 1} 轮 · 模型看到了什么</summary>
      <div class="body">
        <p class="obs">${esc(r.observed)}</p>
        <div class="reasons">
          ${DIMS.map(d => r[d.key + '_reason'] ? `<div class="reason">
            <b>${esc(d.label)}</b><span>${esc(r[d.key + '_reason'])}</span></div>` : '').join('')}
        </div>
        <div class="pills" style="margin-top:12px">${flags}</div>
        ${texts}${issues}
      </div>
    </details>`;
  }).join('');

  document.getElementById('detail').innerHTML = `
    <div class="vidwrap">
      <div class="vidgrid">
        <video id="player" src="${esc(s.video)}" controls preload="metadata" playsinline></video>
        <div class="vidside">
          <h2>${esc(s.scene_id)}</h2>
          <div class="ttl">${esc(ctx.shot_type || '')}${ctx.environment ? ' · ' + esc(ctx.environment) : ''}</div>
          <div class="scorerow">${scores}</div>
        </div>
      </div>
    </div>

    <section>
      <h3>本段该演什么</h3>
      ${beats.map(b => `
        <div class="reason" style="grid-template-columns:86px 1fr">
          <b class="mono">${b.start.toFixed(1)}–${(b.start + b.duration).toFixed(1)}s</b>
          <div>
            <div>${esc(b.visual_beat) || '<span class="empty">（无 visual_beat）</span>'}</div>
            <div class="note" style="margin-top:2px">口播：${esc(b.script) || '（无）'}</div>
          </div>
        </div>`).join('') || '<p class="empty">（无参照物）</p>'}
    </section>

    <section>
      <h3>扣分项${rounds.length > 1 ? '（逐轮列全：同一个问题在不同轮可能定级不同）' : ''}</h3>
      ${dedBlock}
    </section>

    ${renderBlock}

    <section>
      <h3>评审判词</h3>
      ${obsBlock}
      ${s.errors.length ? `<p class="note" style="color:var(--bad)">评审失败：${esc(s.errors[0])}</p>` : ''}
    </section>

    <section>
      <h3>参照材料</h3>
      <details><summary>scene_design（判 design_fidelity 的对照物）</summary>
        <div class="body mono">${esc(ctx.scene_design)}</div></details>
      <details><summary>送进视频模型的完整 prompt</summary>
        <div class="body mono">${esc(ctx.prompt)}</div></details>
      ${ctx.provenance ? `<details><summary>链路溯源 (provenance)</summary>
        <div class="body mono">${esc(JSON.stringify(ctx.provenance, null, 2))}</div></details>` : ''}
    </section>

    ${renderOptimization(s.scene_id)}
  `;

  const player = document.getElementById('player');
  document.querySelectorAll('[data-seek]').forEach(el => el.onclick = () => {
    player.currentTime = +el.dataset.seek;
    player.play();
  });
}

/* ---------- Optimization rendering ---------- */
const OPT = DATA.optimization;
const STAGE_LABEL = {script_to_plan: '脚本→分镜', plan_to_prompt: '分镜→Prompt', prompt_to_video: 'Prompt→视频'};
function renderOptimization(sceneId){
  if (!OPT || !OPT.scenes || !OPT.scenes.length) return '';
  const opt = OPT.scenes.find(s => s.scene_id === sceneId);
  if (!opt) return '';
  if (opt.status === 'unassessable' || opt.status === 'skip')
    return `<section><h3>链路归因 + Prompt 优化</h3><p class="empty">${esc(opt.status)}: ${esc(opt.reason || '')}</p></section>`;
  if (opt.status === 'error')
    return `<section><h3>链路归因 + Prompt 优化</h3><p class="note" style="color:var(--bad)">分析失败: ${esc(opt.error || '')}</p></section>`;

  const findingsHtml = (opt.findings || []).map(f => {
    const grounded = f.quote_grounded ? '✓' : '✗';
    return `<div class="ded ${f.severity === 'high' ? 'critical' : f.severity === 'medium' ? 'major' : 'minor'}">
      <div class="dedhead">
        <span class="sev ${f.severity === 'high' ? 'critical' : f.severity === 'medium' ? 'major' : 'minor'}">${esc(f.severity)}</span>
        <span class="dimname">${esc(STAGE_LABEL[f.stage] || f.stage)} · ${esc(f.status)}</span>
        <span class="mono">[${grounded} quote · ${Number(f.confidence ?? 0).toFixed(2)}]</span>
      </div>
      <p>${esc(f.problem)}</p>
      <p class="exp">根因: ${esc(f.root_cause)}</p>
      <p class="exp">引用 (${esc(f.source_field)}): <code>${esc(f.exact_quote)}</code></p>
    </div>`;
  }).join('');

  const patchesHtml = (opt.valid_patches || []).map(p =>
    `<div class="ded minor"><div class="dedhead"><span class="sev minor">${esc(p.op)}</span>
      <span class="dimname">${esc(p.target)} · confidence ${Number(p.confidence ?? 0).toFixed(2)}</span></div>
      <p>${p.anchor ? `<code>${esc(p.anchor)}</code> → ` : ''}${p.content ? `<code>${esc(p.content)}</code>` : '(delete)'}</p>
      <p class="exp">${esc(p.reason || '')}</p></div>`
  ).join('');

  const rejectedHtml = (opt.rejected_patches || []).map(p =>
    `<div class="ded" style="border-left-color:var(--bad);opacity:.6"><div class="dedhead">
      <span class="sev critical">⚠ rejected</span>
      <span class="dimname">${esc(p.target)}.${esc(p.op)}</span></div>
      <p class="exp">${esc(p.rejection_reason || '')}</p></div>`
  ).join('');

  const actionsHtml = (opt.non_prompt_actions || []).map(a =>
    `<div class="reason"><b>${esc(a.action)}</b><span>${esc(a.reason)}</span></div>`
  ).join('');

  return `<section>
    <h3>链路归因 + Prompt 优化</h3>
    ${findingsHtml || '<p class="empty">无归因发现</p>'}
    ${patchesHtml ? '<h3 style="margin-top:16px">有效修改建议</h3>' + patchesHtml : ''}
    ${rejectedHtml ? '<h3 style="margin-top:16px">⚠ 被拒绝的修改（锚定验证失败）</h3>' + rejectedHtml : ''}
    ${actionsHtml ? '<h3 style="margin-top:16px">非 Prompt 操作建议</h3><div class="reasons">' + actionsHtml + '</div>' : ''}
  </section>`;
}

/* ---------- 交互 ---------- */
document.getElementById('sort').onchange = renderRail;
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey) return;
  const list = visible();
  if (!list.length) return;                       // 筛空了，没有可跳的下一段
  const pos = list.findIndex(x => x.i === current);
  if (e.key === 'j' || e.key === 'ArrowDown'){ e.preventDefault(); select(list[Math.min(pos + 1, list.length - 1)].i); }
  if (e.key === 'k' || e.key === 'ArrowUp'){ e.preventDefault(); select(list[Math.max(pos - 1, 0)].i); }
  if (e.key === ' '){ const p = document.getElementById('player'); if (p){ e.preventDefault(); p.paused ? p.play() : p.pause(); } }
});

/* 左栏和播放器都吸附在顶栏下方。顶栏的 chip 会在窄窗口里折行，
   高度写死就会露出一条缝或盖住一行，所以量出来写进 CSS 变量。*/
function measure(){
  const h = document.querySelector('.topbar').offsetHeight
          + document.getElementById('summary').offsetHeight;
  document.documentElement.style.setProperty('--stick', h + 'px');
}
addEventListener('resize', measure);

renderSummary();
renderRail();
measure();
if (SCENES.length) select(0);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shot_eval.report",
        description="把 shot_eval 的结果 JSON 渲染成自包含 HTML 报告。",
    )
    parser.add_argument("result", help="bench 落盘的 shots-*.json")
    parser.add_argument("-o", "--out", default="", help="输出路径（默认与 JSON 同目录、同名 .html）")
    args = parser.parse_args(argv)

    src = Path(args.result).expanduser().resolve()
    result = json.loads(src.read_text(encoding="utf-8"))
    out = Path(args.out).expanduser() if args.out else src.with_suffix(".html")
    written = write_report(result, out, title=f"逐 shot 评估 · {src.stem}")
    print(f"报告 → {written}")
    print(f"视频按相对路径引用，报告必须和 videos/ 保持当前相对位置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
