"""多模态评审：让模型看视频，判它和分镜说好的是不是一回事。

只做「看画面」这一件事 —— 不听音轨、不做 OCR。评的是片段相对分镜的还原度：
画面在不在做 visual_beat 说的事、渲染模式对不对、风格是否一致、
有没有出现禁止的叠加物，以及作为科普内容准不准、看不看得懂、有没有帮助。

保留逐片段多模态评审，删除整片评估相关的两路
（`judge_narration` / `judge_text_rules`）和对 `EvalCase` 的 import。
`judge_scene()` 的签名一个字没动——上游早就把它和 EvalCase 解开了，
入参只要一个 `style_lock` 字符串，这正是抽取的接缝。
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Sequence
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

from shot_eval import config, prompts
from shot_eval.models import (
    FULL_MARK,
    SCENE_DIMENSIONS,
    SCIENCE_DIMENSIONS,
    SEVERITY_POINTS,
    RenderVerdict,
    SceneSlice,
    SceneVerdict,
    SceneVerdictNoStyle,
)

logger = logging.getLogger("shot_eval.judge")

MAX_CONCURRENCY = 3
_client: genai.Client | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> genai.Client:
    global _client
    async with _client_lock:
        if _client is None:
            _client = genai.Client(
                vertexai=True,
                project=config.vertex_project(),
                location=config.vertex_location(),
            )
        return _client


#: 可选的评审模型。空串 = 用默认值。
#: 都在 Vertex 上验证过可用；`gemini-3.1-pro` 不存在，pro 只有 preview 版。
JUDGE_MODELS = ("", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview")


def judge_model(override: str | None = None) -> str:
    """本次评审用哪个模型。显式入参优先，其次环境变量，最后默认值。

    **必须显式透传，不能只靠环境变量**：可能并发跑多轮评估，`os.environ`
    是进程级的，两轮选了不同模型会互相覆盖，而且事后无从分辨谁污染了谁。
    """
    return (
        (override or "").strip()
        or os.getenv("EVAL_JUDGE_MODEL", "").strip()
        or config.planner_gemini_model()
    )


#: 评审的思考档位。两处调用共用一份，避免改一处漏一处。
#: 空串 / unspecified = 不传 thinking_config，用模型默认。
THINKING_LEVELS = ("", "minimal", "low", "medium", "high")


def thinking_level() -> str:
    return os.getenv("EVAL_THINKING_LEVEL", "").strip().lower()


def _config(schema: type[BaseModel]) -> types.GenerateContentConfig:
    """所有评审请求的统一配置：temperature=0 + 结构化输出 + 可选思考档位。"""
    level = thinking_level()
    thinking = None
    if level and level != "unspecified":
        thinking = types.ThinkingConfig(thinking_level=level.upper())
    return types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=thinking,
    )


#: 配额被打满时重试几次。一批 126 次视频上传打过去，429 是常态而不是意外。
#: 不重试的代价不是「少一条结果」——`judge_scenes` 会把这一轮记成 error，
#: 那个片段的中位数就只剩两轮样本，而重复跑的全部意义就是样本量。
MAX_RETRIES = 4
RETRY_BASE = 4.0


def _retryable(error: Exception) -> bool:
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "503" in text or "UNAVAILABLE" in text


async def _generate(**kwargs):
    """发一次请求，配额类错误按指数退避重试。"""
    client = await _get_client()
    for attempt in range(MAX_RETRIES):
        try:
            return await client.aio.models.generate_content(**kwargs)
        except Exception as error:
            if attempt == MAX_RETRIES - 1 or not _retryable(error):
                raise
            # 抖动是必需的：一批片段是同时出发的，等长退避会让它们
            # 在同一时刻齐刷刷地重试，把刚让开的配额再次打满。
            delay = RETRY_BASE * (2 ** attempt) * (0.6 + random.random() * 0.8)
            logger.warning("judge retry %d/%d in %.1fs: %s",
                           attempt + 1, MAX_RETRIES - 1, delay, str(error)[:120])
            await asyncio.sleep(delay)


def _points(severity: str) -> float:
    return SEVERITY_POINTS.get(str(severity or "").strip().lower(), SEVERITY_POINTS["minor"])


def _score_from(deductions: list[dict[str, object]], dimension: str) -> float:
    """某个维度的得分 = 5 − 该维度所有扣分之和，下限 **0**。

    下限是 0 而不是常规 1～5 量表的 1：留着那 1 分，等于每个维度都有一段
    再烂也扣不掉的分，权重就不再是唯一的价值判断入口。
    """
    lost = sum(
        float(d.get("points") or 0)
        for d in deductions
        if str(d.get("dimension") or "") == dimension
    )
    return round(max(0.0, FULL_MARK - lost), 1)


#: beat_alignment 拿什么当参照物。
#: `visual_beat` 判「片子有没有照分镜演」；`script` 判「这一段有没有把这句口播画出来」——
#: 后者绕过 planner 的转写；`both` 两者都给。
BEAT_SOURCES = ("visual_beat", "script", "both")
BEAT_SOURCE_LABEL = {
    "visual_beat": "分镜运镜描述 visual_beat",
    "script": "口播原文 script",
    "both": "口播原文 + 分镜运镜描述",
}

#: style_adherence 拿什么当参照物。
#: `style_lock` 是全篇统一规范，必须跨渲染模式中立；`scene_design` 是本片段自己声明的，
#: 会随渲染模式变（实拍段用摄影语汇、CG 段用渲染器语汇）。只对 style_lock 判，
#: 查不出「实拍段用了渲染器语汇」。
#: `off` 整个跳过这一维度——它会退出计分。
STYLE_SOURCES = ("style_lock", "scene_design", "both", "off")
STYLE_SOURCE_LABEL = {
    "style_lock": "全篇风格锁 style_lock",
    "scene_design": "本片段的 scene_design",
    "both": "全篇风格锁 + 本片段的 scene_design",
    "off": "（本轮不评）",
}

#: 渲染模式要不要评。参照物永远是 scene_design 声明的 RENDER，没有「换一个」的余地，
#: 所以这里是开关而不是选择器。
#: 关掉能省掉**一半**的调用——盲判是独立的第二次请求。
#: 没有「合进主调用省一次」这个选项：早期版本那么做过，模型被声明值锚死，
#: 同一段素材盲测判 cg、给了声明后判 photoreal 且置信度 0.95。省下的钱换来错的结论。
RENDER_SOURCES = ("blind", "off")
RENDER_SOURCE_LABEL = {
    "blind": "盲判（独立请求，不给任何分镜信息）",
    "off": "（本轮不评）",
}


def _scene_style(scene_design: str) -> str:
    """从 scene_design 里取出风格描述 —— 也就是**最后一句**。

    scene_design 的结构是固定的：声明行 / 景别 / 主体 / 运镜 / 光线 / 风格。
    前面几句归 design_fidelity 管，整段拿来判风格会和那个维度重叠，
    模型可能因为主体不符去扣风格分。所以这里只取最后那句风格描述。
    """
    body = (scene_design or "").strip().rstrip(".")
    return body.rsplit(".", 1)[-1].strip() if "." in body else body


def style_reference(style_lock: str, item: SceneSlice, source: str) -> str:
    """按选定的参照物渲染风格描述，空串表示无参照。"""
    lock = (style_lock or "").strip()
    tail = _scene_style(item.scene_design)
    if source == "scene_design":
        return tail
    if source == "both":
        parts = [f"【全篇】{lock}" if lock else "", f"【本片段】{tail}" if tail else ""]
        return "\n".join(p for p in parts if p)
    return lock


def beat_lines(item: SceneSlice, source: str) -> list[str]:
    """按选定的参照物渲染时间轴。时间是**相对本片段**的，与扣分项同坐标系。"""
    lines: list[str] = []
    for beat in item.beats:
        start = float(beat.get("start_time_ms", 0)) / 1000
        window = f"  {start:.1f}-{start + float(beat.get('duration', 0)):.1f}s: "
        script = str(beat.get("script") or "").strip()
        visual = str(beat.get("visual_beat") or "").strip()
        if source == "script":
            if script:
                lines.append(f"{window}{script}")
        elif source == "both":
            if script or visual:
                parts = [f"口播「{script}」" if script else "", visual]
                lines.append(window + " / ".join(p for p in parts if p))
        elif visual:
            lines.append(f"{window}{visual}")
    return lines


def _script_lines(item: SceneSlice) -> list[str]:
    """本段口播原文。**无条件送进 prompt**，不受 beat_source 影响。

    科学准确性 / 清晰度 / 教育有效性这三条判的是「这一段在讲什么、讲清楚没有」，
    没有口播就无从判起——beat_source 那个开关只决定 beat_alignment 拿什么对齐，
    不该连带把评审的知识背景一起关掉。
    """
    lines = []
    for beat in item.beats:
        text = str(beat.get("script") or "").strip()
        if not text:
            continue
        start = float(beat.get("start_time_ms", 0)) / 1000
        lines.append(f"  {start:.1f}-{start + float(beat.get('duration', 0)):.1f}s: {text}")
    return lines


def _scene_context(item: SceneSlice, source: str) -> str:
    """两份 prompt 共用的素材段。模板在 prompts.yaml 的 `scene.context`。"""
    beats = beat_lines(item, source)
    script = _script_lines(item)
    return prompts.render(
        "scene.context",
        shot_type=item.shot_type or "未声明",
        environment=item.environment or "未声明",
        scene_design=item.scene_design or "（缺失）",
        beat_label=BEAT_SOURCE_LABEL[source],
        beat_lines="\n".join(beats) if beats else "（无）",
        script_lines="\n".join(script) if script else "（无）",
    )


def _criteria_lines(source: str, style_source: str) -> list[str]:
    """六（或五）条判据。文案全部来自 prompts.yaml，代码只负责按开关挑。"""
    crit = prompts.get("scene.criteria")
    style_off = style_source == "off"
    lines = [
        f"{'两' if style_off else '三'}个还原度维度（每个维度都从 5 分起，按下面的规则往下扣）：",
        f"- beat_alignment：{crit['beat_alignment'][source]}",
        f"- design_fidelity：{crit['design_fidelity']}",
    ]
    if not style_off:
        lines.append(f"- style_adherence：{crit['style_adherence'][style_source]}")
    lines.append(prompts.get("scene.science_preamble").rstrip("\n"))
    for key in SCIENCE_DIMENSIONS:
        lines.append(f"- {key}：{crit[key]}")
    return lines


def scene_prompt(
    style_lock: str,
    item: SceneSlice,
    source: str = "visual_beat",
    style_source: str = "style_lock",
) -> str:
    """组装评审 prompt。风格关掉时整段不出现 style_adherence（配 SceneVerdictNoStyle）。

    公开（上游叫 `_scene_prompt`）：`--dry-run` 要把它打出来给人看，
    「这一轮到底问了模型什么」是评估结果可复核的前提。
    """
    style_off = style_source == "off"
    parts = [_scene_context(item, source)]
    if style_off:
        parts.append(prompts.get("scene.style_off_note").rstrip("\n"))
    else:
        parts.append(
            prompts.render(
                "scene.style_block",
                style_label=STYLE_SOURCE_LABEL[style_source],
                style_reference=style_reference(style_lock, item, style_source) or "（无）",
            ).rstrip("\n")
        )
    parts.append("")
    parts.extend(_criteria_lines(source, style_source))
    parts.append(
        prompts.render(
            "scene.rubric",
            dimension_count="五" if style_off else "六",
            critical_examples="、把科学讲反了" if style_off else "、影调与风格参照南辕北辙、把科学讲反了",
        )
    )
    return "\n".join(parts)


async def judge_render_blind(clip: Path, model: str | None = None) -> dict[str, object]:
    """渲染模式盲判：只给视频，不给任何分镜信息。"""
    response = await _generate(
        model=judge_model(model),
        contents=[
            types.Part.from_bytes(data=clip.read_bytes(), mime_type="video/mp4"),
            prompts.get("render_blind"),
        ],
        config=_config(RenderVerdict),
    )
    return RenderVerdict.model_validate_json(response.text).model_dump()


async def judge_scene(
    style_lock: str,
    item: SceneSlice,
    clip: Path,
    beat_source: str = "visual_beat",
    style_source: str = "style_lock",
    render_source: str = "blind",
    model: str | None = None,
) -> dict[str, object]:
    """评审一个分镜片段。

    入参是 `style_lock` 而不是整个用例对象——这一路真正需要的只有那一个字段。
    任何一段 mp4 配一份分镜声明就能评，不必等成片拼出来。
    """
    payload = clip.read_bytes()
    style_off = style_source == "off"
    # 两份 prompt 配两份 schema：不评风格时 style_adherence 在结构上就不存在
    schema = SceneVerdictNoStyle if style_off else SceneVerdict

    async def contextual() -> dict[str, object]:
        response = await _generate(
            model=judge_model(model),
            contents=[
                types.Part.from_bytes(data=payload, mime_type="video/mp4"),
                scene_prompt(style_lock, item, beat_source, style_source),
            ],
            config=_config(schema),
        )
        return schema.model_validate_json(response.text).model_dump()

    render_off = render_source == "off"
    if render_off:
        # 不评就不发第二次请求——这是省钱的大头
        verdict, render = await contextual(), {}
    else:
        verdict, render = await asyncio.gather(
            contextual(), judge_render_blind(clip, model)
        )

    # 分数由扣分项推出来，不让模型直接给——保证每一分都能追到具体画面。
    deductions = []
    for raw in verdict.get("deductions") or []:
        # dimension / severity 都是 Literal 枚举，结构化输出保证取值合法
        raw["points"] = _points(raw.get("severity"))
        deductions.append(raw)
    verdict["deductions"] = deductions
    for dimension in SCENE_DIMENSIONS:
        verdict[dimension] = _score_from(deductions, dimension)
    if style_off:
        verdict["style_adherence"] = None

    # 参照物缺失的维度**直接判 0**，不能让它拿满分。
    # 评审 prompt 里 visual_beat / scene_design / style_lock 缺失时填的是「（无）」，
    # 模型没有可比对的东西，自然一条扣分项也挑不出来，结果就是满分——那不是
    # 「画面好」，是「没东西可对照」，按 0 记才不会误导跨版本对比。
    unreferenced = []
    if not beat_lines(item, beat_source):
        unreferenced.append("beat_alignment")
    if not str(item.scene_design or "").strip():
        unreferenced.append("design_fidelity")
    if not style_off and not style_reference(style_lock, item, style_source).strip():
        unreferenced.append("style_adherence")
    for dimension in unreferenced:
        verdict[dimension] = 0.0
    verdict["unreferenced"] = unreferenced
    verdict["beat_source"] = beat_source
    verdict["style_source"] = style_source
    verdict["render_source"] = render_source

    declared = item.render_mode
    verdict["render_declared"] = declared
    if render_off:
        # None 表示「没评」，与「评了但不匹配」区分开——前者退出计分，后者扣分
        verdict["render_verdict"] = None
        verdict["render_confidence"] = None
        verdict["render_reason"] = None
        verdict["render_match"] = None
    else:
        observed = str(render.get("verdict") or "").lower()
        verdict["render_verdict"] = observed
        verdict["render_confidence"] = render.get("confidence")
        verdict["render_reason"] = render.get("reason")
        verdict["render_match"] = bool(declared) and observed == declared
    verdict["scene_id"] = item.scene_id
    verdict["index"] = item.index
    verdict["start"] = item.start
    verdict["duration"] = item.duration
    return verdict


async def judge_scenes(
    scenes: Sequence[tuple[str, SceneSlice, Path]],
    on_done=None,
    beat_source: str = "visual_beat",
    style_source: str = "style_lock",
    render_source: str = "blind",
    model: str | None = None,
    concurrency: int | None = None,
) -> list[dict[str, object]]:
    """并发评审一批分镜片段，失败的记错误而不是整批中断。

    每项自带 `style_lock`，所以一批里可以混不同的片子。
    返回**按入参顺序**，不重排：跨片子时 `index` 会重复，按它排会把两部片子
    的结果交织在一起。
    """
    semaphore = asyncio.Semaphore(concurrency or MAX_CONCURRENCY)

    async def one(style_lock: str, item: SceneSlice, clip: Path) -> dict[str, object]:
        async with semaphore:
            try:
                result = await judge_scene(
                    style_lock, item, clip, beat_source, style_source, render_source, model
                )
            except Exception as error:  # 单个片段挂掉不该毁掉整轮评估
                logger.exception("scene judge failed, scene_id=%s", item.scene_id)
                result = {
                    "scene_id": item.scene_id,
                    "index": item.index,
                    "start": item.start,
                    "duration": item.duration,
                    "error": str(error),
                }
            if on_done:
                on_done(result)
            return result

    return list(await asyncio.gather(*(one(*scene) for scene in scenes)))
