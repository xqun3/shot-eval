"""把标准 video_prompts.json 生成产物装载成 `SceneTask`。

上游的三个装载器认的是 scheme_c 的 `state.json`（`plan.scenes[].scene_design`
+ `scenes[].timeline`，片段按 `<片名>/<scene_id>.mp4` 找）。我们这条流水线
落盘的是 `runs/<run>/video_prompts.json` + `runs/<run>/videos/<unit_id>.mp4`，
形状对不上，硬套不如就地写一个——装载本来就是整套评估里唯一该跟着数据走的部分。

一份 `video_prompts.json` 里，评估要用的字段：

| SceneSlice | 来源 | 说明 |
|---|---|---|
| `scene_id`    | `unit_id` | shot 粒度是 `A01-01`，scene 粒度是 `A01` |
| `duration`    | `videos[].actual_seconds` | 实际出片时长，不是规划时长 |
| `scene_design`| 见 `--design-source` | 两个都在，语义不同 |
| `prompt`      | `video_prompt` | 送进视频模型的完整 prompt |
| `beats`       | `request.clip_script` + `visual_beat` | 一个 shot 一条 |
| `style_lock`  | `effective_style_lock` | 单独作为 `SceneTask.style_lock` |
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shot_eval.bench import SceneTask
from shot_eval.models import Provenance, SceneSlice, parse_declaration

#: `scene_design` 有两份，判 design_fidelity 时用哪一份是**两个不同的问题**：
#: * `plan`      —— 分镜方案里那份（scene 级，英文，无时间码）。
#: * `generated` —— app.py 产出的那份（shot 级，带时间码）。
#: 默认 `generated`：逐 shot 评的是**每条视频的效果**，指令与成片之间的差距
#: 才是能直接拿去改 prompt 的信息。
DESIGN_SOURCES = ("generated", "plan")


def _clip_path(entry: dict[str, Any], videos: dict[str, dict], videos_dir: Path) -> Path | None:
    """先信 `videos[]` 里记的绝对路径，找不到再按 `<unit_id>.mp4` 在目录里捞。"""
    unit = str(entry.get("unit_id") or "")
    recorded = (videos.get(unit) or {}).get("path")
    if recorded:
        path = Path(str(recorded))
        if path.exists():
            return path
    fallback = videos_dir / f"{unit}.mp4"
    return fallback if fallback.exists() else None


def _text_list(value: Any) -> str:
    """把历史上可能是 str/list 的 beat 字段规范成可逐字引用的文本。"""
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def _script_excerpt(full_script: str, fragment: str, radius: int = 200) -> str:
    """截取当前口播附近上下文；找不到时不假装完成了映射。"""
    if not full_script or not fragment:
        return ""
    pos = full_script.find(fragment)
    if pos < 0:
        return ""
    return full_script[max(0, pos - radius):pos + len(fragment) + radius]


def _recover_plan_fields(
    entry: dict[str, Any],
    storyboard_plan: dict[str, Any] | None,
    full_script: str = "",
) -> dict[str, str]:
    """从 entry 取 plan 字段，缺失时按 scene_id + shot_index 从 plan 回补。"""
    request = entry.get("request") or {}
    script_fragment = str(
        entry.get("script_fragment") or request.get("clip_script") or ""
    ).strip()
    source_script = str(request.get("full_script") or full_script or "")
    result: dict[str, str] = {
        "plan_scene_design": str(entry.get("plan_scene_design") or "").strip(),
        "plan_visual_beats": _text_list(entry.get("plan_visual_beats")),
        "script_fragment": script_fragment,
        "script_context": str(entry.get("script_context") or "").strip()
        or _script_excerpt(source_script, script_fragment),
    }
    if not storyboard_plan:
        return result

    unit = str(entry.get("unit_id") or "")
    scene_id = str(entry.get("scene_id") or "")
    if not scene_id:
        scene_id = unit.rsplit("-", 1)[0] if "-" in unit else unit
    scene_match = next(
        (
            scene for scene in storyboard_plan.get("scenes") or []
            if isinstance(scene, dict) and str(scene.get("scene_id") or "") == scene_id
        ),
        None,
    )
    if not scene_match:
        return result

    if not result["plan_scene_design"]:
        result["plan_scene_design"] = str(scene_match.get("scene_design") or "").strip()
    shots = [s for s in scene_match.get("shots") or [] if isinstance(s, dict)]
    shot_index = entry.get("shot_index")
    selected: list[dict[str, Any]] = []
    if isinstance(shot_index, int) and 0 <= shot_index < len(shots):
        selected = [shots[shot_index]]
    elif unit != scene_id:
        selected = [
            shot for shot in shots
            if str(shot.get("shot_id") or shot.get("unit_id") or "") == unit
        ]
    else:
        selected = shots

    if not result["plan_visual_beats"]:
        result["plan_visual_beats"] = "\n".join(
            str(shot.get("visual_beat") or "").strip()
            for shot in selected
            if str(shot.get("visual_beat") or "").strip()
        )
    if not result["script_fragment"]:
        result["script_fragment"] = " ".join(
            str(shot.get("script") or "").strip()
            for shot in selected
            if str(shot.get("script") or "").strip()
        )
        result["script_context"] = (
            result["script_context"]
            or _script_excerpt(source_script, result["script_fragment"])
        )
    return result


def tasks_from_run(
    target: str | Path,
    *,
    design_source: str = "generated",
    source_name: str = "",
) -> list[SceneTask]:
    """从一个 run 目录（或直接给 `video_prompts.json`）装载全部片段。

    只装载 `ok=True` 且 mp4 真的在盘上的片段。
    """
    path = Path(target).expanduser().resolve()
    run_dir = path if path.is_dir() else path.parent
    if path.is_dir():
        path = path / "video_prompts.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    if design_source not in DESIGN_SOURCES:
        raise ValueError(f"--design-source 只能是 {DESIGN_SOURCES}，收到 {design_source!r}")

    data = json.loads(path.read_text(encoding="utf-8"))
    videos = {
        str(v.get("unit_id")): v
        for v in data.get("videos") or []
        if isinstance(v, dict) and v.get("ok")
    }
    videos_dir = run_dir / "videos"
    source = source_name or run_dir.name

    # Load storyboard_plan for provenance recovery
    storyboard_plan: dict[str, Any] | None = None
    plan_path = run_dir / "storyboard_plan.json"
    if plan_path.exists():
        try:
            storyboard_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if not storyboard_plan:
        storyboard_plan = data.get("storyboard_plan")

    tasks: list[SceneTask] = []
    index = 0
    for entry in data.get("video_prompts") or []:
        if not isinstance(entry, dict) or not entry.get("ok"):
            continue
        clip = _clip_path(entry, videos, videos_dir)
        if clip is None:
            continue
        unit = str(entry.get("unit_id"))
        video = videos.get(unit) or {}

        request = entry.get("request") or {}
        root_input = data.get("input") or {}
        plan_fields = _recover_plan_fields(
            entry,
            storyboard_plan,
            full_script=str(root_input.get("script") or ""),
        )

        key = "scene_design" if design_source == "generated" else "plan_scene_design"
        design = (
            str(entry.get("scene_design") or "")
            if key == "scene_design"
            else plan_fields["plan_scene_design"]
        )
        shot_type, environment, render_mode = parse_declaration(design)
        render_mode = render_mode or str(entry.get("render_mode") or "").strip().lower()

        duration = video.get("actual_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            duration = float(entry.get("duration_seconds") or 0)

        script = str(request.get("clip_script") or plan_fields["script_fragment"]).strip()
        visual = str(entry.get("visual_beat") or "").strip()

        provenance = Provenance(
            script_fragment=script,
            script_context=plan_fields["script_context"],
            plan_scene_design=plan_fields["plan_scene_design"],
            plan_visual_beats=plan_fields["plan_visual_beats"],
            generated_scene_design=str(entry.get("scene_design") or ""),
            generated_visual_beat=visual,
            effective_style_lock=str(
                entry.get("effective_style_lock") or entry.get("style_lock") or ""
            ),
            render_mode=render_mode,
            final_video_prompt=str(entry.get("video_prompt") or ""),
            unit_granularity=str((data.get("meta") or {}).get("granularity") or "shot"),
            analysis=(
                entry.get("analysis")
                if isinstance(entry.get("analysis"), dict)
                else str(entry.get("analysis") or "")
            ),
        )

        tasks.append(
            SceneTask(
                source=source,
                origin="shot",
                clip=clip,
                style_lock=str(
                    entry.get("effective_style_lock") or entry.get("style_lock") or ""
                ),
                provenance=provenance,
                item=SceneSlice(
                    scene_id=unit,
                    index=index,
                    start=0.0,
                    duration=round(float(duration), 3),
                    scene_design=design,
                    shot_type=shot_type,
                    environment=environment,
                    render_mode=render_mode,
                    prompt=str(entry.get("video_prompt") or ""),
                    beats=[{
                        "shot_id": unit,
                        "start_time_ms": 0,
                        "duration": round(float(duration), 3),
                        "script": script,
                        "visual_beat": visual,
                    }],
                    provenance=provenance,
                ),
            )
        )
        index += 1
    if not tasks:
        raise ValueError(f"{path} 里没有可评的片段（ok=True 且 mp4 在盘上）")
    return tasks
