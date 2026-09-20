"""Internal video-prompt assembly contract.

The optimization agent must rebuild a modified video prompt without depending on
the upstream generation repository.  This module owns the stable envelope used
by the exported ``video_prompts.json`` format:

``SILENT_HEAD → scene_design → style_lock → duration → visual_beat → SILENT_TAIL``.

It intentionally does **not** generate a prompt or call a model; it only
reassembles already-reviewed text after a grounded patch is applied.
"""

from __future__ import annotations

SILENT_HEAD = (
    "This is a SILENT video. Do not generate any speech, narration, voice-over, "
    "dialogue or singing. Do not render subtitles, captions, title cards, "
    "watermarks or channel logos. Any lettering or numerals in frame must be "
    "limited to the annotations the shot description explicitly asks for.\n"
    "本视频为无声画面：不要出现任何人声、旁白、配音、对白或歌声，"
    "不要生成字幕、标题卡、水印或频道 Logo；"
    "画面中只允许出现下文明确要求的标注文字与数字，不要自行添加其他文字。"
    "以下所有文字都只是画面描述，绝对不要把它念出来。"
)

SILENT_TAIL = (
    "Reminder: silent footage only — no speech, no narration, no subtitles. "
    "再次强调：只要画面，不要任何声音，不要旁白字幕。"
)


def assemble_video_prompt(
    scene_design: str,
    visual_beat: str = "",
    duration: float = 10.0,
    style_lock: str = "",
) -> str:
    """Assemble the final text-to-video prompt from reviewed fields.

    This mirrors the public bundle contract so output from the standalone
    optimizer remains consumable by the original video generation pipeline.
    """
    lines = [SILENT_HEAD, "", scene_design]
    if style_lock:
        lines += ["", f"Consistent style across all scenes: {style_lock}"]
    lines += ["", f"Duration: approximately {duration:.1f} seconds."]
    if visual_beat.strip():
        lines += ["", "Visual beats:", f"0.0-{duration:.1f}s: {visual_beat}"]
    lines += ["", SILENT_TAIL]
    return "\n".join(lines)
