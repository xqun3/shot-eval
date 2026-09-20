"""数据形状：一个待评片段（`SceneSlice`）+ 模型必须交回的结构化判决。

`SceneSlice` 与基础视频判决 schema 来自上游；本模块另外增加了链路溯源、
prompt 优化和四模式渲染判定 schema。没有包含整片评估使用的
`SentenceCoverage` / `CoherenceDeduction` / `NarrationVerdict`——那三个是
口播覆盖度那一路的，逐片段评审碰不到。

唯一的**行为性**改动在 `parse_declaration()`：见那里的注释。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

# --- 片段 ---------------------------------------------------------------------


#: scene_design 的声明行。上游的版本是 `SHOT: … | ENV: … | RENDER: …`，
#: 中间不允许有别的字段。常见生成流水线的声明行会多一个 LIGHTING 段：
#:   `SHOT: close-up | ENV: 史前岩洞 | LIGHTING: 洞口柔和日光 | RENDER: photoreal`
#: 拿上游那条正则去匹配会**整行失配**，`shot_type` / `environment` / `render_mode`
#: 三个字段一起变空串——后果不是少打印两行，而是 render_match 永远判不成立
#: （declared 为空时 `judge_scene` 直接把 render_match 记 False），
#: 等于凭空多出一个系统性的假问题。所以这里放宽成「ENV 与 RENDER 之间允许若干
#: `KEY: value` 段」，同时仍然要求 RENDER 收在行尾。
DECLARATION = re.compile(
    r"^\s*SHOT:\s*(?P<shot>[^|]+?)\s*\|\s*ENV:\s*(?P<env>[^|]+?)"
    # 中间段必须排除 RENDER：否则这个贪婪的「任意 KEY: value」会把
    # `| RENDER: photoreal` 也当成一个普通段吃掉，render 组落空——
    # 正好复现它要修的那个 bug，而且这次连 LIGHTING 的例子都测不出来。
    r"(?:\s*\|\s*(?!RENDER:)[A-Za-z_]+:\s*[^|]+?)*"
    r"(?:\s*\|\s*RENDER:\s*(?P<render>[A-Za-z_]+))?\s*$",
    re.MULTILINE,
)


def parse_declaration(scene_design: str) -> tuple[str, str, str]:
    """从 scene_design 的声明行里取出 (景别, 环境, 渲染模式)。失配返回三个空串。"""
    match = DECLARATION.search(scene_design or "")
    if not match:
        return "", "", ""
    return (
        match.group("shot").strip(),
        match.group("env").strip(),
        (match.group("render") or "").strip().lower(),
    )


@dataclass(frozen=True, slots=True)
class SceneSlice:
    """一个片段，以及它「本来该演什么」。

    字段与上游一致，含义在逐 shot 这一路上略有收窄：片段本身就是独立 mp4，
    所以 `start` 恒为 0、`duration` 是这条 mp4 的实际时长，
    `beats` 里通常只有一条（一个 shot 一条口播 + 一条运镜描述）。
    """

    scene_id: str
    index: int
    start: float
    duration: float
    scene_design: str
    shot_type: str
    environment: str
    render_mode: str
    prompt: str
    beats: list[dict[str, object]] = field(default_factory=list)
    provenance: Provenance | None = None

    @property
    def end(self) -> float:
        return self.start + self.duration

    def as_context(self) -> dict[str, object]:
        """报告里存一份「对照物」，这样历史报告不依赖源数据还在不在。

        `beats` 的时间是**相对本片段**的，和扣分项里的时间点同一坐标系。
        """
        ctx: dict[str, object] = {
            "scene_id": self.scene_id,
            "index": self.index,
            "shot_type": self.shot_type,
            "environment": self.environment,
            "render_mode": self.render_mode,
            "scene_design": self.scene_design,
            "prompt": self.prompt,
            "beats": [
                {
                    "shot_id": beat.get("shot_id"),
                    "script": beat.get("script"),
                    "visual_beat": beat.get("visual_beat"),
                    "start": round(float(beat.get("start_time_ms") or 0) / 1000, 2),
                    "duration": round(float(beat.get("duration") or 0), 2),
                }
                for beat in self.beats
            ],
        }
        if self.provenance is not None:
            prov = self.provenance
            ctx["provenance"] = {
                "script_fragment": prov.script_fragment,
                "script_context": prov.script_context,
                "plan_scene_design": prov.plan_scene_design,
                "plan_visual_beats": prov.plan_visual_beats,
                "generated_scene_design": prov.generated_scene_design,
                "generated_visual_beat": prov.generated_visual_beat,
                "effective_style_lock": prov.effective_style_lock,
                "render_mode": prov.render_mode,
                "final_video_prompt": prov.final_video_prompt,
                "unit_granularity": prov.unit_granularity,
                "analysis": prov.analysis,
            }
        return ctx


# --- 判决 schema ---------------------------------------------------------------


class RenderVerdict(BaseModel):
    """渲染模式的**盲判**结论：这一问必须在看不到分镜的前提下回答。

    早期版本把声明的 RENDER 值写进了同一条 prompt，模型被锚死——同一段素材盲测
    判 cg、给了声明后判 photoreal 且置信度 0.95。而「声明 photoreal 实际出 CG」
    正是这套评估要抓的头号问题，所以这一问单独发一次请求。
    """

    verdict: Literal["photoreal", "cg", "illustrated", "vox", "unclear"] = Field(
        description="photoreal（实拍质感）、cg（科教 3D 渲染）、"
        "illustrated（风格化叙事动画）、vox（纸艺/拼贴可视化）或 unclear。"
    )
    confidence: float = Field(ge=0, le=1, description="判断置信度。")
    reason: str = Field(description="从材质、光照、物理运动、镜头瑕疵等角度说明依据，中文一句。")


#: 扣分档位。模型只能从这三档里选，避免出现 0.3 分这种无法解释的数字。
SEVERITY_POINTS = {"minor": 0.5, "major": 1.5, "critical": 3.0}
FULL_MARK = 5.0
#: 前三条评「片子有没有照计划演」，后三条评「作为科普作品好不好」。
#: 后三条是科普片的评价框架（准确性 / 清晰度 / 启发性），前三条一条都覆盖不到——
#: 一条把每个分镜都精确还原、却把科学讲反了的片子，在前三条上可以拿满分。
SCENE_DIMENSIONS = (
    "beat_alignment",
    "design_fidelity",
    "style_adherence",
    "science_accuracy",
    "clarity",
    "instructional_value",
)
#: 判后三条必须知道这一段在讲什么，所以口播原文无条件送进评审 prompt，
#: 不受 beat_source 影响——那个开关只决定 beat_alignment 拿什么对齐。
SCIENCE_DIMENSIONS = ("science_accuracy", "clarity", "instructional_value")


Severity = Literal["minor", "major", "critical"]
_SEVERITY_DESC = (
    "严重程度：minor（细节瑕疵，扣 0.5）、major（关键要素缺失或明显偏离，扣 1.5）、"
    "critical（该维度基本没达成，扣 3）。"
)
_TIMESTAMP_DESC = (
    "问题出现的时间点，片段内相对时间，如 `00:03` 或 `00:03-00:05`；整段都有的问题填 `全段`。"
)
_DEDUCTIONS_DESC = (
    "所有扣分项。每个维度各自从 5 分起扣；某个维度没有问题就不要为它编造扣分项，"
    "有问题就必须列出来——不允许出现「说了缺点却不扣分」或「扣了分却说不出问题」。"
)


class Deduction(BaseModel):
    """一条扣分项。分数是**从扣分项推出来的**，不是模型直接给的。

    早期版本让模型直接打 1～5 分再补一句理由，结果是「4 分」配一句纯表扬的话——
    丢的那一分落在哪里没人知道，指标掉了也无从改起。现在反过来：模型只负责把
    问题一条条挑出来并定级，分数由 `_score_from()` 算，每一分都能追到画面。

    `dimension` 用 Literal 而不是 str：结构化输出会把它转成枚举，模型给不出
    列表以外的值。靠描述文字约束再由代码归一化，等于允许它先出错再擦掉。
    """

    dimension: Literal[
        "beat_alignment", "design_fidelity", "style_adherence",
        "science_accuracy", "clarity", "instructional_value",
    ] = Field(
        description="扣在哪个维度：beat_alignment（画面没做参照物描述的事）、"
        "design_fidelity（景别/环境/镜头运动与 scene_design 不符）、"
        "style_adherence（配色/光照/材质偏离风格参照）、"
        "science_accuracy（画面传达的信息与科学事实或口播不符）、"
        "clarity（看不懂：视觉隐喻不当、关键结构没交代、信息过载）、"
        "instructional_value（画面对理解这段内容没有帮助，只是氛围填充）。"
    )
    severity: Severity = Field(description=_SEVERITY_DESC)
    timestamp: str = Field(description=_TIMESTAMP_DESC)
    evidence: str = Field(description="画面上实际看到的、构成这个问题的具体内容，中文一句。")
    expected: str = Field(description="分镜或风格参照在这里要求的是什么，中文一句。")


class DeductionNoStyle(BaseModel):
    """两维度版的扣分项：枚举里**没有** style_adherence，模型给不出来。"""

    dimension: Literal[
        "beat_alignment", "design_fidelity",
        "science_accuracy", "clarity", "instructional_value",
    ] = Field(
        description="扣在哪个维度：beat_alignment（画面没做参照物描述的事）、"
        "design_fidelity（景别/环境/镜头运动与 scene_design 不符）、"
        "science_accuracy（画面传达的信息与科学事实或口播不符）、"
        "clarity（看不懂：视觉隐喻不当、关键结构没交代、信息过载）、"
        "instructional_value（画面对理解这段内容没有帮助，只是氛围填充）。"
    )
    severity: Severity = Field(description=_SEVERITY_DESC)
    timestamp: str = Field(description=_TIMESTAMP_DESC)
    evidence: str = Field(description="画面上实际看到的、构成这个问题的具体内容，中文一句。")
    expected: str = Field(description="分镜在这里要求的是什么，中文一句。")


class _SceneVerdictBase(BaseModel):
    """两份 schema 共有的字段。"""

    observed: str = Field(description="你在这段视频里实际看到了什么，两句话，中文。")
    beat_alignment_reason: str = Field(description="画面与本段参照物的整体吻合情况，中文一句。")
    design_fidelity_reason: str = Field(description="景别、环境、镜头运动与 scene_design 的吻合情况，中文一句。")
    science_accuracy_reason: str = Field(description="画面传达的信息是否与科学事实及口播一致，中文一句。")
    clarity_reason: str = Field(description="不懂这个知识点的人看这段画面能不能看明白，中文一句。")
    instructional_value_reason: str = Field(description="这段画面对理解本段内容有多大帮助，中文一句。")
    has_subtitle_or_caption: bool = Field(description="画面里是否出现了旁白字幕或说明性字幕条。")
    has_watermark_or_logo: bool = Field(description="是否出现水印、频道 Logo 或标题卡。")
    has_brand_name: bool = Field(description="是否出现真实厂商名或商标。")
    visible_text: list[str] = Field(default_factory=list, description="画面中出现的所有文字，逐条列出；没有就空列表。")
    issues: list[str] = Field(default_factory=list, description="其他值得记录的画面问题，如穿帮、结构崩坏、主体漂移。")


class SceneVerdict(_SceneVerdictBase):
    """三维度版。不给分，只列扣分项，分数由代码算。"""

    deductions: list[Deduction] = Field(default_factory=list, description=_DEDUCTIONS_DESC)
    style_adherence_reason: str = Field(description="配色、光照、材质与风格参照的吻合情况，中文一句。")


class SceneVerdictNoStyle(_SceneVerdictBase):
    """两维度版：连 `style_adherence_reason` 字段都没有，风格维度无从产出。"""

    deductions: list[DeductionNoStyle] = Field(default_factory=list, description=_DEDUCTIONS_DESC)


# --- Provenance（链路溯源）----------------------------------------------------

@dataclass(frozen=True, slots=True)
class Provenance:
    """一条链路的完整溯源：从脚本到分镜到 prompt 到视频，每一环的输入输出。

    所有字段 optional，向后兼容：老数据没有就是 None，不影响已有的评审流程。
    adapters 在缺失时尝试从 storyboard_plan 回补。
    """

    # 脚本 → plan
    script_fragment: str = ""
    script_context: str = ""
    plan_scene_design: str = ""
    plan_visual_beats: str = ""

    # plan → prompt
    generated_scene_design: str = ""
    generated_visual_beat: str = ""
    effective_style_lock: str = ""
    render_mode: str = ""
    final_video_prompt: str = ""

    # plan scene_design 是 scene 级；当前生成单元可能是 scene 或 shot。
    unit_granularity: str = "shot"

    # 分析
    analysis: dict[str, object] | str = ""


# --- 三阶段归因 + 优化 schemas -------------------------------------------------

#: 渲染模式的合法取值。generator 支持四种。
RENDER_MODES = ("photoreal", "cg", "illustrated", "vox", "unclear")


class RenderModeEnum(BaseModel):
    """四模式 + unclear 的渲染分类。"""
    mode: Literal["photoreal", "cg", "illustrated", "vox", "unclear"] = Field(
        description="photoreal / cg / illustrated / vox / unclear",
    )


#: 三阶段
ATTRIBUTION_STAGES = ("script_to_plan", "plan_to_prompt", "prompt_to_video")

ProblemSeverity = Literal["low", "medium", "high"]
PatchOp = Literal["replace", "delete", "insert_after", "append"]
PATCH_TARGETS = (
    "plan_scene_design", "plan_visual_beats",
    "generated_scene_design", "generated_visual_beat",
    "style_lock", "render_mode",
)


FindingStatus = Literal["pass", "problem", "unassessable"]
RootCause = Literal[
    "faithful", "missing_content", "meaning_drift", "ambiguous_instruction",
    "conflicting_instruction", "overloaded_instruction", "unvisualizable_instruction",
    "scientific_misdescription", "generator_noncompliance", "insufficient_evidence",
]
FindingSource = Literal[
    "script_fragment", "script_context", "plan_scene_design", "plan_visual_beats",
    "generated_scene_design", "generated_visual_beat", "style_lock", "render_mode",
    "final_video_prompt", "video_evidence",
]


class Finding(BaseModel):
    """一条阶段归因；三个阶段即使无问题也各返回一条。"""
    stage: Literal["script_to_plan", "plan_to_prompt", "prompt_to_video"] = Field(
        description="被检查的阶段。")
    status: FindingStatus = Field(description="该阶段通过、有问题或因缺少参照无法判断。")
    problem: str = Field(description="问题或通过结论。")
    severity: ProblemSeverity = Field(description="严重程度。")
    root_cause: RootCause = Field(description="受限枚举的根因。")
    source_field: FindingSource = Field(description="逐字证据所在字段。")
    exact_quote: str = Field(
        default="", description="problem 时必须逐字引用；video_evidence 引用评审 evidence。")
    confidence: float = Field(ge=0, le=1, description="归因置信度。")


class PromptPatch(BaseModel):
    """一条经过程序锚定后才可执行的最小 prompt 修改。"""
    target: Literal[
        "plan_scene_design", "plan_visual_beats",
        "generated_scene_design", "generated_visual_beat",
        "style_lock", "render_mode",
    ] = Field(description="要修改的字段。")
    op: PatchOp = Field(description="操作类型。")
    anchor: str = Field(default="", description="replace/delete/insert_after 时的逐字锚点。")
    content: str = Field(default="", description="replace/insert_after/append 的新内容。")
    reason: str = Field(description="修改理由。")
    confidence: float = Field(ge=0, le=1, description="该 patch 能解决问题的置信度。")


class NonPromptAction(BaseModel):
    """非 prompt 层面的建议——用于 generator 不合规时的操作建议。"""
    action: str = Field(description="建议操作，如 retry / switch_model / adjust_params。")
    reason: str = Field(description="理由。")


class OptimizationResult(BaseModel):
    """一个 scene 的优化分析结果。"""
    findings: list[Finding] = Field(default_factory=list)
    patches: list[PromptPatch] = Field(default_factory=list)
    non_prompt_actions: list[NonPromptAction] = Field(default_factory=list)
