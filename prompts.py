"""Prompt loader for the standalone shot_eval package.

``prompts.yaml`` is packaged with shot_eval and is the default evaluation
contract. Set ``SHOT_EVAL_PROMPTS`` only when intentionally testing a separate
prompt version. The loader validates, hot-reloads and fingerprints the file so
reports remain comparable only within the same contract.
"""

from __future__ import annotations

import hashlib
import os
import re
import string
import threading
from pathlib import Path
from typing import Any

import yaml

def _resolve() -> Path:
    raw = os.getenv("SHOT_EVAL_PROMPTS", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    local = Path(__file__).with_name("prompts.yaml")
    if local.exists():
        return local
    raise PromptConfigError(
        f"找不到打包的 prompts.yaml：{local}；"
        "可用 SHOT_EVAL_PROMPTS 指定路径"
    )


#: 每段模板允许出现的占位符。写了别的名字加载就失败——比运行时 KeyError 早得多。
PLACEHOLDERS: dict[str, set[str]] = {
    "scene.context": {
        "shot_type", "environment", "scene_design",
        "beat_label", "beat_lines", "script_lines",
    },
    "scene.style_block": {"style_label", "style_reference"},
    "scene.science_preamble": set(),
    "scene.style_off_note": set(),
    "scene.rubric": {"dimension_count", "critical_examples"},
    "render_blind": set(),
    "narration": {"numbered_sentences"},
    "text_rules": set(),
    "optimization": {"provenance_json", "deductions_json"},
}

#: 判据的必需 key。少一个就说明这一版 yaml 配不齐某个维度。
REQUIRED_CRITERIA = {
    "beat_alignment": {"visual_beat", "script", "both"},
    "style_adherence": {"style_lock", "scene_design", "both"},
    "design_fidelity": None,          # 单条字符串
    "science_accuracy": None,
    "clarity": None,
    "instructional_value": None,
}

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_stamp: tuple[str, int, int] | None = None


class PromptConfigError(ValueError):
    """`prompts.yaml` 配错了。消息要能直接指出改哪一行。"""


def _fields(template: str) -> set[str]:
    return {
        name
        for _, name, _, _ in string.Formatter().parse(template)
        if name
    }


def _dig(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise PromptConfigError(f"prompts.yaml 缺少 `{dotted}`")
        node = node[part]
    return node


#: 折叠标量（`>` / `>-` / `>+`）把换行变成空格——对英文正确，对中文是错的：
#: 它会在句读处塞进空格，而文件里看着一切正常。这里**直接禁掉病因**而不是猜症状：
#: 先前试过用正则找「中文之间的空格」，结果既漏（断在 `——` 后面）又误报
#: （`1. brand —— 真实厂商名` 是有意的列表排版）。禁标量本身没有歧义。
_FOLDED = re.compile(r"^\s*[\w.-]+:\s*>[-+]?\s*$", re.MULTILINE)


def _reject_folded(raw: str) -> None:
    hits = [m.group().strip() for m in _FOLDED.finditer(raw)]
    if hits:
        raise PromptConfigError(
            "prompts.yaml 里不允许折叠标量（`>` / `>-`）——它把换行变成空格，"
            f"会在中文句子中间塞进空格。请改用字面块 `|-`。命中：{hits[:3]}"
        )


def _validate(data: dict[str, Any]) -> None:
    if not isinstance(data.get("version"), int):
        raise PromptConfigError("prompts.yaml 顶层需要一个整数 `version`")

    for dotted, allowed in PLACEHOLDERS.items():
        template = _dig(data, dotted)
        if not isinstance(template, str) or not template.strip():
            raise PromptConfigError(f"`{dotted}` 必须是非空字符串")
        unknown = _fields(template) - allowed
        if unknown:
            raise PromptConfigError(
                f"`{dotted}` 里有未定义的占位符 {sorted(unknown)}；"
                f"这一段只能用 {sorted(allowed) or '（无）'}"
            )

    criteria = _dig(data, "scene.criteria")
    for key, subkeys in REQUIRED_CRITERIA.items():
        if key not in criteria:
            raise PromptConfigError(f"`scene.criteria.{key}` 缺失")
        value = criteria[key]
        if subkeys is None:
            if not isinstance(value, str) or not value.strip():
                raise PromptConfigError(f"`scene.criteria.{key}` 必须是非空字符串")
            continue
        if not isinstance(value, dict) or set(value) != subkeys:
            raise PromptConfigError(
                f"`scene.criteria.{key}` 必须正好包含 {sorted(subkeys)}，"
                f"实际是 {sorted(value) if isinstance(value, dict) else type(value).__name__}"
            )
        for sub, text in value.items():
            if not isinstance(text, str) or not text.strip():
                raise PromptConfigError(f"`scene.criteria.{key}.{sub}` 必须是非空字符串")


def path() -> Path:
    """本轮实际加载的判据文件。报告里要打出来——回落到上游时必须看得见。"""
    return _resolve()


def load(force: bool = False) -> dict[str, Any]:
    """读 `prompts.yaml`，按 mtime + size 热加载。校验不过直接抛。"""
    global _cache, _cache_stamp
    target = _resolve()
    stat = target.stat()
    # 路径也进指纹：换了 SHOT_EVAL_PROMPTS 而 mtime 碰巧一样时不能用旧缓存
    stamp = (str(target), stat.st_mtime_ns, stat.st_size)
    with _lock:
        if not force and _cache is not None and _cache_stamp == stamp:
            return _cache
        raw = target.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise PromptConfigError("prompts.yaml 根节点必须是一个映射")
        _reject_folded(raw)
        _validate(data)
        data["_digest"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        _cache, _cache_stamp = data, stamp
        return data


def get(dotted: str) -> Any:
    return _dig(load(), dotted)


def render(dotted: str, **values: object) -> str:
    """取一段模板并填占位符。缺值时报出是哪一段缺哪个字段。"""
    template = _dig(load(), dotted)
    try:
        return template.format(**values)
    except KeyError as error:
        raise PromptConfigError(f"渲染 `{dotted}` 时缺少占位符 {error}") from error


def version_tag() -> str:
    """写进报告的判据指纹：`v{version}+{内容哈希}`。

    只记 version 不够——改了内容忘了改 version 是最常见的情况，
    哈希能兜住；只记哈希又看不出人为的版本意图，所以两个都要。
    """
    data = load()
    return f"v{data['version']}+{data['_digest']}"
