"""Deterministic core: patch application, linting, plan generation.

All functions are pure / side-effect-free (except file I/O at the edges).
Unit-testable without network, ADK, or LLM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("shot_eval.prompt_optimizer_agent.core")

# ---------------------------------------------------------------------------
# Constants / Policy
# ---------------------------------------------------------------------------

DEFAULT_MIN_CONFIDENCE = 0.8
DEFAULT_MIN_REPEATS = 2
AUTO_APPLY_TARGETS = {"generated_scene_design", "generated_visual_beat"}
AUTO_APPLY_OPS = {"replace", "delete", "insert_after", "append"}
REVIEW_ONLY_TARGETS = {"plan_scene_design", "plan_visual_beats", "style_lock", "render_mode"}
REGEN_CATEGORIES = (
    "regenerate_prompt_changed",
    "retry_generator_noncompliance",
    "review_only",
    "skipped",
)

GLOBAL_RULE_LIBRARY = {
    "ambiguous_instruction": (
        "Replace literalizable scientific similes with observable density, direction, "
        "speed and state-change descriptions; add shot-specific must-not-show constraints."
    ),
    "meaning_drift": (
        "Preserve the current narration claim explicitly when converting plan content into "
        "generated_scene_design and generated_visual_beat."
    ),
    "missing_content": (
        "Carry every visually essential subject, action, number and causal result from the "
        "selected plan beat into the generated prompt."
    ),
    "overloaded_instruction": (
        "Limit each shot to one primary content action and one primary camera operation; "
        "prefer scientific content over cinematic camera choreography."
    ),
    "generator_noncompliance": (
        "When an instruction is already explicit, retry first; after repeated failure, "
        "simplify the shot or switch model instead of duplicating synonymous prompt text."
    ),
    "scientific_misdescription": (
        "Add explicit biological containment, relative scale, and source-path-destination-result "
        "constraints for microscopic scientific processes."
    ),
}


@dataclass
class Policy:
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_repeats: int = DEFAULT_MIN_REPEATS
    allow_single_run: bool = False
    # These are always review-only by default
    review_only_targets: frozenset[str] = frozenset(REVIEW_ONLY_TARGETS)


# ---------------------------------------------------------------------------
# Patch application (deterministic, exact-one semantics)
# ---------------------------------------------------------------------------

class PatchError(Exception):
    """A patch cannot be applied deterministically."""


def _count_occurrences(text: str, anchor: str) -> int:
    if not anchor:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx == -1:
            break
        count += 1
        start = idx + len(anchor)
    return count


def apply_patch(text: str, patch: dict[str, Any]) -> str:
    """Apply a single patch to *text*. Raises PatchError on violation."""
    op = patch["op"]
    anchor = str(patch.get("anchor") or "")
    content = str(patch.get("content") or "")

    if op in {"replace", "delete", "insert_after"} and not anchor:
        raise PatchError(f"{op}: non-empty anchor is required")
    if op in {"replace", "insert_after", "append"} and not content.strip():
        raise PatchError(f"{op}: non-empty content is required")

    if op == "replace":
        n = _count_occurrences(text, anchor)
        if n == 0:
            raise PatchError(f"replace: anchor {anchor!r} not found")
        if n > 1:
            raise PatchError(f"replace: anchor {anchor!r} occurs {n} times, need exactly 1")
        return text.replace(anchor, content, 1)

    if op == "delete":
        n = _count_occurrences(text, anchor)
        if n == 0:
            raise PatchError(f"delete: anchor {anchor!r} not found")
        if n > 1:
            raise PatchError(f"delete: anchor {anchor!r} occurs {n} times, need exactly 1")
        return text.replace(anchor, "", 1)

    if op == "insert_after":
        n = _count_occurrences(text, anchor)
        if n == 0:
            raise PatchError(f"insert_after: anchor {anchor!r} not found")
        if n > 1:
            raise PatchError(f"insert_after: anchor {anchor!r} occurs {n} times, need exactly 1")
        idx = text.find(anchor)
        return text[:idx + len(anchor)] + content + text[idx + len(anchor):]

    if op == "append":
        # Deduplicate: do not append if content already present
        if content.strip() and content.strip() in text:
            return text  # already present, dedup
        return text + content

    raise PatchError(f"unknown op: {op!r}")


def detect_overlapping_anchors(patches: list[dict[str, Any]], text: str) -> list[str]:
    """Return list of conflict descriptions for overlapping anchor ranges."""
    positions: list[tuple[int, int, int]] = []
    for i, p in enumerate(patches):
        anchor = p.get("anchor", "")
        if not anchor:
            continue
        idx = text.find(anchor)
        if idx < 0:
            continue
        positions.append((idx, idx + len(anchor), i))
    positions.sort()
    conflicts: list[str] = []
    for j in range(len(positions) - 1):
        if positions[j][1] > positions[j + 1][0]:
            a, b = positions[j][2], positions[j + 1][2]
            conflicts.append(
                f"patches {a} and {b} overlap: "
                f"{patches[a].get('anchor', '')!r} / {patches[b].get('anchor', '')!r}"
            )
    return conflicts


def apply_patches_to_entry(
    entry: dict[str, Any],
    patches: list[dict[str, Any]],
    policy: Policy,
) -> dict[str, Any]:
    """Apply valid patches to a single video_prompts entry.

    Returns a dict with:
      applied: list of applied patch dicts
      skipped: list of {patch, reason} dicts
      modified_fields: set of field names changed
      new_scene_design, new_visual_beat: resulting text (or None if unchanged)
    """
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    modified_fields: set[str] = set()

    # Group patches by target field
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in patches:
        by_target[p["target"]].append(p)

    new_scene_design: str | None = None
    new_visual_beat: str | None = None

    for target, target_patches in by_target.items():
        # Determine the text to patch
        if target == "generated_scene_design":
            text = str(entry.get("scene_design") or "")
        elif target == "generated_visual_beat":
            text = str(entry.get("visual_beat") or "")
        else:
            # review-only targets
            for p in target_patches:
                skipped.append({"patch": p, "reason": f"target {target!r} is review-only"})
            continue

        # Check confidence gate
        low_conf = [p for p in target_patches if p.get("confidence", 0) < policy.min_confidence]
        for p in low_conf:
            skipped.append({
                "patch": p,
                "reason": f"confidence {p.get('confidence', 0)} < {policy.min_confidence}",
            })
        ok_patches = [p for p in target_patches if p.get("confidence", 0) >= policy.min_confidence]
        if not ok_patches:
            continue

        # Detect overlapping anchors
        conflicts = detect_overlapping_anchors(ok_patches, text)
        if conflicts:
            for p in ok_patches:
                skipped.append({"patch": p, "reason": f"overlapping anchors: {'; '.join(conflicts)}"})
            continue

        # Apply patches sequentially
        current = text
        for p in ok_patches:
            try:
                current = apply_patch(current, p)
                applied.append(p)
            except PatchError as exc:
                skipped.append({"patch": p, "reason": str(exc)})

        if current != text:
            modified_fields.add(target)
            if target == "generated_scene_design":
                new_scene_design = current
            elif target == "generated_visual_beat":
                new_visual_beat = current

    return {
        "applied": applied,
        "skipped": skipped,
        "modified_fields": modified_fields,
        "new_scene_design": new_scene_design,
        "new_visual_beat": new_visual_beat,
    }


# ---------------------------------------------------------------------------
# Internal prompt assembly
# ---------------------------------------------------------------------------


def rebuild_video_prompt(
    scene_design: str,
    visual_beat: str,
    duration_seconds: float,
    style_lock: str,
) -> str:
    """Rebuild the final prompt using shot_eval's portable assembly contract."""
    from shot_eval.prompt_assembly import assemble_video_prompt

    return assemble_video_prompt(
        scene_design=scene_design,
        visual_beat=visual_beat,
        duration=duration_seconds,
        style_lock=style_lock,
    )


# ---------------------------------------------------------------------------
# Static linter (advisory only)
# ---------------------------------------------------------------------------

# Timecode pattern: matches HH:MM:SS or MM:SS or 0.0-5.0s patterns
_TIMECODE_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b\d+\.\d+-\d+\.\d+s?\b"
)

# Patterns for each lint rule
_SCIENCE_SIMILE_RE = re.compile(
    r"(如|像|好比|仿佛|就像|宛如|as if|like a|resembling)\s*"
    r"[\u4e00-\u9fff\w]*"
    r"(蜂群|海浪|河流|星辰|火山|原子|粒子|水车|水轮|风车|齿轮|"
    r"ocean|river|swarm|volcano|waterwheel|mill.?wheel|windmill|gear)",
    re.IGNORECASE,
)

_BIO_CONTAINMENT_KEYWORDS = re.compile(
    r"(溢出|泄漏|穿透|穿出|体外|突破|逃逸|外泄|向外喷涌|"
    r"leak|spill|breach|escape|rupture|branching out|outside the body)",
    re.IGNORECASE,
)
_BIO_CONTEXT = re.compile(
    r"(人体|身体|血管|血液|细胞|病毒|细菌|蛋白质|器官|"
    r"body|vessel|blood|cell|virus|bacteria|protein|organ)",
    re.IGNORECASE,
)
_BIO_CONTAINMENT_SAFE = re.compile(
    r"(严格限制在.*(?:体内|人体|身体|细胞|血管)|绝不穿出|不得穿出|"
    r"strictly (?:confined |contained )?within|never (?:cross|extend).*outside|"
    r"without extending outside)",
    re.IGNORECASE,
)

_CAMERA_OPS = re.compile(
    r"(zoom|pan|tilt|dolly|crane|track|push|pull|orbit|rotate|whip|rack focus"
    r"|推|拉|摇|移|升|降|甩|跟)",
    re.IGNORECASE,
)

_EXACT_NUMBER_RE = re.compile(
    r"(?:exactly|precisely|恰好|精确)\s*\d+\s*(?:个|条|根|层|片|pieces|items|layers)?",
    re.IGNORECASE,
)

_GRAPH_MG_RE = re.compile(
    r"(chart|graph|diagram|表格|图表|柱状图|折线图|饼图|bar chart|pie chart|line graph)",
    re.IGNORECASE,
)

_ANNOTATION_RE = re.compile(
    r"(标注|注释|annotation|label|caption|callout|数据标签)",
    re.IGNORECASE,
)
_NO_TEXT_RE = re.compile(
    r"(no text|no subtitle|不要.*文字|不要.*字幕|无文字|no lettering)",
    re.IGNORECASE,
)


@dataclass
class LintWarning:
    rule: str
    message: str
    location: str = ""  # e.g. "scene_design" or "visual_beat"
    severity: str = "advisory"


def _is_timecode_number(text: str, match_start: int, match_end: int) -> bool:
    """Check if a number at this position is part of a timecode."""
    # Look at surrounding context
    ctx_start = max(0, match_start - 20)
    ctx_end = min(len(text), match_end + 20)
    context = text[ctx_start:ctx_end]
    return bool(_TIMECODE_RE.search(context))


def lint_prompt(scene_design: str, visual_beat: str, unit_id: str = "") -> list[LintWarning]:
    """Run static linter on a single prompt unit. Returns advisory warnings."""
    warnings: list[LintWarning] = []
    combined = f"{scene_design}\n{visual_beat}"

    # 1. Literalizable science similes
    for m in _SCIENCE_SIMILE_RE.finditer(combined):
        loc = "scene_design" if m.start() < len(scene_design) else "visual_beat"
        warnings.append(LintWarning(
            rule="literalizable_simile",
            message=f"Science simile may be rendered literally by video model: '{m.group()}'",
            location=loc,
        ))

    # 2. Biological containment risk
    if (
        _BIO_CONTEXT.search(combined)
        and _BIO_CONTAINMENT_KEYWORDS.search(combined)
        and not _BIO_CONTAINMENT_SAFE.search(combined)
    ):
        warnings.append(LintWarning(
            rule="biological_containment_risk",
            message="Containment-risk language near biological context may generate unsafe imagery",
            location="scene_design",
        ))

    # 3. Missing relative scale
    # Check if describing micro/macro subjects without scale reference
    micro_re = re.compile(r"(细胞|分子|原子|蛋白质|enzyme|molecule|atom|cell|protein)", re.IGNORECASE)
    scale_re = re.compile(r"(相对|对比|比例|scale|relative|compared to|reference)", re.IGNORECASE)
    if micro_re.search(combined) and not scale_re.search(combined):
        warnings.append(LintWarning(
            rule="missing_relative_scale",
            message="Micro/macro subject without explicit scale reference",
            location="scene_design",
        ))

    # 4. Source-path-destination/result continuity. Only inspect scientific
    # particles/flows; camera "toward" language does not need a material source.
    process_re = re.compile(
        r"(分子|颗粒|微球|光流|血流|血液|葡萄糖|胰岛素|食糜|"
        r"molecule|particle|sphere|blood flow|glucose|insulin|chyme)",
        re.IGNORECASE,
    )
    destination_re = re.compile(
        r"(进入|流向|汇入|注入|穿过|到达|\binto\b|\btoward\b|\bthrough\b|\breaches?\b)",
        re.IGNORECASE,
    )
    source_re = re.compile(
        r"(从|源自|离开|由.*释放|\bfrom\b|\boriginat(?:e|es|ing)\b|\breleased? by\b)",
        re.IGNORECASE,
    )
    for sentence in re.split(r"[。！？\n]|(?<=[.!?])\s+", combined):
        if (
            process_re.search(sentence)
            and destination_re.search(sentence)
            and not source_re.search(sentence)
        ):
            snippet = sentence.strip()[:100]
            warnings.append(LintWarning(
                rule="source_path_continuity",
                message=f"Scientific flow has a destination but no explicit source: '{snippet}'",
                location="scene_design",
            ))
            break

    # 5. Camera overload
    camera_matches = list(_CAMERA_OPS.finditer(combined))
    # Filter out timecode-adjacent matches
    real_camera = [m for m in camera_matches
                   if not _is_timecode_number(combined, m.start(), m.end())]
    if len(real_camera) > 3:
        warnings.append(LintWarning(
            rule="camera_overload",
            message=f"Too many camera operations ({len(real_camera)}) in single shot",
            location="scene_design",
        ))

    # 6. Exact-number/graph MG routing
    if _EXACT_NUMBER_RE.search(combined):
        warnings.append(LintWarning(
            rule="exact_number_mg_routing",
            message="Exact numeric count may not render correctly in AI video; consider MG overlay",
            location="scene_design",
        ))
    if _GRAPH_MG_RE.search(combined):
        warnings.append(LintWarning(
            rule="graph_mg_routing",
            message="Graph/chart description should route to MG pipeline, not AI video",
            location="scene_design",
        ))

    # 7. Annotation / no-text conflict
    has_annotation = _ANNOTATION_RE.search(combined)
    has_no_text = _NO_TEXT_RE.search(combined)
    if has_annotation and has_no_text:
        warnings.append(LintWarning(
            rule="annotation_no_text_conflict",
            message="Prompt requests annotations/labels but also says no text",
            location="scene_design",
        ))

    return warnings


# ---------------------------------------------------------------------------
# Global rule candidates
# ---------------------------------------------------------------------------

def aggregate_global_candidates(
    scene_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate repeated root causes into global rule candidates."""
    cause_counter: Counter[str] = Counter()
    cause_evidence: dict[str, list[str]] = defaultdict(list)
    cause_confidence: dict[str, list[float]] = defaultdict(list)

    for scene in scene_results:
        findings = scene.get("findings") or []
        for f in findings:
            if f.get("status") != "problem":
                continue
            rc = f.get("root_cause", "unknown")
            cause_counter[rc] += 1
            cause_evidence[rc].append(scene.get("unit_id", "?"))
            cause_confidence[rc].append(f.get("confidence", 0.0))

    candidates: list[dict[str, Any]] = []
    for cause, count in cause_counter.most_common():
        if count < 2:
            continue
        avg_conf = sum(cause_confidence[cause]) / len(cause_confidence[cause])
        candidates.append({
            "root_cause": cause,
            "count": count,
            "evidence_units": sorted(set(cause_evidence[cause])),
            "confidence": round(avg_conf, 3),
            "status": "candidate",
            "suggested_rule": GLOBAL_RULE_LIBRARY.get(
                cause,
                "Review repeated evidence and define a scoped generation rule before adoption.",
            ),
        })
    return candidates


# ---------------------------------------------------------------------------
# Core optimization pipeline
# ---------------------------------------------------------------------------

def _compute_bundle_hash(bundle: dict[str, Any]) -> str:
    """SHA-256 of the JSON-serialized bundle (deterministic key order)."""
    raw = json.dumps(bundle, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _eval_stem(eval_path: str) -> str:
    """Derive output dir name from evaluation file path."""
    p = Path(eval_path)
    return p.stem


def optimize_video_prompts(
    video_prompts_path: str,
    eval_json_path: str,
    output_dir: str = "",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    allow_single_run: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Deterministic prompt optimization: load, match, apply, rebuild, write.

    This is the core function wrapped by FunctionTool for the ADK agent.

    Args:
        video_prompts_path: Path to video_prompts.json (the source bundle).
        eval_json_path: Path to an evaluation JSON with optimization.scenes[].
        output_dir: Where to write outputs. Default: run/optimizations/<eval-stem>/.
        min_confidence: Minimum patch confidence for auto-apply.
        allow_single_run: If True, allow auto-apply even with repeats<2.
        dry_run: If True, compute plan but write nothing.

    Returns:
        Dictionary with optimization_report, regeneration_plan, etc.
    """
    # --- Load ---
    vp_path = Path(video_prompts_path).expanduser().resolve()
    eval_path = Path(eval_json_path).expanduser().resolve()

    if not vp_path.exists():
        return {"error": f"video_prompts not found: {vp_path}"}
    if not eval_path.exists():
        return {"error": f"eval JSON not found: {eval_path}"}

    bundle = json.loads(vp_path.read_text(encoding="utf-8"))
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

    bundle_hash = _compute_bundle_hash(bundle)
    entries = bundle.get("video_prompts") or []
    entry_map = {str(e.get("unit_id")): e for e in entries if isinstance(e, dict)}

    optimization = eval_data.get("optimization") or {}
    opt_scenes = optimization.get("scenes") or []
    eval_config = eval_data.get("config") or {}
    repeats = eval_data.get("repeats", 1)

    policy = Policy(
        min_confidence=min_confidence,
        allow_single_run=allow_single_run,
    )

    # --- Match and process ---
    scene_results: list[dict[str, Any]] = []
    regen_entries: list[dict[str, Any]] = []
    any_applied = False

    for opt_scene in opt_scenes:
        scene_id = opt_scene.get("scene_id", "")
        entry = entry_map.get(scene_id)
        result: dict[str, Any] = {
            "unit_id": scene_id,
            "status": "skipped",
            "reason": "",
            "patches_applied": [],
            "patches_skipped": [],
            "lint_warnings": [],
            "prompt_rebuilt": False,
        }

        if not entry:
            result["reason"] = f"scene_id {scene_id!r} not found in bundle"
            result["status"] = "skipped"
            scene_results.append(result)
            regen_entries.append({
                "unit_id": scene_id,
                "category": "skipped",
                "reasons": [result["reason"]],
                "priority": 0,
                "patches": [],
                "non_prompt_actions": [],
            })
            continue

        # Collect patches from valid_patches only (never trust rejected_patches)
        valid_patches = opt_scene.get("valid_patches") or []
        rejected_patches = opt_scene.get("rejected_patches") or []
        non_prompt_actions = opt_scene.get("non_prompt_actions") or []
        findings = opt_scene.get("findings") or []

        # Filter to auto-apply-eligible patches
        eligible: list[dict[str, Any]] = []
        review: list[dict[str, Any]] = []

        for p in valid_patches:
            target = p.get("target", "")
            op = p.get("op", "")
            source = "valid_patches"  # already grounded

            if target in policy.review_only_targets:
                review.append({**p, "_skip_reason": f"target {target!r} is review-only"})
                continue
            if target not in AUTO_APPLY_TARGETS:
                review.append({**p, "_skip_reason": f"target {target!r} not auto-applicable"})
                continue
            if op not in AUTO_APPLY_OPS:
                review.append({**p, "_skip_reason": f"op {op!r} not auto-applicable"})
                continue
            if p.get("confidence", 0) < policy.min_confidence:
                review.append({**p, "_skip_reason": f"confidence below {policy.min_confidence}"})
                continue

            # Revalidate anchor against current bundle
            if op in ("replace", "delete", "insert_after"):
                anchor = p.get("anchor", "")
                if target == "generated_scene_design":
                    field_val = str(entry.get("scene_design") or "")
                elif target == "generated_visual_beat":
                    field_val = str(entry.get("visual_beat") or "")
                else:
                    field_val = ""
                if anchor and anchor not in field_val:
                    review.append({**p, "_skip_reason": "stale anchor — not found in current bundle"})
                    continue

            eligible.append(p)

        # Single-run gating
        is_plan_only = repeats < policy.min_repeats and not policy.allow_single_run
        if is_plan_only and eligible:
            for p in eligible:
                review.append({**p, "_skip_reason": f"plan-only: repeats={repeats} < {policy.min_repeats}"})
            eligible = []

        # Apply eligible patches
        if eligible:
            apply_result = apply_patches_to_entry(entry, eligible, policy)
            result["patches_applied"] = apply_result["applied"]
            result["patches_skipped"] = [
                {"patch": s["patch"], "reason": s["reason"]}
                for s in apply_result["skipped"]
            ]

            # Rebuild video_prompt if anything changed
            if apply_result["modified_fields"]:
                any_applied = True
                new_sd = apply_result["new_scene_design"] or str(entry.get("scene_design") or "")
                new_vb = apply_result["new_visual_beat"] or str(entry.get("visual_beat") or "")
                dur = float(entry.get("duration_seconds") or 10.0)
                sl = str(entry.get("effective_style_lock") or entry.get("style_lock") or "")

                try:
                    new_prompt = rebuild_video_prompt(new_sd, new_vb, dur, sl)
                    result["prompt_rebuilt"] = True
                    result["new_video_prompt"] = new_prompt
                    result["new_scene_design"] = new_sd
                    result["new_visual_beat"] = new_vb
                    result["status"] = "applied"
                    result["reason"] = "grounded patches applied and prompt rebuilt"
                except Exception as exc:
                    result["status"] = "error"
                    result["reason"] = f"rebuild failed: {exc}"
                    logger.exception("rebuild failed for %s", scene_id)
            else:
                result["status"] = "review_only" if review else "skipped"
                result["reason"] = "no patches changed text"
        else:
            result["status"] = "review_only" if (review or non_prompt_actions) else "skipped"
            result["reason"] = "no eligible patches"

        # Add review entries to skipped
        for r in review:
            skip_reason = r.pop("_skip_reason", "review-only")
            result["patches_skipped"].append({"patch": r, "reason": skip_reason})

        # Lint
        sd = result.get("new_scene_design") or str(entry.get("scene_design") or "")
        vb = result.get("new_visual_beat") or str(entry.get("visual_beat") or "")
        lint_warnings = lint_prompt(sd, vb, scene_id)
        result["lint_warnings"] = [
            {"rule": w.rule, "message": w.message, "location": w.location}
            for w in lint_warnings
        ]

        scene_results.append(result)

        # Build regen entry
        has_noncompliance = any(
            f.get("root_cause") == "generator_noncompliance"
            for f in findings if f.get("status") == "problem"
        )
        if result["status"] == "applied":
            category = "regenerate_prompt_changed"
            priority = 1
        elif has_noncompliance:
            category = "retry_generator_noncompliance"
            priority = 2
        elif result["status"] == "review_only":
            category = "review_only"
            priority = 3
        else:
            category = "skipped"
            priority = 4

        regen_entries.append({
            "unit_id": scene_id,
            "category": category,
            "reasons": [result.get("reason", "")],
            "priority": priority,
            "patches": result["patches_applied"],
            "non_prompt_actions": non_prompt_actions,
            "lint_warnings": result["lint_warnings"],
        })

    # --- Build outputs ---
    global_candidates = aggregate_global_candidates(
        [{"unit_id": s["unit_id"], "findings": (opt_scenes[i].get("findings") or []) if i < len(opt_scenes) else []}
         for i, s in enumerate(scene_results)]
    )

    # Build optimized bundle (copy, clear videos, record metadata)
    optimized_bundle = copy.deepcopy(bundle)
    old_videos = optimized_bundle.get("videos") or []
    optimized_bundle["videos"] = []  # Clear — old videos invalidated
    optimized_bundle["_optimization_meta"] = {
        "source_eval": str(eval_path),
        "source_bundle_hash": bundle_hash,
        "policy": {
            "min_confidence": policy.min_confidence,
            "min_repeats": policy.min_repeats,
            "allow_single_run": policy.allow_single_run,
        },
        "old_videos_invalidated": [
            {"unit_id": v.get("unit_id"), "path": v.get("path")}
            for v in old_videos if isinstance(v, dict)
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Apply changes to the optimized bundle
    for sr in scene_results:
        uid = sr["unit_id"]
        opt_entry = next(
            (e for e in optimized_bundle.get("video_prompts") or []
             if isinstance(e, dict) and str(e.get("unit_id")) == uid),
            None,
        )
        if opt_entry and sr["status"] == "applied":
            if sr.get("new_scene_design") is not None:
                opt_entry["scene_design"] = sr["new_scene_design"]
            if sr.get("new_visual_beat") is not None:
                opt_entry["visual_beat"] = sr["new_visual_beat"]
            if sr.get("new_video_prompt"):
                opt_entry["video_prompt"] = sr["new_video_prompt"]

    regeneration_plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_eval": str(eval_path),
        "source_bundle_hash": bundle_hash,
        "policy": {
            "min_confidence": policy.min_confidence,
            "min_repeats": policy.min_repeats,
            "allow_single_run": policy.allow_single_run,
        },
        "entries": sorted(regen_entries, key=lambda e: e["priority"]),
        "note": "This plan does NOT generate paid videos. Use it to guide regeneration.",
    }

    # Build markdown report
    report_lines = [
        f"# Prompt Optimization Report",
        f"",
        f"**Source evaluation**: `{eval_path.name}`  ",
        f"**Source bundle hash**: `{bundle_hash}`  ",
        f"**Eval repeats**: {repeats}  ",
        f"**Policy**: min_confidence={policy.min_confidence}, "
        f"min_repeats={policy.min_repeats}, "
        f"allow_single_run={policy.allow_single_run}  ",
        f"**Plan-only mode**: {'yes' if (repeats < policy.min_repeats and not policy.allow_single_run) else 'no'}  ",
        f"",
    ]
    counts = Counter(s["status"] for s in scene_results)
    report_lines.append(f"## Summary")
    report_lines.append(f"")
    for status, cnt in counts.most_common():
        report_lines.append(f"- **{status}**: {cnt} scenes")
    report_lines.append(f"")

    report_lines.append(f"## Scene Details")
    report_lines.append(f"")
    for sr in scene_results:
        uid = sr["unit_id"]
        st = sr["status"]
        report_lines.append(f"### {uid} — {st}")
        if sr.get("reason"):
            report_lines.append(f"Reason: {sr['reason']}")
        if sr["patches_applied"]:
            report_lines.append(f"Applied patches: {len(sr['patches_applied'])}")
            for p in sr["patches_applied"]:
                report_lines.append(f"  - `{p.get('target')}.{p.get('op')}`: "
                                    f"{p.get('anchor', '')[:40]!r} → {p.get('content', '')[:40]!r}")
        if sr["patches_skipped"]:
            report_lines.append(f"Skipped patches: {len(sr['patches_skipped'])}")
            for s in sr["patches_skipped"]:
                report_lines.append(f"  - {s['reason']}")
        if sr["lint_warnings"]:
            report_lines.append(f"Lint warnings:")
            for w in sr["lint_warnings"]:
                report_lines.append(f"  - ⚠ [{w['rule']}] {w['message']}")
        if sr.get("prompt_rebuilt"):
            report_lines.append(f"✅ Prompt rebuilt via assemble_video_prompt")
        report_lines.append(f"")

    if global_candidates:
        report_lines.append(f"## Global Rule Candidates")
        report_lines.append(f"")
        for gc in global_candidates:
            report_lines.append(
                f"- **{gc['root_cause']}**: {gc['count']} occurrences, "
                f"confidence={gc['confidence']}, "
                f"units={', '.join(gc['evidence_units'])}"
            )
            report_lines.append(f"  - Candidate rule: {gc['suggested_rule']}")
        report_lines.append(f"")

    report_md = "\n".join(report_lines)

    # --- Output ---
    if not output_dir:
        run_dir = vp_path.parent
        output_dir = str(run_dir / "optimizations" / _eval_stem(str(eval_path)))

    out_path = Path(output_dir)

    artifacts = {
        "output_dir": str(out_path),
        "optimized_video_prompts": optimized_bundle,
        "regeneration_plan": regeneration_plan,
        "optimization_report": report_md,
        "global_rule_candidates": global_candidates,
        "scene_results": scene_results,
        "summary": {
            "total_scenes": len(scene_results),
            "applied": counts.get("applied", 0),
            "review_only": counts.get("review_only", 0),
            "skipped": counts.get("skipped", 0),
            "errors": counts.get("error", 0),
            "plan_only": repeats < policy.min_repeats and not policy.allow_single_run,
            "lint_warning_count": sum(len(s["lint_warnings"]) for s in scene_results),
            "global_candidates": len(global_candidates),
        },
    }

    if dry_run:
        artifacts["dry_run"] = True
        return artifacts

    # Write files
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "optimized_video_prompts.json").write_text(
        json.dumps(optimized_bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_path / "regeneration_plan.json").write_text(
        json.dumps(regeneration_plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_path / "optimization_report.md").write_text(report_md, encoding="utf-8")
    (out_path / "global_rule_candidates.json").write_text(
        json.dumps(global_candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artifacts["files_written"] = [
        str(out_path / "optimized_video_prompts.json"),
        str(out_path / "regeneration_plan.json"),
        str(out_path / "optimization_report.md"),
        str(out_path / "global_rule_candidates.json"),
    ]
    return artifacts
