"""Post-evaluation prompt optimization: grounded attribution + actionable patches.

After all judge repeats finish, exactly one TEXT-ONLY structured-output call per
scene (when enabled) receives provenance and all successful round deductions.
Scores and verdicts remain untouched.

Programmatically validates exact quotes and patch anchors against their target
fields; valid and rejected patches are kept separate.  `append` may work on
empty fields.  Invalid patches must never appear actionable.

Missing provenance yields ``unassessable`` / skip — not crash.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from shot_eval import prompts
from shot_eval.models import (
    ATTRIBUTION_STAGES,
    PATCH_TARGETS,
    Finding,
    NonPromptAction,
    OptimizationResult,
    PromptPatch,
    Provenance,
)

logger = logging.getLogger("shot_eval.optimize")


# ---------------------------------------------------------------------------
# Grounding validation
# ---------------------------------------------------------------------------

def _resolve_target_value(provenance: Provenance, target: str) -> str:
    """Get the text value of a provenance field by patch target name."""
    # Map patch target names to provenance field names
    field_map = {"style_lock": "effective_style_lock"}
    field_name = field_map.get(target, target)
    return str(getattr(provenance, field_name, "") or "")


def validate_quote(
    finding: Finding,
    provenance: Provenance,
    video_evidence: str = "",
) -> bool:
    """验证 problem finding 的逐字证据；pass/unassessable 不强迫伪造引用。"""
    if finding.status != "problem":
        return True
    if not finding.exact_quote:
        return False
    if finding.source_field == "video_evidence":
        return finding.exact_quote in video_evidence
    field_value = _resolve_target_value(provenance, finding.source_field)
    return finding.exact_quote in field_value


def validate_patch(patch: PromptPatch, provenance: Provenance) -> bool:
    """校验 patch 是否可机械执行，而不只是“建议看起来合理”。"""
    if patch.target not in PATCH_TARGETS:
        return False
    if patch.op in {"replace", "insert_after", "append"} and not patch.content.strip():
        return False
    field_value = _resolve_target_value(provenance, patch.target)
    if patch.op == "append":
        return True
    if not patch.anchor:
        return False
    return patch.anchor in field_value


def ground_optimization(
    raw: OptimizationResult,
    provenance: Provenance,
    video_evidence: str = "",
) -> dict[str, Any]:
    """校验 finding 和 patch；未通过的 patch 只进入 rejected_patches。"""
    valid_patches: list[dict[str, Any]] = []
    rejected_patches: list[dict[str, Any]] = []

    for patch in raw.patches:
        dump = patch.model_dump()
        if validate_patch(patch, provenance):
            valid_patches.append(dump)
        else:
            if patch.op in {"replace", "insert_after", "append"} and not patch.content.strip():
                reason = f"{patch.op} requires non-empty content"
            elif patch.op != "append":
                reason = f"anchor {patch.anchor!r} not found in {patch.target}"
            else:
                reason = f"invalid target {patch.target!r}"
            dump["rejection_reason"] = reason
            rejected_patches.append(dump)

    findings_out: list[dict[str, Any]] = []
    for finding in raw.findings:
        dump = finding.model_dump()
        dump["quote_grounded"] = validate_quote(finding, provenance, video_evidence)
        findings_out.append(dump)

    seen_stages = {f["stage"] for f in findings_out}
    for stage in ATTRIBUTION_STAGES:
        if stage not in seen_stages:
            findings_out.append({
                "stage": stage,
                "status": "unassessable",
                "problem": "优化模型未返回该阶段结论",
                "severity": "low",
                "root_cause": "insufficient_evidence",
                "source_field": "script_fragment",
                "exact_quote": "",
                "confidence": 0.0,
                "quote_grounded": True,
            })

    return {
        "findings": findings_out,
        "valid_patches": valid_patches,
        "rejected_patches": rejected_patches,
        "non_prompt_actions": [a.model_dump() for a in raw.non_prompt_actions],
    }


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def build_optimization_prompt(
    provenance: Provenance,
    deductions: list[list[dict[str, Any]]],
) -> str:
    """Build the TEXT-ONLY optimization prompt from provenance + deductions.

    Uses the ``optimization`` template from prompts.yaml.
    """
    # 保留轮次，优化器才能区分稳定问题与单轮噪声。
    all_deductions: list[dict[str, Any]] = []
    for round_index, round_deds in enumerate(deductions, 1):
        for d in round_deds:
            all_deductions.append({"round": round_index, **d})

    prov_block = json.dumps({
        "script_fragment": provenance.script_fragment,
        "script_context": provenance.script_context,
        "plan_scene_design": provenance.plan_scene_design,
        "plan_visual_beats": provenance.plan_visual_beats,
        "generated_scene_design": provenance.generated_scene_design,
        "generated_visual_beat": provenance.generated_visual_beat,
        "effective_style_lock": provenance.effective_style_lock,
        "render_mode": provenance.render_mode,
        "final_video_prompt": provenance.final_video_prompt,
        "unit_granularity": provenance.unit_granularity,
        "analysis": provenance.analysis,
    }, ensure_ascii=False, indent=2)

    ded_block = json.dumps(all_deductions, ensure_ascii=False, indent=2)

    return prompts.render(
        "optimization",
        provenance_json=prov_block,
        deductions_json=ded_block,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def optimize_scenes(
    scenes_data: list[dict[str, Any]],
    tasks: Any,
    *,
    model: str | None = None,
    optimization_model: str | None = None,
) -> list[dict[str, Any]]:
    """Run exactly one optimization call per scene.

    Returns a list of optimization results (one per scene), parallel to
    ``scenes_data``.  Missing provenance yields an ``unassessable`` skip.
    Errors are caught per-scene and recorded, never crash the batch.
    """
    from shot_eval import judge

    used_model = judge.judge_model(optimization_model or model)
    results: list[dict[str, Any]] = []

    for i, (scene_result, task) in enumerate(zip(scenes_data, tasks)):
        provenance = task.provenance or task.item.provenance
        scene_id = task.item.scene_id

        if provenance is None:
            results.append({
                "scene_id": scene_id,
                "status": "unassessable",
                "reason": "missing provenance",
                "findings": [],
                "valid_patches": [],
                "rejected_patches": [],
                "non_prompt_actions": [],
            })
            continue

        # 收集所有成功轮次；空 deductions 仍需检查两个上游阶段。
        rounds = scene_result.get("rounds") or []
        round_deductions = [
            r.get("deductions") or []
            for r in rounds
            if "error" not in r
        ]

        try:
            prompt_text = build_optimization_prompt(provenance, round_deductions)
            response = await judge._generate(
                model=used_model,
                contents=[prompt_text],
                config=judge._config(OptimizationResult),
            )
            raw = OptimizationResult.model_validate_json(response.text)
            video_evidence = "\n".join(
                str(d.get("evidence") or "")
                for round_items in round_deductions
                for d in round_items
                if d.get("evidence")
            )
            grounded = ground_optimization(raw, provenance, video_evidence)
            grounded["scene_id"] = scene_id
            grounded["status"] = "ok"
            grounded["model"] = used_model
            results.append(grounded)

        except Exception as exc:
            logger.exception("optimization failed for scene %s", scene_id)
            results.append({
                "scene_id": scene_id,
                "status": "error",
                "error": str(exc),
                "findings": [],
                "valid_patches": [],
                "rejected_patches": [],
                "non_prompt_actions": [],
            })

    return results
