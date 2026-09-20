"""整片总分：把逐 shot 评审结果聚合成一个 0–100 的加权诊断分。

设计要点
--------
1. **逐片段先聚合、再跨片段取算术平均**——而不是先攒全部分数再取跨片段中位数。
   中位数天然抗极端值，一个 critical 连扣 3 分的灾难段会被 20 个正常段淹掉，
   诊断用途下掩盖坏 shot 比偶尔受异常值拉低更危险。

2. 每个维度块（dimension block）= 各个已评场景中位分的算术平均 / 5 × 100。
   「已评场景」要求：该维度在本场景有 aggregated median，且该维度不在
   unreferenced 列表里。

3. render_match 块 = 全部已评场景的 matched / judged × 100。
   visual_compliance 块 = 成功原始轮次中三项合规标志全 False 的占比 × 100。

4. 覆盖权重仅包含有数据的块；缺失块降权而不是判 0。
   总分 = Σ(block_pct × weight) / covered_weight，四舍五入到两位。

5. 只用标准库。不依赖 pydantic、numpy 等外部包。

公共接口
--------
score_result(result) → dict   唯一入口，接收 bench.run_batch 或落盘 JSON 的顶层 dict。
"""

from __future__ import annotations

import statistics
from typing import Any

# ---------------------------------------------------------------------------
# 权重表
# ---------------------------------------------------------------------------

BLOCK_WEIGHTS: dict[str, int] = {
    "science_accuracy": 20,
    "beat_alignment": 12,
    "clarity": 10,
    "design_fidelity": 8,
    "instructional_value": 8,
    "style_adherence": 4,
    "render_match": 6,
    "visual_compliance": 4,
}

MAX_WEIGHT: int = sum(BLOCK_WEIGHTS.values())  # 72

#: 这六个维度走「逐场景中位分→算术平均」路径
DIMENSION_BLOCKS: tuple[str, ...] = (
    "science_accuracy",
    "beat_alignment",
    "clarity",
    "design_fidelity",
    "instructional_value",
    "style_adherence",
)

# ---------------------------------------------------------------------------
# 等级
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
]


def _grade(score: float) -> str:
    for threshold, letter in _GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "E"


# ---------------------------------------------------------------------------
# 维度块评分
# ---------------------------------------------------------------------------


def _dimension_block_pct(scenes: list[dict[str, Any]], dim: str) -> tuple[float | None, int]:
    """单维度块百分制分数 + 样本场景数。

    算法：遍历每个场景，如果该场景对该维度有 aggregated median 且不在
    unreferenced 列表中，取该 median。跨场景做算术平均（不是中位数），
    再除以 5 × 100 转百分制。

    返回 (pct_or_None, sample_count)。
    """
    medians: list[float] = []
    for scene in scenes:
        dims = scene.get("dims") or {}
        unreferenced = scene.get("unreferenced") or []
        if dim in unreferenced:
            continue
        stat = dims.get(dim)
        if not stat:
            continue
        med = stat.get("median")
        if med is None:
            continue
        medians.append(float(med))
    if not medians:
        return None, 0
    mean_score = sum(medians) / len(medians)
    pct = mean_score / 5.0 * 100.0
    return round(pct, 4), len(medians)


# ---------------------------------------------------------------------------
# render_match 块
# ---------------------------------------------------------------------------


def _render_block_pct(scenes: list[dict[str, Any]]) -> tuple[float | None, int]:
    """render_match 百分制 = matched / judged × 100。

    返回 (pct_or_None, judged_count)。
    """
    total_judged = 0
    total_matched = 0
    for scene in scenes:
        rm = scene.get("render_match")
        if not rm:
            continue
        judged = rm.get("judged", 0)
        matched = rm.get("matched", 0)
        total_judged += judged
        total_matched += matched
    if total_judged == 0:
        return None, 0
    pct = total_matched / total_judged * 100.0
    return round(pct, 4), total_judged


# ---------------------------------------------------------------------------
# visual_compliance 块
# ---------------------------------------------------------------------------

_COMPLIANCE_FLAGS = ("has_subtitle_or_caption", "has_watermark_or_logo", "has_brand_name")


def _compliance_block_pct(scenes: list[dict[str, Any]]) -> tuple[float | None, int]:
    """visual_compliance = 成功轮次中三项合规标志全为 False 的百分比。

    只看无 error 的轮次（「成功的原始轮次」），且该轮次必须包含所有三个字段。
    返回 (pct_or_None, total_rounds_checked)。
    """
    total = 0
    clean = 0
    for scene in scenes:
        rounds = scene.get("rounds") or []
        for r in rounds:
            if "error" in r:
                continue
            # 必须三个字段都存在才算有效轮次
            if not all(k in r for k in _COMPLIANCE_FLAGS):
                continue
            total += 1
            if not any(r.get(k) for k in _COMPLIANCE_FLAGS):
                clean += 1
    if total == 0:
        return None, 0
    pct = clean / total * 100.0
    return round(pct, 4), total


# ---------------------------------------------------------------------------
# 严重度计数
# ---------------------------------------------------------------------------


def _severity_counts(scenes: list[dict[str, Any]]) -> dict[str, int]:
    """统计全部场景所有成功轮次中的 critical / major / minor 扣分项数。"""
    counts: dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
    for scene in scenes:
        for r in scene.get("rounds") or []:
            if "error" in r:
                continue
            for d in r.get("deductions") or []:
                sev = d.get("severity", "")
                if sev in counts:
                    counts[sev] += 1
    return counts


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def score_result(result: dict[str, Any]) -> dict[str, Any]:
    """为一份 bench 结果计算加权总分。

    Parameters
    ----------
    result : dict
        ``bench.run_batch`` 返回的顶层字典，或落盘后重新加载的 JSON。
        至少需要 ``scenes`` 键；缺失时返回空分。

    Returns
    -------
    dict 包含:
        score       : float | None 0–100；无有效覆盖时为 None
        grade       : str | None   A/B/C/D/E；无有效覆盖时为 None
        covered_weight : int      实际参与计算的权重之和
        max_weight  : int         72
        complete    : bool        covered_weight == max_weight
        blocks      : dict[str, block_info]
        severity    : dict        {critical, major, minor}
    """
    scenes = result.get("scenes") or []

    blocks: dict[str, dict[str, Any]] = {}
    covered_weight = 0
    weighted_sum = 0.0

    # 1) 六个维度块
    for dim in DIMENSION_BLOCKS:
        pct, sample = _dimension_block_pct(scenes, dim)
        weight = BLOCK_WEIGHTS[dim]
        if pct is not None:
            contribution = round(pct * weight, 4)
            covered_weight += weight
            weighted_sum += contribution
            blocks[dim] = {
                "pct": round(pct, 2),
                "weight": weight,
                "contribution": round(contribution, 2),
                "sample": sample,
            }
        else:
            blocks[dim] = {
                "pct": None,
                "weight": weight,
                "contribution": 0,
                "sample": 0,
            }

    # 2) render_match 块
    rm_pct, rm_sample = _render_block_pct(scenes)
    rm_weight = BLOCK_WEIGHTS["render_match"]
    if rm_pct is not None:
        contribution = round(rm_pct * rm_weight, 4)
        covered_weight += rm_weight
        weighted_sum += contribution
        blocks["render_match"] = {
            "pct": round(rm_pct, 2),
            "weight": rm_weight,
            "contribution": round(contribution, 2),
            "sample": rm_sample,
        }
    else:
        blocks["render_match"] = {
            "pct": None,
            "weight": rm_weight,
            "contribution": 0,
            "sample": 0,
        }

    # 3) visual_compliance 块
    vc_pct, vc_sample = _compliance_block_pct(scenes)
    vc_weight = BLOCK_WEIGHTS["visual_compliance"]
    if vc_pct is not None:
        contribution = round(vc_pct * vc_weight, 4)
        covered_weight += vc_weight
        weighted_sum += contribution
        blocks["visual_compliance"] = {
            "pct": round(vc_pct, 2),
            "weight": vc_weight,
            "contribution": round(contribution, 2),
            "sample": vc_sample,
        }
    else:
        blocks["visual_compliance"] = {
            "pct": None,
            "weight": vc_weight,
            "contribution": 0,
            "sample": 0,
        }

    # 4) 总分
    if covered_weight > 0:
        score: float | None = round(weighted_sum / covered_weight, 2)
        grade: str | None = _grade(score)
    else:
        score = None
        grade = None

    # 5) 严重度统计
    severity = _severity_counts(scenes)

    return {
        "score": score,
        "grade": grade,
        "covered_weight": covered_weight,
        "max_weight": MAX_WEIGHT,
        "complete": covered_weight == MAX_WEIGHT,
        "blocks": blocks,
        "severity": severity,
    }
