"""No-network unit tests for shot_eval.scoring and its integration points."""

from __future__ import annotations

import json
import unittest
from typing import Any

from shot_eval.scoring import (
    BLOCK_WEIGHTS,
    DIMENSION_BLOCKS,
    MAX_WEIGHT,
    score_result,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic result dicts without touching any JSON files.
# ---------------------------------------------------------------------------

def _make_round(
    *,
    dims: dict[str, float] | None = None,
    deductions: list[dict[str, Any]] | None = None,
    has_subtitle: bool = False,
    has_watermark: bool = False,
    has_brand: bool = False,
    render_verdict: str | None = None,
    render_match: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a single round verdict dict."""
    r: dict[str, Any] = {
        "observed": "test",
        "has_subtitle_or_caption": has_subtitle,
        "has_watermark_or_logo": has_watermark,
        "has_brand_name": has_brand,
        "deductions": deductions or [],
    }
    if dims:
        for k, v in dims.items():
            r[k] = v
            r[f"{k}_reason"] = "test"
    if render_verdict:
        r["render_verdict"] = render_verdict
    if render_match is not None:
        r["render_match"] = render_match
    if error is not None:
        r["error"] = error
    return r


def _make_scene(
    scene_id: str,
    *,
    dim_medians: dict[str, float] | None = None,
    unreferenced: list[str] | None = None,
    render_match: dict[str, Any] | None = None,
    rounds: list[dict[str, Any]] | None = None,
    judged: int = 1,
) -> dict[str, Any]:
    """Build a scene aggregate dict (the shape that bench._aggregate produces)."""
    dims = {}
    for dim, med in (dim_medians or {}).items():
        dims[dim] = {"median": med, "spread": 0.0, "values": [med]}
    return {
        "key": f"test/{scene_id}",
        "source": "test",
        "origin": "shot",
        "scene_id": scene_id,
        "render_declared": "photoreal",
        "context": {},
        "rounds": rounds or [_make_round()],
        "errors": [],
        "dims": dims,
        "hits": {},
        "unreferenced": unreferenced or [],
        "render_match": render_match,
        "judged": judged,
        "deduction_count": [0],
    }


def _make_result(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenes": scenes,
        "judge_model": "test-model",
        "thinking_level": "none",
        "prompt_version_tag": "test-v1",
        "repeats": 1,
        "elapsed": 1.0,
        "config": {
            "beat_source": "visual_beat",
            "style_source": "style_lock",
            "render_source": "blind",
        },
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestPerfectScore(unittest.TestCase):
    """All blocks present, all scoring 100% → overall = 100."""

    def test_perfect_100(self):
        scene = _make_scene(
            "S01",
            dim_medians={d: 5.0 for d in DIMENSION_BLOCKS},
            render_match={"judged": 3, "matched": 3, "observed": ["photoreal"] * 3},
            rounds=[
                _make_round(
                    dims={d: 5.0 for d in DIMENSION_BLOCKS},
                    render_match=True,
                    render_verdict="photoreal",
                ),
                _make_round(
                    dims={d: 5.0 for d in DIMENSION_BLOCKS},
                    render_match=True,
                    render_verdict="photoreal",
                ),
                _make_round(
                    dims={d: 5.0 for d in DIMENSION_BLOCKS},
                    render_match=True,
                    render_verdict="photoreal",
                ),
            ],
            judged=3,
        )
        result = _make_result([scene])
        ov = score_result(result)
        self.assertEqual(ov["score"], 100.0)
        self.assertEqual(ov["grade"], "A")
        self.assertTrue(ov["complete"])
        self.assertEqual(ov["covered_weight"], MAX_WEIGHT)
        self.assertEqual(ov["max_weight"], 72)


class TestDimensionWeightedLoss(unittest.TestCase):
    """One dimension scores below 5 → overall drops proportionally."""

    def test_single_dimension_loss(self):
        """science_accuracy at 3.0/5 (60%) with weight 20 → overall < 100."""
        medians = {d: 5.0 for d in DIMENSION_BLOCKS}
        medians["science_accuracy"] = 3.0
        scene = _make_scene(
            "S01",
            dim_medians=medians,
            render_match={"judged": 1, "matched": 1, "observed": ["photoreal"]},
            rounds=[_make_round(
                dims=medians,
                render_match=True,
                render_verdict="photoreal",
            )],
        )
        result = _make_result([scene])
        ov = score_result(result)
        # science_accuracy: 3/5*100 = 60, weight 20
        # rest: 100, their weights sum to 72-20 = 52
        # render_match: 100, weight 6
        # compliance: 100, weight 4
        # total = (60*20 + 100*52) / 72 = (1200 + 5200) / 72 ≈ 88.89
        self.assertAlmostEqual(ov["score"], 88.89, places=2)
        self.assertEqual(ov["grade"], "B")
        self.assertTrue(ov["complete"])

    def test_multi_scene_mean_not_median(self):
        """Two scenes: one perfect, one terrible → mean exposes the bad one."""
        good = _make_scene("S01", dim_medians={"science_accuracy": 5.0},
                           rounds=[_make_round(dims={"science_accuracy": 5.0})])
        bad = _make_scene("S02", dim_medians={"science_accuracy": 1.0},
                          rounds=[_make_round(dims={"science_accuracy": 1.0})])
        result = _make_result([good, bad])
        ov = score_result(result)
        # mean = (5+1)/2 = 3.0 → 60%
        blk = ov["blocks"]["science_accuracy"]
        self.assertAlmostEqual(blk["pct"], 60.0, places=1)
        self.assertEqual(blk["sample"], 2)


class TestRenderMismatch(unittest.TestCase):
    """render_match block handles mismatches and absences."""

    def test_partial_mismatch(self):
        scene = _make_scene(
            "S01",
            dim_medians={"beat_alignment": 5.0},
            render_match={"judged": 3, "matched": 1, "observed": ["cg", "cg", "photoreal"]},
            rounds=[
                _make_round(render_match=False, render_verdict="cg"),
                _make_round(render_match=False, render_verdict="cg"),
                _make_round(render_match=True, render_verdict="photoreal"),
            ],
        )
        result = _make_result([scene])
        ov = score_result(result)
        blk = ov["blocks"]["render_match"]
        self.assertAlmostEqual(blk["pct"], 33.33, places=2)

    def test_no_render_eval_omits_weight(self):
        scene = _make_scene("S01", dim_medians={"beat_alignment": 5.0})
        result = _make_result([scene])
        ov = score_result(result)
        self.assertIsNone(ov["blocks"]["render_match"]["pct"])
        self.assertNotIn(BLOCK_WEIGHTS["render_match"],
                         [ov["covered_weight"]])  # not counted


class TestComplianceViolation(unittest.TestCase):
    """visual_compliance block handles violations and missing data."""

    def test_subtitle_violation(self):
        scene = _make_scene(
            "S01",
            dim_medians={"clarity": 5.0},
            rounds=[
                _make_round(has_subtitle=True),
                _make_round(has_subtitle=False),
            ],
        )
        result = _make_result([scene])
        ov = score_result(result)
        blk = ov["blocks"]["visual_compliance"]
        self.assertAlmostEqual(blk["pct"], 50.0)
        self.assertEqual(blk["sample"], 2)

    def test_all_clean(self):
        scene = _make_scene(
            "S01",
            dim_medians={"clarity": 5.0},
            rounds=[_make_round(), _make_round()],
        )
        result = _make_result([scene])
        ov = score_result(result)
        blk = ov["blocks"]["visual_compliance"]
        self.assertAlmostEqual(blk["pct"], 100.0)

    def test_no_compliance_data_omits(self):
        """Error rounds don't count; if all rounds are errors, block is omitted."""
        scene = _make_scene(
            "S01",
            dim_medians={"clarity": 5.0},
            rounds=[_make_round(error="boom")],
        )
        result = _make_result([scene])
        ov = score_result(result)
        self.assertIsNone(ov["blocks"]["visual_compliance"]["pct"])


class TestMissingUnreferencedReweighting(unittest.TestCase):
    """Missing or unreferenced blocks properly reduce covered_weight."""

    def test_unreferenced_dimension_excluded(self):
        scene = _make_scene(
            "S01",
            dim_medians={"beat_alignment": 5.0, "science_accuracy": 5.0},
            unreferenced=["beat_alignment"],
        )
        result = _make_result([scene])
        ov = score_result(result)
        # beat_alignment excluded → only science_accuracy counted
        blk = ov["blocks"]["beat_alignment"]
        self.assertIsNone(blk["pct"])
        self.assertEqual(blk["sample"], 0)
        # covered_weight should not include beat_alignment weight
        self.assertFalse(ov["complete"])

    def test_all_unreferenced_excludes_all(self):
        scene = _make_scene(
            "S01",
            dim_medians={d: 5.0 for d in DIMENSION_BLOCKS},
            unreferenced=list(DIMENSION_BLOCKS),
        )
        result = _make_result([scene])
        ov = score_result(result)
        # All dims unreferenced → only compliance block might be present
        for d in DIMENSION_BLOCKS:
            self.assertIsNone(ov["blocks"][d]["pct"])

    def test_missing_dimension_data(self):
        """Scene has no data for a dimension at all → block gets None."""
        scene = _make_scene("S01", dim_medians={"beat_alignment": 4.0})
        result = _make_result([scene])
        ov = score_result(result)
        self.assertIsNone(ov["blocks"]["science_accuracy"]["pct"])
        self.assertIsNotNone(ov["blocks"]["beat_alignment"]["pct"])

    def test_coverage_renormalization(self):
        """With only beat_alignment (weight 12) and compliance (weight 4),
        total = (100*12 + 100*4) / 16 = 100."""
        scene = _make_scene(
            "S01",
            dim_medians={"beat_alignment": 5.0},
            rounds=[_make_round(dims={"beat_alignment": 5.0})],
        )
        result = _make_result([scene])
        ov = score_result(result)
        # covered = beat_alignment(12) + compliance(4)
        self.assertEqual(ov["covered_weight"], 16)
        self.assertAlmostEqual(ov["score"], 100.0)


class TestNoData(unittest.TestCase):
    """Empty or missing scenes → no score rather than a misleading E."""

    def test_empty_scenes(self):
        result = _make_result([])
        ov = score_result(result)
        self.assertIsNone(ov["score"])
        self.assertIsNone(ov["grade"])
        self.assertEqual(ov["covered_weight"], 0)
        self.assertFalse(ov["complete"])

    def test_missing_scenes_key(self):
        ov = score_result({})
        self.assertIsNone(ov["score"])
        self.assertIsNone(ov["grade"])

    def test_none_scenes(self):
        ov = score_result({"scenes": None})
        self.assertIsNone(ov["score"])


class TestSeverityCounts(unittest.TestCase):
    """Severity counts aggregate across all successful rounds."""

    def test_counts(self):
        rounds = [
            _make_round(deductions=[
                {"dimension": "science_accuracy", "severity": "critical",
                 "timestamp": "00:01", "evidence": "x", "expected": "y"},
                {"dimension": "clarity", "severity": "major",
                 "timestamp": "00:02", "evidence": "x", "expected": "y"},
            ]),
            _make_round(deductions=[
                {"dimension": "clarity", "severity": "minor",
                 "timestamp": "00:01", "evidence": "x", "expected": "y"},
            ]),
            _make_round(error="boom"),  # should be ignored
        ]
        scene = _make_scene("S01", dim_medians={"science_accuracy": 2.0},
                            rounds=rounds)
        result = _make_result([scene])
        ov = score_result(result)
        self.assertEqual(ov["severity"]["critical"], 1)
        self.assertEqual(ov["severity"]["major"], 1)
        self.assertEqual(ov["severity"]["minor"], 1)

    def test_no_deductions(self):
        scene = _make_scene("S01", dim_medians={"clarity": 5.0})
        result = _make_result([scene])
        ov = score_result(result)
        self.assertEqual(ov["severity"]["critical"], 0)
        self.assertEqual(ov["severity"]["major"], 0)
        self.assertEqual(ov["severity"]["minor"], 0)


class TestGrades(unittest.TestCase):
    """Grade thresholds."""

    def test_grade_A(self):
        from shot_eval.scoring import _grade
        self.assertEqual(_grade(90.0), "A")
        self.assertEqual(_grade(100.0), "A")

    def test_grade_B(self):
        from shot_eval.scoring import _grade
        self.assertEqual(_grade(80.0), "B")
        self.assertEqual(_grade(89.99), "B")

    def test_grade_C(self):
        from shot_eval.scoring import _grade
        self.assertEqual(_grade(70.0), "C")
        self.assertEqual(_grade(79.99), "C")

    def test_grade_D(self):
        from shot_eval.scoring import _grade
        self.assertEqual(_grade(60.0), "D")
        self.assertEqual(_grade(69.99), "D")

    def test_grade_E(self):
        from shot_eval.scoring import _grade
        self.assertEqual(_grade(59.99), "E")
        self.assertEqual(_grade(0.0), "E")


class TestOldReports(unittest.TestCase):
    """Old reports without new fields should not crash."""

    def test_no_rounds_key(self):
        """Scene without rounds → compliance block omitted, no crash."""
        scene = {"key": "old/S01", "source": "old", "scene_id": "S01",
                 "dims": {"beat_alignment": {"median": 4.0, "spread": 0, "values": [4.0]}},
                 "unreferenced": [], "hits": {}, "judged": 1,
                 "deduction_count": [0], "errors": []}
        result = _make_result([scene])
        ov = score_result(result)
        self.assertIsNotNone(ov["score"])
        self.assertIn("grade", ov)

    def test_no_deduction_count(self):
        scene = _make_scene("S01", dim_medians={"clarity": 5.0})
        del scene["deduction_count"]
        result = _make_result([scene])
        # Should not crash
        ov = score_result(result)
        self.assertIsNotNone(ov["score"])

    def test_scene_with_error_rounds_only(self):
        scene = _make_scene(
            "S01",
            dim_medians={"clarity": 3.0},
            rounds=[_make_round(error="timeout"), _make_round(error="ratelimit")],
        )
        result = _make_result([scene])
        ov = score_result(result)
        # No crash, compliance block omitted (all rounds errored)
        self.assertIsNone(ov["blocks"]["visual_compliance"]["pct"])


class TestReportPayloadHTML(unittest.TestCase):
    """Integration: report.build_payload and render_html include overall."""

    def test_build_payload_includes_overall(self):
        from shot_eval import report
        scene = _make_scene("S01", dim_medians={"beat_alignment": 5.0})
        result = _make_result([scene])
        result["tasks"] = [{"key": "test/S01", "clip": "/tmp/S01.mp4"}]
        payload = report.build_payload(result)
        self.assertIn("overall", payload)
        self.assertEqual(payload["overall"]["max_weight"], 72)
        self.assertIn("score", payload["overall"])

    def test_render_html_contains_overall(self):
        from shot_eval import report
        scene = _make_scene("S01", dim_medians={"beat_alignment": 5.0})
        result = _make_result([scene])
        result["tasks"] = [{"key": "test/S01", "clip": "/tmp/S01.mp4"}]
        payload = report.build_payload(result)
        html = report.render_html(payload, "test")
        self.assertIn("OVERALL", html)
        self.assertIn("Overall", html)

    def test_summarize_includes_overall(self):
        from shot_eval import report
        scene = _make_scene("S01", dim_medians={"beat_alignment": 4.0})
        result = _make_result([scene])
        brief = report.summarize(result)
        self.assertIn("overall", brief)
        self.assertIn("score", brief["overall"])
        self.assertIn("grade", brief["overall"])


class TestServeIndex(unittest.TestCase):
    """Integration: serve.index_page includes Overall column."""

    def test_index_has_overall_header(self):
        from shot_eval import serve
        # Craft minimal Store-like object
        class FakeStore:
            def load(self, path):
                scene = _make_scene("S01", dim_medians={d: 5.0 for d in DIMENSION_BLOCKS})
                return _make_result([scene])

        from pathlib import Path
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "shots-test.json"
            fp.write_text("{}")  # content unused, store is fake
            run = serve.Run(rid="test", path=fp, root=Path(td))
            page = serve.index_page([run], FakeStore())
            text = page.decode("utf-8")
            self.assertIn("<th>Overall</th>", text)
            self.assertIn("Overall", text)


class TestBlockWeights(unittest.TestCase):
    """Sanity: weight table matches spec."""

    def test_max_weight_is_72(self):
        self.assertEqual(MAX_WEIGHT, 72)

    def test_weights_match(self):
        self.assertEqual(BLOCK_WEIGHTS["science_accuracy"], 20)
        self.assertEqual(BLOCK_WEIGHTS["beat_alignment"], 12)
        self.assertEqual(BLOCK_WEIGHTS["clarity"], 10)
        self.assertEqual(BLOCK_WEIGHTS["design_fidelity"], 8)
        self.assertEqual(BLOCK_WEIGHTS["instructional_value"], 8)
        self.assertEqual(BLOCK_WEIGHTS["style_adherence"], 4)
        self.assertEqual(BLOCK_WEIGHTS["render_match"], 6)
        self.assertEqual(BLOCK_WEIGHTS["visual_compliance"], 4)


class TestReturnShape(unittest.TestCase):
    """Full return dict has all required keys."""

    def test_all_keys_present(self):
        scene = _make_scene("S01", dim_medians={"beat_alignment": 4.0})
        result = _make_result([scene])
        ov = score_result(result)
        self.assertIn("score", ov)
        self.assertIn("grade", ov)
        self.assertIn("covered_weight", ov)
        self.assertIn("max_weight", ov)
        self.assertIn("complete", ov)
        self.assertIn("blocks", ov)
        self.assertIn("severity", ov)
        # Check block shape
        for bname, binfo in ov["blocks"].items():
            self.assertIn("pct", binfo)
            self.assertIn("weight", binfo)
            self.assertIn("contribution", binfo)
            self.assertIn("sample", binfo)


class TestComplexScenario(unittest.TestCase):
    """Integration test with multiple scenes and all block types."""

    def test_three_scenes_mixed(self):
        """Three scenes: one perfect, one with bad science, one with render mismatch."""
        perfect = _make_scene(
            "S01",
            dim_medians={d: 5.0 for d in DIMENSION_BLOCKS},
            render_match={"judged": 1, "matched": 1, "observed": ["photoreal"]},
            rounds=[_make_round(
                dims={d: 5.0 for d in DIMENSION_BLOCKS},
                render_match=True, render_verdict="photoreal",
            )],
        )
        bad_sci = _make_scene(
            "S02",
            dim_medians={**{d: 5.0 for d in DIMENSION_BLOCKS}, "science_accuracy": 2.0},
            render_match={"judged": 1, "matched": 1, "observed": ["photoreal"]},
            rounds=[_make_round(
                dims={**{d: 5.0 for d in DIMENSION_BLOCKS}, "science_accuracy": 2.0},
                render_match=True, render_verdict="photoreal",
            )],
        )
        render_bad = _make_scene(
            "S03",
            dim_medians={d: 5.0 for d in DIMENSION_BLOCKS},
            render_match={"judged": 1, "matched": 0, "observed": ["cg"]},
            rounds=[_make_round(
                dims={d: 5.0 for d in DIMENSION_BLOCKS},
                render_match=False, render_verdict="cg",
            )],
        )
        result = _make_result([perfect, bad_sci, render_bad])
        ov = score_result(result)
        # science_accuracy mean = (5+2+5)/3 = 4.0 → 80%
        self.assertAlmostEqual(ov["blocks"]["science_accuracy"]["pct"], 80.0, places=1)
        # render_match = 2/3 → 66.67%
        self.assertAlmostEqual(ov["blocks"]["render_match"]["pct"], 66.67, places=2)
        # All dim blocks present + render + compliance → complete
        self.assertTrue(ov["complete"])
        self.assertTrue(0 < ov["score"] < 100)
        self.assertIn(ov["grade"], ("A", "B", "C", "D", "E"))


if __name__ == "__main__":
    unittest.main()


class TestDimensionSummaryMean(unittest.TestCase):
    """Frontend summary uses arithmetic mean as primary while retaining median."""

    def test_summarize_has_distinct_mean_and_median(self):
        from shot_eval.report import summarize

        scenes = [
            _make_scene("S1", dim_medians={"science_accuracy": 5.0}),
            _make_scene("S2", dim_medians={"science_accuracy": 5.0}),
            _make_scene("S3", dim_medians={"science_accuracy": 3.5}),
        ]
        stat = summarize(_make_result(scenes))["dims"]["science_accuracy"]
        self.assertEqual(stat["mean"], 4.5)
        self.assertEqual(stat["median"], 5.0)

    def test_html_labels_mean_and_median(self):
        from pathlib import Path
        from shot_eval.report import build_payload, render_html

        result = _make_result([_make_scene("S1", dim_medians={"science_accuracy": 5.0})])
        html = render_html(build_payload(result, html_path=Path("/tmp/report.html")), "test")
        self.assertIn("均值", html)
        self.assertIn("中位", html)


class TestOptimizationSummary(unittest.TestCase):
    def test_absent_returns_none(self):
        from shot_eval.report import summarize_optimization
        self.assertIsNone(summarize_optimization({"scenes": []}))

    def test_counts_stages_patches_actions_and_grounding(self):
        from shot_eval.report import summarize_optimization
        result = {"optimization": {"model": "gemini-3.8-flash", "errors": [], "scenes": [
            {"status": "ok", "valid_patches": [{}, {}], "rejected_patches": [{}],
             "non_prompt_actions": [{}], "findings": [
                {"stage": "script_to_plan", "status": "problem", "quote_grounded": True},
                {"stage": "plan_to_prompt", "status": "unassessable", "quote_grounded": True},
                {"stage": "prompt_to_video", "status": "problem", "quote_grounded": False},
             ]}
        ]}}
        value = summarize_optimization(result)
        self.assertEqual(value["scene_count"], 1)
        self.assertEqual(value["stage_problems"], {"script_to_plan": 1, "plan_to_prompt": 0, "prompt_to_video": 1})
        self.assertEqual(value["stage_unassessable"]["plan_to_prompt"], 1)
        self.assertEqual(value["valid_patches"], 2)
        self.assertEqual(value["rejected_patches"], 1)
        self.assertEqual(value["non_prompt_actions"], 1)
        self.assertEqual(value["ungrounded_findings"], 1)

    def test_payload_and_html_include_overview(self):
        from pathlib import Path
        from shot_eval.report import build_payload, render_html
        result = _make_result([_make_scene("S1", dim_medians={"science_accuracy": 5.0})])
        result["optimization"] = {"model": "m", "scenes": [{"scene_id": "S1", "status": "ok", "findings": [], "valid_patches": [], "rejected_patches": [], "non_prompt_actions": []}], "errors": []}
        payload = build_payload(result, html_path=Path("/tmp/report.html"))
        self.assertEqual(payload["optimization_summary"]["scene_count"], 1)
        html = render_html(payload, "test")
        self.assertIn("优化归因", html)
        self.assertIn("stage_problems", html)
