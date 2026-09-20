"""No-network unit tests for shot_eval provenance, optimization, grounding,
orchestration, backward report rendering, and four-mode render schema.

All LLM calls are mocked — these tests run without any network access.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video_prompts(
    entries: list[dict[str, Any]] | None = None,
    storyboard_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal video_prompts.json payload."""
    if entries is None:
        entries = [
            {
                "unit_id": "A01-01",
                "ok": True,
                "scene_design": "SHOT: close-up | ENV: lab | RENDER: photoreal",
                "plan_scene_design": "Plan-level design for A01",
                "visual_beat": "Camera zooms into cell",
                "video_prompt": "Full prompt for A01-01",
                "effective_style_lock": "cinematic sci-fi",
                "render_mode": "photoreal",
                "duration_seconds": 5.0,
                "request": {"clip_script": "细胞在显微镜下分裂"},
            }
        ]
    data: dict[str, Any] = {
        "video_prompts": entries,
        "videos": [
            {"unit_id": e["unit_id"], "ok": True, "actual_seconds": 5.0}
            for e in entries if e.get("ok")
        ],
    }
    if storyboard_plan is not None:
        data["storyboard_plan"] = storyboard_plan
    return data


def _write_run_dir(tmp: str, data: dict[str, Any]) -> Path:
    """Write video_prompts.json and dummy mp4 files to a temp dir."""
    run_dir = Path(tmp) / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "video_prompts.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    vids = run_dir / "videos"
    vids.mkdir(exist_ok=True)
    for entry in data.get("video_prompts") or []:
        if entry.get("ok"):
            (vids / f"{entry['unit_id']}.mp4").write_bytes(b"\x00" * 16)
    return run_dir


# ===========================================================================
# 1. Provenance recovery
# ===========================================================================

class TestProvenanceRecovery(unittest.TestCase):
    """Provenance is built from entry fields; missing fields recovered from plan."""

    def test_provenance_from_entry_fields(self):
        """When entry has all fields, provenance uses them directly."""
        from shot_eval.adapters import _recover_plan_fields

        entry = {
            "unit_id": "A01-01",
            "plan_scene_design": "from entry",
            "plan_visual_beats": "entry beats",
            "script_fragment": "entry script",
            "script_context": "entry context",
        }
        result = _recover_plan_fields(entry, None)
        self.assertEqual(result["plan_scene_design"], "from entry")
        self.assertEqual(result["plan_visual_beats"], "entry beats")

    def test_provenance_recovery_from_storyboard_plan(self):
        """When entry fields are absent, recover from storyboard_plan."""
        from shot_eval.adapters import _recover_plan_fields

        plan = {
            "scenes": [{
                "scene_id": "A01",
                "scene_design": "plan level design",
                "shots": [{
                    "shot_id": "A01-01",
                    "script": "recovered script",
                    "visual_beat": "recovered beat",
                }],
            }]
        }
        entry = {"unit_id": "A01-01"}
        result = _recover_plan_fields(entry, plan)
        self.assertEqual(result["plan_scene_design"], "plan level design")
        self.assertEqual(result["plan_visual_beats"], "recovered beat")
        self.assertEqual(result["script_fragment"], "recovered script")

    def test_shot_index_recovery_without_shot_ids(self):
        """真实 plan 的 shots 没有 shot_id 时必须按 entry.shot_index 取当前 shot。"""
        from shot_eval.adapters import _recover_plan_fields

        plan = {"scenes": [{
            "scene_id": "A01",
            "scene_design": "scene design",
            "shots": [
                {"script": "first", "visual_beat": "beat one"},
                {"script": "second", "visual_beat": "beat two"},
            ],
        }]}
        entry = {
            "unit_id": "A01-02", "scene_id": "A01", "shot_index": 1,
            "request": {"clip_script": "second"},
        }
        result = _recover_plan_fields(entry, plan)
        self.assertEqual(result["script_fragment"], "second")
        self.assertEqual(result["plan_visual_beats"], "beat two")
        self.assertNotIn("beat one", result["plan_visual_beats"])

    def test_plan_visual_beats_list_is_normalized(self):
        from shot_eval.adapters import _recover_plan_fields

        result = _recover_plan_fields(
            {"unit_id": "A01", "plan_visual_beats": ["beat one", "beat two"]},
            None,
        )
        self.assertEqual(result["plan_visual_beats"], "beat one\nbeat two")

    def test_scene_level_granularity(self):
        """Scene-level unit_id (no dash) matches scene directly."""
        from shot_eval.adapters import _recover_plan_fields

        plan = {
            "scenes": [{
                "scene_id": "A01",
                "scene_design": "scene design",
                "shots": [
                    {"shot_id": "A01-01", "visual_beat": "beat1"},
                    {"shot_id": "A01-02", "visual_beat": "beat2"},
                ],
            }]
        }
        entry = {"unit_id": "A01"}
        result = _recover_plan_fields(entry, plan)
        self.assertEqual(result["plan_scene_design"], "scene design")
        self.assertIn("beat1", result["plan_visual_beats"])
        self.assertIn("beat2", result["plan_visual_beats"])

    def test_full_adapter_builds_provenance(self):
        """tasks_from_run attaches provenance to each SceneSlice."""
        with tempfile.TemporaryDirectory() as tmp:
            data = _make_video_prompts()
            run_dir = _write_run_dir(tmp, data)
            from shot_eval.adapters import tasks_from_run

            tasks = tasks_from_run(str(run_dir))
            self.assertEqual(len(tasks), 1)
            prov = tasks[0].item.provenance
            self.assertIsNotNone(prov)
            self.assertIs(tasks[0].provenance, prov)
            self.assertEqual(prov.generated_scene_design,
                             "SHOT: close-up | ENV: lab | RENDER: photoreal")
            self.assertEqual(prov.effective_style_lock, "cinematic sci-fi")

    def test_provenance_in_as_context(self):
        """as_context includes provenance when present."""
        from shot_eval.models import Provenance, SceneSlice

        prov = Provenance(
            script_fragment="frag",
            plan_scene_design="plan",
        )
        s = SceneSlice(
            scene_id="A01-01", index=0, start=0, duration=5,
            scene_design="design", shot_type="close", environment="lab",
            render_mode="cg", prompt="prompt", beats=[], provenance=prov,
        )
        ctx = s.as_context()
        self.assertIn("provenance", ctx)
        self.assertEqual(ctx["provenance"]["script_fragment"], "frag")

    def test_provenance_absent_no_key(self):
        """as_context omits provenance when None — backward compat."""
        from shot_eval.models import SceneSlice

        s = SceneSlice(
            scene_id="X", index=0, start=0, duration=1,
            scene_design="", shot_type="", environment="",
            render_mode="", prompt="", beats=[],
        )
        ctx = s.as_context()
        self.assertNotIn("provenance", ctx)


# ===========================================================================
# 2. Prompt rendering (optimization template)
# ===========================================================================

class TestPromptRendering(unittest.TestCase):
    """Optimization prompt template loads and renders correctly."""

    def test_optimization_template_loads(self):
        """prompts.yaml has a valid `optimization` key after version bump."""
        from shot_eval import prompts

        prompts.load(force=True)
        tmpl = prompts.get("optimization")
        self.assertIsInstance(tmpl, str)
        self.assertIn("{provenance_json}", tmpl)
        self.assertIn("{deductions_json}", tmpl)

    def test_optimization_prompt_renders(self):
        """build_optimization_prompt produces a non-empty string."""
        from shot_eval.models import Provenance
        from shot_eval.optimize import build_optimization_prompt

        prov = Provenance(
            script_fragment="cells divide",
            generated_scene_design="SHOT: close-up | ENV: lab",
            final_video_prompt="full prompt here",
        )
        deds = [[{"dimension": "beat_alignment", "severity": "major",
                   "evidence": "no cell shown", "expected": "show cell"}]]
        text = build_optimization_prompt(prov, deds)
        self.assertIn("cells divide", text)
        self.assertIn("beat_alignment", text)
        # The prompt instructs the model NOT to evaluate duration/audio
        self.assertIn("不要评价", text)

    def test_prompt_version_bumped(self):
        """Version should be 3 after the bump."""
        from shot_eval import prompts

        data = prompts.load(force=True)
        self.assertEqual(data["version"], 3)

    def test_no_duration_or_audio_in_optimization(self):
        """optimization template explicitly excludes duration and audio."""
        from shot_eval import prompts

        tmpl = prompts.get("optimization")
        # The template says "不要评价时长或音频" — it tells the model to exclude these
        self.assertIn("不要评价", tmpl)
        self.assertIn("音频", tmpl)


# ===========================================================================
# 3. Grounding validation
# ===========================================================================

class TestGroundingValidation(unittest.TestCase):
    """Exact quote and patch anchor validation against provenance fields."""

    def test_validate_quote_present(self):
        from shot_eval.models import Finding, Provenance
        from shot_eval.optimize import validate_quote

        prov = Provenance(generated_scene_design="The camera zooms into a cell.")
        f = Finding(
            stage="plan_to_prompt", status="problem",
            problem="wrong", severity="medium", root_cause="meaning_drift",
            source_field="generated_scene_design",
            exact_quote="zooms into a cell", confidence=0.8,
        )
        self.assertTrue(validate_quote(f, prov))

    def test_validate_quote_absent(self):
        from shot_eval.models import Finding, Provenance
        from shot_eval.optimize import validate_quote

        prov = Provenance(generated_scene_design="The camera pans left.")
        f = Finding(
            stage="plan_to_prompt", status="problem",
            problem="wrong", severity="medium", root_cause="meaning_drift",
            source_field="generated_scene_design",
            exact_quote="zooms into a cell", confidence=0.8,
        )
        self.assertFalse(validate_quote(f, prov))

    def test_validate_video_evidence_quote(self):
        from shot_eval.models import Finding, Provenance
        from shot_eval.optimize import validate_quote

        finding = Finding(
            stage="prompt_to_video", status="problem", problem="动作未执行",
            severity="high", root_cause="generator_noncompliance",
            source_field="video_evidence", exact_quote="镜头始终静止", confidence=0.9,
        )
        self.assertTrue(validate_quote(
            finding, Provenance(), "画面中镜头始终静止，没有推进"
        ))
        self.assertFalse(validate_quote(finding, Provenance(), "画面正常推进"))

    def test_validate_patch_replace(self):
        from shot_eval.models import Provenance, PromptPatch
        from shot_eval.optimize import validate_patch

        prov = Provenance(generated_scene_design="Camera zooms into cell.")
        p = PromptPatch(
            target="generated_scene_design", op="replace",
            anchor="zooms into", content="pans across", reason="test", confidence=0.8,
        )
        self.assertTrue(validate_patch(p, prov))

    def test_validate_patch_replace_missing_anchor(self):
        from shot_eval.models import Provenance, PromptPatch
        from shot_eval.optimize import validate_patch

        prov = Provenance(generated_scene_design="Camera pans left.")
        p = PromptPatch(
            target="generated_scene_design", op="replace",
            anchor="zooms into", content="pans across", reason="test", confidence=0.8,
        )
        self.assertFalse(validate_patch(p, prov))

    def test_validate_patch_append_requires_content(self):
        from shot_eval.models import Provenance, PromptPatch
        from shot_eval.optimize import validate_patch

        patch = PromptPatch(
            target="style_lock", op="append", content=" ", reason="bad", confidence=0.2,
        )
        self.assertFalse(validate_patch(patch, Provenance()))

    def test_validate_patch_append_empty_field(self):
        """append works even on empty fields."""
        from shot_eval.models import Provenance, PromptPatch
        from shot_eval.optimize import validate_patch

        prov = Provenance(effective_style_lock="")
        p = PromptPatch(
            target="style_lock", op="append",
            content="cinematic lighting", reason="test", confidence=0.8,
        )
        self.assertTrue(validate_patch(p, prov))

    def test_validate_patch_delete_no_anchor(self):
        from shot_eval.models import Provenance, PromptPatch
        from shot_eval.optimize import validate_patch

        prov = Provenance()
        p = PromptPatch(
            target="plan_scene_design",
            op="delete", anchor="nonexistent", reason="test", confidence=0.8,
        )
        self.assertFalse(validate_patch(p, prov))

    def test_ground_optimization_separates_valid_rejected(self):
        from shot_eval.models import (
            OptimizationResult, Provenance, PromptPatch, Finding,
            NonPromptAction,
        )
        from shot_eval.optimize import ground_optimization

        prov = Provenance(
            generated_scene_design="Wide shot of forest clearing.",
            effective_style_lock="dark moody",
        )
        raw = OptimizationResult(
            findings=[Finding(
                stage="plan_to_prompt", status="problem",
                problem="too wide", severity="medium", root_cause="meaning_drift",
                source_field="generated_scene_design",
                exact_quote="Wide shot", confidence=0.9,
            )],
            patches=[
                PromptPatch(target="generated_scene_design", op="replace",
                            anchor="Wide shot", content="Close-up", reason="fix", confidence=0.9),
                PromptPatch(target="generated_scene_design", op="replace",
                            anchor="NONEXISTENT", content="something", reason="fix", confidence=0.5),
            ],
            non_prompt_actions=[NonPromptAction(action="retry", reason="flicker")],
        )
        result = ground_optimization(raw, prov)
        self.assertEqual(len(result["valid_patches"]), 1)
        self.assertEqual(len(result["rejected_patches"]), 1)
        self.assertIn("rejection_reason", result["rejected_patches"][0])
        self.assertEqual(len(result["non_prompt_actions"]), 1)
        self.assertTrue(result["findings"][0]["quote_grounded"])
        self.assertEqual(
            {finding["stage"] for finding in result["findings"]},
            {"script_to_plan", "plan_to_prompt", "prompt_to_video"},
        )


# ===========================================================================
# 4. One-call-per-scene orchestration via mocks
# ===========================================================================

class TestOptimizationOrchestration(unittest.TestCase):
    """optimize_scenes makes exactly one call per scene, mocked."""

    def _make_task(self, scene_id: str, has_provenance: bool = True):
        from shot_eval.models import Provenance, SceneSlice
        from shot_eval.bench import SceneTask

        prov = Provenance(
            generated_scene_design="design text here",
            final_video_prompt="prompt text here",
        ) if has_provenance else None
        item = SceneSlice(
            scene_id=scene_id, index=0, start=0, duration=5,
            scene_design="design", shot_type="close", environment="lab",
            render_mode="cg", prompt="prompt", beats=[], provenance=prov,
        )
        return SceneTask(
            source="test", origin="shot", item=item,
            clip=Path("/tmp/fake.mp4"), style_lock="lock",
        )

    def test_missing_provenance_yields_unassessable(self):
        """Scene without provenance is skipped, not crashed."""
        from shot_eval.optimize import optimize_scenes

        task = self._make_task("A01-01", has_provenance=False)
        scene_data = {"rounds": [{"deductions": [{"dimension": "beat_alignment"}]}]}
        results = asyncio.run(optimize_scenes([scene_data], [task]))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "unassessable")

    def test_no_deductions_still_attributes_upstream(self):
        """没有视频扣分也必须做一次调用，才能发现两个上游阶段的问题。"""
        from shot_eval.optimize import optimize_scenes

        task = self._make_task("A01-01")
        scene_data = {"rounds": [{"deductions": []}]}
        response = MagicMock(text=json.dumps({
            "findings": [], "patches": [], "non_prompt_actions": [],
        }))
        with patch(
            "shot_eval.judge._get_client",
            new=AsyncMock(return_value=MagicMock(
                aio=MagicMock(models=MagicMock(
                    generate_content=AsyncMock(return_value=response)
                ))
            )),
        ) as get_client:
            results = asyncio.run(optimize_scenes([scene_data], [task]))
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(len(results[0]["findings"]), 3)
        self.assertEqual(get_client.return_value.aio.models.generate_content.call_count, 1)

    def test_one_call_per_scene(self):
        """Exactly one LLM call per scene with deductions."""
        from shot_eval.optimize import optimize_scenes

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "findings": [], "patches": [], "non_prompt_actions": [],
        })
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("shot_eval.judge._get_client", new=AsyncMock(return_value=mock_client)):
            tasks = [self._make_task("A01-01"), self._make_task("A02-01")]
            scenes_data = [
                {"rounds": [{"deductions": [{"dimension": "beat_alignment", "severity": "major"}]}]},
                {"rounds": [{"deductions": [{"dimension": "clarity", "severity": "minor"}]}]},
            ]
            results = asyncio.run(optimize_scenes(scenes_data, tasks))
            self.assertEqual(len(results), 2)
            self.assertEqual(mock_client.aio.models.generate_content.call_count, 2)

    def test_error_is_caught_per_scene(self):
        """A failing LLM call records error, doesn't crash batch."""
        from shot_eval.optimize import optimize_scenes

        mock_client = AsyncMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with patch("shot_eval.judge._get_client", new=AsyncMock(return_value=mock_client)):
            task = self._make_task("A01-01")
            scene_data = {"rounds": [{"deductions": [{"dimension": "x", "severity": "major"}]}]}
            results = asyncio.run(optimize_scenes([scene_data], [task]))
            self.assertEqual(results[0]["status"], "error")
            self.assertIn("boom", results[0]["error"])


# ===========================================================================
# 5. Backward report rendering
# ===========================================================================

class TestBackwardReportRendering(unittest.TestCase):
    """Old reports (without optimization or provenance) still render."""

    def test_old_format_renders(self):
        """build_payload handles result without optimization key."""
        from shot_eval.report import build_payload, render_html

        result = {
            "judge_model": "test",
            "thinking_level": "low",
            "prompt_version_tag": "v1+abc123",
            "prompts_path": "/test",
            "repeats": 1,
            "elapsed": 10,
            "design_source": "generated",
            "config": {"beat_source": "visual_beat", "style_source": "style_lock",
                       "render_source": "blind"},
            "tasks": [{"key": "test/A01-01", "source": "test", "origin": "shot",
                       "scene_id": "A01-01", "clip": "/tmp/fake.mp4", "duration": 5,
                       "render_declared": "cg"}],
            "scenes": [{
                "key": "test/A01-01", "source": "test", "origin": "shot",
                "scene_id": "A01-01", "render_declared": "cg",
                "context": {
                    "scene_id": "A01-01", "index": 0,
                    "shot_type": "close", "environment": "lab",
                    "render_mode": "cg", "scene_design": "design",
                    "prompt": "prompt", "beats": [],
                },
                "rounds": [{"observed": "something", "deductions": [],
                             "beat_alignment": 5, "design_fidelity": 5,
                             "science_accuracy": 5, "clarity": 5,
                             "instructional_value": 5,
                             "beat_alignment_reason": "ok",
                             "design_fidelity_reason": "ok",
                             "science_accuracy_reason": "ok",
                             "clarity_reason": "ok",
                             "instructional_value_reason": "ok",
                             "has_subtitle_or_caption": False,
                             "has_watermark_or_logo": False,
                             "has_brand_name": False,
                             }],
                "errors": [], "dims": {"beat_alignment": {"median": 5.0, "spread": 0, "values": [5.0]}},
                "hits": {}, "unreferenced": [], "judged": 1, "deduction_count": [0],
                "render_match": None,
            }],
        }
        payload = build_payload(result)
        html = render_html(payload, "Test Report")
        self.assertIn("Test Report", html)
        self.assertIn("A01-01", html)
        # No optimization key — should still render fine
        self.assertIsNone(payload.get("optimization"))

    def test_new_format_with_optimization_renders(self):
        """Report with optimization data renders without error."""
        from shot_eval.report import build_payload, render_html

        result = {
            "judge_model": "test", "thinking_level": "low",
            "prompt_version_tag": "v2+xyz", "prompts_path": "/test",
            "repeats": 1, "elapsed": 5, "design_source": "generated",
            "config": {"beat_source": "visual_beat", "style_source": "off",
                       "render_source": "off"},
            "tasks": [], "scenes": [],
            "optimization": {
                "model": "test-model",
                "config": {"enabled": True},
                "scenes": [{"scene_id": "A01-01", "status": "ok",
                             "findings": [], "valid_patches": [],
                             "rejected_patches": [], "non_prompt_actions": []}],
                "errors": [],
            },
        }
        payload = build_payload(result)
        self.assertIsNotNone(payload.get("optimization"))
        html = render_html(payload, "Opt Report")
        self.assertIn("Opt Report", html)


# ===========================================================================
# 6. Four-mode render schema
# ===========================================================================

class TestFourModeRenderSchema(unittest.TestCase):
    """RenderVerdict and RENDER_MODES support four modes + unclear."""

    def test_render_modes_constant(self):
        from shot_eval.models import RENDER_MODES

        self.assertEqual(set(RENDER_MODES),
                         {"photoreal", "cg", "illustrated", "vox", "unclear"})

    def test_render_verdict_accepts_all_modes(self):
        from shot_eval.models import RenderVerdict

        for mode in ("photoreal", "cg", "illustrated", "vox", "unclear"):
            v = RenderVerdict(verdict=mode, confidence=0.9, reason="test")
            self.assertEqual(v.verdict, mode)

    def test_render_verdict_rejects_unknown_mode(self):
        from pydantic import ValidationError
        from shot_eval.models import RenderVerdict

        with self.assertRaises(ValidationError):
            RenderVerdict(verdict="pixel_art", confidence=0.9, reason="test")

    def test_render_blind_prompt_has_four_modes(self):
        from shot_eval import prompts

        prompts.load(force=True)
        text = prompts.get("render_blind")
        for mode in ("photoreal", "cg", "illustrated", "vox"):
            self.assertIn(mode, text)

    def test_render_mode_enum_model(self):
        from shot_eval.models import RenderModeEnum

        for mode in ("photoreal", "cg", "illustrated", "vox", "unclear"):
            m = RenderModeEnum(mode=mode)
            self.assertEqual(m.mode, mode)


# ===========================================================================
# 7. Three-stage schema validation
# ===========================================================================

class TestThreeStageSchemas(unittest.TestCase):
    """Finding, PromptPatch, NonPromptAction, OptimizationResult schemas."""

    def test_finding_schema(self):
        from shot_eval.models import Finding

        f = Finding(
            stage="script_to_plan", status="problem",
            problem="desc", severity="high", root_cause="missing_content",
            source_field="plan_scene_design", exact_quote="quote", confidence=0.7,
        )
        d = f.model_dump()
        self.assertEqual(d["stage"], "script_to_plan")
        self.assertEqual(d["severity"], "high")

    def test_prompt_patch_schema(self):
        from shot_eval.models import PromptPatch

        p = PromptPatch(
            target="generated_scene_design", op="replace",
            anchor="old text", content="new text", reason="fix", confidence=0.8,
        )
        d = p.model_dump()
        self.assertEqual(d["op"], "replace")

    def test_non_prompt_action_schema(self):
        from shot_eval.models import NonPromptAction

        a = NonPromptAction(action="retry", reason="flicker detected")
        self.assertEqual(a.action, "retry")

    def test_optimization_result_schema(self):
        from shot_eval.models import (
            Finding, NonPromptAction, OptimizationResult, PromptPatch,
        )

        result = OptimizationResult(
            findings=[Finding(
                stage="prompt_to_video", status="problem",
                problem="p", severity="low", root_cause="generator_noncompliance",
                source_field="render_mode", exact_quote="cg", confidence=0.8,
            )],
            patches=[PromptPatch(
                target="render_mode", op="replace",
                anchor="cg", content="photoreal", reason="fix", confidence=0.8,
            )],
            non_prompt_actions=[NonPromptAction(action="switch_model", reason="test")],
        )
        d = result.model_dump()
        self.assertEqual(len(d["findings"]), 1)
        self.assertEqual(len(d["patches"]), 1)
        self.assertEqual(len(d["non_prompt_actions"]), 1)

    def test_attribution_stages_constant(self):
        from shot_eval.models import ATTRIBUTION_STAGES

        self.assertEqual(ATTRIBUTION_STAGES,
                         ("script_to_plan", "plan_to_prompt", "prompt_to_video"))


# ===========================================================================
# 8. CLI flags
# ===========================================================================

class TestCLIFlags(unittest.TestCase):
    """CLI parser accepts new optimization flags."""

    def test_optimize_prompt_flag(self):
        from shot_eval.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["test_run", "--optimize-prompt"])
        self.assertTrue(args.optimize_prompt)

    def test_optimization_model_flag(self):
        from shot_eval.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["test_run", "--optimization-model", "gemini-3.8-flash"])
        self.assertEqual(args.optimization_model, "gemini-3.8-flash")

    def test_default_no_optimization(self):
        from shot_eval.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["test_run"])
        self.assertFalse(args.optimize_prompt)
        self.assertEqual(args.optimization_model, "")


if __name__ == "__main__":
    unittest.main()
