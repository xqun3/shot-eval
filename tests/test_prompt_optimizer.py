"""Comprehensive no-network tests for shot_eval.prompt_optimizer_agent.

Covers: ADK root_agent type/tool, Runner wiring via mock, stable 3-round apply,
single-run gating, confidence gate, stale anchors, exact-one replacement,
append dedupe, overlap conflict, review-only targets, prompt rebuilt by source
function, videos cleared, source unchanged, linter cases, global candidates,
output paths, and backward/missing data handling.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
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

_SAMPLE_ENTRY = {
    "unit_id": "A01-01",
    "ok": True,
    "scene_design": "SHOT: close-up | ENV: lab | RENDER: photoreal\nCamera zooms into cell membrane.",
    "visual_beat": "Camera slowly pushes in, revealing membrane structure.",
    "video_prompt": "Original prompt text",
    "effective_style_lock": "cinematic sci-fi dark moody",
    "render_mode": "photoreal",
    "duration_seconds": 5.0,
    "request": {"clip_script": "细胞在显微镜下分裂"},
    "plan_scene_design": "Plan-level design for A01",
    "plan_visual_beats": "Plan beats",
    "scene_id": "A01",
    "scene_type": "ai_concept",
    "shot_index": 0,
}

_SAMPLE_EVAL_SCENE = {
    "scene_id": "A01-01",
    "status": "ok",
    "findings": [
        {
            "stage": "plan_to_prompt",
            "status": "problem",
            "problem": "Simile rendered literally",
            "severity": "medium",
            "root_cause": "meaning_drift",
            "source_field": "generated_scene_design",
            "exact_quote": "zooms into cell",
            "confidence": 0.9,
            "quote_grounded": True,
        },
        {
            "stage": "script_to_plan",
            "status": "pass",
            "problem": "Faithful",
            "severity": "low",
            "root_cause": "faithful",
            "source_field": "script_fragment",
            "exact_quote": "",
            "confidence": 1.0,
            "quote_grounded": True,
        },
        {
            "stage": "prompt_to_video",
            "status": "problem",
            "problem": "Generator did not comply",
            "severity": "high",
            "root_cause": "generator_noncompliance",
            "source_field": "video_evidence",
            "exact_quote": "lens stayed static",
            "confidence": 0.85,
            "quote_grounded": True,
        },
    ],
    "valid_patches": [
        {
            "target": "generated_scene_design",
            "op": "replace",
            "anchor": "zooms into cell membrane",
            "content": "pushes slowly toward cell membrane",
            "reason": "Avoid zoom being ignored by model",
            "confidence": 0.9,
        },
    ],
    "rejected_patches": [
        {
            "target": "render_mode",
            "op": "replace",
            "anchor": "photoreal",
            "content": "cg",
            "reason": "Wrong mode",
            "confidence": 0.3,
            "rejection_reason": "low confidence",
        },
    ],
    "non_prompt_actions": [
        {"action": "retry", "reason": "Generator noncompliance detected"},
    ],
}


def _make_bundle(entries=None):
    if entries is None:
        entries = [copy.deepcopy(_SAMPLE_ENTRY)]
    return {
        "meta": {"granularity": "shot"},
        "input": {"script": "完整脚本"},
        "video_prompts": entries,
        "videos": [
            {"unit_id": e["unit_id"], "ok": True, "actual_seconds": 5.0, "path": "/tmp/fake.mp4"}
            for e in entries if e.get("ok")
        ],
    }


def _make_eval(scenes=None, repeats=3):
    if scenes is None:
        scenes = [copy.deepcopy(_SAMPLE_EVAL_SCENE)]
    return {
        "config": {"prompt_optimization": True},
        "repeats": repeats,
        "optimization": {
            "model": "test-model",
            "scenes": scenes,
        },
    }


def _write_files(tmp, bundle, eval_data):
    d = Path(tmp)
    vp = d / "video_prompts.json"
    ev = d / "eval.json"
    vp.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    ev.write_text(json.dumps(eval_data, ensure_ascii=False), encoding="utf-8")
    return str(vp), str(ev)


# ===========================================================================
# 1. ADK root_agent type and tool
# ===========================================================================

class TestADKAgentType(unittest.TestCase):
    """root_agent is a google.adk.agents.Agent with FunctionTool."""

    def test_root_agent_is_agent(self):
        from google.adk.agents import Agent
        from shot_eval.prompt_optimizer_agent import root_agent
        self.assertIsInstance(root_agent, Agent)

    def test_root_agent_has_tool(self):
        from shot_eval.prompt_optimizer_agent.agent import root_agent, optimize_tool
        self.assertIn(optimize_tool, root_agent.tools)

    def test_tool_is_function_tool(self):
        from google.adk.tools import FunctionTool
        from shot_eval.prompt_optimizer_agent.agent import optimize_tool
        self.assertIsInstance(optimize_tool, FunctionTool)

    def test_root_agent_has_model(self):
        from shot_eval.prompt_optimizer_agent.agent import root_agent
        self.assertTrue(root_agent.model)

    def test_default_model_from_env(self):
        """SHOT_OPTIMIZER_AGENT_MODEL env var overrides default."""
        os.environ["SHOT_OPTIMIZER_AGENT_MODEL"] = "test-model-override"
        try:
            # Re-import to pick up new env
            import importlib
            import shot_eval.prompt_optimizer_agent.agent as agent_mod
            importlib.reload(agent_mod)
            self.assertEqual(agent_mod._DEFAULT_MODEL, "test-model-override")
        finally:
            os.environ.pop("SHOT_OPTIMIZER_AGENT_MODEL", None)
            import importlib
            importlib.reload(agent_mod)

    def test_instruction_requires_one_tool_call(self):
        from shot_eval.prompt_optimizer_agent.agent import root_agent
        instr = root_agent.instruction
        upper = instr.upper() if instr else ""
        self.assertIn("EXACTLY ONCE", upper)


# ===========================================================================
# 2. Runner wiring via mock
# ===========================================================================

class TestRunnerWiring(unittest.TestCase):
    """ADK Runner can be instantiated with our agent (mock session)."""

    def test_runner_instantiation(self):
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from shot_eval.prompt_optimizer_agent.agent import root_agent

        session_service = InMemorySessionService()
        runner = Runner(
            agent=root_agent,
            app_name="test_optimizer",
            session_service=session_service,
        )
        self.assertIsNotNone(runner)

    def test_runner_session_creation(self):
        from google.adk.sessions import InMemorySessionService

        service = InMemorySessionService()
        session = asyncio.run(
            service.create_session(app_name="test", user_id="u1")
        )
        self.assertIsNotNone(session.id)


# ===========================================================================
# 3. Stable 3-round apply
# ===========================================================================

class TestStableApply(unittest.TestCase):
    """3 repeats + valid patches → applied."""

    def test_three_round_apply(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=3)
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            self.assertNotIn("error", result)
            sr = result["scene_results"]
            self.assertEqual(len(sr), 1)
            self.assertEqual(sr[0]["status"], "applied")
            self.assertTrue(sr[0]["prompt_rebuilt"])


# ===========================================================================
# 4. Single-run gating
# ===========================================================================

class TestSingleRunGating(unittest.TestCase):
    """repeats=1 without allow_single_run → plan-only."""

    def test_single_run_is_plan_only(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=1)
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"]
            # Should NOT be "applied" — plan-only
            self.assertNotEqual(sr[0]["status"], "applied")
            self.assertTrue(result["summary"]["plan_only"])

    def test_single_run_with_allow(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=1)
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, allow_single_run=True, dry_run=True)
            sr = result["scene_results"]
            self.assertEqual(sr[0]["status"], "applied")


# ===========================================================================
# 5. Confidence gate
# ===========================================================================

class TestConfidenceGate(unittest.TestCase):
    """Patches below min_confidence are skipped."""

    def test_low_confidence_skipped(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["valid_patches"][0]["confidence"] = 0.5
            bundle = _make_bundle()
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, min_confidence=0.8, dry_run=True)
            sr = result["scene_results"][0]
            self.assertNotEqual(sr["status"], "applied")
            self.assertTrue(any("confidence" in s["reason"] for s in sr["patches_skipped"]))

    def test_high_confidence_applied(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=3)
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, min_confidence=0.5, dry_run=True)
            self.assertEqual(result["scene_results"][0]["status"], "applied")


# ===========================================================================
# 6. Stale anchors
# ===========================================================================

class TestStaleAnchors(unittest.TestCase):
    """Patches with anchors not in current bundle are review/skipped."""

    def test_stale_anchor_becomes_review(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            entry = copy.deepcopy(_SAMPLE_ENTRY)
            entry["scene_design"] = "Something completely different."
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["valid_patches"][0]["anchor"] = "nonexistent text"

            bundle = _make_bundle([entry])
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"][0]
            self.assertNotEqual(sr["status"], "applied")
            self.assertTrue(any("stale" in s["reason"] for s in sr["patches_skipped"]))


# ===========================================================================
# 7. Exact-one replacement
# ===========================================================================

class TestExactOneReplacement(unittest.TestCase):
    """replace/delete/insert_after must match exactly one occurrence."""

    def test_replace_exactly_one(self):
        from shot_eval.prompt_optimizer_agent.core import apply_patch
        text = "The cat sat on the mat."
        result = apply_patch(text, {"op": "replace", "anchor": "cat", "content": "dog"})
        self.assertEqual(result, "The dog sat on the mat.")

    def test_replace_multiple_raises(self):
        from shot_eval.prompt_optimizer_agent.core import apply_patch, PatchError
        text = "cat cat cat"
        with self.assertRaises(PatchError):
            apply_patch(text, {"op": "replace", "anchor": "cat", "content": "dog"})

    def test_replace_none_raises(self):
        from shot_eval.prompt_optimizer_agent.core import apply_patch, PatchError
        text = "The dog sat."
        with self.assertRaises(PatchError):
            apply_patch(text, {"op": "replace", "anchor": "cat", "content": "dog"})

    def test_delete_exactly_one(self):
        from shot_eval.prompt_optimizer_agent.core import apply_patch
        text = "Remove this word please."
        result = apply_patch(text, {"op": "delete", "anchor": "this word "})
        self.assertEqual(result, "Remove please.")

    def test_insert_after_exactly_one(self):
        from shot_eval.prompt_optimizer_agent.core import apply_patch
        text = "Camera zooms in."
        result = apply_patch(text, {"op": "insert_after", "anchor": "zooms", "content": " slowly"})
        self.assertEqual(result, "Camera zooms slowly in.")


# ===========================================================================
# 8. Append dedupe
# ===========================================================================

class TestAppendDedupe(unittest.TestCase):
    """append must not duplicate existing content."""

    def test_append_new_content(self):
        from shot_eval.prompt_optimizer_agent.core import apply_patch
        text = "Existing text."
        result = apply_patch(text, {"op": "append", "content": " New content."})
        self.assertEqual(result, "Existing text. New content.")

    def test_append_deduplicates(self):
        from shot_eval.prompt_optimizer_agent.core import apply_patch
        text = "Existing text. New content."
        result = apply_patch(text, {"op": "append", "content": "New content."})
        # Should not duplicate
        self.assertEqual(result, text)


# ===========================================================================
# 9. Overlap conflict
# ===========================================================================

class TestOverlapConflict(unittest.TestCase):
    """Overlapping anchors cause all patches in that group to be skipped."""

    def test_overlapping_anchors_detected(self):
        from shot_eval.prompt_optimizer_agent.core import detect_overlapping_anchors
        patches = [
            {"anchor": "zooms into cell", "op": "replace", "content": "x"},
            {"anchor": "into cell membrane", "op": "replace", "content": "y"},
        ]
        text = "Camera zooms into cell membrane."
        conflicts = detect_overlapping_anchors(patches, text)
        self.assertTrue(len(conflicts) > 0)

    def test_non_overlapping_ok(self):
        from shot_eval.prompt_optimizer_agent.core import detect_overlapping_anchors
        patches = [
            {"anchor": "Camera", "op": "replace", "content": "x"},
            {"anchor": "membrane", "op": "replace", "content": "y"},
        ]
        text = "Camera zooms into cell membrane."
        conflicts = detect_overlapping_anchors(patches, text)
        self.assertEqual(len(conflicts), 0)

    def test_overlap_causes_skip_in_pipeline(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["valid_patches"] = [
                {
                    "target": "generated_scene_design",
                    "op": "replace",
                    "anchor": "zooms into cell",
                    "content": "A",
                    "reason": "r",
                    "confidence": 0.9,
                },
                {
                    "target": "generated_scene_design",
                    "op": "replace",
                    "anchor": "into cell membrane",
                    "content": "B",
                    "reason": "r",
                    "confidence": 0.9,
                },
            ]
            bundle = _make_bundle()
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"][0]
            self.assertNotEqual(sr["status"], "applied")
            self.assertTrue(any("overlap" in s["reason"] for s in sr["patches_skipped"]))


# ===========================================================================
# 10. Review-only targets
# ===========================================================================

class TestReviewOnlyTargets(unittest.TestCase):
    """plan_scene_design, plan_visual_beats, style_lock, render_mode → review-only."""

    def test_plan_target_is_review_only(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["valid_patches"] = [{
                "target": "plan_scene_design",
                "op": "replace",
                "anchor": "something",
                "content": "new",
                "reason": "test",
                "confidence": 0.95,
            }]
            bundle = _make_bundle()
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"][0]
            self.assertNotEqual(sr["status"], "applied")
            self.assertTrue(any("review-only" in s["reason"] for s in sr["patches_skipped"]))


# ===========================================================================
# 11. Prompt rebuilt by source function
# ===========================================================================

class TestPromptRebuilt(unittest.TestCase):
    """Changed entries get prompt rebuilt via assemble_video_prompt."""

    def test_prompt_uses_assemble(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=3)
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"][0]
            if sr["status"] == "applied":
                self.assertTrue(sr["prompt_rebuilt"])
                new_prompt = sr.get("new_video_prompt", "")
                # Must contain the SILENT header from assemble_video_prompt
                self.assertIn("SILENT", new_prompt)
                self.assertIn("Duration:", new_prompt)

    def test_rebuilt_contains_style_lock(self):
        from shot_eval.prompt_optimizer_agent.core import rebuild_video_prompt
        prompt = rebuild_video_prompt("Scene design", "Visual beat", 5.0, "dark moody")
        self.assertIn("dark moody", prompt)
        self.assertIn("Scene design", prompt)


# ===========================================================================
# 12. Videos cleared
# ===========================================================================

class TestVideoCleared(unittest.TestCase):
    """Optimized bundle must have videos=[] and invalidation metadata."""

    def test_videos_cleared(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=3)
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            opt_bundle = result["optimized_video_prompts"]
            self.assertEqual(opt_bundle["videos"], [])
            meta = opt_bundle["_optimization_meta"]
            self.assertIn("source_bundle_hash", meta)
            self.assertIn("old_videos_invalidated", meta)
            self.assertTrue(len(meta["old_videos_invalidated"]) > 0)


# ===========================================================================
# 13. Source unchanged
# ===========================================================================

class TestSourceUnchanged(unittest.TestCase):
    """Original bundle file must not be modified."""

    def test_source_file_unchanged(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=3)
            vp, ev = _write_files(tmp, bundle, eval_data)

            original = Path(vp).read_text(encoding="utf-8")
            optimize_video_prompts(vp, ev, output_dir=str(Path(tmp) / "out"), dry_run=False)
            after = Path(vp).read_text(encoding="utf-8")
            self.assertEqual(original, after)


# ===========================================================================
# 14. Linter cases
# ===========================================================================

class TestLinterCases(unittest.TestCase):
    """Static linter detects known patterns."""

    def test_literalizable_simile(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt("酶分子如蜂群般附着", "")
        rules = [w.rule for w in warnings]
        self.assertIn("literalizable_simile", rules)

    def test_biological_containment(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt("细胞膜突破泄漏", "")
        rules = [w.rule for w in warnings]
        self.assertIn("biological_containment_risk", rules)

    def test_literalizable_waterwheel(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt("ATP 合成酶如纳米水车般旋转", "")
        self.assertIn("literalizable_simile", [w.rule for w in warnings])

    def test_camera_overload(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt(
            "zoom in, pan left, tilt up, dolly forward, crane shot", ""
        )
        rules = [w.rule for w in warnings]
        self.assertIn("camera_overload", rules)

    def test_exact_number_mg(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt("exactly 5 pieces of candy", "")
        rules = [w.rule for w in warnings]
        self.assertIn("exact_number_mg_routing", rules)

    def test_graph_mg_routing(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt("Display a bar chart showing values", "")
        rules = [w.rule for w in warnings]
        self.assertIn("graph_mg_routing", rules)

    def test_annotation_no_text_conflict(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt("Add annotation labels but no text allowed", "")
        rules = [w.rule for w in warnings]
        self.assertIn("annotation_no_text_conflict", rules)

    def test_timecode_not_flagged_as_annotation(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt(
            "0.0-5.0s: Camera pushes in slowly. No text.", ""
        )
        # Should NOT flag annotation conflict (timecodes are not annotations)
        rules = [w.rule for w in warnings]
        self.assertNotIn("annotation_no_text_conflict", rules)

    def test_clean_prompt_no_warnings(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt
        warnings = lint_prompt("Simple scene with trees and mountains.", "")
        # Should have minimal warnings
        self.assertTrue(len(warnings) <= 1)  # maybe missing_relative_scale not triggered

    def test_explicit_containment_suppresses_risk(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt

        warnings = lint_prompt(
            "Cyan blood vessels branching out strictly within the translucent body, "
            "without extending outside the skin.",
            "",
        )
        self.assertNotIn("biological_containment_risk", [w.rule for w in warnings])

    def test_camera_words_do_not_trigger_flow_continuity(self):
        from shot_eval.prompt_optimizer_agent.core import lint_prompt

        warnings = lint_prompt(
            "SHOT: close-up | RENDER: photoreal. Camera dollies toward the subject.",
            "",
        )
        self.assertNotIn("source_path_continuity", [w.rule for w in warnings])


# ===========================================================================
# 15. Global candidates
# ===========================================================================

class TestGlobalCandidates(unittest.TestCase):
    """Repeated root causes become global rule candidates."""

    def test_repeated_cause_becomes_candidate(self):
        from shot_eval.prompt_optimizer_agent.core import aggregate_global_candidates
        scenes = [
            {"unit_id": "A01-01", "findings": [
                {"status": "problem", "root_cause": "meaning_drift", "confidence": 0.9},
            ]},
            {"unit_id": "A02-01", "findings": [
                {"status": "problem", "root_cause": "meaning_drift", "confidence": 0.8},
            ]},
        ]
        candidates = aggregate_global_candidates(scenes)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["root_cause"], "meaning_drift")
        self.assertEqual(candidates[0]["count"], 2)
        self.assertEqual(candidates[0]["status"], "candidate")

    def test_single_cause_not_candidate(self):
        from shot_eval.prompt_optimizer_agent.core import aggregate_global_candidates
        scenes = [
            {"unit_id": "A01-01", "findings": [
                {"status": "problem", "root_cause": "meaning_drift", "confidence": 0.9},
            ]},
        ]
        candidates = aggregate_global_candidates(scenes)
        self.assertEqual(len(candidates), 0)

    def test_pass_findings_ignored(self):
        from shot_eval.prompt_optimizer_agent.core import aggregate_global_candidates
        scenes = [
            {"unit_id": "A01-01", "findings": [
                {"status": "pass", "root_cause": "faithful", "confidence": 1.0},
            ]},
            {"unit_id": "A02-01", "findings": [
                {"status": "pass", "root_cause": "faithful", "confidence": 1.0},
            ]},
        ]
        candidates = aggregate_global_candidates(scenes)
        self.assertEqual(len(candidates), 0)


# ===========================================================================
# 16. Output paths
# ===========================================================================

class TestOutputPaths(unittest.TestCase):
    """Output goes to run/optimizations/<eval-stem>/ by default."""

    def test_default_output_dir(self):
        from shot_eval.prompt_optimizer_agent.core import _eval_stem
        stem = _eval_stem("/path/to/shots-carbs_shot-20260916-034929.json")
        self.assertEqual(stem, "shots-carbs_shot-20260916-034929")

    def test_files_written(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=3)
            vp, ev = _write_files(tmp, bundle, eval_data)
            out = str(Path(tmp) / "out")

            result = optimize_video_prompts(vp, ev, output_dir=out, dry_run=False)
            self.assertIn("files_written", result)
            written = result["files_written"]
            names = [Path(f).name for f in written]
            self.assertIn("optimized_video_prompts.json", names)
            self.assertIn("regeneration_plan.json", names)
            self.assertIn("optimization_report.md", names)
            self.assertIn("global_rule_candidates.json", names)

            # Verify files actually exist
            for f in written:
                self.assertTrue(Path(f).exists(), f"{f} should exist")


# ===========================================================================
# 17. Backward / missing data handling
# ===========================================================================

class TestBackwardCompat(unittest.TestCase):
    """Handles missing optimization data, missing fields gracefully."""

    def test_no_optimization_key(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = {"config": {}, "repeats": 1}  # No optimization key
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            self.assertEqual(result["summary"]["total_scenes"], 0)

    def test_scene_not_in_bundle(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["scene_id"] = "NONEXISTENT"
            bundle = _make_bundle()
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"][0]
            self.assertEqual(sr["status"], "skipped")
            self.assertIn("not found", sr["reason"])

    def test_missing_fields_in_entry(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            entry = {"unit_id": "A01-01", "ok": True}  # Minimal
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["valid_patches"] = [{
                "target": "generated_scene_design",
                "op": "append",
                "content": "New content here",
                "reason": "test",
                "confidence": 0.9,
            }]
            bundle = _make_bundle([entry])
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            # Should not crash
            result = optimize_video_prompts(vp, ev, dry_run=True)
            self.assertNotIn("error", result)

    def test_empty_valid_patches(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["valid_patches"] = []
            bundle = _make_bundle()
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"][0]
            self.assertIn(sr["status"], ("review_only", "skipped"))


# ===========================================================================
# 18. Regeneration plan categories
# ===========================================================================

class TestRegenerationPlan(unittest.TestCase):
    """Regen plan has correct categories and structure."""

    def test_categories(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval(repeats=3)
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            plan = result["regeneration_plan"]
            entries = plan.get("entries", [])
            self.assertTrue(len(entries) > 0)
            for e in entries:
                self.assertIn(e["category"], [
                    "regenerate_prompt_changed",
                    "retry_generator_noncompliance",
                    "review_only",
                    "skipped",
                ])
                self.assertIn("unit_id", e)
                self.assertIn("priority", e)
                self.assertIn("patches", e)
                self.assertIn("non_prompt_actions", e)

    def test_plan_note_no_paid_videos(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            bundle = _make_bundle()
            eval_data = _make_eval()
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            plan = result["regeneration_plan"]
            self.assertIn("NOT generate paid videos", plan.get("note", ""))


# ===========================================================================
# 19. CLI parser
# ===========================================================================

class TestCLIParser(unittest.TestCase):
    """CLI accepts all required flags."""

    def test_all_flags(self):
        from shot_eval.prompt_optimizer_agent.__main__ import build_parser
        p = build_parser()
        args = p.parse_args([
            "--eval", "/tmp/eval.json",
            "--run", "/tmp/run",
            "--out", "/tmp/out",
            "--min-confidence", "0.7",
            "--allow-single-run",
            "--agent-model", "gemini-test",
            "--project", "my-project",
            "--location", "us-central1",
            "--dry-run",
        ])
        self.assertEqual(args.eval, "/tmp/eval.json")
        self.assertEqual(args.run, "/tmp/run")
        self.assertTrue(args.dry_run)
        self.assertTrue(args.allow_single_run)
        self.assertEqual(args.min_confidence, 0.7)
        self.assertEqual(args.agent_model, "gemini-test")


# ===========================================================================
# 20. Never trust rejected_patches
# ===========================================================================

class TestNeverTrustRejected(unittest.TestCase):
    """rejected_patches from eval must never be applied."""

    def test_rejected_not_applied(self):
        from shot_eval.prompt_optimizer_agent.core import optimize_video_prompts

        with tempfile.TemporaryDirectory() as tmp:
            scene = copy.deepcopy(_SAMPLE_EVAL_SCENE)
            scene["valid_patches"] = []
            scene["rejected_patches"] = [{
                "target": "generated_scene_design",
                "op": "replace",
                "anchor": "zooms into cell membrane",
                "content": "something bad",
                "reason": "bad patch",
                "confidence": 0.99,
                "rejection_reason": "already rejected",
            }]
            bundle = _make_bundle()
            eval_data = _make_eval([scene])
            vp, ev = _write_files(tmp, bundle, eval_data)

            result = optimize_video_prompts(vp, ev, dry_run=True)
            sr = result["scene_results"][0]
            self.assertNotEqual(sr["status"], "applied")
            # No patches should be applied
            self.assertEqual(len(sr["patches_applied"]), 0)


class TestPatchInputCompleteness(unittest.TestCase):
    """Malformed grounded patches must fail fast without loops or silent edits."""

    def test_empty_anchor_rejected(self):
        from shot_eval.prompt_optimizer_agent.core import PatchError, apply_patch

        with self.assertRaises(PatchError):
            apply_patch("abc", {"op": "replace", "anchor": "", "content": "x"})

    def test_empty_append_content_rejected(self):
        from shot_eval.prompt_optimizer_agent.core import PatchError, apply_patch

        with self.assertRaises(PatchError):
            apply_patch("abc", {"op": "append", "anchor": "", "content": " "})


class TestCompactADKToolResponse(unittest.TestCase):
    """FunctionTool response must not send the full optimized bundle back to the LLM."""

    def test_tool_returns_compact_summary(self):
        from shot_eval.prompt_optimizer_agent.agent import _optimize_video_prompts_tool

        with tempfile.TemporaryDirectory() as tmp:
            vp, ev = _write_files(tmp, _make_bundle(), _make_eval(repeats=3))
            result = _optimize_video_prompts_tool(vp, ev, output_dir=str(Path(tmp) / "out"))
            self.assertEqual(result["status"], "ok")
            self.assertIn("summary", result)
            self.assertIn("files_written", result)
            self.assertNotIn("optimized_video_prompts", result)
            self.assertNotIn("scene_results", result)


class TestGlobalCandidateRuleText(unittest.TestCase):
    def test_candidate_contains_reviewable_rule(self):
        from shot_eval.prompt_optimizer_agent.core import aggregate_global_candidates

        scenes = [
            {"unit_id": "A", "findings": [{"status": "problem", "root_cause": "generator_noncompliance", "confidence": 0.9}]},
            {"unit_id": "B", "findings": [{"status": "problem", "root_cause": "generator_noncompliance", "confidence": 0.8}]},
        ]
        candidate = aggregate_global_candidates(scenes)[0]
        self.assertEqual(candidate["status"], "candidate")
        self.assertIn("suggested_rule", candidate)
        self.assertTrue(candidate["suggested_rule"])


if __name__ == "__main__":
    unittest.main()

class TestStandalonePromptAssembly(unittest.TestCase):
    """Optimizer core must rebuild prompts without sibling repositories."""

    def test_rebuild_uses_internal_assembly(self):
        from shot_eval.prompt_optimizer_agent.core import rebuild_video_prompt

        prompt = rebuild_video_prompt("SCENE", "BEAT", 4.0, "STYLE")
        self.assertIn("This is a SILENT video.", prompt)
        self.assertIn("Consistent style across all scenes: STYLE", prompt)
        self.assertIn("Duration: approximately 4.0 seconds.", prompt)
        self.assertIn("Visual beats:", prompt)

    def test_core_has_no_parent_repository_loader(self):
        from shot_eval.prompt_optimizer_agent import core

        source = Path(core.__file__).read_text(encoding="utf-8")
        self.assertNotIn("peanut" + "-cut", source)
        self.assertNotIn("peanut" + "_cut", source)
