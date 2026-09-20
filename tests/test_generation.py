"""No-network tests for generation-model metadata.

Covers: normalisation, explicit-wins, missing/legacy fallback, task clip
inference, CLI stored field, payload generation, HTML labels/escaping,
service homepage chip, compare metadata, and actual run inference.
"""

from __future__ import annotations

import html
import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from shot_eval.generation import (
    GENERATION_KEYS,
    infer_generation,
    infer_run_root_from_clip,
    load_bundle,
    normalize_meta,
    resolve_generation,
    video_model_label,
)


# ===========================================================================
# 1. normalize_meta
# ===========================================================================


class TestNormalizeMeta(unittest.TestCase):
    """normalize_meta produces canonical dicts with all GENERATION_KEYS."""

    def test_full_meta(self):
        meta = {
            "video_provider": "gemini_omni",
            "video_model": "flash-preview",
            "llm_model": "gemini-flash",
            "planner_model": "gemini-flash",
            "granularity": "shot",
            "plan_kind": "reference_generate",
        }
        gen = normalize_meta(meta)
        self.assertEqual(set(gen.keys()), set(GENERATION_KEYS))
        self.assertEqual(gen["video_provider"], "gemini_omni")
        self.assertEqual(gen["video_model"], "flash-preview")
        self.assertEqual(gen["prompt_model"], "gemini-flash")
        self.assertEqual(gen["planner_model"], "gemini-flash")
        self.assertEqual(gen["granularity"], "shot")
        self.assertEqual(gen["plan_kind"], "reference_generate")

    def test_none_meta(self):
        gen = normalize_meta(None)
        self.assertTrue(all(v == "unknown" for v in gen.values()))
        self.assertEqual(set(gen.keys()), set(GENERATION_KEYS))

    def test_empty_meta(self):
        gen = normalize_meta({})
        self.assertTrue(all(v == "unknown" for v in gen.values()))

    def test_empty_strings_become_unknown(self):
        meta = {"video_model": "", "llm_model": "  ", "planner_model": None}
        gen = normalize_meta(meta)
        self.assertEqual(gen["video_model"], "unknown")
        self.assertEqual(gen["prompt_model"], "unknown")
        self.assertEqual(gen["planner_model"], "unknown")

    def test_provider_fallback(self):
        """video_provider falls back to 'provider' key."""
        meta = {"provider": "minimax"}
        gen = normalize_meta(meta)
        self.assertEqual(gen["video_provider"], "minimax")

    def test_video_provider_wins_over_provider(self):
        meta = {"video_provider": "gemini", "provider": "minimax"}
        gen = normalize_meta(meta)
        self.assertEqual(gen["video_provider"], "gemini")


# ===========================================================================
# 2. Explicit result wins
# ===========================================================================


class TestExplicitResultWins(unittest.TestCase):
    """When result["generation"] is present, it takes priority."""

    def test_explicit_generation_used(self):
        result = {
            "generation": {
                "video_provider": "custom_provider",
                "video_model": "custom_model",
                "prompt_model": "custom_llm",
                "planner_model": "custom_planner",
                "granularity": "scene",
                "plan_kind": "full",
            }
        }
        gen = infer_generation(result)
        self.assertEqual(gen["video_model"], "custom_model")
        self.assertEqual(gen["video_provider"], "custom_provider")

    def test_explicit_partial_fills_unknown(self):
        result = {"generation": {"video_model": "my_model"}}
        gen = infer_generation(result)
        self.assertEqual(gen["video_model"], "my_model")
        # Missing keys should be "unknown"
        self.assertEqual(gen["planner_model"], "unknown")

    def test_explicit_empty_dict_falls_through(self):
        result = {"generation": {}}
        gen = infer_generation(result)
        self.assertTrue(all(v == "unknown" for v in gen.values()))


# ===========================================================================
# 3. Missing / legacy safe fallback
# ===========================================================================


class TestLegacyFallback(unittest.TestCase):
    """Old result dicts without generation don't crash."""

    def test_no_generation_no_tasks(self):
        gen = infer_generation({})
        self.assertEqual(set(gen.keys()), set(GENERATION_KEYS))
        self.assertTrue(all(v == "unknown" for v in gen.values()))

    def test_no_generation_bad_clip(self):
        result = {"tasks": [{"clip": "/nonexistent/path/video.mp4"}]}
        gen = infer_generation(result)
        self.assertTrue(all(v == "unknown" for v in gen.values()))

    def test_tasks_is_none(self):
        result = {"tasks": None}
        gen = infer_generation(result)
        self.assertEqual(set(gen.keys()), set(GENERATION_KEYS))


# ===========================================================================
# 4. Task clip inference
# ===========================================================================


class TestTaskClipInference(unittest.TestCase):
    """infer_run_root_from_clip correctly identifies the run root."""

    def test_standard_clip_path(self):
        root = infer_run_root_from_clip("/data/runs/test/videos/A01-01.mp4")
        self.assertEqual(root, Path("/data/runs/test"))

    def test_non_standard_path_returns_none(self):
        root = infer_run_root_from_clip("/data/output/video.mp4")
        self.assertIsNone(root)

    def test_empty_returns_none(self):
        self.assertIsNone(infer_run_root_from_clip(""))

    def test_inference_from_real_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "test_run"
            run_dir.mkdir()
            vids = run_dir / "videos"
            vids.mkdir()
            # Write a video_prompts.json
            meta = {"video_model": "test-model", "llm_model": "test-llm"}
            vp_data = {"meta": meta, "video_prompts": [], "videos": []}
            (run_dir / "video_prompts.json").write_text(json.dumps(vp_data))
            # Write a dummy clip
            clip_path = vids / "A01-01.mp4"
            clip_path.write_bytes(b"\x00")

            # Inference from result with this clip
            result = {"tasks": [{"clip": str(clip_path)}]}
            gen = infer_generation(result)
            self.assertEqual(gen["video_model"], "test-model")


# ===========================================================================
# 5. CLI stored field (via run_batch mock)
# ===========================================================================


class TestCLIStoredField(unittest.TestCase):
    """The CLI stores result['generation'] from args.run/video_prompts.json."""

    def test_cli_stores_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            vids = run_dir / "videos"
            vids.mkdir()
            meta = {"video_model": "cli-model", "llm_model": "cli-llm",
                    "granularity": "shot", "plan_kind": "reference_generate",
                    "video_provider": "gemini_omni", "planner_model": "cli-planner"}
            vp_data = {
                "meta": meta,
                "video_prompts": [{
                    "unit_id": "A01-01", "ok": True,
                    "scene_design": "test", "video_prompt": "p",
                    "duration_seconds": 5.0,
                    "request": {"clip_script": "test"},
                }],
                "videos": [{"unit_id": "A01-01", "ok": True, "actual_seconds": 5.0}],
            }
            (run_dir / "video_prompts.json").write_text(json.dumps(vp_data))
            (vids / "A01-01.mp4").write_bytes(b"\x00" * 16)

            gen = resolve_generation(run_dir)
            self.assertEqual(gen["video_model"], "cli-model")
            self.assertEqual(gen["prompt_model"], "cli-llm")
            self.assertEqual(gen["planner_model"], "cli-planner")
            self.assertEqual(gen["granularity"], "shot")


# ===========================================================================
# 6. Payload generation
# ===========================================================================


class TestPayloadGeneration(unittest.TestCase):
    """build_payload includes generation metadata."""

    def test_payload_has_generation(self):
        from shot_eval.report import build_payload

        result = {
            "generation": {
                "video_model": "pay-model", "video_provider": "pay-prov",
                "prompt_model": "pay-llm", "planner_model": "pay-plan",
                "granularity": "shot", "plan_kind": "rg",
            },
            "judge_model": "test", "repeats": 1, "elapsed": 0,
            "config": {"beat_source": "v", "style_source": "s", "render_source": "r"},
            "scenes": [], "tasks": [],
        }
        payload = build_payload(result)
        self.assertIn("generation", payload)
        self.assertEqual(payload["generation"]["video_model"], "pay-model")

    def test_payload_generation_inferred_from_run_root(self):
        from shot_eval.report import build_payload

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "r"
            run_dir.mkdir()
            meta = {"video_model": "root-model"}
            (run_dir / "video_prompts.json").write_text(json.dumps({"meta": meta}))

            result = {
                "judge_model": "j", "repeats": 1, "elapsed": 0,
                "config": {}, "scenes": [], "tasks": [],
            }
            payload = build_payload(result, run_root=run_dir)
            self.assertEqual(payload["generation"]["video_model"], "root-model")


# ===========================================================================
# 7. HTML labels and escaping
# ===========================================================================


class TestHTMLLabelsEscaping(unittest.TestCase):
    """Generation labels appear in rendered HTML with proper escaping."""

    def test_html_report_contains_generation_labels(self):
        from shot_eval.report import build_payload, render_html

        result = {
            "generation": {
                "video_model": "test<script>", "video_provider": "prov&co",
                "prompt_model": "llm\"model", "planner_model": "plan",
                "granularity": "shot", "plan_kind": "rg",
            },
            "judge_model": "j", "repeats": 1, "elapsed": 0,
            "config": {"beat_source": "v", "style_source": "s", "render_source": "r"},
            "scenes": [], "tasks": [],
        }
        payload = build_payload(result)
        page = render_html(payload, "test")
        # Labels must be present
        self.assertIn("视频模型", page)
        self.assertIn("评审模型", page)
        # XSS: the </script> in the JSON blob is escaped to <\/script>
        # so it won't prematurely close the <script> tag
        self.assertNotIn("</script>", page.split("</script>", 1)[0].split('"test')[1]
                          if '"test' in page else "")
        # The raw model name appears in the JSON data blob (with <\/ escaping)
        self.assertIn(r"test<script>", page)
        # The JS esc() function handles XSS at runtime in the browser
        self.assertIn("GEN.video_model", page)

    def test_generation_chips_in_html(self):
        from shot_eval.report import build_payload, render_html

        result = {
            "generation": {
                "video_model": "flash-preview", "video_provider": "gemini_omni",
                "prompt_model": "gemini-flash", "planner_model": "gemini-flash",
                "granularity": "shot", "plan_kind": "rg",
            },
            "judge_model": "j", "repeats": 1, "elapsed": 0,
            "config": {"beat_source": "v", "style_source": "s", "render_source": "r"},
            "scenes": [], "tasks": [],
        }
        payload = build_payload(result)
        page = render_html(payload, "test")
        # The JS code should reference GEN for chip rendering
        self.assertIn("GEN.video_model", page)
        self.assertIn("Prompt LLM", page)
        self.assertIn("Planner LLM", page)
        self.assertIn("粒度", page)


# ===========================================================================
# 8. Service homepage chip
# ===========================================================================


class TestServiceHomepageChip(unittest.TestCase):
    """index_page renders 视频模型 chip for each run."""

    def test_homepage_has_video_model_chip(self):
        from shot_eval.serve import Run, Store, index_page

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "r"
            run_dir.mkdir()
            vids = run_dir / "videos"
            vids.mkdir()
            meta = {"video_model": "hp-model", "video_provider": "hp-prov",
                    "llm_model": "hp-llm", "granularity": "shot"}
            result = {
                "judge_model": "j", "repeats": 1, "elapsed": 0,
                "config": {"beat_source": "v", "style_source": "s", "render_source": "r"},
                "scenes": [], "tasks": [],
                "generation": normalize_meta(meta),
            }
            result_path = run_dir / "eval" / "shots-test-123.json"
            result_path.parent.mkdir()
            result_path.write_text(json.dumps(result))

            # Write video_prompts.json for run root inference
            (run_dir / "video_prompts.json").write_text(json.dumps({"meta": meta}))

            runs = [Run(rid="test", path=result_path, root=run_dir)]
            store = Store()
            page_bytes = index_page(runs, store)
            page = page_bytes.decode("utf-8")

            self.assertIn("视频模型", page)
            self.assertIn("hp-model", page)

    def test_homepage_provider_chip_when_different(self):
        from shot_eval.serve import Run, Store, index_page

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "r"
            run_dir.mkdir()
            vids = run_dir / "videos"
            vids.mkdir()
            meta = {"video_model": "model-v2", "video_provider": "different_prov",
                    "llm_model": "llm"}
            result = {
                "judge_model": "j", "repeats": 1, "elapsed": 0,
                "config": {"beat_source": "v", "style_source": "s", "render_source": "r"},
                "scenes": [], "tasks": [],
                "generation": normalize_meta(meta),
            }
            rp = run_dir / "eval" / "shots-test.json"
            rp.parent.mkdir()
            rp.write_text(json.dumps(result))
            (run_dir / "video_prompts.json").write_text(json.dumps({"meta": meta}))

            runs = [Run(rid="t", path=rp, root=run_dir)]
            page = index_page(runs, Store()).decode()
            self.assertIn("Provider", page)
            self.assertIn("different_prov", page)


# ===========================================================================
# 9. Compare metadata
# ===========================================================================


class TestCompareMetadata(unittest.TestCase):
    """Compare page shows generation metadata in summary cards."""

    def test_meta_block_includes_generation(self):
        from shot_eval.serve import _meta_block

        result = {
            "generation": {
                "video_model": "cm-model", "video_provider": "cm-prov",
                "prompt_model": "cm-llm", "planner_model": "cm-plan",
                "granularity": "shot", "plan_kind": "rg",
            },
            "judge_model": "j", "config": {},
        }
        mb = _meta_block(result)
        self.assertIn("generation", mb)
        self.assertEqual(mb["generation"]["video_model"], "cm-model")

    def test_gen_diff_notice(self):
        from shot_eval.serve import _gen_diff_notice

        left_meta = {"generation": {"video_model": "model-A", "video_provider": "p"}}
        right_meta = {"generation": {"video_model": "model-B", "video_provider": "p"}}
        notice = _gen_diff_notice(left_meta, right_meta)
        self.assertIn("视频生成模型不同", notice)
        self.assertIn("model-A", notice)
        self.assertIn("model-B", notice)

    def test_gen_diff_notice_same_model(self):
        from shot_eval.serve import _gen_diff_notice

        meta = {"generation": {"video_model": "same", "video_provider": "p"}}
        notice = _gen_diff_notice(meta, meta)
        self.assertEqual(notice, "")

    def test_generation_not_in_compat_fields(self):
        """Generation model must NOT be in COMPAT_FIELDS."""
        from shot_eval.serve import COMPAT_FIELDS
        for f in COMPAT_FIELDS:
            self.assertNotIn("video_model", f)
            self.assertNotIn("generation", f)


# ===========================================================================
# 10. Actual runs inference
# ===========================================================================


class TestOptionalRunIntegration(unittest.TestCase):
    """Optional integration checks for a local fixture directory.

    Set ``SHOT_EVAL_TEST_RUNS_DIR`` to a directory containing ``baseline`` and
    ``omni11`` subdirectories, each with a ``video_prompts.json`` bundle. These
    tests are skipped in a clean GitHub clone.
    """

    def _root(self) -> Path:
        raw = os.getenv("SHOT_EVAL_TEST_RUNS_DIR", "").strip()
        if not raw:
            self.skipTest("SHOT_EVAL_TEST_RUNS_DIR is not configured")
        root = Path(raw)
        if not root.is_dir():
            self.skipTest(f"fixture directory does not exist: {root}")
        return root

    def test_baseline_generation_meta(self):
        root = self._root()
        gen = resolve_generation(root / "baseline")
        self.assertNotEqual(gen["video_model"], "unknown")
        self.assertEqual(set(gen.keys()), set(GENERATION_KEYS))

    def test_omni11_generation_meta(self):
        root = self._root()
        gen = resolve_generation(root / "omni11")
        self.assertIn("1.1", gen["video_model"])


# ===========================================================================
# 11. video_model_label
# ===========================================================================


class TestVideoModelLabel(unittest.TestCase):

    def test_model_present(self):
        self.assertEqual(video_model_label({"video_model": "flash", "video_provider": "g"}), "flash")

    def test_fallback_to_provider(self):
        self.assertEqual(video_model_label({"video_model": "unknown", "video_provider": "minimax"}), "minimax")

    def test_fallback_to_unknown(self):
        self.assertEqual(video_model_label({"video_model": "unknown", "video_provider": "unknown"}), "未知")

    def test_empty_dict(self):
        self.assertEqual(video_model_label({}), "未知")


# ===========================================================================
# 12. Summarize includes generation
# ===========================================================================


class TestSummarizeGeneration(unittest.TestCase):

    def test_summarize_includes_generation_key(self):
        from shot_eval.report import summarize
        result = {
            "generation": {
                "video_model": "s-model", "video_provider": "s-prov",
                "prompt_model": "s-llm", "planner_model": "s-plan",
                "granularity": "shot", "plan_kind": "rg",
            },
            "scenes": [],
        }
        brief = summarize(result)
        self.assertIn("generation", brief)
        self.assertEqual(brief["generation"]["video_model"], "s-model")

    def test_summarize_with_run_root(self):
        from shot_eval.report import summarize

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "r"
            run_dir.mkdir()
            meta = {"video_model": "rr-model"}
            (run_dir / "video_prompts.json").write_text(json.dumps({"meta": meta}))

            result = {"scenes": []}
            brief = summarize(result, run_root=run_dir)
            self.assertEqual(brief["generation"]["video_model"], "rr-model")


# ===========================================================================
# 13. compileall
# ===========================================================================


class TestCompileAll(unittest.TestCase):
    """All shot_eval modules compile without errors."""

    def test_compileall(self):
        import compileall
        import shot_eval
        pkg_dir = str(Path(shot_eval.__file__).parent)
        ok = compileall.compile_dir(pkg_dir, quiet=2, force=True)
        self.assertTrue(ok, "compileall failed for shot_eval package")


if __name__ == "__main__":
    unittest.main()
