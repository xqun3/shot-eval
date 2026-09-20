"""No-network unit tests for the report comparison feature in shot_eval.serve."""

from __future__ import annotations

import html
import io
import json
import re
import tempfile
import textwrap
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from shot_eval import report
from shot_eval.serve import (
    COMPAT_FIELDS,
    Handler,
    Run,
    Store,
    _compat_warning,
    _compare_shell,
    _meta_block,
    build_paired_data,
    compare_page,
    discover,
    index_page,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene(
    scene_id: str,
    source: str = "test",
    *,
    key: str | None = None,
    dim_medians: dict[str, float] | None = None,
    unreferenced: list[str] | None = None,
    deduction_count: list[int] | None = None,
    judged: int = 1,
    rounds: list[dict[str, Any]] | None = None,
    render_match: dict[str, Any] | None = None,
) -> dict[str, Any]:
    k = key or f"{source}/{scene_id}"
    dims = {}
    for d in report.DIMENSIONS:
        dk = d["key"]
        med = (dim_medians or {}).get(dk, 5.0)
        dims[dk] = {"median": med, "mean": med, "spread": 0}
    return {
        "key": k,
        "scene_id": scene_id,
        "source": source,
        "dims": dims,
        "unreferenced": unreferenced or [],
        "deduction_count": deduction_count or [0],
        "hits": {},
        "judged": judged,
        "errors": [],
        "render_match": render_match,
        "render_declared": None,
        "rounds": rounds or [],
    }


def _make_task(
    scene_id: str,
    source: str = "test",
    *,
    key: str | None = None,
    clip: str = "",
    duration: float | None = 5.0,
) -> dict[str, Any]:
    return {
        "key": key or f"{source}/{scene_id}",
        "scene_id": scene_id,
        "source": source,
        "clip": clip,
        "duration": duration,
    }


def _make_result(
    scenes: list[dict[str, Any]] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    *,
    judge_model: str = "gemini-3.1-pro",
    prompt_version_tag: str = "v3+abc",
    design_source: str = "generated",
    beat_source: str = "script",
    style_source: str = "default",
    render_source: str = "default",
) -> dict[str, Any]:
    return {
        "judge_model": judge_model,
        "prompt_version_tag": prompt_version_tag,
        "design_source": design_source,
        "config": {
            "beat_source": beat_source,
            "style_source": style_source,
            "render_source": render_source,
        },
        "repeats": 3,
        "elapsed": 42,
        "scenes": scenes or [],
        "tasks": tasks or [],
    }


def _make_run(rid: str, path: Path, root: Path | None = None) -> Run:
    return Run(rid=rid, path=path, root=root or path.parent)


# ---------------------------------------------------------------------------
# Tests: compare_page — no selection
# ---------------------------------------------------------------------------


class TestNoSelection(unittest.TestCase):
    """When neither left nor right is provided, show the selection form."""

    def test_no_selection_returns_html(self):
        page = compare_page([], Store(), None, None)
        self.assertIsInstance(page, bytes)
        text = page.decode("utf-8")
        self.assertIn("请选择两份报告", text)
        self.assertIn("<form", text)
        self.assertIn('method="get"', text)
        self.assertIn('action="/compare"', text)

    def test_no_selection_has_select_controls(self):
        page = compare_page([], Store(), None, None)
        text = page.decode("utf-8")
        self.assertIn('name="left"', text)
        self.assertIn('name="right"', text)


# ---------------------------------------------------------------------------
# Tests: valid same-report pair
# ---------------------------------------------------------------------------


class TestValidSameReportPair(unittest.TestCase):
    """Comparing a report against itself."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        scenes = [_make_scene(f"S{i:02d}") for i in range(3)]
        tasks = [_make_task(f"S{i:02d}", clip=f"/videos/s{i}.mp4") for i in range(3)]
        result = _make_result(scenes, tasks)
        self.json_path = self.root / "eval" / "shots-test.json"
        self.json_path.parent.mkdir(parents=True)
        self.json_path.write_text(json.dumps(result), encoding="utf-8")
        self.run = _make_run("test-rid", self.json_path, self.root)
        self.store = Store()

    def test_same_report_no_compat_warning(self):
        page = compare_page([self.run], self.store, "test-rid", "test-rid")
        text = page.decode("utf-8")
        self.assertNotIn("compat-warn", text)
        # Should contain summary cards
        self.assertIn("LEFT", text)
        self.assertIn("RIGHT", text)

    def test_same_report_has_paired_scenes(self):
        page = compare_page([self.run], self.store, "test-rid", "test-rid")
        text = page.decode("utf-8")
        # 3 scenes should be paired
        for i in range(3):
            self.assertIn(f"test/S{i:02d}", text)

    def test_returns_bytes(self):
        page = compare_page([self.run], self.store, "test-rid", "test-rid")
        self.assertIsInstance(page, bytes)


# ---------------------------------------------------------------------------
# Tests: pair alignment despite swapped orders
# ---------------------------------------------------------------------------


class TestPairAlignmentSwappedOrder(unittest.TestCase):
    """Scenes appear in different order in left vs right; pairing by key still works."""

    def test_pairing_by_key_not_position(self):
        left_scenes = [_make_scene("S01"), _make_scene("S02"), _make_scene("S03")]
        right_scenes = [_make_scene("S03"), _make_scene("S01"), _make_scene("S02")]
        left_result = _make_result(left_scenes,
                                   [_make_task("S01"), _make_task("S02"), _make_task("S03")])
        right_result = _make_result(right_scenes,
                                    [_make_task("S03"), _make_task("S01"), _make_task("S02")])
        pairs = build_paired_data(left_result, right_result, "L", "R")
        self.assertEqual(len(pairs), 3)
        # Order follows left original
        self.assertEqual(pairs[0]["key"], "test/S01")
        self.assertEqual(pairs[1]["key"], "test/S02")
        self.assertEqual(pairs[2]["key"], "test/S03")
        # All paired
        for p in pairs:
            self.assertIsNotNone(p["left"])
            self.assertIsNotNone(p["right"])

    def test_pairing_deterministic(self):
        """Same inputs produce same output regardless of call order."""
        scenes_a = [_make_scene("S01"), _make_scene("S02")]
        scenes_b = [_make_scene("S02"), _make_scene("S01")]
        r_a = _make_result(scenes_a, [_make_task("S01"), _make_task("S02")])
        r_b = _make_result(scenes_b, [_make_task("S02"), _make_task("S01")])
        p1 = build_paired_data(r_a, r_b, "L", "R")
        p2 = build_paired_data(r_a, r_b, "L", "R")
        self.assertEqual(
            [(p["key"], p["left"] is not None, p["right"] is not None) for p in p1],
            [(p["key"], p["left"] is not None, p["right"] is not None) for p in p2],
        )


# ---------------------------------------------------------------------------
# Tests: right-only / left-only
# ---------------------------------------------------------------------------


class TestOneSidedScenes(unittest.TestCase):
    """Scenes present only on one side are explicitly marked."""

    def test_left_only_scene(self):
        left_scenes = [_make_scene("S01"), _make_scene("S02")]
        right_scenes = [_make_scene("S01")]
        lr = _make_result(left_scenes, [_make_task("S01"), _make_task("S02")])
        rr = _make_result(right_scenes, [_make_task("S01")])
        pairs = build_paired_data(lr, rr, "L", "R")
        self.assertEqual(len(pairs), 2)
        s02 = next(p for p in pairs if p["key"] == "test/S02")
        self.assertIsNotNone(s02["left"])
        self.assertIsNone(s02["right"])

    def test_right_only_scene(self):
        left_scenes = [_make_scene("S01")]
        right_scenes = [_make_scene("S01"), _make_scene("S03")]
        lr = _make_result(left_scenes, [_make_task("S01")])
        rr = _make_result(right_scenes, [_make_task("S01"), _make_task("S03")])
        pairs = build_paired_data(lr, rr, "L", "R")
        self.assertEqual(len(pairs), 2)
        # Left-first, then right-only
        self.assertEqual(pairs[0]["key"], "test/S01")
        self.assertEqual(pairs[1]["key"], "test/S03")
        self.assertIsNone(pairs[1]["left"])
        self.assertIsNotNone(pairs[1]["right"])

    def test_right_only_in_rendered_page(self):
        """Missing pane shows '缺失' in the rendered page."""
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        left_scenes = [_make_scene("S01")]
        right_scenes = [_make_scene("S01"), _make_scene("S99")]
        lr = _make_result(left_scenes, [_make_task("S01")])
        rr = _make_result(right_scenes, [_make_task("S01"), _make_task("S99")])
        lp = root / "left" / "shots-l.json"
        rp = root / "right" / "shots-r.json"
        lp.parent.mkdir(parents=True)
        rp.parent.mkdir(parents=True)
        lp.write_text(json.dumps(lr), encoding="utf-8")
        rp.write_text(json.dumps(rr), encoding="utf-8")
        lrun = _make_run("left-rid", lp, root / "left")
        rrun = _make_run("right-rid", rp, root / "right")
        store = Store()
        page = compare_page([lrun, rrun], store, "left-rid", "right-rid").decode("utf-8")
        self.assertIn("缺失", page)


# ---------------------------------------------------------------------------
# Tests: incompatible config/model warning
# ---------------------------------------------------------------------------


class TestCompatWarning(unittest.TestCase):
    """Compatibility banner logic."""

    def test_identical_config_no_warning(self):
        r = _make_result()
        self.assertEqual(_compat_warning(r, r), "")

    def test_model_only_difference(self):
        r1 = _make_result(judge_model="model-A")
        r2 = _make_result(judge_model="model-B")
        w = _compat_warning(r1, r2)
        self.assertIn("Judge 模型不同", w)
        self.assertIn("model-A", w)
        self.assertIn("model-B", w)
        self.assertNotIn("口径不同", w)

    def test_multiple_field_mismatch(self):
        r1 = _make_result(judge_model="A", beat_source="b1")
        r2 = _make_result(judge_model="B", beat_source="b2")
        w = _compat_warning(r1, r2)
        self.assertIn("口径不同", w)
        self.assertIn("judge_model", w)
        self.assertIn("beat_source", w)

    def test_beat_only_mismatch(self):
        r1 = _make_result(beat_source="script")
        r2 = _make_result(beat_source="storyboard")
        w = _compat_warning(r1, r2)
        self.assertIn("口径不同", w)
        self.assertIn("beat_source", w)

    def test_compat_fields_covers_required_set(self):
        required = {"judge_model", "prompt_version_tag", "beat_source",
                     "style_source", "render_source", "design_source"}
        self.assertTrue(required.issubset(set(COMPAT_FIELDS)))


# ---------------------------------------------------------------------------
# Tests: XSS safety — correct escaped values
# ---------------------------------------------------------------------------


class TestXSSEscaping(unittest.TestCase):
    """Dynamic user/model text must be escaped."""

    def test_model_name_escaped(self):
        r = _make_result(judge_model='<script>alert("xss")</script>')
        meta = _meta_block(r)
        # meta_block doesn't escape; compare_page does when rendering
        self.assertIn("<script>", meta["judge_model"])

    def test_page_escapes_model_name(self):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        xss = '<img src=x onerror=alert(1)>'
        result = _make_result(judge_model=xss, scenes=[_make_scene("S01")])
        p = root / "shots-xss.json"
        p.write_text(json.dumps(result), encoding="utf-8")
        run = _make_run("xss-rid", p, root)
        store = Store()
        page = compare_page([run], store, "xss-rid", "xss-rid").decode("utf-8")
        # The raw XSS should NOT appear unescaped
        self.assertNotIn('<img src=x', page)
        self.assertIn(html.escape(xss), page)

    def test_scene_key_escaped_in_table(self):
        xss_key = '<b>bad</b>'
        scene = _make_scene("S01", key=xss_key)
        result = _make_result([scene], [_make_task("S01", key=xss_key)])
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        p = root / "shots-esc.json"
        p.write_text(json.dumps(result), encoding="utf-8")
        run = _make_run("esc-rid", p, root)
        store = Store()
        page = compare_page([run], store, "esc-rid", "esc-rid").decode("utf-8")
        self.assertNotIn('<b>bad</b>', page)
        self.assertIn(html.escape(xss_key), page)

    def test_js_data_escapes_script_close(self):
        """</script> in data must not close the script tag."""
        scene = _make_scene("S01", rounds=[{
            "deductions": [{"evidence": "</script><img>", "severity": "minor"}]
        }])
        result = _make_result([scene], [_make_task("S01")])
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        p = root / "shots-js.json"
        p.write_text(json.dumps(result), encoding="utf-8")
        run = _make_run("js-rid", p, root)
        store = Store()
        page = compare_page([run], store, "js-rid", "js-rid").decode("utf-8")
        # The literal </script> should be escaped as <\/script> in JSON blob
        self.assertNotIn('</script><img>', page)


# ---------------------------------------------------------------------------
# Tests: media href route
# ---------------------------------------------------------------------------


class TestMediaHref(unittest.TestCase):
    """Paired data must use /media/<rid>/<filename> URL pattern."""

    def test_video_url_format(self):
        scenes = [_make_scene("S01")]
        tasks = [_make_task("S01", clip="/some/path/videos/clip.mp4")]
        result = _make_result(scenes, tasks)
        pairs = build_paired_data(result, result, "my-rid", "my-rid")
        self.assertEqual(pairs[0]["left"]["video_url"], "/media/my-rid/clip.mp4")

    def test_empty_clip_gives_empty_url(self):
        scenes = [_make_scene("S01")]
        tasks = [_make_task("S01", clip="")]
        result = _make_result(scenes, tasks)
        pairs = build_paired_data(result, result, "rid", "rid")
        self.assertEqual(pairs[0]["left"]["video_url"], "")


# ---------------------------------------------------------------------------
# Tests: homepage Compare link
# ---------------------------------------------------------------------------


class TestHomepageCompareLink(unittest.TestCase):
    """The index page must have a link/button to /compare."""

    def test_index_has_compare_link(self):
        page = index_page([], Store())
        text = page.decode("utf-8")
        self.assertIn('/compare', text)
        self.assertIn('Compare', text)

    def test_compare_link_is_anchor(self):
        page = index_page([], Store())
        text = page.decode("utf-8")
        self.assertRegex(text, r'<a\s[^>]*href="/compare"')


# ---------------------------------------------------------------------------
# Tests: invalid rids
# ---------------------------------------------------------------------------


class TestInvalidRids(unittest.TestCase):
    """Invalid or missing rids produce a helpful selection page."""

    def test_invalid_left_rid(self):
        page = compare_page([], Store(), "nonexistent", None).decode("utf-8")
        self.assertIn("请选择两份报告", page)

    def test_both_invalid(self):
        page = compare_page([], Store(), "bad-left", "bad-right").decode("utf-8")
        self.assertIn("无效", page)

    def test_invalid_does_not_crash(self):
        # Must return bytes, not raise
        page = compare_page([], Store(), "???", "!!!")
        self.assertIsInstance(page, bytes)

    def test_left_valid_right_invalid(self):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        result = _make_result([_make_scene("S01")])
        p = root / "shots-a.json"
        p.write_text(json.dumps(result), encoding="utf-8")
        run = _make_run("valid-rid", p, root)
        store = Store()
        page = compare_page([run], store, "valid-rid", "bogus").decode("utf-8")
        self.assertIn("无效", page)
        self.assertIn("bogus", page)


# ---------------------------------------------------------------------------
# Tests: compare route response and Content-Type
# ---------------------------------------------------------------------------


class TestCompareRoute(unittest.TestCase):
    """Test the Handler._compare method integration."""

    def _make_handler(self, path: str, runs: list[Run] | None = None,
                       roots: list[Path] | None = None) -> tuple[Handler, MagicMock]:
        """Create a Handler with mocked wfile for testing."""
        handler = Handler.__new__(Handler)
        handler.path = path
        handler.headers = {}
        handler.wfile = io.BytesIO()
        handler.requestline = f"GET {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.request_version = "HTTP/1.1"
        handler.command = "GET"

        # Set class-level attributes
        Handler.runs = runs or []
        Handler.roots = roots or []
        Handler.store = Store()

        # Mock send_response and send_header to capture response
        responses = []
        headers = {}

        original_send_response = handler.send_response
        original_send_header = handler.send_header
        original_end_headers = handler.end_headers

        def mock_send_response(code, message=None):
            responses.append(code)

        def mock_send_header(key, value):
            headers[key] = value

        def mock_end_headers():
            pass

        handler.send_response = mock_send_response
        handler.send_header = mock_send_header
        handler.end_headers = mock_end_headers

        mock = MagicMock()
        mock.responses = responses
        mock.headers = headers
        return handler, mock

    def test_compare_no_params_returns_200(self):
        handler, mock = self._make_handler("/compare")
        handler._compare("")
        body = handler.wfile.getvalue()
        self.assertGreater(len(body), 0)
        self.assertEqual(mock.responses[0], HTTPStatus.OK)
        self.assertIn("text/html", mock.headers.get("Content-Type", ""))

    def test_compare_with_valid_params(self):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        result = _make_result([_make_scene("S01")])
        p = root / "eval" / "shots-x.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(result), encoding="utf-8")
        # discover expects roots; the run's rid will be URL-encoded relative path
        import urllib.parse
        rid = urllib.parse.quote(str(p.relative_to(root)), safe="")
        run = Run(rid=rid, path=p.resolve(), root=root)
        handler, mock = self._make_handler(
            f"/compare?left={rid}&right={rid}", [run], [root])
        handler._compare(f"left={rid}&right={rid}")
        body = handler.wfile.getvalue()
        self.assertIn(b"LEFT", body)
        self.assertIn(b"RIGHT", body)


# ---------------------------------------------------------------------------
# Tests: build_paired_data edge cases
# ---------------------------------------------------------------------------


class TestBuildPairedData(unittest.TestCase):
    """Detailed tests for the pairing logic."""

    def test_empty_both_sides(self):
        pairs = build_paired_data(_make_result(), _make_result(), "L", "R")
        self.assertEqual(pairs, [])

    def test_fallback_key_from_source_scene_id(self):
        """When key is missing, use source/scene_id as fallback."""
        scene = {"scene_id": "X01", "source": "mycase", "dims": {}, "unreferenced": [],
                 "deduction_count": [], "hits": {}, "judged": 0, "errors": [],
                 "render_match": None, "render_declared": None, "rounds": []}
        lr = _make_result([scene])
        rr = _make_result([scene])
        pairs = build_paired_data(lr, rr, "L", "R")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["key"], "mycase/X01")

    def test_130_scene_pairing(self):
        """Verify pairing works with 130 scenes (the real workload)."""
        scenes = [_make_scene(f"A{i:03d}") for i in range(130)]
        tasks = [_make_task(f"A{i:03d}") for i in range(130)]
        result = _make_result(scenes, tasks)
        pairs = build_paired_data(result, result, "L", "R")
        self.assertEqual(len(pairs), 130)
        for p in pairs:
            self.assertIsNotNone(p["left"])
            self.assertIsNotNone(p["right"])

    def test_slim_scene_keeps_narration_and_video_prompt_only(self):
        """Comparison keeps requested narration/prompt, never whole context/provenance."""
        scene = _make_scene("S01")
        scene["context"] = {
            "beats": [{"start": 0.0, "duration": 3.0, "script": "原始口播", "visual_beat": "action"}],
            "prompt": "This is the full video prompt.",
            "provenance": {"secret": "must not be in compare payload"},
        }
        task = _make_task("S01", clip="/videos/v.mp4")
        result = _make_result([scene], [task])
        pairs = build_paired_data(result, result, "rid", "rid")
        slim = pairs[0]["left"]
        self.assertEqual(slim["scripts"][0]["script"], "原始口播")
        self.assertEqual(slim["video_prompt"], "This is the full video prompt.")
        self.assertNotIn("context", slim)
        self.assertNotIn("provenance", slim)

    def test_union_of_keys_order(self):
        """Left order first, then right-only."""
        l_scenes = [_make_scene("A"), _make_scene("C")]
        r_scenes = [_make_scene("B"), _make_scene("C"), _make_scene("D")]
        lr = _make_result(l_scenes, [_make_task("A"), _make_task("C")])
        rr = _make_result(r_scenes, [_make_task("B"), _make_task("C"), _make_task("D")])
        pairs = build_paired_data(lr, rr, "L", "R")
        keys = [p["key"] for p in pairs]
        # Left order: A, C; then right-only: B, D
        self.assertEqual(keys, ["test/A", "test/C", "test/B", "test/D"])

    def test_deductions_included_in_rounds(self):
        """Round deductions are preserved in paired data."""
        rounds = [{
            "deductions": [
                {"dimension": "clarity", "severity": "major", "points": 1.5,
                 "timestamp": 3.5, "evidence": "blurry", "expected": "sharp"}
            ]
        }]
        scene = _make_scene("S01", rounds=rounds)
        result = _make_result([scene], [_make_task("S01")])
        pairs = build_paired_data(result, result, "L", "R")
        ded = pairs[0]["left"]["rounds"][0]["deductions"][0]
        self.assertEqual(ded["severity"], "major")
        self.assertEqual(ded["points"], 1.5)
        self.assertEqual(ded["timestamp"], 3.5)
        self.assertEqual(ded["evidence"], "blurry")
        self.assertEqual(ded["expected"], "sharp")


# ---------------------------------------------------------------------------
# Tests: JS syntax validation / structural test
# ---------------------------------------------------------------------------


class TestJSSyntax(unittest.TestCase):
    """Structural checks on the embedded JavaScript."""

    def _get_page_with_data(self):
        scenes = [_make_scene("S01", rounds=[{
            "deductions": [{"severity": "minor", "dimension": "clarity",
                            "points": 0.5, "evidence": "ok"}]
        }])]
        result = _make_result(scenes, [_make_task("S01", clip="/v/s01.mp4")])
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        p = root / "shots-js.json"
        p.write_text(json.dumps(result), encoding="utf-8")
        run = _make_run("js-rid", p, root)
        store = Store()
        return compare_page([run], store, "js-rid", "js-rid").decode("utf-8")

    def test_script_tag_present(self):
        page = self._get_page_with_data()
        self.assertIn("<script>", page)

    def test_pairs_json_is_valid(self):
        """The embedded PAIRS data must be valid JSON."""
        page = self._get_page_with_data()
        # Extract the PAIRS assignment
        match = re.search(r'var PAIRS = (.+?);$', page, re.MULTILINE)
        self.assertIsNotNone(match, "PAIRS assignment not found")
        data = json.loads(match.group(1))
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_dims_json_is_valid(self):
        page = self._get_page_with_data()
        match = re.search(r'var DIMS = (.+?);$', page, re.MULTILINE)
        self.assertIsNotNone(match, "DIMS assignment not found")
        data = json.loads(match.group(1))
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 6)

    def test_functions_defined(self):
        """Key JS functions must be present."""
        page = self._get_page_with_data()
        for fn in ("selectPair", "renderSide", "seekVideo", "scoreColor", "esc", "fmtTime"):
            self.assertIn(f"function {fn}", page)

    def test_no_unclosed_braces(self):
        """Basic brace-balance check on the script block."""
        page = self._get_page_with_data()
        script_match = re.search(r'<script>(.*?)</script>', page, re.DOTALL)
        self.assertIsNotNone(script_match)
        js = script_match.group(1)
        self.assertEqual(js.count("{"), js.count("}"),
                         "Unbalanced braces in JS")


# ---------------------------------------------------------------------------
# Tests: meta_block extraction
# ---------------------------------------------------------------------------


class TestMetaBlock(unittest.TestCase):
    def test_extracts_all_fields(self):
        result = _make_result(
            judge_model="test-model",
            prompt_version_tag="v3+xyz",
            design_source="generated",
            beat_source="script",
            style_source="default",
            render_source="default",
        )
        meta = _meta_block(result)
        self.assertEqual(meta["judge_model"], "test-model")
        self.assertEqual(meta["prompt_version_tag"], "v3+xyz")
        self.assertEqual(meta["beat_source"], "script")

    def test_missing_config_returns_empty(self):
        result = {"judge_model": "m"}
        meta = _meta_block(result)
        self.assertEqual(meta["beat_source"], "")


# ---------------------------------------------------------------------------
# Tests: old tests still work (regression guard)
# ---------------------------------------------------------------------------


class TestOldBehavior(unittest.TestCase):
    """Existing single report/media/raw/home behavior stays working."""

    def test_index_page_still_returns_bytes(self):
        page = index_page([], Store())
        self.assertIsInstance(page, bytes)
        self.assertIn(b"<!DOCTYPE html>", page)

    def test_index_page_has_table_structure(self):
        page = index_page([], Store())
        text = page.decode("utf-8")
        # Should have the original structure elements
        self.assertIn("逐 shot 评估", text)


# ---------------------------------------------------------------------------
# Tests: compare_page shell
# ---------------------------------------------------------------------------


class TestCompareShell(unittest.TestCase):
    def test_shell_returns_valid_html(self):
        page = _compare_shell("sel", "body").decode("utf-8")
        self.assertIn("<!DOCTYPE html>", page)
        self.assertIn("sel", page)
        self.assertIn("body", page)
        self.assertIn("返回总览", page)

    def test_shell_has_back_link(self):
        page = _compare_shell("", "").decode("utf-8")
        self.assertIn('href="/"', page)


# ---------------------------------------------------------------------------
# Tests: URL encoding preservation
# ---------------------------------------------------------------------------


class TestURLEncoding(unittest.TestCase):
    """Chinese characters and special chars in rids must be properly handled."""

    def test_chinese_rid_in_select_option(self):
        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        result = _make_result()
        p = root / "shots-淀粉-test.json"
        p.write_text(json.dumps(result), encoding="utf-8")
        rid = "shots-%E6%B7%80%E7%B2%89-test.json"
        run = _make_run(rid, p, root)
        store = Store()
        page = compare_page([run], store, None, None).decode("utf-8")
        # The option should contain the escaped rid as value
        self.assertIn(rid, page)


# ---------------------------------------------------------------------------
# Tests: compat_warning HTML safety
# ---------------------------------------------------------------------------


class TestCompatWarningHTML(unittest.TestCase):
    def test_warning_escapes_model_name(self):
        xss = '<script>alert(1)</script>'
        r1 = _make_result(judge_model=xss)
        r2 = _make_result(judge_model="safe")
        w = _compat_warning(r1, r2)
        self.assertNotIn('<script>', w)
        self.assertIn(html.escape(xss), w)


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# Tests: fully expanded comparison details
# ---------------------------------------------------------------------------

class TestFullyExpandedComparison(unittest.TestCase):
    """All paired videos and per-round evaluation content are present by default."""

    def test_slim_round_keeps_full_evaluation_content(self):
        from shot_eval.serve import build_paired_data

        left = _make_result([_make_scene("S01")])
        left["tasks"] = [{"key": "test/S01", "clip": "/tmp/S01.mp4", "duration": 3.0}]
        left["scenes"][0]["rounds"] = [{
            "observed": "observed content",
            "science_accuracy_reason": "science reason",
            "has_subtitle_or_caption": True,
            "has_watermark_or_logo": False,
            "has_brand_name": False,
            "visible_text": ["TEXT"],
            "issues": ["artifact"],
            "render_verdict": "cg",
            "render_match": True,
            "render_reason": "render reason",
            "deductions": [],
        }]
        pairs = build_paired_data(left, _make_result([]), "l", "r")
        rnd = pairs[0]["left"]["rounds"][0]
        self.assertEqual(rnd["observed"], "observed content")
        self.assertEqual(rnd["science_accuracy_reason"], "science reason")
        self.assertEqual(rnd["visible_text"], ["TEXT"])
        self.assertEqual(rnd["issues"], ["artifact"])
        self.assertEqual(rnd["render_reason"], "render reason")

    def test_page_renders_all_pairs_by_default(self):
        from shot_eval.serve import Run, Store, compare_page

        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        result = _make_result([_make_scene("S01"), _make_scene("S02")])
        result["tasks"] = [
            {"key": "test/S01", "clip": "/tmp/S01.mp4", "duration": 3.0},
            {"key": "test/S02", "clip": "/tmp/S02.mp4", "duration": 3.0},
        ]
        path = root / "eval" / "shots-x.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        run = Run("x", path.resolve(), root)
        page = compare_page([run], Store(), "x", "x").decode("utf-8")
        self.assertIn('id="all-pair-details"', page)
        self.assertIn("renderAllPairs();", page)
        self.assertIn('preload="metadata"', page)
        self.assertIn("全部逐 shot 对比", page)


class TestNarrationAndPromptComparison(unittest.TestCase):
    """Compare UI exposes per-shot narration and full model prompt in details."""

    def test_page_has_expandable_narration_and_prompt(self):
        from shot_eval.serve import Run, Store, compare_page

        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        scene = _make_scene("S01")
        scene["context"] = {
            "beats": [{"start": 0.0, "duration": 3.0, "script": "原始口播句子", "visual_beat": "act"}],
            "prompt": "COMPLETE VIDEO PROMPT",
        }
        result = _make_result([scene], [_make_task("S01", clip="/videos/S01.mp4")])
        path = root / "eval" / "shots-x.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        run = Run("x", path.resolve(), root)
        page = compare_page([run], Store(), "x", "x").decode("utf-8")
        self.assertIn("原始口播（评估输入）", page)
        self.assertIn("送入视频模型的完整 Prompt", page)
        self.assertIn("原始口播句子", page)
        self.assertIn("COMPLETE VIDEO PROMPT", page)
        self.assertIn('<details open', page)
