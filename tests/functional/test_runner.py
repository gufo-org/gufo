"""Fast checks for the functional runner's failure reporting and process ownership."""

import contextlib
from copy import deepcopy
import base64
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/functional"))
spec = importlib.util.spec_from_file_location(
    "functional", ROOT / "tests/functional/run.py")
functional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functional)
from metrics import (CaseComplete, Recorder, canonical, compare, join_server_timings,
                     qualify, summarize, validate_tool_events)
from progress import ProgressTrace
from tool_reasoning import ARGUMENTS, assert_edit, assert_terminal_call, assert_no_envelope_framing
from discovery import assert_model_listing
from image_inputs import assert_color, image_cases, invalid_image_cases
from cache_concurrency import check_cache_concurrency
from cache_shared_prefix import check_cache_shared_prefix
from cache_bridge import check_cache_bridge
from cache_compaction import check_cache_compaction, ARCHIVE_LINE
from cache_disk_spacing import check_disk_spacing
from cache_growth import check_cache_growth
from cache_depth import check_cache_depth
from cache_rotation import check_cache_rotation, check_snapshot_budget, host_available_bytes
from server_metrics import (COUNTERS, TYPES, PROMPT, GENERATED, PROCESSING,
                            CACHED, MAX_SEQUENCE, DRAFT_ROUNDS, DRAFTS, ACCEPTED, PROMPT_SECONDS,
                            GENERATED_SECONDS, KV_USAGE, assert_slots,
                            parse_metrics, assert_accounting, validate_metrics_report)


class FunctionalRunnerTest(unittest.TestCase):
    def test_continuation_restore_reports_observed_equality(self):
        import continuation

        def result(text, prefill=37, cached=672, disk=True):
            return {"choices": [{"message": {"role": "assistant", "content": text}}],
                    "usage": {"cached_tokens": cached, "gufo": {
                        "prefill_tokens": prefill, "ttft_ms": 1, "cache_disk_hit": disk}}}

        cases = [
            ({"resume_equal": False}, None),
            ({"followup_equal": False}, None),
            ({}, None),
            ({"followup_equal": None}, None),
            ({"resume_equal": False, "followup_equal": None}, None),
            ({"temperature": 0}, None),
            ({"prefill": 0}, None),
            ({"followup_cached": 671}, "disk-restored third turn differs"),
            ({"prefill": 2048}, "disk restore lost too much"),
            ({"disk": False}, "restart did not restore disk state"),
            ({"cached": 0}, "restart did not restore disk state"),
        ]
        for strict in ({"temperature": 0}, {"prefill": 0}):
            cases += [({**strict, "resume_equal": False}, "disk restore changed seeded output"),
                      ({**strict, "followup_equal": False}, "disk-restored third turn differs")]
        for changes, error in cases:
            settings = {"temperature": .8, "prefill": 37, "cached": 672, "disk": True,
                        "resume_equal": True, "followup_equal": True, "followup_cached": 672,
                        **changes}
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "restored.json"
                previous = {"case": "synthetic", "request": {
                    "temperature": settings["temperature"]}, "sha256": continuation.digest(result("7"))}
                replies = [result("7" if settings["resume_equal"] else "8",
                                  settings["prefill"], settings["cached"], settings["disk"])]
                if settings["followup_equal"] is not None:
                    previous["followup"] = {"request": {"temperature": settings["temperature"]},
                                            "sha256": continuation.digest(result("seven"))}
                    replies.append(result("seven" if settings["followup_equal"] else "eight",
                                          cached=settings["followup_cached"]))
                source = root / "previous.json"
                source.write_text(json.dumps([previous]))
                argv = ["continuation.py", "--restore", str(source), "--output", str(output)]
                with patch.object(sys, "argv", argv), \
                        patch.object(continuation.TRACE, "path", None), \
                        patch.object(continuation, "call", side_effect=replies) as request, \
                        patch("http.client.HTTPConnection.request",
                              side_effect=AssertionError("HTTP is forbidden")) as http, \
                        contextlib.redirect_stdout(io.StringIO()):
                    if error:
                        with self.assertRaisesRegex(RuntimeError, error):
                            continuation.main()
                        self.assertFalse(output.exists())
                    else:
                        continuation.main()
                        self.assertEqual(request.call_count, len(replies))
                        report, = json.loads(output.read_text())
                        self.assertEqual(report["exact"], settings["resume_equal"]
                                         and settings["followup_equal"] is not False)
                        self.assertEqual(report.get("status"), "passed")
                        self.assertEqual(report.get("exact_required"),
                                         settings["temperature"] == 0 or settings["prefill"] == 0)
                    http.assert_not_called()

    def test_continuation_drop_reasoning_omits_seeded_tool_thought(self):
        import continuation

        class FirstRequestCaptured(Exception):
            pass

        control = None
        for drop_reasoning in (False, True):
            with self.subTest(drop_reasoning=drop_reasoning), \
                    tempfile.TemporaryDirectory() as directory:
                captured = []

                def capture(url, body, stop_field=None):
                    captured.append((deepcopy(body), stop_field))
                    raise FirstRequestCaptured

                argv = ["continuation.py", "--url", "http://example.invalid:1",
                        "--output", str(Path(directory) / "unused.json"), "--tools",
                        "--case", "reasoning_content-preserve1-sampled0",
                        "--prefix-repetitions", "1"]
                if drop_reasoning:
                    argv.append("--drop-reasoning")
                with patch.object(sys, "argv", argv), \
                        patch.object(continuation.TRACE, "path", None), \
                        patch.object(continuation, "call", side_effect=capture) as request, \
                        patch("http.client.HTTPConnection.request",
                              side_effect=AssertionError("HTTP is forbidden")) as http:
                    with self.assertRaises(FirstRequestCaptured):
                        continuation.main()
                    request.assert_called_once()
                    http.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])
                body, stop_field = captured[0]
                self.assertEqual(stop_field, "reasoning_content")
                messages = body["messages"]
                self.assertEqual([m["role"] for m in messages],
                                 ["system", "user", "assistant", "tool"])
                self.assertEqual(messages[2]["tool_calls"], [{
                    "id": "fixture-call", "type": "function",
                    "function": {"name": "read_fixture", "arguments": "{}"}}])
                self.assertEqual(messages[3], {
                    "role": "tool", "tool_call_id": "fixture-call",
                    "content": "The fixture is ready. Answer the user's request directly."})
                if drop_reasoning:
                    self.assertNotIn("reasoning_content", messages[2])
                    del control["messages"][2]["reasoning_content"]
                    self.assertEqual(body, control)
                else:
                    self.assertEqual(messages[2]["reasoning_content"], "Read the fixture.")
                    control = body

    def test_pi_proxy_discovery_and_missing_content_type(self):
        import http.client
        import http.server
        import threading
        from pi_agent import Recorder as PiRecorder

        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                if "untyped" not in self.path:
                    self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"object":"list","data":[]}')

        with tempfile.TemporaryDirectory() as directory, \
                http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as upstream:
            with PiRecorder(f"http://127.0.0.1:{upstream.server_port}",
                            Path(directory)) as proxy:
                workers = [threading.Thread(target=s.serve_forever, daemon=True)
                           for s in (upstream, proxy)]
                for worker in workers:
                    worker.start()
                try:
                    for suffix, content_type in (
                            ("", "application/json"),
                            ("?untyped=1", "application/octet-stream")):
                        connection = http.client.HTTPConnection(
                            "127.0.0.1", proxy.server_port, timeout=5)
                        try:
                            connection.request("GET", "/v1/models" + suffix)
                            response = connection.getresponse()
                            self.assertEqual(response.status, 200)
                            self.assertEqual(response.getheader("Content-Type"), content_type)
                            self.assertEqual(json.loads(response.read()),
                                             {"object": "list", "data": []})
                        finally:
                            connection.close()
                    self.assertEqual(proxy.requests, [])
                    self.assertEqual(list(Path(directory).iterdir()), [])
                finally:
                    for server in (proxy, upstream):
                        server.shutdown()
                    for worker in workers:
                        worker.join(timeout=5)

    def test_image_inputs_require_a_projector_before_starting_a_server(self):
        for suite in ("image-inputs", "image-count", "tool-images"):
            argv = ["run.py", "--output", "/unused", "--sampling-preset", "qwen38",
                    "--suite", suite, "--record-baseline", "--", "gufo", "serve", "llm"]
            with patch.object(sys, "argv", argv), patch.object(functional, "server") as start, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                functional.main()
            self.assertEqual(error.exception.code, 2)
            start.assert_not_called()

    def test_image_inputs_cover_real_formats_and_require_the_correct_color(self):
        def image(color):
            return {"image_url": {"url": "data:image/png;base64,AQID"}}
        cases = dict(image_cases(image))
        self.assertEqual(len(cases), 7)
        jpeg = base64.b64decode(cases["jpeg"].split(",", 1)[1], validate=True)
        self.assertEqual(len(jpeg), 384)
        self.assertTrue(jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"))
        for name in ("webp_lossless", "webp_lossy"):
            data = base64.b64decode(cases[name].split(",", 1)[1], validate=True)
            self.assertEqual(data[:4], b"RIFF")
            self.assertEqual(int.from_bytes(data[4:8], "little"), len(data) - 8)
            self.assertEqual(data[8:12], b"WEBP")
        self.assertEqual(len(invalid_image_cases(image)), 5)
        result = {"text": "red", "reasoning": "", "finish": "stop", "usage": {
            "prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0},
            "gufo": {"prefill_tokens": 100}}}
        assert_color(result, "red")
        for change in ({"text": "blue"}, {"text": "not red"}, {"reasoning": "leaked"},
                       {"finish": "length"}, {"usage": {**result["usage"],
                        "prompt_tokens_details": {"cached_tokens": 100}}}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                assert_color({**result, **change}, "red")

    def test_tool_reasoning_requires_the_bug_trigger_and_exact_edit(self):
        result = {"text": "", "reasoning": "Quoted <tool_call> is file data.",
                  "finish": "tool_calls", "tools": [{"function": {
                      "name": "edit", "arguments": json.dumps(ARGUMENTS)}}]}
        assert_edit(result)
        for change in ({"reasoning": "No quoted tag."}, {"text": "leaked reasoning"},
                       {"tools": []}, {"finish": "length"},
                       {"tools": [{"function": {"name": "edit", "arguments": "{}"}}]}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                assert_edit({**result, **change})

    def test_envelope_workflow_rejects_missing_calls_and_damaged_arguments(self):
        command = "printf '%s' '</invoke>'"
        call = {"function": {"name": "terminal", "arguments": json.dumps({"command": command})}}
        result = {"text": "", "finish": "tool_calls", "tools": [call]}
        assert_terminal_call(result, command)
        for change in ({"tools": []}, {"tools": [call, call]}, {"finish": "stop"},
                       {"tools": [{"function": {"name": "other", "arguments": call["function"]["arguments"]}}]},
                       {"tools": [{"function": {"name": "terminal", "arguments":
                                   json.dumps({"command": "echo '</' + 'invoke>'"})}}]}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                assert_terminal_call({**result, **change}, command)
        for text in ("</invoke>", '<invoke name="terminal">', '<parameter name="command">pwd'):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                assert_no_envelope_framing({**result, "text": text})

    def test_discovery_requires_an_explicit_expectation_before_starting_a_server(self):
        for suite in ("discovery", "all"):
            argv = ["run.py", "--output", "/unused", "--sampling-preset", "qwen38",
                    "--suite", suite, "--record-baseline", "--", "gufo", "serve", "llm"]
            with self.subTest(suite=suite), patch.object(sys, "argv", argv), \
                    patch.object(functional, "server") as start, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                functional.main()
            self.assertEqual(error.exception.code, 2)
            start.assert_not_called()

    def test_discovery_requires_expected_capabilities_and_model_metadata(self):
        entry = {"id": "test", "object": "model", "created": 1, "owned_by": "gufo",
                 "context_length": 8192, "architecture": {"input_modalities": ["text"]}}
        listing = {"object": "list", "data": [entry]}
        assert_model_listing(listing, "test", 8192, ["text"])
        image = {**entry, "architecture": {"input_modalities": ["text", "image"]}}
        assert_model_listing({**listing, "data": [image]}, "test", 8192, ["text", "image"])
        for invalid in (
            {**listing, "data": []}, {**listing, "data": [entry, entry]},
            {**listing, "data": [{k: v for k, v in entry.items() if k != "architecture"}]},
            *({**listing, "data": [{**entry, **change}]} for change in (
                {"id": "wrong"}, {"context_length": 4096}, {"owned_by": "wrong"},
                {"created": True}, {"architecture": {"input_modalities": ["image"]}},
                {"architecture": {"input_modalities": ["text", "image"]}})),
        ):
            with self.subTest(invalid=invalid), self.assertRaises((AssertionError, KeyError)):
                assert_model_listing(invalid, "test", 8192, ["text"])

    def test_discovery_fingerprints_ignore_only_creation_time(self):
        def measurement(value, endpoint="/v1/models"):
            parts = [(1, json.dumps(value).encode())]
            return summarize(parts, False, True, (endpoint, {}, 200))
        listing = {"object": "list", "data": [{"id": "test", "created": 1,
                   "context_length": 8192, "architecture": {"input_modalities": ["text"]}}]}
        _, first = measurement(listing)
        entry = listing["data"][0]
        self.assertEqual(first, measurement({**listing, "data": [{**entry, "created": 2}]})[1])
        for change in ({"id": "other"}, {"context_length": 4096},
                       {"architecture": {"input_modalities": ["text", "image"]}}):
            self.assertNotEqual(first, measurement({**listing, "data": [{**entry, **change}]})[1])
        self.assertNotEqual(measurement({"status": "ok"}, "/health")[1],
                            measurement({"status": "broken"}, "/health")[1])

    def test_discovery_timings_do_not_require_generation_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "discovery.requests.json").write_text(json.dumps({"requests": [{
                "index": 0, "endpoint": "/v1/models", "request_id": "r1",
                "output_sha256": "models", "metrics": {}}]}))
            (root / "server.log").write_text(
                "[http] request=r1 event=completed duration_ms=0.2\n")
            join_server_timings(root)
            row = json.loads((root / "discovery.requests.json").read_text())["requests"][0]
            self.assertEqual(row["metrics"], {"server_duration_ms": .2})
            (root / "server.log").write_text("")
            with self.assertRaisesRegex(ValueError, "no completed"):
                join_server_timings(root)
            for endpoint in ("/health", "/v1/health", "/ready", "/v1/ready"):
                (root / "discovery.requests.json").write_text(json.dumps({"requests": [{
                    "index": 0, "endpoint": endpoint, "request_id": "r1",
                    "output_sha256": "probe", "wall_ms": 1, "metrics": {}}]}))
                join_server_timings(root)

    def test_cache_rotation_checks_host_headroom_and_requested_limits(self):
        gib = 1024**3
        def log(capacity, sessions=1, entries=128):
            return (f"event=snapshot_cache_configured sessions={sessions} "
                    f"snapshot_entries={entries} capacity_bytes={capacity}\n")
        self.assertEqual(host_available_bytes("MemTotal: 9 kB\nMemAvailable: 4096 kB\n"),
                         4096 * 1024)
        for invalid in ("", "MemAvailable: 0 kB\n", "MemAvailable: invalid kB\n"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                host_available_bytes(invalid)
        for available, requested, capacity in (
            (44 * gib, 0, 22 * gib), (128 * gib, 0, 32 * gib),
            (44 * gib, 20 * gib, 20 * gib), (44 * gib, 64 * gib, 22 * gib),
            (44 * gib, 30 * gib, 30 * gib),  # Explicit limits may pass half.
            (44 * gib, 64 * gib, 40 * gib), (128 * gib, 48 * gib, 48 * gib),
        ):
            result = check_snapshot_budget(log(capacity), available, requested, 1)
            self.assertEqual(result["capacity_bytes"], capacity)
        for output, available, requested in (
            (log(32 * gib), 44 * gib, 0),  # A fixed cap can exceed host headroom.
            (log(64 * gib), 44 * gib, 64 * gib),  # Overrides keep 4 GiB free.
            (log(41 * gib), 44 * gib, 64 * gib),
            (log(48 * gib), 128 * gib, 0), (log(21 * gib), 44 * gib, 20 * gib),
            (log(0), 44 * gib, 0), (log(20 * gib, sessions=4), 44 * gib, 0),
            (log(20 * gib, entries=8), 44 * gib, 0), ("", 44 * gib, 0),
            (log(20 * gib) * 2, 44 * gib, 0),
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                check_snapshot_budget(output, available, requested, 1)

    def test_disk_spacing_checks_drained_logs(self):
        def event(tokens, reason="saved"):
            action = "stored" if reason == "saved" else "skipped"
            return (f"[cache] event=disk_cache action={action} reason={reason} "
                    f"file_bytes=1 payload_bytes=1 tokens={tokens} retained_bytes=1\n")
        totals = (2400, 2680, 2960, 3240, 3520, 5960, 6240)
        report = {"turns": [{"measured": {"total": total}} for total in totals]}
        skips = "".join(event(total, "min_step") for total in totals[1:5] + totals[6:])
        grown = event(310) + event(2396) + skips + event(5956)
        restored = event(3012)
        summary = check_disk_spacing(grown, restored, report)
        self.assertEqual(summary["stored_tokens"], [2396, 5956])
        self.assertEqual(summary["shared_boundary_tokens"], [3012])
        for label, grown_log, restored_log in (
            ("every turn written", grown + event(2700), restored),
            ("long turn missing", event(2396) + skips, restored),
            ("first turn missing", skips + event(5956), restored),
            ("no skips", event(2396) + event(5956), restored),
            ("shared boundary skipped", grown, event(3012, "min_step")),
        ):
            with self.subTest(label), self.assertRaises(ValueError):
                check_disk_spacing(grown_log, restored_log, report)

    def run_cache_depth(self, lost=False, retry_work=False, contaminated=False,
                        recorder=None, concurrency=4):
        owner = threading.get_ident()

        class Checks(dict):
            def __setitem__(inner, label, value):
                self.assertEqual(threading.get_ident(), owner)
                super(Checks, inner).__setitem__(label, value)
                if recorder:
                    recorder.mark(label)

        requests, checks, seen = [], Checks(), set()

        def chat_result(client, body):
            requests.append(deepcopy(body))
            if recorder:
                recorder.begin("/v1/chat/completions", body).row["status"] = "complete"
            messages = body["messages"]
            side = messages[0]["content"].startswith("cache_depth_side")
            total = 64 if side else 20000 + (len(messages) - 2) * 100
            cold = body.get("extra_body", {}).get("cache_prompt") is False
            key = json.dumps(messages, sort_keys=True)
            repeated = key in seen and not cold
            cached = (0 if cold or lost else total if repeated else max(0, total - 200))
            if repeated and retry_work:
                cached = total - 1
            seen.add(key)
            code = messages[-1]["content"].rsplit("only ", 1)[-1].rstrip(".")
            if contaminated and cached:
                code = "WRONG"
            return {"text": code, "reasoning": "", "tools": [], "finish": "stop",
                    "usage": {"prompt_tokens": total, "cached_tokens": cached,
                              "completion_tokens": 2,
                              "gufo": {"prefill_tokens": total - cached}}}

        with contextlib.redirect_stderr(io.StringIO()):
            check_cache_depth(None, "fixture", checks, chat_result, concurrency)
        return requests, checks

    def test_cache_depth_replays_branches_before_cold_controls(self):
        requests, checks = self.run_cache_depth()
        self.assertEqual(len(checks), 17)
        self.assertEqual(len(requests), 16)
        self.assertTrue(all(body.get("extra_body", {}).get("cache_prompt") is False
                            for body in requests[-7:]))
        for code in ("RED", "GREEN", "BLUE"):
            self.assertEqual(checks["depth_branch_" + code]["text"], code)
        self.assertEqual(requests[7]["messages"], requests[8]["messages"])
        self.assertEqual(requests[7]["messages"], requests[-1]["messages"])

    def test_cache_depth_records_one_concurrent_history_group(self):
        import metrics
        recorder = Recorder(None)
        requests, _ = self.run_cache_depth(recorder=recorder)
        cohort = [row for row in recorder.rows if row.get("case") == "depth_branches"]
        self.assertEqual(len(cohort), 3)
        self.assertEqual({row["request_sha256"] for row in cohort},
                         {metrics.digest({"endpoint": "/v1/chat/completions", "body": body})
                          for body in requests[3:6]})

    def test_cache_depth_single_slot_keeps_branch_admission_order(self):
        requests, checks = self.run_cache_depth(concurrency=1)
        self.assertEqual([body["messages"][-1]["content"] for body in requests[3:6]],
                         ["Reply with only RED.", "Reply with only GREEN.",
                          "Reply with only BLUE."])
        self.assertEqual(set(checks["depth_branches"]), {"RED", "GREEN", "BLUE"})

    def test_cache_depth_rejects_lost_frontiers_and_retry_work(self):
        for options in ({"lost": True}, {"retry_work": True}, {"contaminated": True}):
            with self.subTest(options=options), self.assertRaises(AssertionError):
                self.run_cache_depth(**options)

    def run_cache_rotation(self, lost=False, contaminated=False):
        requests, checks, previous = [], {}, {}

        def chat_result(client, body):
            requests.append(json.loads(json.dumps(body)))
            label = body["messages"][0]["content"].splitlines()[0]
            side = "_side_" in label
            total = (64 if side else 12000 if label.endswith("main") else 3000) \
                + (len(body["messages"]) - 2) * 100
            cold = body.get("extra_body", {}).get("cache_prompt") is False
            cached = 0 if cold or lost else max(0, previous.get(label, 0) - 5)
            previous[label] = total
            code = "ALPHA" if side else "BETA" if label.endswith("main") \
                else label.rsplit("_", 1)[-1]
            if contaminated and cached:
                code = "WRONG"
            return {"text": code, "reasoning": "", "tools": [], "finish": "stop",
                    "usage": {"prompt_tokens": total, "cached_tokens": cached,
                              "completion_tokens": 2,
                              "gufo": {"prefill_tokens": total - cached}}}

        with contextlib.redirect_stderr(io.StringIO()):
            check_cache_rotation(None, "fixture", checks, chat_result)
        return requests, checks

    def test_cache_rotation_delays_controls_and_replays_actual_answers(self):
        requests, checks = self.run_cache_rotation()
        self.assertEqual(len(checks), 27)
        self.assertEqual([body["messages"][0]["content"].splitlines()[0]
                          for body in requests[:4]],
                         ["cache_rotation_" + code for code in ("RED", "GREEN", "BLUE", "GOLD")])
        self.assertTrue(all(body.get("extra_body", {}).get("cache_prompt") is False
                            for body in requests[-5:]))
        for warm, cold in zip([requests[21], *requests[8:12]], requests[-5:]):
            self.assertEqual(warm["messages"], cold["messages"])
        for body in requests[4:12]:
            code = body["messages"][0]["content"].splitlines()[0].rsplit("_", 1)[-1]
            self.assertTrue(all(message["content"] == code for message in body["messages"]
                                if message["role"] == "assistant"))

    def test_cache_rotation_rejects_lost_history(self):
        with self.assertRaisesRegex(AssertionError, "lost its checkpoint"):
            self.run_cache_rotation(lost=True)

    def test_cache_rotation_rejects_cross_conversation_answers(self):
        with self.assertRaises(AssertionError):
            self.run_cache_rotation(contaminated=True)

    def run_cache_growth(self, pinned=False, missing_reasoning=False):
        requests, checks, previous = [], {}, {}

        def chat_result(client, body):
            # Model-independent checks of request ordering and failure reporting.
            requests.append(json.loads(json.dumps(body)))
            label = body["messages"][0]["content"].splitlines()[0]
            total = 3000 + (len(body["messages"]) - 2) * 100
            last = previous.get(label, 0)
            if body["extra_body"].get("cache_prompt") is False:
                cached = 0
            elif total == last:
                cached = total
            else:
                cached = 2995 if pinned and "drop_reasoning" in label else last - 5
            previous[label] = total
            return {"text": "BETA", "reasoning": "The code word is BETA."
                    if body.get("reasoning_effort", "low") != "none"
                    and not missing_reasoning else "",
                    "tools": [], "finish": "stop", "usage": {
                        "prompt_tokens": total, "cached_tokens": cached,
                        "completion_tokens": 8,
                        "gufo": {"prefill_tokens": total - cached}}}

        class MessagesClient:
            """Messages replay reproduces the previous turn, completion included."""
            def __init__(self):
                self.previous = {}

            def post(self, path, body, cast_to):
                assert path == "/messages" and cast_to is object
                label = body["system"].splitlines()[0]
                total = 3000 + (len(body["messages"]) - 1) * 100
                last = self.previous.get(label, 0)
                cached = total if total == last else min(total, last + 8) if last else 0
                self.previous[label] = total
                thinking = body["thinking"]["type"] == "enabled"
                blocks = ([{"type": "thinking", "thinking": "The code word is BETA.",
                            "signature": ""}] if thinking else [])
                return {"content": blocks + [{"type": "text", "text": "BETA"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": total, "cache_read_input_tokens": cached,
                                  "output_tokens": 8},
                        "timings": {"prompt_n": total - cached}}

        with contextlib.redirect_stderr(io.StringIO()):
            check_cache_growth(MessagesClient(), "fixture", checks, chat_result)
        return requests, checks

    def test_cache_growth_uses_real_replay_shapes_and_delays_cold_controls(self):
        requests, checks = self.run_cache_growth()
        self.assertEqual(len(checks), 54)
        for offset, replay in enumerate(("drop_reasoning", "keep_reasoning",
                                        "discard_reasoning", "thinking_off")):
            history = requests[offset * 9:(offset + 1) * 9]
            self.assertEqual([len(body["messages"]) for body in history[:4]], [2, 4, 6, 8])
            self.assertIs(history[0]["extra_body"]["cache_prompt"], False)
            self.assertTrue(all("cache_prompt" not in body["extra_body"]
                                for body in history[1:5]))
            self.assertTrue(all(body["extra_body"]["cache_prompt"] is False
                                for body in history[5:]))
            self.assertEqual(history[4]["messages"], history[3]["messages"])
            self.assertIs(history[0]["extra_body"]["chat_template_kwargs"]["preserve_thinking"],
                          replay != "discard_reasoning")
            for warm, cold in zip(history[:4], history[5:]):
                self.assertEqual(warm["messages"], cold["messages"])
            for message in history[3]["messages"]:
                if message["role"] == "assistant":
                    self.assertEqual("reasoning_content" in message,
                                     replay in ("keep_reasoning", "discard_reasoning"))

    def test_cache_growth_rejects_a_frozen_checkpoint(self):
        with self.assertRaisesRegex(AssertionError, "cache did not advance"):
            self.run_cache_growth(pinned=True)

    def test_cache_growth_requires_actual_reasoning(self):
        with self.assertRaises(AssertionError):
            self.run_cache_growth(missing_reasoning=True)

    def run_cache_concurrency(self, regress=None):
        import re
        import threading
        lock, seen, previous, requests = threading.Lock(), set(), {}, []

        def tokens(messages):
            return sum(len(m["content"]) // 4 + 10 for m in messages)

        def chat_result(client, body, streaming=False):
            # Model-independent server model: the first arrival per system
            # prompt prefills, later ones restore the shared system prompt.
            messages = body["messages"]
            system, label = messages[0]["content"], messages[0]["content"].split("\n")[0]
            total = tokens(messages)
            cold = body.get("extra_body", {}).get("cache_prompt") is False
            key = json.dumps(messages)
            with lock:
                requests.append(json.loads(json.dumps(body)))
                cached, wait = 0, 0.0
                if cold:
                    pass
                elif len(messages) > 2:
                    cached = previous[json.dumps(messages[:-2])]
                elif len(system) > 2000 and label in seen:
                    cached, wait = tokens(messages[:1]), 5.0
                    if regress == "prefill":
                        cached = 0
                elif key in previous:
                    cached = total
                elif regress == "wait" and "short_shared" in label and label in seen:
                    wait = 5.0
                seen.add(label)
                if not cold:
                    previous[key] = total
            code = re.search(r"code (\w+)\.$", messages[-1]["content"])
            return {"text": code[1] if code else "Z", "reasoning": "", "tools": [], "finish": "stop",
                    "usage": {"prompt_tokens": total, "cached_tokens": cached,
                              "completion_tokens": 2,
                              "gufo": {"prefill_tokens": total - cached,
                                       "shared_prefix_wait_ms": wait}}}

        owner = threading.get_ident()

        class Checks(dict):
            def __setitem__(inner, label, value):
                self.assertEqual(threading.get_ident(), owner)
                super(Checks, inner).__setitem__(label, value)

        checks = Checks()
        with contextlib.redirect_stderr(io.StringIO()):
            check_cache_concurrency(None, "fixture", checks, chat_result, 4,
                                    abandon=lambda client, body, delay: None)
        return requests, checks

    def test_cache_concurrency_sends_groups_and_cold_controls(self):
        requests, checks = self.run_cache_concurrency()
        for group in ("identical", "fanout", "long_tasks", "short_shared", "unrelated"):
            self.assertTrue(all(f"{group}_{index}" in checks for index in range(4)))
            self.assertTrue(all(f"{group}_cold_{index}" in checks for index in range(4)))
        self.assertIn("history_turn_1_0", checks)
        self.assertEqual(sum(name.startswith("cancelled_leader_") for name in checks), 3)
        cold = [body for body in requests
                if body.get("extra_body", {}).get("cache_prompt") is False]
        self.assertTrue(cold and all(body["temperature"] == 0 for body in cold))

    def test_cache_concurrency_admits_distinct_tasks_in_a_fixed_order(self):
        requests, _ = self.run_cache_concurrency()
        tasks = [body["messages"][-1]["content"] for body in requests
                 if body["messages"][0]["content"].startswith("cache_concurrency_fanout\n")
                 and not body.get("extra_body", {}).get("cache_prompt") is False]
        self.assertEqual(tasks, [f"Task {index}. Reply with only the code {code}."
                                 for index, code in enumerate(("ALPHA", "BETA", "GAMMA", "DELTA"))])

    def test_cache_concurrency_rejects_followers_that_prefill_everything(self):
        with self.assertRaisesRegex(AssertionError, "after restoring the shared prefix"):
            self.run_cache_concurrency(regress="prefill")

    def test_cache_concurrency_rejects_waiting_without_a_shared_prefix(self):
        with self.assertRaisesRegex(AssertionError, "waited without a shared prefix"):
            self.run_cache_concurrency(regress="wait")

    def run_cache_shared_prefix(self, regress=False):
        import re
        seen, requests = {}, []

        def tokens(messages):
            return sum(len(m["content"]) // 4 + 10 for m in messages)

        def chat_result(client, body):
            # Model-independent server model: the second conversation under a
            # system prompt learns its boundary, later ones restore it.
            requests.append(json.loads(json.dumps(body)))
            messages = body["messages"]
            label = messages[0]["content"].split("\n")[0]
            total = tokens(messages)
            cached = 0
            if body.get("extra_body", {}).get("cache_prompt") is not False:
                count = seen.get(label, 0)
                seen[label] = count + 1
                if count == 1:
                    cached = 2048
                elif count >= 2:
                    cached = 2048 if regress else tokens(messages[:1])
            code = re.search(r"code (\w+)\.$", messages[-1]["content"])
            return {"text": code[1] if code else "Z", "reasoning": "", "tools": [],
                    "finish": "stop",
                    "usage": {"prompt_tokens": total, "cached_tokens": cached,
                              "completion_tokens": 2,
                              "gufo": {"prefill_tokens": total - cached}}}

        checks = {}
        with contextlib.redirect_stderr(io.StringIO()):
            check_cache_shared_prefix(None, "fixture", checks, chat_result)
        return requests, checks

    def test_cache_shared_prefix_runs_conversations_then_cold_controls(self):
        requests, checks = self.run_cache_shared_prefix()
        for group in ("long_tasks", "short_tasks"):
            self.assertTrue(all(f"{group}_{index}" in checks for index in range(4)))
            self.assertTrue(all(f"{group}_cold_{index}" in checks for index in range(4)))
        warm = [body for body in requests if "extra_body" not in body]
        self.assertEqual(len(warm), 8)

    def test_cache_shared_prefix_rejects_grid_only_reuse(self):
        with self.assertRaisesRegex(AssertionError, "of the shared system prompt"):
            self.run_cache_shared_prefix(regress=True)

    def run_cache_compaction(self, regression=None):
        requests, checks, seen = [], {}, set()
        stem_total = compact_total = None

        def chat_result(client, body):
            nonlocal stem_total, compact_total
            requests.append(deepcopy(body))
            messages = body["messages"]
            cold = body.get("extra_body", {}).get("cache_prompt") is False
            compacted = "Compacted archive summary:" in messages[1]["content"]
            lines = sum(message.get("content", "").count(ARCHIVE_LINE)
                        for message in messages if message["role"] == "tool")
            total = 3000 + 12 * lines + (len(messages) - 2) * 24
            if compacted:
                total += 48
                compact_total = compact_total or total
            stem_total = stem_total or total
            key = json.dumps(body, sort_keys=True)
            repeated = key in seen and not cold
            cached = (0 if cold or len(requests) == 1 else total if repeated
                      else stem_total - 48 if len(messages) == 2
                      else compact_total - 8 if compacted else stem_total - 16)
            if compacted and not cold and not repeated:
                if regression == "lost":
                    cached = 0
                elif regression == "false_hit":
                    cached = stem_total + 1
                elif regression == "pinned" and len(messages) > 2:
                    cached = stem_total - 48
            if repeated and regression == "retry_work":
                cached -= 1
            seen.add(key)
            code = ("GAMMA" if messages[-1]["content"] == "Return exactly GAMMA."
                    else "BETA" if compacted else "ALPHA")
            if compacted and not cold and regression == "stale_answer":
                code = "ALPHA"
            return {"text": code, "reasoning": "", "tools": [], "finish": "stop",
                    "usage": {"prompt_tokens": total, "cached_tokens": cached,
                              "completion_tokens": 2,
                              "gufo": {"prefill_tokens": total - cached}}}

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                check_cache_compaction(None, "fixture", checks, chat_result, 32768)
        finally:
            self.compaction_requests, self.compaction_checks = requests, checks
        return requests, checks

    def test_cache_compaction_keeps_system_tools_and_delays_controls(self):
        requests, checks = self.run_cache_compaction()
        self.assertEqual(len(requests), 16)
        self.assertTrue(all(body.get("extra_body", {}).get("cache_prompt") is False
                            for body in requests[8:]))
        for body in requests:
            self.assertEqual(body["messages"][0], requests[0]["messages"][0])
            self.assertEqual(body["tools"], requests[0]["tools"])
        self.assertEqual(requests[2]["messages"][2]["content"],
                         checks["compaction_seed"]["text"])
        self.assertEqual(requests[3], requests[4])
        self.assertEqual(requests[5], requests[6])
        self.assertEqual(len(requests[5]["messages"]), 2)
        self.assertIn("Compacted archive summary:", requests[5]["messages"][1]["content"])
        self.assertEqual(requests[7]["messages"][2]["content"],
                         checks["compaction_summary"]["text"])
        bounds = checks["compaction_boundaries"]
        self.assertGreaterEqual(bounds["near_limit_tokens"], 32768 * .75)
        self.assertGreaterEqual(bounds["dropped_tokens"], 32768 // 2)

    def test_cache_compaction_rejects_misses_and_false_hits_after_controls(self):
        for regression in ("lost", "false_hit", "pinned"):
            with self.subTest(regression=regression), self.assertRaises(AssertionError):
                self.run_cache_compaction(regression)
            self.assertEqual(len(self.compaction_requests), 16)
            self.assertTrue(self.compaction_checks["compaction_boundaries"]["failures"])

    def test_cache_compaction_rejects_retry_work_and_abandoned_answers(self):
        for regression in ("retry_work", "stale_answer"):
            with self.subTest(regression=regression), self.assertRaises(AssertionError):
                self.run_cache_compaction(regression)

    def test_cache_compaction_requires_room_for_a_large_drop(self):
        with self.assertRaisesRegex(ValueError, "at least 16384"):
            check_cache_compaction(None, "fixture", {}, None, 8192)

    def run_cache_bridge(self, lost=False, restore_bytes=100, capacity=1000):
        requests = []

        def chat_result(client, body):
            requests.append(json.loads(json.dumps(body)))
            messages = body["messages"]
            cold = body.get("extra_body", {}).get("cache_prompt") is False
            if messages[0]["content"].startswith("cache_bridge_side_"):
                return {"text": "OK", "reasoning": "", "tools": [], "finish": "stop",
                        "usage": {"prompt_tokens": 200, "cached_tokens": 0,
                                  "completion_tokens": 1,
                                  "gufo": {"prefill_tokens": 200,
                                           "cache_snapshot_bytes": 300,
                                           "cache_restore_bytes": 0}}}
            if messages[-1]["content"] == "Z":
                total, code = 5000, "Z"
            else:
                turn = (len(messages) + 1) // 2
                total, code = 5790 + (turn - 1) * 70, f"T{turn:02d}"
            cached = 0 if cold or lost or code == "Z" or code in ("T01", "T02") else 4990
            return {"text": code, "reasoning": "", "tools": [], "finish": "stop",
                    "usage": {"prompt_tokens": total, "cached_tokens": cached,
                              "completion_tokens": 2,
                              "gufo": {"prefill_tokens": total - cached,
                                       "cache_snapshot_bytes": 200,
                                       "cache_restore_bytes": restore_bytes if cached else 0}}}

        checks = {}
        with contextlib.redirect_stderr(io.StringIO()):
            check_cache_bridge(None, "fixture", checks, chat_result, capacity)
        return requests, checks

    def test_cache_bridge_fills_budget_then_rewrites_history_and_delays_controls(self):
        requests, checks = self.run_cache_bridge()
        self.assertTrue(all(body["messages"][0]["content"].startswith("cache_bridge_side_")
                            for body in requests[:5]))
        turns = [body for body in requests if body["messages"][0]["content"].startswith(
            "cache_bridge\n") and "extra_body" not in body]
        self.assertEqual(len(turns), 6)
        for index, body in enumerate(turns, 1):
            live = body["messages"][-1]["content"]
            self.assertIn(f"cache-bridge-{index:02d}", live)
            self.assertTrue(live.endswith(f"Reply with only the code T{index:02d}."))
            copies = [message["content"] for message in body["messages"][1:-1:2]]
            self.assertTrue(all(copy.startswith("[Telegram User") for copy in copies))
            self.assertEqual(len(copies), index - 1)
        controls = [body for body in requests
                    if body.get("extra_body", {}).get("cache_prompt") is False]
        self.assertEqual(len(controls), 7)
        self.assertEqual(requests[-7:], controls)
        self.assertEqual(controls[0]["messages"][-1]["content"], "Z")
        self.assertEqual([body["messages"] for body in controls[1:]],
                         [body["messages"] for body in turns])
        self.assertEqual([row["floor"] for row in checks["bridge_evidence"]["turns"]],
                         [4392, 4462, 4532, 4602])

    def test_cache_bridge_rejects_a_lost_divergence_point(self):
        with self.assertRaisesRegex(AssertionError, "restored 0 tokens"):
            self.run_cache_bridge(lost=True)

    def test_cache_bridge_rejects_a_budget_without_pressure(self):
        with self.assertRaisesRegex(AssertionError, "unqualified"):
            self.run_cache_bridge(restore_bytes=10)
        with self.assertRaisesRegex(AssertionError, "snapshot capacity"):
            self.run_cache_bridge(capacity=None)

    def test_prompt_progress_contract_and_output_order(self):
        def event(processed, elapsed=0):
            return {"prompt_progress": {"total": 10, "cache": 2,
                                        "processed": processed, "time_ms": elapsed},
                    "choices": [{"delta": {}, "finish_reason": None}]}
        trace = ProgressTrace(True)
        trace(event(2))
        trace(event(10, 5))
        trace({"choices": [{"delta": {"content": "ok"}}]})
        trace.finish({"prompt_tokens": 10, "cached_tokens": 2})
        with self.assertRaises(AssertionError):
            trace(event(10, 6))
        for invalid in (event(1), event(11), event(3, -1)):
            with self.subTest(event=invalid), self.assertRaises(AssertionError):
                ProgressTrace(True)(invalid)
        with self.assertRaises(AssertionError):
            ProgressTrace(False)(event(2))
        with self.assertRaises(AssertionError):
            ProgressTrace(True).finish({"prompt_tokens": 10, "cached_tokens": 2})
        ProgressTrace(True, allow_missing=True).finish({})

    def test_prometheus_contract_rejects_missing_and_invalid_metrics(self):
        text = "".join(f"# HELP {name} Description\n# TYPE {name} {kind}\n{name} 0\n"
                       for name, kind in TYPES.items())
        self.assertEqual(parse_metrics(text), dict.fromkeys(TYPES, 0))
        large = str(2**60 + 1)
        self.assertEqual(parse_metrics(text.replace(f"{PROMPT} 0", f"{PROMPT} {large}"))[PROMPT],
                         int(large))
        for bad in (
            text.replace(f"{PROMPT} 0\n", ""),
            text + f"{GENERATED} 1\n",
            text.replace(f"# TYPE {PROMPT} counter", f"# TYPE {PROMPT} gauge"),
            text.replace(f"{PROCESSING} 0", f"{PROCESSING} -1"),
            text.replace(f"{PROCESSING} 0", f"{PROCESSING} 0.5"),
            text.replace(f"{GENERATED} 0", f"{GENERATED} nan"),
            text.replace(f"{GENERATED} 0", f"{GENERATED} inf"),
            text.replace(f"{DRAFT_ROUNDS} 0", f"{DRAFT_ROUNDS} 0.5"),
            text.replace(f"{KV_USAGE} 0", f"{KV_USAGE} 1.1"),
        ):
            with self.subTest(text=bad), self.assertRaises(ValueError):
                parse_metrics(bad)

    def test_prometheus_accounting_uses_uncached_work_once(self):
        before = dict.fromkeys(COUNTERS, 10)
        rows = [
            {"http_status": 200, "status": "complete", "metrics": {
                "prompt_tokens": 100, "cached_tokens": 98,
                "prefill_tokens": 2, "completion_tokens": 7,
                "draft_rounds": 3, "draft_tokens": 8, "draft_tokens_accepted": 5, "prefill_ms": 100, "decode_ms": 200}},
            {"http_status": 200, "status": "complete", "metrics": {
                "prompt_tokens": 100, "cached_tokens": 100,
                "prefill_tokens": 0, "completion_tokens": 3,
                "draft_rounds": 2, "draft_tokens": 4, "draft_tokens_accepted": 2, "prefill_ms": 0, "decode_ms": 100}},
            {"http_status": 400, "status": "complete", "metrics": {}},
        ]
        after = {**before, PROMPT: 12, GENERATED: 20, CACHED: 208,
                 MAX_SEQUENCE: 107, DRAFT_ROUNDS: 15, DRAFTS: 22, ACCEPTED: 17,
                 PROMPT_SECONDS: 10.1, GENERATED_SECONDS: 10.3}
        assert_accounting(before, after, rows)
        for bad in ({PROMPT: 210, GENERATED: 20}, {PROMPT: 14, GENERATED: 30},
                    {PROMPT: 12, GENERATED: 19}):
            with self.subTest(after=bad), self.assertRaises(AssertionError):
                assert_accounting(before, {**after, **bad}, rows)
        for metric in (CACHED, MAX_SEQUENCE, DRAFT_ROUNDS, DRAFTS, ACCEPTED, PROMPT_SECONDS, GENERATED_SECONDS):
            with self.subTest(metric=metric), self.assertRaises(AssertionError):
                assert_accounting(before, {**after, metric: after[metric] + 1}, rows)
        rows[0]["status"] = "disconnected"
        with self.assertRaisesRegex(AssertionError, "missing completed"):
            assert_accounting(before, after, rows)

    def test_prometheus_cancelled_work_matches_terminal_server_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.requests.json").write_text(json.dumps({"requests": [
                {"request_id": "r1", "http_status": 200, "status": "disconnected"},
                {"request_id": "r2", "http_status": 200, "status": "complete"},
                {"request_id": "r3", "http_status": 400, "status": "complete"}]}))
            (root / "metrics.json").write_text(json.dumps({"checks": {
                "metrics_chat_cold": {"before": {**dict.fromkeys(COUNTERS, 0), PROMPT: 100, GENERATED: 200}},
                "metrics_live_queue_cancel": {"cancelled": [{"cancelled": True}]},
                "metrics_after_cancel_cached": {"after": {**dict.fromkeys(COUNTERS, 0),
                    PROMPT: 112, GENERATED: 207, CACHED: 20, MAX_SEQUENCE: 24, DRAFT_ROUNDS: 3, DRAFTS: 8, ACCEPTED: 4}}}}))
            log = (
                "[http] request=r1 event=completed status=200 prefill_tokens=12 "
                "generated_tokens=3 prompt_tokens=12 cached_tokens=0 draft_rounds=1 draft_proposed=4 draft_accepted=2 finish=cancelled\n"
                "[http] request=r2 event=completed status=200 prefill_tokens=0 "
                "generated_tokens=4 prompt_tokens=20 cached_tokens=20 draft_rounds=2 draft_proposed=4 draft_accepted=2 finish=length\n"
                "[http] request=r3 event=completed status=400\n")
            (root / "server.log").write_text(log)
            self.assertEqual(validate_metrics_report(root), {"requests": 2, "cancelled": 1})
            # A disconnect can hide its final usage from the client; its tokens
            # must still be counted exactly once in the process-wide total.
            (root / "server.log").write_text(log.replace("generated_tokens=3", "generated_tokens=0"))
            with self.assertRaises(AssertionError):
                validate_metrics_report(root)

    def test_live_slots_reject_stale_identity_progress_and_prompt_disclosure(self):
        slot = {"id": 0, "model": "test", "n_ctx": 4096, "speculative": True,
                "is_processing": True, "id_task": 7, "task_id": 7, "state": 1,
                "n_prompt_tokens": 20, "n_prompt_tokens_cache": 12,
                "n_prompt_tokens_processed": 8, "prompt": "",
                "next_token": [{"has_next_token": True, "has_new_line": False,
                                "n_remain": 6, "n_decoded": 2}]}
        assert_slots([slot], 1, "test", 4096, True)
        for change in ({"prompt": "private"}, {"task_id": 0},
                       {"n_prompt_tokens_processed": 7}, {"speculative": False},
                       {"is_processing": False}, {"id": 2}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                assert_slots([{**slot, **change}], 1, "test", 4096, True)
        with self.assertRaises(AssertionError):
            assert_slots([slot, {**slot, "id": 1}], 2, "test", 4096, True)

    def test_fingerprints_preserve_schema_payload_ids(self):
        for key in ("schema", "parameters", "metadata", "arguments"):
            self.assertNotEqual(canonical({key: {"id": "a", "call_id": "x"}}),
                                canonical({key: {"id": "b", "call_id": "x"}}))
        def request(value):
            return {"tools": [{"type": "function", "function": {
                "name": "f", "parameters": {"type": "object", "const": {"id": value}}}}]}
        self.assertNotEqual(canonical(request("alpha")), canonical(request("beta")))
        self.assertNotEqual(canonical({"id": "user-data-a"}), canonical({"id": "user-data-b"}))
        def history(identifier):
            return {"input": [
                {"type": "function_call", "id": identifier + "-item", "call_id": identifier,
                 "name": "f", "arguments": '{"id":"literal"}'},
                {"type": "function_call_output", "call_id": identifier, "output": "ok"}]}
        self.assertEqual(canonical(history("call_a")), canonical(history("call_b")))

    def test_tool_event_arguments_and_identifiers_match_final_output(self):
        added = {"type": "function_call", "id": "fc1", "call_id": "call1",
                 "name": "f", "arguments": "", "status": "in_progress"}
        done = {**added, "arguments": '{"id":"literal"}', "status": "completed"}
        events = [
            {"type": "response.output_item.added", "output_index": 0, "item": added},
            {"type": "response.function_call_arguments.delta", "output_index": 0,
             "item_id": "fc1", "delta": '{"id":'},
            {"type": "response.function_call_arguments.delta", "output_index": 0,
             "item_id": "fc1", "delta": '"literal"}'},
            {"type": "response.function_call_arguments.done", "output_index": 0,
             "item_id": "fc1", "name": "f", "arguments": done["arguments"]},
            {"type": "response.output_item.done", "output_index": 0, "item": done}]
        validate_tool_events(events, {"output": [done]})
        validate_tool_events(events[:2], None)  # A cancelled stream may be partial.
        for index, key, value in ((1, "delta", '{"id":"wrong"}'), (2, "item_id", "other"),
                                   (3, "name", "wrong"), (4, "output_index", 1)):
            corrupt = json.loads(json.dumps(events))
            corrupt[index][key] = value
            with self.assertRaises(ValueError):
                validate_tool_events(corrupt, {"output": [done]})
        with self.assertRaises(ValueError):
            validate_tool_events(events, {"output": [{**done, "call_id": "other"}]})
        with self.assertRaises(ValueError):
            validate_tool_events(events, {"output": []})

    def test_mode_coverage_checks_loader_restart_and_executed_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "server.log"
            records = root / "state-edges.requests.json"
            def prepare(mode, proposed, accepted):
                log.write_text(f"[loader] event=load_completed kind=text speculative={mode}\n")
                records.write_text(json.dumps({"requests": [{"metrics": {
                    "completion_tokens": 32, "draft_tokens": proposed,
                    "draft_tokens_accepted": accepted}}]}))
            prepare("off", 0, 0)
            self.assertFalse(functional.execution_coverage(root, "off")["draft_execution_observed"])
            for mode in ("dflash2", "mtp", "dspark"):
                prepare(mode, 7, 4)
                self.assertTrue(functional.execution_coverage(root, mode)["draft_execution_observed"])
                with self.assertRaisesRegex(ValueError, "loaded modes"):
                    functional.execution_coverage(root, "off")
            prepare("off", 7, 4)
            with self.assertRaisesRegex(ValueError, "AR request"):
                functional.execution_coverage(root, "off")
            prepare("mtp", 4, 7)
            with self.assertRaisesRegex(ValueError, "counters"):
                functional.execution_coverage(root, "mtp")
            prepare("mtp", 7, 4)
            (root / "server-restarted.log").write_text(
                "[loader] event=load_completed kind=text speculative=off\n")
            with self.assertRaisesRegex(ValueError, "server-restarted"):
                functional.execution_coverage(root, "mtp")

    def test_each_endpoint_timing_location_and_split_usage_chunk(self):
        timing = {"prompt_n": 1, "prompt_ms": 2, "predicted_ms": 3,
                  "cache_restore_ms": 0, "cache_snapshot_ms": 0, "cache_disk_enqueue_ms": 0}
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                 "cached_tokens": 0}
        for endpoint in ("completions", "responses"):
            if endpoint == "completions":
                events = [
                    {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
                     "timings": timing},
                    {"choices": [], "usage": usage},
                ]
            else:
                response = {"status": "completed", "output": [{"type": "message",
                    "content": [{"type": "output_text", "text": "ok"}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                              "input_tokens_details": {"cached_tokens": 0}},
                    "timings": timing}
                events = [
                    {"type": "response.created", "sequence_number": 0},
                    {"type": "response.completed", "sequence_number": 1, "response": response},
                ]
            data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
            measured, fingerprint = summarize([(5, data)], True, True,
                ("/v1/" + endpoint, {"stream": True}, 200))
            self.assertEqual(measured["decode_ms"], 3)
            self.assertEqual(measured["prefill_ms"], 2)
            self.assertIsNotNone(fingerprint)

    def messages_fixture(self):
        request = {"model": "fixture", "max_tokens": 2, "temperature": 0, "seed": 31,
                   "stop_sequences": ["END", "HALT"],
                   "messages": [{"role": "user", "content": "Reply BETA."}]}
        # Buffered Messages uses input_tokens including cached work, no total_tokens,
        # ordered content blocks, and the separate GenerationTimings object.
        response = {"id": "msg_fixture", "type": "message", "role": "assistant",
                    "model": "fixture", "content": [
                        {"type": "thinking", "thinking": "The code is BETA.", "signature": ""},
                        {"type": "text", "text": "BETA"}],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 2,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 6},
                    "timings": {"prompt_n": 4, "prompt_ms": 2, "prompt_per_token_ms": .5,
                                "prompt_per_second": 2000, "predicted_n": 2, "predicted_ms": 3,
                                "predicted_per_token_ms": 1.5, "predicted_per_second": 2000 / 3,
                                "cache_n": 6, "cache_restore_ms": 1, "cache_snapshot_ms": 1,
                                "cache_disk_enqueue_ms": 0, "draft_rounds": 1,
                                "draft_n": 3, "draft_n_accepted": 1}}
        return request, response

    def test_buffered_messages_preserves_output_cache_and_timings(self):
        request, response = self.messages_fixture()
        measured, fingerprint = summarize([(5, json.dumps(response).encode())], False, True,
                                          ("/v1/messages", request, 200))
        expected = {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 6,
                    "prefill_tokens": 4, "prefill_ms": 2, "decode_ms": 3,
                    "prefill_ms_per_token": .5, "decode_ms_per_token": 1.5,
                    "cache_restore_ms": 1, "cache_snapshot_ms": 1, "cache_disk_enqueue_ms": 0,
                    "draft_rounds": 1, "draft_tokens": 3, "draft_tokens_accepted": 1}
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertEqual(measured.get(key), value)
        self.assertIsInstance(fingerprint, str)
        self.assertEqual(len(fingerprint), 64)

    def test_buffered_messages_comparison_detects_output_changes(self):
        request, response = self.messages_fixture()
        text, thinking = deepcopy(response), deepcopy(response)
        text["content"][1]["text"] = "OTHER"
        thinking["content"][0]["thinking"] = "The code is OTHER."
        stopped = {**response, "stop_reason": "stop_sequence", "stop_sequence": "END"}
        cases = [
            ("text", response, text, "failed"),
            ("thinking", response, thinking, "failed"),
            ("stop_reason", response, {**response, "stop_reason": "max_tokens"}, "failed"),
            ("stop_sequence", stopped, {**stopped, "stop_sequence": "HALT"}, "failed"),
            ("generated_id", response, {**response, "id": "msg_other"}, "passed"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory) / "baseline", Path(directory) / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            for label, before, after, expected in cases:
                with self.subTest(change=label):
                    for root, payload in ((baseline, before), (candidate, after)):
                        with patch("metrics.time.monotonic", return_value=1):
                            recorder = Recorder(root / "messages.requests.json")
                            step = recorder.begin("/v1/messages", request)
                            step.row["http_status"] = 200
                            step.feed(json.dumps(payload).encode())
                            step.ended = True
                            step.finish()
                    result = compare(baseline, candidate)
                    self.assertEqual(result["status"], expected)
                    if expected == "failed":
                        self.assertTrue(any("output_sha256 changed" in issue
                                            for issue in result["quality_or_coverage_changes"]))
                    else:
                        self.assertFalse(result["quality_or_coverage_changes"])

    def test_missing_timings_or_logs_cannot_qualify(self):
        completed = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                  "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        with self.assertRaisesRegex(ValueError, "prefill_ms"):
            summarize([(1, json.dumps(completed).encode())], False, True,
                      ("/v1/chat/completions", {}, 200))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "responses.requests.json"
            path.write_text(json.dumps({"version": 1, "requests": [{
                "index": 0, "endpoint": "/v1/responses", "request_id": "r7",
                "metrics": {}, "output_sha256": "output"}]}))
            (root / "server.log").write_text(
                "[INFO] request=r7 event=completed duration_ms=12 queue_ms=1 ttft_ms=3\n")
            join_server_timings(root)
            self.assertEqual(json.loads(path.read_text())["requests"][0]["metrics"]["queue_ms"], 1)
            (root / "server.log").write_text("")
            with self.assertRaisesRegex(ValueError, "no completed"):
                join_server_timings(root)
        record = Recorder(None)
        rejected = record.begin("/v1/chat/completions", {"stream": True})
        rejected.row["http_status"] = 400
        rejected.feed(b'{"error":{"type":"invalid_request_error","message":"bad schema"}}')
        rejected.ended = True
        rejected.finish()  # Errors stay ordinary JSON even for stream:true.
        self.assertEqual(rejected.row["status"], "complete")
        baseline = {"startup_ms": 100, "restart_ms": 100}
        result = {"status": "passed", "measurements": []}
        functional.compare_lifecycle(result, baseline,
                                     {"startup_ms": 125, "restart_ms": 75})
        self.assertEqual(result["status"], "inconclusive")
        for candidate in ({}, {"startup_ms": 100}, {"startup_ms": float("nan")},
                          {"startup_ms": 100, "restart_ms": 0}):
            with self.assertRaisesRegex(ValueError, "server"):
                functional.compare_lifecycle(result, baseline, candidate)
        with self.assertRaisesRegex(ValueError, "restart_ms"):
            functional.compare_lifecycle(result,
                {"startup_ms": 100, "suites": {"text-cancel-disk": {}}},
                {"startup_ms": 100, "suites": {"text-cancel-disk": {}}})

    def test_nullable_parallel_responses_keeps_default(self):
        response = {"status": "completed", "output": [
            {"type": "function_call", "name": "f", "call_id": f"call_{i}",
             "arguments": "{}"} for i in range(2)],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
                      "input_tokens_details": {"cached_tokens": 0}},
            "timings": {"prompt_n": 1, "prompt_ms": 2, "predicted_ms": 3,
                        "cache_restore_ms": 0, "cache_snapshot_ms": 0,
                        "cache_disk_enqueue_ms": 0}}
        body = {"tools": [{"type": "function", "name": "f", "parameters": {}}],
                "parallel_tool_calls": None}
        data = [(1, json.dumps(response).encode())]
        summarize(data, False, True, ("/v1/responses", body, 200))
        body["parallel_tool_calls"] = False
        with self.assertRaisesRegex(ValueError, "parallel_tool_calls"):
            summarize(data, False, True, ("/v1/responses", body, 200))

    def test_fragmented_stream_metrics_and_output_identity(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 2,
                 "gufo": {"prefill_tokens": 10, "prefill_ms": 20, "decode_ms": 8}}
        payloads = [
            {"choices": [{"index": 0, "delta": {"content": "é"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
        data = b"".join(b"data: " + json.dumps(item, ensure_ascii=False).encode()
                        + b"\n\n" for item in payloads) + b"data: [DONE]\n\n"
        split = data.index("é".encode()) + 1
        measured, fingerprint = summarize([(1, data[:split]), (7, data[split:])], True, True)
        self.assertEqual(measured["client_ttft_ms"], 7)
        self.assertEqual(measured["decode_ms_per_token"], 4)
        progress = b'data: {"choices":[{"index":0,"delta":{},"finish_reason":null}],"prompt_progress":{"total":10,"cache":0,"processed":5,"time_ms":1}}\n\n'
        with_progress, same_output = summarize(
            [(.5, progress), (1, data[:split]), (7, data[split:])], True, True)
        self.assertEqual(with_progress, measured)
        self.assertEqual(same_output, fingerprint)
        buffered = json.dumps({"choices": [{"index": 0, "message": {"content": "é"},
                                           "finish_reason": "stop"}], "usage": usage}).encode()
        self.assertEqual(summarize([(9, buffered)], False, True)[1], fingerprint)
        self.assertIsNone(summarize([(1, data[:split])], True, False)[1])
        with self.assertRaises(ValueError):
            summarize([(1, b'data: {"broken"\n\n')], True, True)

    def test_comparison_catches_slow_steps_quality_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a", root / "b"
            a.mkdir()
            b.mkdir()
            record = Recorder(a / "tools.requests.json")
            step = record.begin("/v1/chat/completions", {"seed": 3})
            step.row.update(http_status=200)
            step.feed(json.dumps({"choices": [{"index": 0, "message": {
                                                    "role": "assistant", "content": "safe"},
                                                "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                            "total_tokens": 2, "cached_tokens": 0,
                                            "gufo": {"prefill_tokens": 1, "prefill_ms": 1,
                                                     "decode_ms": 100, "queue_ms": 0,
                                                     "ttft_ms": 2, "cache_restore_ms": 0,
                                                     "cache_snapshot_ms": 0,
                                                     "cache_disk_enqueue_ms": 0}}}).encode())
            step.ended = True
            step.finish()
            original = (a / "tools.requests.json").read_text()
            (b / "tools.requests.json").write_text(original)
            self.assertEqual(compare(a, b)["status"], "passed")
            payload = json.loads(original)
            payload["requests"][0]["metrics"]["decode_ms"] = 125
            payload["requests"][0]["metrics"]["decode_ms_per_token"] = 125
            # An equal improvement elsewhere must never cancel this regression.
            baseline_payload = json.loads(original)
            second = json.loads(json.dumps(baseline_payload["requests"][0]))
            second.update(index=1, request_sha256="second-request")
            baseline_payload["requests"].append(second)
            (a / "tools.requests.json").write_text(json.dumps(baseline_payload))
            faster = json.loads(json.dumps(second))
            faster["metrics"]["decode_ms"] = 75
            faster["metrics"]["decode_ms_per_token"] = 75
            payload["requests"].append(faster)
            (b / "tools.requests.json").write_text(json.dumps(payload))
            result = compare(a, b)
            self.assertEqual(result["status"], "inconclusive")
            self.assertTrue(any(row["metric"] == "decode_ms" and row["exceeds_margin"]
                                for row in result["measurements"]))
            self.assertFalse(result["quality_or_coverage_changes"])
            (a / "tools.requests.json").write_text(original)
            payload = json.loads(original)
            payload["requests"][0]["output_sha256"] = "changed"
            (b / "tools.requests.json").write_text(json.dumps(payload))
            self.assertTrue(compare(a, b)["quality_or_coverage_changes"])
            payload["requests"][0]["metrics"]["decode_ms"] = float("nan")
            (b / "tools.requests.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                compare(a, b)
            (b / "tools.requests.json").unlink()
            self.assertEqual(compare(a, b)["status"], "failed")
        self.assertNotEqual(canonical({"arguments": '{"id":"a"}'}),
                            canonical({"arguments": '{"id":"b"}'}))

    def test_timing_evidence_does_not_hide_outliers_or_average_requests(self):
        def pair(old, new):
            result = {"status": "passed", "measurements": [],
                      "quality_or_coverage_changes": [], "slowdown_tolerance": .05, "noise_ms": 3}
            functional.compare_lifecycle(result, {"startup_ms": old}, {"startup_ms": new})
            return result

        self.assertEqual(qualify([pair(100, 105)])["status"], "passed")
        self.assertEqual(qualify([pair(1, 4)])["status"], "passed")
        self.assertEqual(qualify([pair(100, 106)])["status"], "inconclusive")
        self.assertEqual(qualify([pair(100, 120), pair(101, 119)])["status"], "failed")
        noisy = qualify([pair(100, 700), pair(700, 100)])
        self.assertEqual(noisy["status"], "inconclusive")
        self.assertEqual(len(noisy["measurements"][0]["samples"]), 2)
        # A lucky repetition cannot erase the original slower observation.
        self.assertEqual(qualify([pair(100, 700), pair(100, 99)])["status"], "inconclusive")
        # Main itself reproduced the stall: neither failure nor qualification.
        control = qualify([pair(100, 120), pair(101, 119)], [pair(100, 800)])
        self.assertEqual(control["status"], "inconclusive")
        self.assertEqual(control["measurements"][0]["control_flags"], 1)
        bad = pair(100, 100)
        bad["quality_or_coverage_changes"] = ["unexpected full prefill"]
        self.assertEqual(qualify([bad, pair(100, 90)])["status"], "failed")
        changed_margin = pair(100, 100)
        changed_margin["noise_ms"] = 50
        with self.assertRaisesRegex(ValueError, "margins"):
            qualify([changed_margin])
        faster_peer = pair(100, 50)["measurements"][0]
        faster_peer.update(request_key=["other-request"], case="other")
        slow = pair(100, 120)
        slow["measurements"].append(faster_peer)
        repeated = pair(100, 120)
        repeated["measurements"].append(faster_peer)
        result = qualify([slow, repeated])
        self.assertEqual([row["status"] for row in result["measurements"]], ["failed", "passed"])

    def test_comparison_preserves_history_but_allows_parallel_submission_order(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.mkdir()
            b.mkdir()
            rows = [{"index": index, "case": case, "request_sha256": str(index),
                     "status": "complete", "wall_ms": 1, "metrics": {}}
                    for index, case in enumerate(("warmup", "batch", "batch", "retry"))]

            def write(path, items):
                (path / "batch.requests.json").write_text(json.dumps(
                    {"version": 1, "requests": items}))

            write(a, rows)
            write(b, [rows[0], rows[2], rows[1], rows[3]])
            self.assertEqual(compare(a, b)["status"], "passed")
            write(b, [rows[3], *rows[:3]])
            self.assertEqual(compare(a, b)["status"], "failed")
            write(a, [rows[0], rows[3]])
            write(b, [rows[0], rows[3]])
            followup = compare(a, b)
            write(a, rows)
            write(b, rows)
            with self.assertRaisesRegex(ValueError, "history"):
                qualify([compare(a, b), followup])
            for path in (a, b):
                (path / "report.json").write_text(json.dumps(
                    {"suites": {"warmup": {}, "batch": {}}}))
                (path / "warmup.requests.json").write_text(json.dumps(
                    {"version": 1, "requests": rows[:2]}))
            original = compare(a, b)
            for path in (a, b):
                (path / "warmup.requests.json").write_text(json.dumps(
                    {"version": 1, "requests": rows[:1]}))
            with self.assertRaisesRegex(ValueError, "history"):
                qualify([original, compare(a, b)])

    def test_identical_concurrent_requests_match_cache_work_without_hiding_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.mkdir()
            b.mkdir()
            rows = [{"index": index, "case": "cohort", "request_sha256": "same",
                     "status": "complete", "wall_ms": 100,
                     "metrics": {"prefill_tokens": 100 if index == 0 else 0,
                                 "cached_tokens": 0 if index == 0 else 100,
                                 "decode_ms": 100}} for index in range(3)]

            def write(path, items):
                (path / "batch.requests.json").write_text(json.dumps(
                    {"version": 1, "requests": items}))

            write(a, rows)
            swapped = json.loads(json.dumps(rows))
            swapped[0]["metrics"], swapped[2]["metrics"] = (
                swapped[2]["metrics"], swapped[0]["metrics"])
            write(b, swapped)
            self.assertEqual(compare(a, b)["status"], "passed")
            swapped[1]["metrics"]["decode_ms"] = 125
            write(b, swapped)
            self.assertEqual(compare(a, b)["status"], "inconclusive")
            swapped[1]["metrics"]["prefill_tokens"] = 100
            swapped[1]["metrics"]["cached_tokens"] = 0
            write(b, swapped)
            self.assertTrue(compare(a, b)["quality_or_coverage_changes"])

    def test_evidence_requires_independent_runs_and_identical_builds(self):
        from compare import evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def record(name, binary, elapsed):
                path = root / name
                path.mkdir()
                (path / "report.json").write_text(json.dumps({
                    "status": "passed", "binary": binary, "startup_ms": elapsed,
                    "suites": {"tools": {"status": "passed"}},
                    "comparison_command": ["serve", "llm", "--sessions", "4"]}))
                (path / "tools.requests.json").write_text(json.dumps({
                    "version": 1, "requests": [{"index": 0, "case": "literal",
                        "request_sha256": "prompt", "status": "complete", "wall_ms": elapsed}]}))
                return path

            a, b = record("a", "main", 100), record("b", "pr", 125)
            c, d = record("c", "main", 101), record("d", "pr", 124)
            result = evidence([(a, b), (c, d)])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["evidence"]), 2)
            with self.assertRaisesRegex(ValueError, "reused"):
                evidence([(a, b), (a, d)])
            with self.assertRaisesRegex(ValueError, "itself"):
                evidence([(a, a)])
            wrong = record("wrong", "different-build", 125)
            with self.assertRaisesRegex(ValueError, "builds"):
                evidence([(a, b), (c, wrong)])
            stalled = record("stalled", "main", 700)
            self.assertEqual(evidence([(a, b), (c, d)], [(a, stalled)])["status"], "inconclusive")

    def test_focused_replay_finishes_case_before_stopping_the_next_request(self):
        recorder = Recorder(None, "retry")
        for case in ("warmup", "retry"):
            request = recorder.begin("/v1/models", {})
            request.row["http_status"] = 200
            request.feed(b'{"data":[],"object":"list"}')
            request.ended = True
            request.finish()
            recorder.mark(case)
            # Marking a case must not interrupt the assertions that follow it.
            self.assertEqual(request.row["case"], case)
        with self.assertRaises(CaseComplete):
            recorder.begin("/v1/models", {})
        self.assertEqual(len(recorder.rows), 2)

    def test_explicit_zero_and_neutral_overrides_are_not_dropped(self):
        self.assertEqual(functional.sampling_overrides([
            "--temperature", "0", "--top-k=0", "--top-p", "1",
            "--presence-penalty", "0", "--seed", "123",
        ]), {"temperature": 0, "top_k": 0, "top_p": 1,
             "presence_penalty": 0, "seed": 123})
        with self.assertRaises(ValueError):
            functional.sampling_overrides(["--temperature", "0", "--temperature=1"])

    def test_runner_requires_explicit_suites_before_starting_server(self):
        args = ["run.py", "--record-baseline", "--output", "/unused",
                "--sampling-preset", "qwen38", "--", "gufo", "serve", "llm",
                "--model", "fixture.gguf"]
        with patch.object(sys, "argv", args), patch.object(functional, "server") as start, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            functional.main()
        self.assertEqual(error.exception.code, 2)
        start.assert_not_called()

    def test_runner_inconclusive_is_nonzero_and_correctness_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run(name, elapsed, output_hash="same", baseline=None):
                child = f'''import json,sys
from pathlib import Path
p = Path(sys.argv[sys.argv.index("--output") + 1])
p.write_text(json.dumps({{"status": "passed"}}))
p.with_suffix(".requests.json").write_text(json.dumps({{
    "version": 1, "requests": [{{"index": 0, "case": "reply", "status": "complete",
    "endpoint": "/v1/chat/completions", "wall_ms": {elapsed},
    "request_sha256": "same", "output_sha256": "{output_hash}"}}]}}))
'''
                (root / "openai_sdk.py").write_text(child)
                args = ["run.py", "--output", str(root / name), "--sampling-preset", "qwen38",
                        "--suite", "responses"]
                args += ["--baseline", str(baseline)] if baseline else ["--record-baseline"]
                args += ["--", sys.executable, "serve", "llm", "--model", "fixture.gguf"]
                with patch.object(sys, "argv", args), patch.object(functional, "TESTS", root), \
                     patch.object(functional, "server", return_value=contextlib.nullcontext()), \
                     patch.object(functional, "provenance", return_value={}), \
                     patch.object(functional, "join_server_timings"), \
                     patch.object(functional, "execution_coverage", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    code = functional.main()
                return code, json.loads((root / name / "report.json").read_text())

            self.assertEqual(run("main", 100)[0], 0)
            code, report = run("pr", 150, baseline=root / "main")
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "inconclusive")
            self.assertEqual(report["functional_status"], "passed")
            code, report = run("wrong", 150, "different", root / "main")
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")

    def test_failed_or_incomplete_child_is_not_reported_as_a_pass(self):
        for child, expected in (
            ('import sys; sys.exit(7)', "failed"),
            ('pass', "failed"),
            ('import time; time.sleep(10)', "failed"),
            ('import json,sys; from pathlib import Path; '
             'Path(sys.argv[sys.argv.index("--output")+1]).write_text('
             'json.dumps({"status":"failed"}))', "failed"),
            ('import sys; from pathlib import Path; '
             'Path(sys.argv[sys.argv.index("--output")+1]).write_text("[]")', "failed"),
            ('import json,sys; from pathlib import Path; '
             'Path(sys.argv[sys.argv.index("--output")+1]).write_text('
             'json.dumps({"status":"passed"}))', "failed"),
            ('import json,sys; from pathlib import Path; '
             'p=Path(sys.argv[sys.argv.index("--output")+1]); '
             'p.write_text(json.dumps({"status":"passed"})); '
             'p.with_suffix(".requests.json").write_text(json.dumps({"version":1,'
             '"requests":[{"status":"complete","wall_ms":1,"endpoint":"/v1/models",'
             '"request_sha256":"fixture"}]}))', "passed"),
        ):
            with self.subTest(child=child), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "openai_sdk.py").write_text(child)
                args = ["run.py", "--record-baseline", "--output", str(root / "report"),
                        "--sampling-preset", "qwen38", "--suite", "responses",
                        "--suite-timeout", ".2", "--",
                        sys.executable, "serve", "llm", "--model", "fixture.gguf"]
                with patch.object(sys, "argv", args), patch.object(functional, "TESTS", root), \
                     patch.object(functional, "server", return_value=contextlib.nullcontext()), \
                     patch.object(functional, "provenance", return_value={}), \
                     patch.object(functional, "join_server_timings"), \
                     patch.object(functional, "execution_coverage", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    status = functional.main()
                report = json.loads((root / "report/report.json").read_text())
                self.assertEqual(report["status"], expected)
                self.assertEqual(status, 0 if expected == "passed" else 1)

    def test_disk_cache_is_removed_after_restart_checks(self):
        for exit_code, continuation_report, spacing_exact, expected in (
            (0, {"status": "passed", "exact": True}, True, "passed"),
            (7, {"status": "passed", "exact": True}, True, "failed"),
            (0, {"status": "passed", "exact": False}, True, "passed"),
            (0, {"exact": True}, True, "failed"),
            (0, {"status": "failed", "exact": True}, True, "failed"),
            (0, {"status": "passed", "exact": True}, False, "failed"),
        ):
            with self.subTest(exit_code=exit_code, continuation=continuation_report,
                              spacing_exact=spacing_exact), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                # Stands in for the server filling its runner-owned disk cache.
                child = f'''
import json, sys
from pathlib import Path
output = Path(sys.argv[sys.argv.index("--output") + 1])
(output.parent / "disk").mkdir(exist_ok=True)
(output.parent / "disk" / "entry.kvc").write_bytes(b"x")
payload = ({continuation_report!r} if Path(__file__).name == "continuation.py"
           else {{"exact": {spacing_exact!r}}})
output.write_text(json.dumps([payload]))
output.with_suffix(".requests.json").write_text(json.dumps({{"version": 1, "requests": [{{
    "status": "complete", "wall_ms": 1, "endpoint": "/v1/chat/completions",
    "request_sha256": "fixture"}}]}}))
sys.exit({exit_code})
'''
                for script in ("continuation.py", "cache_disk_spacing.py"):
                    (root / script).write_text(child)

                def server(command, log, timeout):
                    log.touch()
                    return contextlib.nullcontext()

                args = ["run.py", "--record-baseline", "--output", str(root / "report"),
                        "--sampling-preset", "qwen38", "--suite", "cache", "--",
                        sys.executable, "serve", "llm", "--model", "fixture.gguf"]
                with patch.object(sys, "argv", args), patch.object(functional, "TESTS", root), \
                     patch.object(functional, "server", side_effect=server), \
                     patch.object(sys.modules["cache_disk_spacing"], "check_disk_spacing",
                                  return_value={}), \
                     patch.object(functional, "provenance", return_value={}), \
                     patch.object(functional, "join_server_timings"), \
                     patch.object(functional, "execution_coverage", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    functional.main()
                report = json.loads((root / "report/report.json").read_text())
                self.assertEqual(report["status"], expected, report)
                continuation_passed = not exit_code and continuation_report.get("status") == "passed"
                self.assertEqual(report["suites"]["text-cancel"]["status"],
                                 "passed" if continuation_passed else "failed")
                self.assertEqual(report["suites"]["disk-spacing"]["status"],
                                 "passed" if not exit_code and spacing_exact else "failed")
                self.assertIn("--cache-disk", report["command"])
                self.assertFalse((root / "report/disk").exists())
                self.assertTrue((root / "report/text-cancel.json").is_file())

    def test_server_is_reaped_on_startup_timeout_and_test_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as reserve:
                reserve.bind(("127.0.0.1", 0))
                port = reserve.getsockname()[1]
            command = [sys.executable, "-c",
                       "import time; time.sleep(20)", "--port", str(port)]
            with self.assertRaises(TimeoutError):
                with functional.server(command, Path(directory) / "timeout.log", .1):
                    self.fail("unready server accepted")
            code = ("from http.server import BaseHTTPRequestHandler,HTTPServer\n"
                    "class Handler(BaseHTTPRequestHandler):\n"
                    " def do_GET(self):\n"
                    "  self.send_response(200); self.end_headers()\n"
                    f"HTTPServer(('127.0.0.1',{port}),Handler).serve_forever()\n")
            command = [sys.executable, "-c", code, "--port", str(port)]
            process = None
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with functional.server(command, Path(directory) / "ready.log", 5) as process:
                    raise RuntimeError("synthetic failure")
            self.assertIsNotNone(process.returncode)


class PiWatchdogTest(unittest.TestCase):
    def test_repeated_command_can_make_progress(self):
        from pi_agent import repeated_actions

        def history(outputs):
            result = []
            for index, output in enumerate(outputs):
                result += [
                    {"role": "assistant", "content": [{
                        "type": "toolCall", "id": str(index), "name": "bash",
                        "arguments": {"command": "node test.js"}}]},
                    {"role": "toolResult", "toolCallId": str(index),
                     "content": [{"type": "text", "text": output}]}]
            return result

        self.assertEqual(repeated_actions(history(["same error"] * 6)), 6)
        self.assertEqual(repeated_actions(history([
            "missing module", "case 1 fails", "case 2 fails", "ok"])), 1)


if __name__ == "__main__":
    unittest.main()
