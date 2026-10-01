"""Fast checks for the functional runner's failure reporting and process ownership."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
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


class FunctionalRunnerTest(unittest.TestCase):
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
                "index": 0, "request_id": "r7", "metrics": {}, "output_sha256": "output"}]}))
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


if __name__ == "__main__":
    unittest.main()
