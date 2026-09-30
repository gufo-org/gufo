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
from metrics import Recorder, canonical, compare, join_server_timings, summarize


class FunctionalRunnerTest(unittest.TestCase):
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
                                     {"startup_ms": 125, "restart_ms": 75}, .05, 3)
        self.assertEqual(result["status"], "failed")
        for candidate in ({}, {"startup_ms": 100}, {"startup_ms": float("nan")},
                          {"startup_ms": 100, "restart_ms": 0}):
            with self.assertRaisesRegex(ValueError, "server"):
                functional.compare_lifecycle(result, baseline, candidate, .05, 3)
        with self.assertRaisesRegex(ValueError, "restart_ms"):
            functional.compare_lifecycle(result,
                {"startup_ms": 100, "suites": {"text-cancel-disk": {}}},
                {"startup_ms": 100, "suites": {"text-cancel-disk": {}}}, .05, 3)

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
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any(row["metric"] == "decode_ms" and row["regression"]
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

    def test_explicit_zero_and_neutral_overrides_are_not_dropped(self):
        self.assertEqual(functional.sampling_overrides([
            "--temperature", "0", "--top-k=0", "--top-p", "1",
            "--presence-penalty", "0", "--seed", "123",
        ]), {"temperature": 0, "top_k": 0, "top_p": 1,
             "presence_penalty": 0, "seed": 123})
        with self.assertRaises(ValueError):
            functional.sampling_overrides(["--temperature", "0", "--temperature=1"])

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
                     patch.object(functional, "join_server_timings"), \
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
