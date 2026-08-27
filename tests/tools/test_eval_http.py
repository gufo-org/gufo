#!/usr/bin/env python3
"""HTTP fixture tests for the OpenAI-compatible gufo eval client."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import tempfile
import threading
from typing import Any
import unittest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gufo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


ARGS, UNITTEST_ARGS = parse_args(), ["test_eval_http.py"]


def user_prompt(case: dict[str, Any]) -> str:
    prompt = case["question"] + "\n"
    if case["kind"] == "mcq":
        prompt += "\nChoices:\n"
        for index, choice in enumerate(case["choices"]):
            prompt += f"{chr(ord('A') + index)}. {choice}\n"
        prompt += (
            "\nSolve the question. At the end, write exactly one final line "
            "in this format and do not write anything after it:\n"
            "Answer: <letter>"
        )
    elif case["kind"] == "linespec":
        prompt += (
            "\nAt the end, write exactly one final line in this format and do "
            "not write anything after it:\n"
            "Answer: <line number or comma-separated line numbers>"
        )
    else:
        prompt += (
            "\nSolve the problem. At the end, write exactly one final line in "
            "this format and do not write anything after it:\n"
            "Answer: <integer>"
        )
    return prompt


class FixtureState:
    def __init__(
        self,
        responses: list[dict[str, Any]],
        model_count: int = 1,
        api_key: str = "",
    ) -> None:
        self.responses = responses
        self.model_count = model_count
        self.api_key = api_key
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def next_response(self) -> dict[str, Any]:
        with self.lock:
            if not self.responses:
                raise AssertionError("unexpected extra completion request")
            return self.responses.pop(0)


class FixtureHandler(BaseHTTPRequestHandler):
    server: "FixtureServer"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def authorize(self) -> bool:
        expected = self.server.state.api_key
        if not expected:
            return True
        return self.headers.get("Authorization") == f"Bearer {expected}"

    def do_GET(self) -> None:
        if not self.authorize():
            self.send_json(401, {"error": {"code": "unauthorized"}})
            return
        if self.path != "/v1/models":
            self.send_json(404, {"error": {"code": "not_found"}})
            return
        models = [
            {
                "id": f"fixture-model-{index}",
                "object": "model",
                "owned_by": "fixture",
                "path": "/home/private/model.gguf",
                "gufo": {
                    "artifact_id": "sha256:fixture",
                    "repository_revision": "fixture-revision",
                },
            }
            for index in range(self.server.state.model_count)
        ]
        self.send_json(200, {"object": "list", "data": models})

    def do_POST(self) -> None:
        if not self.authorize():
            self.send_json(401, {"error": {"code": "unauthorized"}})
            return
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"code": "not_found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        self.server.state.requests.append(request)
        response = self.server.state.next_response()
        mode = response.pop("mode", "json")
        if mode == "disconnect":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.close_connection = True
            return
        self.send_json(response.pop("status", 200), response)


class FixtureServer(ThreadingHTTPServer):
    def __init__(self, state: FixtureState) -> None:
        super().__init__(("127.0.0.1", 0), FixtureHandler)
        self.state = state


@contextmanager
def fixture_server(state: FixtureState) -> Iterator[str]:
    server = FixtureServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def completion(
    answer: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 3,
    reasoning: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": answer}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "total_tokens": 100 + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 12},
            "gufo": {"queue_ms": 1.5, "execution_plan": "must-not-copy"},
        },
        "timings": {"prompt_ms": 2.5, "predicted_ms": 3.5},
        "metrics": {"queue_time_ms": 1.5, "private_path": "/home/private"},
    }


class EvalHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(ARGS.data.read_text(encoding="utf-8"))

    def run_eval(
        self,
        base_url: str,
        questions: int,
        *,
        greedy: bool = False,
        environment: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "result.json"
        command = [
            str(ARGS.gufo),
            "eval",
            "--base-url",
            base_url,
            "--questions",
            str(questions),
            "--output",
            str(output),
        ]
        if greedy:
            command.append("--greedy")
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            check=False,
        )
        report = (
            json.loads(output.read_text(encoding="utf-8"))
            if output.exists()
            else None
        )
        return process, report, output

    def test_default_requests_and_grading(self) -> None:
        state = FixtureState(
            [
                completion("Answer: B", reasoning="careful reasoning"),
                completion("Answer: C"),
                completion("Answer: 70"),
                completion("Answer: C"),
            ]
        )
        with fixture_server(state) as base_url:
            process, report, _ = self.run_eval(base_url, 4)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["summary"]["passed"], 4)
        self.assertEqual(report["summary"]["execution_errors"], 0)
        self.assertEqual(report["summary"]["completion_tokens"], 12)
        self.assertEqual(
            report["run"]["command"],
            "gufo eval --base-url <endpoint> --questions 4 --output <output>",
        )
        self.assertEqual(len(state.requests), 4)
        for index, request in enumerate(state.requests):
            self.assertNotIn("temperature", request)
            self.assertEqual(request["max_completion_tokens"], 16000)
            self.assertEqual(request["model"], "fixture-model-0")
            self.assertFalse(request["stream"])
            self.assertEqual(
                request["messages"],
                [
                    {
                        "role": "system",
                        "content": self.document["system_prompt"],
                    },
                    {
                        "role": "user",
                        "content": user_prompt(self.document["cases"][index]),
                    },
                ],
            )
        self.assertEqual(
            report["cases"][0]["response"]["reasoning_content"],
            "careful reasoning",
        )
        self.assertNotIn("execution_plan", report["cases"][0]["usage"]["gufo"])
        self.assertNotIn("private_path", report["cases"][0]["metrics"])
        serialized = json.dumps(report)
        self.assertNotIn(base_url, serialized)
        self.assertNotIn("127.0.0.1", serialized)

    def test_greedy_sends_exact_temperature_zero(self) -> None:
        state = FixtureState([completion("Answer: B")])
        with fixture_server(state) as base_url:
            process, report, _ = self.run_eval(base_url, 1, greedy=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIsNotNone(report)
        self.assertEqual(state.requests[0]["temperature"], 0)
        assert report is not None
        self.assertEqual(report["run"]["temperature"], 0)
        self.assertEqual(report["run"]["temperature_policy"], "greedy")

    def test_failure_classes_are_distinct(self) -> None:
        state = FixtureState(
            [
                completion("Answer: B", finish_reason="length"),
                {
                    "status": 400,
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "maximum context length exceeded",
                    },
                },
                {"choices": []},
                {"mode": "disconnect"},
            ]
        )
        with fixture_server(state) as base_url:
            process, report, _ = self.run_eval(base_url, 4)
        self.assertEqual(process.returncode, 1)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(
            [case["execution"]["status"] for case in report["cases"]],
            [
                "length",
                "context_rejected",
                "malformed_response",
                "transport_error",
            ],
        )
        self.assertEqual(report["summary"]["length_finishes"], 1)
        self.assertEqual(report["summary"]["execution_errors"], 3)
        self.assertEqual(report["cases"][0]["grade"]["verdict"], "passed")
        self.assertEqual(report["cases"][1]["grade"]["verdict"], "not_graded")

    def test_model_discovery_requires_exactly_one_model(self) -> None:
        for count in (0, 2):
            with self.subTest(count=count):
                state = FixtureState([], model_count=count)
                with fixture_server(state) as base_url:
                    process, report, _ = self.run_eval(base_url, 1)
                self.assertEqual(process.returncode, 1)
                self.assertIsNone(report)
                self.assertIn(
                    f"expected exactly one model from GET /v1/models, found {count}",
                    process.stderr,
                )

    def test_artifact_redacts_secrets_paths_and_private_addresses(self) -> None:
        secret = "fixture-super-secret"
        state = FixtureState(
            [
                completion(
                    "Answer: B\n"
                    f"credential={secret} path=/home/alice/model.gguf "
                    "endpoint=http://192.168.1.50:8080/v1"
                )
            ],
            api_key=secret,
        )
        environment = dict(os.environ)
        environment["OPENAI_API_KEY"] = secret
        environment["HOME"] = "/home/alice"
        environment["GUFO_EVAL_SERVER_CONFIG"] = (
            "gufo serve --host 192.168.1.50 "
            "--model /home/alice/model.gguf"
        )
        with fixture_server(state) as base_url:
            process, report, output = self.run_eval(
                base_url, 1, environment=environment
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIsNotNone(report)
        serialized = output.read_text(encoding="utf-8")
        self.assertNotIn(secret, serialized)
        self.assertNotIn("/home/alice", serialized)
        self.assertNotIn("192.168.1.50", serialized)
        self.assertNotIn(base_url, serialized)
        assert report is not None
        self.assertEqual(
            report["run"]["server"]["configuration"],
            "gufo serve --host <private-address> --model <home>/model.gguf",
        )
        self.assertGreater(report["cases"][0]["response"]["redactions"], 0)


if __name__ == "__main__":
    unittest.main(argv=UNITTEST_ARGS)
