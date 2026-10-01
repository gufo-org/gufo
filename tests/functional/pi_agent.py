#!/usr/bin/env python3
"""Real Pi agent regression for #368; executes generated code in disposable fixtures.

Requires Pi 0.87.0 and Node, and an already-running local Gufo server. Keeps Pi
sessions, exact HTTP bodies/SSE, task files and individual request/task timings.
"""

import argparse
import collections
import hashlib
import http.client
import http.server
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.parse

from metrics import summarize


PROMPTS = {
    "simple": "Réponds en un seul mot : quelle est la capitale de la France ?",
    "tools": "Crée un fichier hello.txt contenant exactement le mot bonjour, affiche-le avec cat, puis relis-le avec l'outil read et confirme son contenu en une phrase.",
    "edit": "Lis src/calc.py, corrige le bug évident avec l'outil edit, puis affiche le fichier corrigé avec cat.",
    "creation": "Crée un module CommonJS stats.js qui exporte mean(tableau) et median(tableau) (médiane correcte pour un nombre pair d'éléments), puis test.js qui les vérifie avec node:assert sur quatre cas, exécute node test.js jusqu'à ce qu'il passe.",
    "bugfix": "Lance node slugify.test.js : il échoue. Corrige slugify.js (accents retirés, tout caractère non alphanumérique devient un tiret, tirets fusionnés et retirés aux extrémités) sans modifier slugify.test.js, et relance jusqu'à ce que ça passe.",
}
SLUG_TEST = """const assert = require("node:assert");
const slugify = require("./slugify");
assert.strictEqual(slugify("Hello World"), "hello-world");
assert.strictEqual(slugify("  Déjà   vu ! "), "deja-vu");
assert.strictEqual(slugify("--a--b--"), "a-b");
console.log("ok");
"""


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Recorder(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, upstream, output):
        super().__init__(("127.0.0.1", 0), Proxy)
        self.upstream = urllib.parse.urlsplit(upstream)
        self.output = output
        self.requests = []
        self.lock = threading.Lock()
        self.case = "startup"


class Proxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path != "/v1/models":
            self.send_error(404, "Only /v1/models is available for discovery")
            return
        conn = http.client.HTTPConnection(
            self.server.upstream.hostname, self.server.upstream.port, timeout=30)
        try:
            conn.request("GET", self.path)
            response = conn.getresponse()
            body = response.read()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type") or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (OSError, http.client.HTTPException):
            self.send_error(502, "Model discovery failed upstream")
        finally:
            conn.close()

    def do_POST(self):
        server = self.server
        with server.lock:
            index = len(server.requests)
            row = {"index": index, "case": server.case, "path": self.path}
            server.requests.append(row)
        stem = server.output / f"request-{index:04d}"
        body = self.rfile.read(int(self.headers["Content-Length"]))
        stem.with_suffix(".json").write_bytes(body)
        conn = http.client.HTTPConnection(
            server.upstream.hostname, server.upstream.port, timeout=180
        )
        start = time.monotonic()
        try:
            conn.request("POST", self.path, body, {"Content-Type": "application/json"})
            response = conn.getresponse()
            row["status"] = response.status
            row["request_id"] = response.getheader("X-Request-ID")
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type") or "application/octet-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = []
            parts = []
            with stem.with_suffix(".sse").open("wb") as raw:
                while chunk := response.read1(65536):
                    chunks.append(chunk)
                    parts.append(((time.monotonic() - start) * 1000, chunk))
                    raw.write(chunk)
                    raw.flush()
                    self.wfile.write(chunk)
                    self.wfile.flush()
            text = b"".join(chunks).decode("utf-8", errors="strict")
            for line in text.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line[6:])
                    if event.get("usage"):
                        row["usage"] = event["usage"]
                    if event.get("response", {}).get("usage"):
                        row["usage"] = event["response"]["usage"]
            row["metrics"], row["output_sha256"] = summarize(
                parts, json.loads(body).get("stream", False), True,
                (self.path, json.loads(body), response.status))
            if response.status != 200:
                row["error"] = text
        except Exception as error:
            row["error"] = repr(error)
        finally:
            row["wall_ms"] = (time.monotonic() - start) * 1000
            conn.close()
            self.close_connection = True
            save(stem.with_suffix(".timing.json"), row)


def messages(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a live writer may not have completed the final line
            if record.get("type") == "message":
                result.append(record["message"])
    return result


def calls_from(history):
    return [
        part
        for message in history
        if message.get("role") == "assistant"
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "toolCall"
    ]


def repeated_actions(history):
    """Count identical calls/results, allowing tests to be retried after progress."""
    pending, counts = {}, collections.Counter()
    for message in history:
        if message.get("role") == "assistant":
            for call in calls_from([message]):
                pending[call["id"]] = call
        elif message.get("role") == "toolResult":
            call = pending.get(message.get("toolCallId"))
            if call:
                counts[(call["name"], json.dumps(call["arguments"], sort_keys=True),
                        json.dumps(message.get("content"), sort_keys=True))] += 1
    return max(counts.values(), default=0)


def validate_requests(rows, server_log=None):
    assert rows, "Pi did not send a request"
    completed = {}
    if server_log:
        for line in server_log.read_text().splitlines():
            if "event=completed " in line:
                fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
                completed[fields.get("request")] = fields
    for row in rows:
        assert row.get("status") == 200 and not row.get("error"), row
        usage = row["usage"]
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        generated = usage.get("completion_tokens", usage.get("output_tokens"))
        assert prompt > 0 and generated > 0, row
        assert usage["total_tokens"] == prompt + generated, row
        metrics = row.get("metrics", usage.get("gufo", {}))
        if server_log:
            fields = completed[row["request_id"]]
            for field in ("queue_ms", "ttft_ms", "duration_ms"):
                metrics[field] = float(fields[field])
        for field in ("queue_ms", "prefill_ms", "decode_ms", "ttft_ms",
                      "cache_restore_ms", "cache_snapshot_ms", "cache_disk_enqueue_ms"):
            value = metrics[field]
            assert isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, row
        assert math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0, row
        if row is not rows[0]:
            # Every subsequent request in this task appends a tool result.
            # A full re-prefill is a deterministic failure, independent of noise.
            cached = usage.get("cached_tokens", usage.get("input_tokens_details", {}).get("cached_tokens", 0))
            assert cached > 0 and metrics["prefill_tokens"] < prompt, row


def validate_task(name, cwd, history):
    calls = calls_from(history)
    for call in calls:
        assert call["name"] in {"read", "write", "edit", "bash"}, call
        assert isinstance(call["arguments"], dict), call
        path = call["arguments"].get("path", "")
        assert not any(tag in path for tag in ("<tool_call", "</tool_call", "</think")), call
        if call["name"] == "edit":
            edits = call["arguments"]["edits"]
            assert isinstance(edits, list) and edits, call
            assert all(isinstance(e.get(k), str) for e in edits for k in ("oldText", "newText")), call
    assistants = [m for m in history if m.get("role") == "assistant"]
    assert assistants and assistants[-1].get("stopReason") == "stop", assistants[-1:]
    text = "\n".join(
        p["text"] for m in assistants for p in m.get("content", [])
        if isinstance(p, dict) and p.get("type") == "text"
    )
    assert not any(tag in text for tag in ("</tool_call>", "</function>", "</parameter>", "</think>")), text
    names = {c["name"] for c in calls}
    if name == "simple":
        assert not calls and text.strip().rstrip(".") == "Paris", text
    elif name == "tools":
        assert (cwd / "hello.txt").read_bytes() == b"bonjour"
        assert {"write", "bash", "read"} <= names, names
    elif name == "edit":
        namespace = {}
        exec(compile((cwd / "src/calc.py").read_text(), "calc.py", "exec"), namespace)
        assert namespace["add"](2, 3) == 5 and namespace["add"](-4, 1) == -3
        assert {"read", "edit", "bash"} <= names, names
    elif name == "creation":
        assert (cwd / "stats.js").is_file() and (cwd / "test.js").is_file()
        subprocess.run(["node", "test.js"], cwd=cwd, check=True, capture_output=True, timeout=10)
        # Independent checks cannot be weakened by the generated test.js.
        subprocess.run(["node", "-e", """
const a=require('node:assert/strict'), s=require('./stats');
for (const [xs, mean, median] of [
  [[1,2,3],2,2], [[4,1,7,2],3.5,3], [[-3,-1,2,6],1,0.5], [[9],9,9]]) {
  a.equal(s.mean(xs), mean); a.equal(s.median(xs), median);
}
"""], cwd=cwd, check=True, capture_output=True, timeout=10)
    elif name == "bugfix":
        assert (cwd / "slugify.test.js").read_text() == SLUG_TEST
        subprocess.run(["node", "slugify.test.js"], cwd=cwd, check=True, capture_output=True, timeout=10)
        subprocess.run(["node", "-e", """
const a=require('node:assert/strict'), s=require('./slugify');
a.equal(s('Été à Paris!?'), 'ete-a-paris'); a.equal(s('---'), '');
a.equal(s('a__b ... c'), 'a-b-c');
"""], cwd=cwd, check=True, capture_output=True, timeout=10)


def run_case(args, recorder, env, index, name):
    case = args.output / f"{index:02d}-{name}"
    case.mkdir()
    cwd = args.output / "work" if args.conversation else case / "work"
    cwd.mkdir(parents=True, exist_ok=args.conversation)
    if name == "edit":
        (cwd / "src").mkdir(exist_ok=True)
        (cwd / "src/calc.py").write_text("def add(a, b):\n    return a - b\n")
    elif name == "bugfix":
        (cwd / "slugify.js").write_text('module.exports = function slugify(title) {\n  return title.toLowerCase().replace(/ /g, "-");\n};\n')
        (cwd / "slugify.test.js").write_text(SLUG_TEST)
    session = (args.output if args.conversation else case) / "session.jsonl"
    previous_messages = len(messages(session))
    command = [
        str(args.pi), "--offline", "--no-extensions", "--no-skills",
        "--no-prompt-templates", "--no-themes", "--no-context-files",
        "--provider", "gufo-regression", "--model", args.model,
        "--thinking", args.thinking, "--session", str(session),
        "--mode", "json", "--print", PROMPTS[name],
    ]
    if args.context_file:
        command += ["--append-system-prompt", str(args.context_file)]
    save(case / "command.json", command)
    recorder.case = case.name
    row = {"case": case.name, "start_request": len(recorder.requests)}
    start = time.monotonic()
    proc = None
    try:
        with (case / "stdout.jsonl").open("w") as out, (case / "stderr.log").open("w") as err:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
            while proc.poll() is None:
                if time.monotonic() - start > args.timeout:
                    raise AssertionError("Pi task timeout")
                history = messages(session)[previous_messages:]
                if len(calls_from(history)) > 80 or repeated_actions(history) > 5:
                    raise AssertionError("Repeated tool action / excessive tool turns")
                time.sleep(0.5)
            assert proc.returncode == 0, (case / "stderr.log").read_text()
        history = messages(session)[previous_messages:]
        validate_task(name, cwd, history)
        row["status"] = "pass"
    except Exception as error:
        row.update(status="fail", error=repr(error))
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    row["wall_ms"] = (time.monotonic() - start) * 1000
    history = messages(session)[previous_messages:]
    row["tool_calls"] = len(calls_from(history))
    row["tool_errors"] = [m for m in history if m.get("role") == "toolResult" and m.get("isError")]
    row["requests"] = recorder.requests[row["start_request"]:]
    try:
        validate_requests(row["requests"], args.server_log)
    except Exception as error:
        row.update(status="fail", error=f"HTTP/usage/cache validation: {error!r}")
    save(case / "result.json", row)
    print(f"{case.name}: {row['status']} {row['wall_ms']/1000:.1f}s, {row['tool_calls']} tool calls", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--pi", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True, help="Gufo informational log for request timing correlation")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--thinking", default="off", choices=["off", "low", "medium", "high"])
    parser.add_argument("--conversation", action="store_true", help="Retain one Pi session across tasks")
    parser.add_argument("--context-file", type=Path, help="Append background context to the system prompt")
    parser.add_argument("--api", default="openai-completions", choices=["openai-completions", "openai-responses"])
    parser.add_argument("--sampling", type=json.loads, default={}, help="Optional request overrides as JSON; otherwise server defaults")
    args = parser.parse_args()
    upstream = urllib.parse.urlsplit(args.base_url)
    assert upstream.scheme == "http" and upstream.hostname in {"127.0.0.1", "localhost", "::1"}, "Loopback Gufo only"
    assert not args.output.exists(), "Use a fresh output directory; preserve previous evidence"
    args.output.mkdir(parents=True)
    version = subprocess.check_output([str(args.pi), "--version"], text=True).strip()
    assert version == "0.87.0", f"Expected reporter's Pi 0.87.0, got {version}"
    metadata = {
        "pi_version": version,
        "node_version": subprocess.check_output(["node", "--version"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    wire = args.output / "wire"
    wire.mkdir()
    recorder = Recorder(args.base_url, wire)
    thread = threading.Thread(target=recorder.serve_forever, daemon=True)
    thread.start()
    config = args.output / "pi-config"
    config.mkdir()
    save(config / "models.json", {"providers": {"gufo-regression": {
        "baseUrl": f"http://127.0.0.1:{recorder.server_port}/v1", "api": args.api,
        "apiKey": "local-functional-test", "compat": {"supportsStrictMode": False},
        "models": [{"id": args.model, "reasoning": args.thinking != "off",
                    "input": ["text", "image"], "contextWindow": 262144,
                    "maxTokens": 8192, "samplingParams": args.sampling}],
    }}})
    save(config / "settings.json", {"packages": [], "quietStartup": True})
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "SSL_CERT_FILE"}}
    env.update(PI_CODING_AGENT_DIR=str(config), PI_OFFLINE="1", PI_TELEMETRY="0")
    rows = []
    cases = ["simple"] + list(PROMPTS) * args.passes
    try:
        for index, name in enumerate(cases):
            rows.append(run_case(args, recorder, env, index, name))
            save(args.output / "report.json", {**metadata, "cases": rows, "requests": recorder.requests})
            if rows[-1]["status"] != "pass":
                break
    finally:
        recorder.shutdown()
        recorder.server_close()
    return 0 if len(rows) == len(cases) and all(r["status"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
