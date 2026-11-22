#!/usr/bin/env python3
"""Regression checks for the interface benchmark's wire timing."""

import contextlib
import hashlib
import io
import json
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gufo import interface_bench

TOKEN_DELAY = 0.035
USAGE = (
    'data: {"choices":[],"usage":{"prompt_tokens":4,'
    '"completion_tokens":3,"total_tokens":7,'
    '"prompt_tokens_details":{"cached_tokens":2},"gufo":{'
    '"queue_ms":1.0,"prefill_ms":2.0,"decode_ms":3.0,'
    '"ttft_ms":10.0,"mean_inter_token_ms":4.0,'
    '"max_inter_token_ms":5.0}}}\n\n'
)


MODELS_BODY = b'{"object":"list","data":[{"id":"test-model"}]}'


def stream_events(server):
    events = ['data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n']
    if server.error:
        events.append('data: {"error":{"message":"overloaded"}}\n\n')
    for index in range(3):
        events.append(
            'data: {"choices":[{"delta":{"' + server.delta_key
            + '":"' + chr(ord("a") + index) + '"}}]}\n\n'
        )
    events += [USAGE, "data: [DONE]\n\n"]
    return [event.encode("utf-8") for event in events]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def _head_fields(head):
    lines = head.split(b"\r\n")
    path = lines[0].split(b" ")[1].decode("ascii")
    headers = {}
    for line in lines[1:]:
        name, separator, value = line.partition(b":")
        if separator:
            headers[name.decode("latin-1").strip().lower()] = (
                value.decode("latin-1").strip()
            )
    return path, headers


class FakeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *, framing="chunked", token_delay=TOKEN_DELAY,
                 force_close=False, silent_close=False, base_path="",
                 delta_key="content", error=False, one_write=False,
                 models_chunked=False):
        super().__init__(("127.0.0.1", 0), FakeHandler)
        # Stream body framing: "chunked", "length" or "close".
        self.framing = framing
        self.token_delay = token_delay
        self.force_close = force_close or framing == "close"
        # Close after each response without announcing `Connection: close`.
        self.silent_close = silent_close
        self.base_path = base_path
        self.delta_key = delta_key
        self.error = error
        # Send the head and the whole body in one write.
        self.one_write = one_write
        self.models_chunked = models_chunked
        self.accepts = 0
        self.paths = []
        self.bodies = []
        self.lock = threading.Lock()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request, client_address):
        # Clients close streams after [DONE] or an error event; the peer
        # reset that follows is expected, not a server failure.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


@contextlib.contextmanager
def running(**options):
    server = FakeServer(**options)
    # A short poll keeps shutdown() from waiting the default half second.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class FakeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        assert isinstance(self.server, FakeServer)
        with self.server.lock:
            self.server.accepts += 1
        buffer = bytearray()
        while True:
            while b"\r\n\r\n" not in buffer:
                chunk = self.request.recv(65536)
                if not chunk:
                    return
                buffer.extend(chunk)
            end = buffer.find(b"\r\n\r\n")
            head = bytes(buffer[:end])
            del buffer[:end + 4]
            path, headers = _head_fields(head)
            length = int(headers.get("content-length", "0"))
            while len(buffer) < length:
                chunk = self.request.recv(65536)
                if not chunk:
                    return
                buffer.extend(chunk)
            body = bytes(buffer[:length])
            del buffer[:length]
            with self.server.lock:
                self.server.paths.append(path)
                if body:
                    self.server.bodies.append(json.loads(body))
            close = (
                self.server.force_close
                or headers.get("connection", "").lower() == "close"
            )
            base = self.server.base_path
            if path == base + "/v1/models":
                self._models(close)
            elif path == base + "/v1/chat/completions":
                self._stream(close)
            else:
                self._error(close)
            if close or self.server.silent_close:
                return

    def _head(self, status, content_type, close, framing):
        return (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"{framing}"
            f"Connection: {'close' if close else 'keep-alive'}\r\n\r\n"
        ).encode("ascii")

    def _models(self, close):
        body = MODELS_BODY
        if self.server.models_chunked:
            half = len(body) // 2
            payload = b"".join(
                f"{len(part):x}\r\n".encode("ascii") + part + b"\r\n"
                for part in (body[:half], body[half:])
            ) + b"0\r\n\r\n"
            framing = "Transfer-Encoding: chunked\r\n"
        else:
            payload = body
            framing = f"Content-Length: {len(body)}\r\n"
        self.request.sendall(
            self._head("200 OK", "application/json", close, framing) + payload
        )

    def _error(self, close):
        body = b'{"error":"unknown route"}'
        self.request.sendall(
            self._head("404 Not Found", "application/json", close,
                       f"Content-Length: {len(body)}\r\n")
            + body
        )

    def _stream(self, close):
        framing = self.server.framing
        events = stream_events(self.server)
        if framing == "chunked":
            pieces = [
                f"{len(event):x}\r\n".encode("ascii") + event + b"\r\n"
                for event in events
            ]
            header = "Transfer-Encoding: chunked\r\n"
            tail = b"0\r\n\r\n"
        else:
            pieces = list(events)
            length = sum(len(event) for event in events)
            header = f"Content-Length: {length}\r\n" if framing == "length" else ""
            tail = b""
        head = self._head("200 OK", "text/event-stream", close, header)
        if self.server.one_write:
            self.request.sendall(head + b"".join(pieces) + tail)
            return
        self.request.sendall(head)
        token_indexes = {
            index for index, event in enumerate(events)
            if self.server.delta_key.encode("ascii") in event
        }
        first_token = min(token_indexes)
        for index, piece in enumerate(pieces):
            if index in token_indexes and index != first_token:
                threading.Event().wait(self.server.token_delay)
            self.request.sendall(piece)
        if tail:
            self.request.sendall(tail)


def stream_sample(server, base_path="", **options):
    return interface_bench.measure_stream(
        interface_bench.Endpoint.parse(server.url + base_path),
        model="test-model",
        prompt="hello",
        max_tokens=3,
        timeout=5.0,
        request_index=0,
        **options,
    )


def overhead(server, base_path="", requests=3, reuse=True):
    return interface_bench.measure_overhead(
        interface_bench.Endpoint.parse(server.url + base_path),
        path="/v1/models",
        requests=requests,
        reuse=reuse,
        timeout=5.0,
    )


def check_timed_stream(sample, label):
    check(sample.event_count == 6, f"{label} events: {sample.event_count}")
    check(len(sample.gap_ms) == 2, f"{label} gaps: {sample.gap_ms}")
    check(sample.gap_ms[-1] >= 25.0, f"{label} token delay lost: {sample.gap_ms}")
    # Measured from the first token: the last token must land inside the
    # request, which fails if TTFT is taken from a later token.
    check(
        sample.client_ttft_ms + sum(sample.gap_ms) <= sample.total_ms + 1e-6,
        f"{label} TTFT is not the first token: {sample}",
    )


# The full benchmark over a keep-alive, chunked server.
with running() as server:
    report = interface_bench.run_benchmark(
        base_url=server.url,
        model="test-model",
        mode="both",
        overhead_path="/v1/models",
        requests=3,
        concurrency_levels=[1, 2],
        max_tokens=3,
        prompt="hello",
        timeout=5.0,
    )
    serialized = json.dumps(report)
    check(report["benchmarkSchema"] == interface_bench.BENCHMARK_SCHEMA,
          "benchmark schema")
    check(report["artifactType"] == "interfaceBenchmark", "artifact type")
    check(report["model"]["id"] == "test-model", "served model not recorded")
    for leaked in ("127.0.0.1", str(server.server_address[1]), "hello",
                   '"abc"', "basePath"):
        check(leaked not in serialized, f"artifact leaks {leaked!r}")
    check(report["endpoint"] == {"tls": False, "hasBasePath": False},
          f"endpoint record: {report['endpoint']}")
    workload = report["workload"]
    check(
        workload["promptSha256"] == hashlib.sha256(b"hello").hexdigest(),
        "prompt identity",
    )
    check(workload["concurrency"] == [1, 2], "concurrency levels")
    check(workload["cachePrompt"] is None, "default cache policy")

    fresh = report["overhead"]["newConnection"]
    reused = report["overhead"]["reusedConnection"]
    check(not fresh["connectionReused"], "new-connection mode reused a socket")
    check(fresh["connect_ms"]["count"] == 3, "fresh connects not measured")
    check(reused["connectionReused"], "reuse mode did not reuse a socket")
    check(reused["reconnects"] == 0, "reuse mode reconnected on keep-alive")
    check(
        reused["connect_ms"]["count"] == 1,
        f"reused requests must not report a connect: {reused['connect_ms']}",
    )
    # Stale read times from the previous response would make this negative.
    check(reused["ttfb_ms"]["minimum"] > 0, "reused TTFB used a stale read")

    check([level["concurrency"] for level in report["streams"]] == [1, 2],
          "one stream report per level")
    for stream in report["streams"]:
        label = f"C={stream['concurrency']}"
        check(stream["requests"] == 3, f"{label} request count")
        check(stream["eventCount"]["median"] == 6,
              f"{label} SSE event count: {stream['eventCount']}")
        check(stream["gap_ms"]["maximum"] >= 25.0,
              f"{label} token delay lost: {stream['gap_ms']}")
        thresholds = stream["gapThresholds"]
        check(thresholds["over_10ms"] >= thresholds["over_30ms"] >= 1,
              f"{label} gap thresholds: {thresholds}")
        for key, value in (("server_ttft_ms", 10.0), ("queue_ms", 1.0),
                           ("prefill_ms", 2.0), ("decode_ms", 3.0),
                           ("cachedPromptTokens", 2)):
            check(stream[key]["median"] == value,
                  f"{label} {key} not read from usage: {stream[key]}")
        check(stream["client_minus_server_ttft_ms"]["count"] == 3,
              f"{label} TTFT delta missing")
    # Three fresh overhead connections, one reused, and one per stream.
    check(server.accepts == 10, f"unexpected connection count: {server.accepts}")
    sampling = server.bodies[-1]
    check(sampling["top_k"] == 0 and sampling["repeat_penalty"] == 1.0,
          f"Gufo sampling was not pinned: {sampling}")
    check("cache_prompt" not in sampling, "default must not send cache_prompt")

    sample = stream_sample(server, cache_prompt=False)
    check_timed_stream(sample, "chunked")
    check(
        sample.client_minus_server_ttft_ms == sample.client_ttft_ms - 10.0,
        f"TTFT delta sign: {sample.client_minus_server_ttft_ms}",
    )
    check(server.bodies[-1]["cache_prompt"] is False, "cache_prompt not sent")

    with running(framing="close") as reference:
        stream_sample(reference, endpoint_profile="openai")
        sampling = reference.bodies[-1]
        check("top_k" not in sampling, f"openai profile sent Gufo fields: {sampling}")

# Servers that announce `Connection: close`, or close without saying so, are
# reconnected and never reported as reused.
for options in ({"force_close": True}, {"silent_close": True}):
    with running(**options) as server:
        result = overhead(server)
        check(not result["connectionReused"], f"{options} reused: {result}")
        check(result["reconnects"] == 2, f"{options} reconnects: {result}")

# A chunked overhead response is drained, so the next reused request parses.
with running(models_chunked=True) as server:
    result = overhead(server)
    check(result["reconnects"] == 0, f"chunked overhead reconnects: {result}")
    check(result["responseBytes"]["median"] == len(MODELS_BODY),
          f"chunked overhead body: {result['responseBytes']}")

# A base URL path prefixes the overhead probe and the stream; a missing route
# fails the stream instead of producing an empty sample.
with running(base_path="/proxy", token_delay=0) as server:
    overhead(server, "/proxy", requests=1, reuse=False)
    stream_sample(server, "/proxy")
    check(
        server.paths == ["/proxy/v1/models", "/proxy/v1/chat/completions"],
        f"base path was not applied: {server.paths}",
    )
    try:
        stream_sample(server)
    except RuntimeError as exception:
        check("404" in str(exception), f"non-200 error text: {exception}")
    else:
        raise AssertionError("a 404 stream was accepted")

# Thinking models stream reasoning first; the server's TTFT counts it, and the
# role event is not a token.
with running(delta_key="reasoning_content", token_delay=0) as server:
    sample = stream_sample(server)
    check(sample.client_ttft_ms is not None, "reasoning TTFT missing")
    check(len(sample.gap_ms) == 2, f"reasoning gaps: {sample.gap_ms}")

# An error event fails the request instead of yielding a truncated sample.
with running(error=True, token_delay=0) as server:
    try:
        stream_sample(server)
    except RuntimeError as exception:
        check("overloaded" in str(exception), f"error text: {exception}")
    else:
        raise AssertionError("an SSE error event was ignored")

# Length- and close-delimited bodies are parsed as bytes arrive, so token
# times survive without chunk framing.
for framing in ("length", "close"):
    with running(framing=framing) as server:
        check_timed_stream(stream_sample(server), framing)

# The head and the whole body in one read: buffered events must not be lost
# when the server then closes, and the coalesced write is visible.
for framing in ("length", "close", "chunked"):
    with running(framing=framing, one_write=True) as server:
        sample = stream_sample(server)
        check(sample.event_count == 6,
              f"one-write {framing} events: {sample.event_count}")
        check(sample.events_per_read >= 3.0,
              f"one-write {framing} events/read: {sample.events_per_read}")
        # Chunk framing adds a few bytes per event; the HTTP head would add
        # far more.
        body_bytes = sum(len(event) + 8 for event in stream_events(server)) + 5
        check(sample.bytes_per_read <= body_bytes,
              f"one-write {framing} counted the head: {sample.bytes_per_read}")

# SSE line endings: mixed CRLF and LF separators, and a CRLF split by a read.
accumulator = interface_bench.SseAccumulator()
events = accumulator.feed(b"data: A\r\n\r\ndata: B\n\n", 0, 0.0)
check([event for event, _, _ in events] == ["A", "B"], f"mixed: {events}")
check(accumulator.feed(b"data: C\r", 1, 1.0) == [], "event ended early")
events = accumulator.feed(b"\n\r\n", 2, 2.0)
check([(event, time) for event, _, time in events] == [("C", 2.0)],
      f"split CRLF: {events}")

check(
    interface_bench._useful_event(
        '{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}'
    )[0],
    "tool-call deltas are tokens",
)
check(interface_bench.distribution([])["count"] == 0, "empty distribution")

# The CLI discovers the served model, writes the artifact and reports errors.
with running(token_delay=0) as server, tempfile.TemporaryDirectory() as root:
    output = Path(root) / "nested" / "interface.json"
    with contextlib.redirect_stdout(io.StringIO()):
        status = interface_bench.main([
            "--base-url", server.url, "--requests", "1", "--concurrency", "1",
            "--max-tokens", "3", "--prompt", "hello", "--prompt-repeat", "2",
            "--output", str(output),
        ])
    check(status == 0, f"CLI exit status: {status}")
    artifact = json.loads(output.read_text(encoding="utf-8"))
    check(artifact["model"]["id"] == "test-model", "model was not discovered")
    check(artifact["workload"]["promptRepeat"] == 2, "prompt repeat")
    check(
        artifact["workload"]["promptSha256"]
        == hashlib.sha256(b"hello\n\nhello").hexdigest(),
        "repeated prompt identity",
    )
with contextlib.redirect_stderr(io.StringIO()) as errors:
    status = interface_bench.main(["--base-url", "ftp://host", "--mode", "overhead"])
check(status == 1 and "http://" in errors.getvalue(),
      f"CLI error path: {status} {errors.getvalue()!r}")
print("interface_bench_test: ok")
