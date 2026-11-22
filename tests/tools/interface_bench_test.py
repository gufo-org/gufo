#!/usr/bin/env python3
"""Regression checks for the interface benchmark's wire timing."""

import json
import socketserver
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gufo import interface_bench


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

    def __init__(self, force_close=False, token_delay=0.035, chunked=True):
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.accepts = 0
        self.force_close = force_close
        self.token_delay = token_delay
        self.chunked = chunked
        self.lock = threading.Lock()


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
            del buffer[:length]
            close = (
                self.server.force_close
                or headers.get("connection", "").lower() == "close"
            )
            if path == "/v1/models":
                self._models(close)
            elif path == "/v1/chat/completions":
                self._stream(close)
            else:
                self._error(close)
            if close:
                return

    def _models(self, close):
        body = b'{"object":"list","data":[]}'
        connection = "close" if close else "keep-alive"
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: {connection}\r\n\r\n"
        ).encode("ascii")
        self.request.sendall(head + body)

    def _error(self, close):
        body = b'{"error":"unknown route"}'
        connection = "close" if close else "keep-alive"
        head = (
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: {connection}\r\n\r\n"
        ).encode("ascii")
        self.request.sendall(head + body)

    def _stream(self, close):
        connection = "close" if close else "keep-alive"
        chunked = self.server.chunked
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            + ("Transfer-Encoding: chunked\r\n" if chunked else "")
            + f"Connection: {connection}\r\n\r\n"
        ).encode("ascii")
        self.request.sendall(head)

        def event(payload):
            data = payload.encode("utf-8")
            if chunked:
                self.request.sendall(
                    f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n"
                )
            else:
                self.request.sendall(data)

        event('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        for index in range(3):
            if index:
                threading.Event().wait(self.server.token_delay)
            event(
                'data: {"choices":[{"delta":{"content":"'
                + chr(ord("a") + index)
                + '"}}]}\n\n'
            )
        event(
            'data: {"choices":[],"usage":{"prompt_tokens":4,'
            '"completion_tokens":3,"total_tokens":7,"gufo":{'
            '"queue_ms":1.0,"prefill_ms":2.0,"decode_ms":3.0,'
            '"ttft_ms":10.0,"mean_inter_token_ms":4.0,'
            '"max_inter_token_ms":5.0}}}\n\n'
        )
        event("data: [DONE]\n\n")
        if chunked:
            self.request.sendall(b"0\r\n\r\n")


def start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


with FakeServer() as server:
    start(server)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    report = interface_bench.run_benchmark(
        base_url=base_url,
        model="test-model",
        mode="both",
        overhead_path="/v1/models",
        requests=3,
        concurrency=2,
        max_tokens=3,
        prompt="hello",
        timeout=5.0,
    )
    assert report["schema"] == interface_bench.BENCHMARK_SCHEMA
    check("127.0.0.1" not in json.dumps(report), "artifact leaks the host")
    check(
        report["workload"]["model"] == "test-model",
        "artifact does not record the served model",
    )
    overhead = report["overhead"]
    check(
        not overhead["newConnection"]["connectionReused"],
        "new-connection mode reused a socket",
    )
    check(
        overhead["reusedConnection"]["connectionReused"],
        "reuse mode did not reuse a socket",
    )
    check(
        overhead["reusedConnection"]["reconnects"] == 0,
        "reuse mode reconnected on a keep-alive server",
    )
    check(
        overhead["reusedConnection"]["connect_ms"]["count"] == 3,
        "connect time must be measured for every request",
    )
    check(
        overhead["reusedConnection"]["ttfb_ms"]["p50"] >= 0,
        "reused TTFT used a previous response's read time",
    )
    stream = report["stream"]
    check(stream["requests"] == 3, "stream request count")
    check(
        stream["eventCount"]["p50"] == 6,
        f"unexpected SSE event count: {stream['eventCount']}",
    )
    check(
        stream["gap_ms"]["max"] >= 25.0,
        f"token delay was not measured: {stream['gap_ms']}",
    )
    check(
        stream["gapThresholds"]["over_30ms"] >= 3,
        f"gap threshold count is wrong: {stream['gapThresholds']}",
    )
    check(
        stream["server_ttft_ms"]["p50"] == 10.0,
        "server TTFT was not read from terminal usage",
    )
    check(
        stream["interfaceOverhead_ms"]["count"] == 3,
        "client/server TTFT delta was not computed",
    )
    # Reuse across requests: three overhead requests in each mode plus three
    # streams, with the reused mode needing one connection instead of three.
    check(server.accepts <= 9, f"unexpected connection count: {server.accepts}")

with FakeServer(force_close=True) as server:
    start(server)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    result = interface_bench.measure_overhead(
        interface_bench.Endpoint.parse(base_url),
        path="/v1/models",
        requests=3,
        reuse=True,
        timeout=5.0,
    )
    check(not result["connectionReused"], "close-per-response was not detected")
    check(result["reconnects"] == 2, f"reconnect count: {result['reconnects']}")

# HTTP/1.0-style streams have no length or chunk framing: the close ends them.
with FakeServer(force_close=True, chunked=False) as server:
    start(server)
    sample = interface_bench.measure_stream(
        interface_bench.Endpoint.parse(
            f"http://127.0.0.1:{server.server_address[1]}"
        ),
        model="test-model",
        prompt="hello",
        max_tokens=3,
        timeout=5.0,
        request_index=0,
    )
    check(sample.event_count == 6, f"close-delimited events: {sample.event_count}")
    check(sample.client_ttft_ms is not None, "close-delimited TTFT missing")
    check(sample.gap_ms[-1] >= 25.0, f"close-delimited gaps: {sample.gap_ms}")

check(
    interface_bench.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5,
    "percentile",
)
check(
    interface_bench.distribution([])["count"] == 0,
    "empty distribution",
)
print("interface_bench_test: ok")
