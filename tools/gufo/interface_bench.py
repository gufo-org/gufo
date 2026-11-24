#!/usr/bin/env python3
"""Measure the HTTP serving interface, not the model.

The canonical serving benchmark answers "how fast does the model run?". This
tool answers a different question: how much of what a streaming client sees is
added after generation, by connection setup, HTTP framing, scheduling of the
socket writer, or the client's own read loop.

It speaks raw HTTP/1.1 over a socket so connection setup, the first response
byte, every SSE event and every socket read get separate timestamps. The report
keeps those stages apart:

* connect_ms: TCP connect (and TLS handshake when enabled)
* ttfb_ms: request written -> first response byte
* first_event_ms: request written -> first SSE data event of any kind
* client_ttft_ms: request written -> first token event (content, reasoning
  or tool call)
* gap_ms: token event arrival intervals, including p99 and >threshold
  counts, so Nagle/delayed-ACK stalls are visible instead of averaged away
* events_per_read / bytes_per_read: whether the server's writes coalesce,
  over the reads that carried body bytes
* client_minus_server_ttft_ms: client TTFT minus the server's `usage.gufo`
  ttft_ms. The server clock starts at backend entry and stops when it selects
  the first token, so the delta covers HTTP parsing and admission before the
  backend, SSE emission and transport, not transport alone.

Reports omit endpoint hosts and paths, prompts (only a SHA-256 is kept),
generated text, raw token IDs and timestamps.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import socket
import ssl
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from gufo.serving_bench import (
    DEFAULT_PROMPT,
    ENDPOINT_PROFILES,
    _authorization_headers,
    atomic_json,
    benchmark_sampling,
    discover_model,
    parse_concurrency_levels,
    source_identity,
)
from gufo.serving_bench import distribution as serving_distribution

BENCHMARK_SCHEMA = "gufo.interface-benchmark.v1"
GAP_THRESHOLDS_MS = (10.0, 30.0, 100.0)


def distribution(values: Iterable[float | None]) -> dict[str, float | None]:
    """The serving benchmark's distribution, plus a sample count and mean."""
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(value)
    ]
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        **serving_distribution(finite),
    }


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    tls: bool
    base_path: str

    @classmethod
    def parse(cls, base_url: str) -> "Endpoint":
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("base URL must start with http:// or https://")
        if not parsed.hostname:
            raise ValueError("base URL has no host")
        return cls(
            host=parsed.hostname,
            port=parsed.port
            or (443 if parsed.scheme == "https" else 80),
            tls=parsed.scheme == "https",
            base_path=parsed.path.rstrip("/"),
        )

    def public(self) -> dict[str, Any]:
        # The artifact never records where the server runs; a base path can
        # carry a proxy route or token, so only its presence is kept.
        return {"tls": self.tls, "hasBasePath": bool(self.base_path)}


@dataclass(frozen=True)
class ResponseHead:
    status: int
    headers: dict[str, str]


@dataclass(frozen=True)
class OverheadSample:
    connect_ms: float | None
    ttfb_ms: float
    total_ms: float
    bytes: int
    reused: bool


@dataclass(frozen=True)
class StreamSample:
    connect_ms: float
    ttfb_ms: float
    first_event_ms: float | None
    client_ttft_ms: float | None
    total_ms: float
    event_count: int
    gap_ms: tuple[float, ...]
    events_per_read: float
    bytes_per_read: float
    server_ttft_ms: float | None
    server_inter_token_ms: float | None
    server_max_inter_token_ms: float | None
    queue_ms: float | None
    prefill_ms: float | None
    decode_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_prompt_tokens: int | None
    client_minus_server_ttft_ms: float | None


class HttpConnection:
    """One HTTP/1.1 connection with byte-arrival timestamps."""

    def __init__(self, endpoint: Endpoint, timeout: float,
                 clock: Callable[[], float]) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.clock = clock
        self.socket: socket.socket | None = None
        self.buffer = bytearray()
        self.read_times: list[float] = []
        self.read_sizes: list[int] = []
        self.last_read_time = 0.0
        self.last_read_index = -1
        # Reads that carried only the response head, not body bytes.
        self.head_only_reads = 0
        self.head_body_bytes = 0
        self.connect_ms = 0.0

    def connect(self) -> None:
        started = self.clock()
        raw = socket.create_connection(
            (self.endpoint.host, self.endpoint.port), timeout=self.timeout
        )
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.endpoint.tls:
            context = ssl.create_default_context()
            raw = context.wrap_socket(raw, server_hostname=self.endpoint.host)
        raw.settimeout(self.timeout)
        self.socket = raw
        self.buffer.clear()
        self.read_times.clear()
        self.read_sizes.clear()
        self.connect_ms = (self.clock() - started) * 1000.0

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            finally:
                self.socket = None

    def send_request(self, method: str, path: str, headers: dict[str, str],
                     body: bytes) -> float:
        assert self.socket is not None
        # Read timestamps are per response, not per socket, so a reused
        # connection reports its own first byte rather than the previous one's.
        self.read_times.clear()
        self.read_sizes.clear()
        self.last_read_index = -1
        self.last_read_time = 0.0
        lines = [f"{method} {path} HTTP/1.1"]
        host = self.endpoint.host
        if ":" in host:
            host = f"[{host}]"
        lines.append(f"Host: {host}:{self.endpoint.port}")
        for name, value in headers.items():
            lines.append(f"{name}: {value}")
        if body:
            lines.append(f"Content-Length: {len(body)}")
        lines.append("")
        lines.append("")
        request = ("\r\n".join(lines)).encode("ascii") + body
        self.socket.sendall(request)
        return self.clock()

    def _fill(self) -> bool:
        assert self.socket is not None
        chunk = self.socket.recv(1 << 16)
        now = self.clock()
        if not chunk:
            return False
        self.buffer.extend(chunk)
        self.read_times.append(now)
        self.read_sizes.append(len(chunk))
        self.last_read_time = now
        self.last_read_index = len(self.read_times) - 1
        return True

    def _take_line(self) -> bytes:
        while True:
            end = self.buffer.find(b"\r\n")
            if end >= 0:
                line = bytes(self.buffer[:end])
                del self.buffer[:end + 2]
                return line
            if not self._fill():
                raise ConnectionError("connection closed while reading a line")

    def read_head(self) -> tuple[ResponseHead, float]:
        """Return the response head and the time its first byte arrived."""
        while True:
            end = self.buffer.find(b"\r\n\r\n")
            if end >= 0:
                break
            if not self._fill():
                raise ConnectionError("connection closed before response head")
        first_byte = self.read_times[0] if self.read_times else self.clock()
        raw = bytes(self.buffer[:end])
        del self.buffer[:end + 4]
        self.head_only_reads = max(
            len(self.read_times) - (1 if self.buffer else 0), 0
        )
        # The read that completed the head may also carry body bytes; only
        # those count toward body read sizes.
        self.head_body_bytes = len(self.buffer)
        lines = raw.split(b"\r\n")
        try:
            status = int(lines[0].split(b" ", 2)[1])
        except (IndexError, ValueError) as exception:
            raise RuntimeError("malformed HTTP status line") from exception
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, separator, value = line.partition(b":")
            if separator:
                headers[name.decode("latin-1").strip().lower()] = (
                    value.decode("latin-1").strip()
                )
        return ResponseHead(status=status, headers=headers), first_byte

    def body_read_sizes(self) -> list[int]:
        """Bytes of response body delivered by each read after the head."""
        sizes = self.read_sizes[self.head_only_reads:]
        if sizes and self.head_body_bytes:
            sizes = [self.head_body_bytes, *sizes[1:]]
        return sizes

    def read_body(self, head: ResponseHead) -> tuple[bytes, bool]:
        """Read a whole response body; return it and whether it ended at
        connection close, which makes the connection unusable."""
        if "chunked" in head.headers.get("transfer-encoding", "").lower():
            reader = ChunkReader(self)
            parts = []
            while fragment := reader.read():
                parts.append(fragment)
            return b"".join(parts), False
        if "content-length" in head.headers:
            length = int(head.headers["content-length"])
            while len(self.buffer) < length:
                if not self._fill():
                    raise ConnectionError(
                        "connection closed inside response body"
                    )
            body = bytes(self.buffer[:length])
            del self.buffer[:length]
            return body, False
        if head.status in (204, 304) or 100 <= head.status < 200:
            return b"", False
        # Close-delimited: the body ends when the server closes.
        while self._fill():
            pass
        body = bytes(self.buffer)
        self.buffer.clear()
        return body, True


class ChunkReader:
    """Decode `Transfer-Encoding: chunked` payload without hiding read times."""

    def __init__(self, connection: HttpConnection) -> None:
        self.connection = connection
        self.remaining = 0
        self.finished = False

    def read(self) -> bytes:
        """Return the next payload fragment, or b"" at the last chunk."""
        if self.finished:
            return b""
        buffer = self.connection.buffer
        if self.remaining == 0:
            size_line = self.connection._take_line()
            try:
                size = int(size_line.split(b";", 1)[0], 16)
            except ValueError as exception:
                raise RuntimeError("malformed chunk size") from exception
            if size == 0:
                # Consume the (empty) trailer section.
                while self.connection._take_line():
                    pass
                self.finished = True
                return b""
            self.remaining = size
        # Return what has arrived rather than waiting for the whole chunk, so
        # events inside a large chunk keep their own read times.
        while not buffer:
            if not self.connection._fill():
                raise ConnectionError("connection closed inside chunk")
        take = min(len(buffer), self.remaining)
        payload = bytes(buffer[:take])
        del buffer[:take]
        self.remaining -= take
        if self.remaining == 0:
            while len(buffer) < 2:
                if not self.connection._fill():
                    raise ConnectionError("connection closed inside chunk")
            if buffer[:2] != b"\r\n":
                raise RuntimeError("malformed chunk terminator")
            del buffer[:2]
        return payload


class SseAccumulator:
    """Split a byte stream into SSE data events, keeping read timestamps."""

    def __init__(self) -> None:
        self.pending = bytearray()

    def feed(self, data: bytes, read_index: int, read_time: float
             ) -> list[tuple[str, int, float]]:
        self.pending.extend(data)
        # SSE lines end in CRLF, LF or CR. Normalize to LF so mixed endings
        # split at the first blank line; a trailing CR may begin a CRLF split
        # across reads, so it waits for the next read.
        tail = self.pending.endswith(b"\r")
        if tail:
            del self.pending[-1:]
        normalized = bytes(self.pending).replace(b"\r\n", b"\n").replace(
            b"\r", b"\n"
        )
        self.pending[:] = normalized + (b"\r" if tail else b"")
        events: list[tuple[str, int, float]] = []
        while True:
            end = self.pending.find(b"\n\n")
            if end < 0:
                return events
            raw = bytes(self.pending[:end])
            del self.pending[:end + 2]
            data_lines = []
            for line in raw.splitlines():
                if line.startswith(b"data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                events.append((
                    b"\n".join(data_lines).decode("utf-8", errors="replace"),
                    read_index,
                    read_time,
                ))


def _useful_event(payload: str) -> tuple[bool, dict[str, Any] | None]:
    if payload == "[DONE]":
        return False, None
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(chunk, dict):
        return False, None
    if "error" in chunk:
        raise RuntimeError(
            "serving stream failed: "
            + json.dumps(chunk["error"], sort_keys=True)
        )
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, chunk
    choice = choices[0]
    if not isinstance(choice, dict):
        return False, chunk
    delta = choice.get("delta")
    if isinstance(delta, dict):
        # The server's ttft_ms stops at its first selected token, which is a
        # reasoning token for thinking models, so reasoning deltas count too.
        for key in ("content", "reasoning_content"):
            text = delta.get(key)
            if isinstance(text, str) and text:
                return True, chunk
        if delta.get("tool_calls"):
            return True, chunk
    return False, chunk


def _usage_metrics(chunk: dict[str, Any] | None) -> dict[str, Any] | None:
    if not chunk:
        return None
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return None
    return usage


def measure_overhead(
    endpoint: Endpoint,
    *,
    path: str,
    requests: int,
    reuse: bool,
    timeout: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    samples: list[OverheadSample] = []
    request_path = endpoint.base_path + path
    headers = {**_authorization_headers(), "Accept": "application/json"}
    reused = False
    connections = 0
    connection = HttpConnection(endpoint, timeout, clock)
    try:
        for index in range(requests):
            fresh = not reuse or connection.socket is None
            if fresh:
                connection.close()
                connection.connect()
                connections += 1
                reused = False
            try:
                sent_at = connection.send_request("GET", request_path, headers, b"")
                head, first_byte = connection.read_head()
                body, closed = connection.read_body(head)
            except TimeoutError:
                # A slow server is not a closed one; retrying would send the
                # request twice.
                raise
            except (ConnectionError, OSError):
                # A server that closes without `Connection: close` cannot be
                # reused; only a socket that already served a response retries.
                if fresh:
                    raise
                connection.close()
                connection.connect()
                connections += 1
                fresh = True
                reused = False
                sent_at = connection.send_request("GET", request_path, headers, b"")
                head, first_byte = connection.read_head()
                body, closed = connection.read_body(head)
            finished = clock()
            samples.append(OverheadSample(
                # A reused socket has no connect; a zero would pull the
                # distribution toward it.
                connect_ms=connection.connect_ms if fresh else None,
                ttfb_ms=(first_byte - sent_at) * 1000.0,
                total_ms=(finished - sent_at) * 1000.0,
                bytes=len(body),
                reused=reused,
            ))
            reused = True
            if closed or head.headers.get("connection", "").lower() == "close":
                connection.close()
                reused = False
    finally:
        connection.close()
    return {
        "path": path,
        "requests": requests,
        "reuse": reuse,
        "connections": connections,
        "reconnects": max(connections - 1, 0),
        "connectionReused": any(sample.reused for sample in samples),
        "connect_ms": distribution(sample.connect_ms for sample in samples),
        "ttfb_ms": distribution(sample.ttfb_ms for sample in samples),
        "total_ms": distribution(sample.total_ms for sample in samples),
        "responseBytes": distribution(sample.bytes for sample in samples),
    }


def measure_stream(
    endpoint: Endpoint,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    request_index: int,
    temperature: float = 0.0,
    endpoint_profile: str = "gufo",
    cache_prompt: bool | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamSample:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        **benchmark_sampling(temperature, endpoint_profile),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection = HttpConnection(endpoint, timeout, clock)
    connection.connect()
    try:
        sent_at = connection.send_request(
            "POST", endpoint.base_path + "/v1/chat/completions",
            {
                **_authorization_headers(),
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "X-Client-ID": f"interface-bench-{request_index}",
            },
            body,
        )
        head, first_byte = connection.read_head()
        if head.status != 200:
            raise RuntimeError(f"stream request returned HTTP {head.status}")
        chunked = "chunked" in head.headers.get(
            "transfer-encoding", ""
        ).lower()
        reader = ChunkReader(connection) if chunked else None
        # Bytes left in a length-delimited body; None when close-delimited.
        remaining = (
            int(head.headers["content-length"])
            if reader is None and "content-length" in head.headers
            else None
        )
        accumulator = SseAccumulator()
        useful_times: list[float] = []
        usage: dict[str, Any] | None = None
        event_count = 0
        finished_at = sent_at
        first_event: float | None = None
        queue_ms = None
        prefill_ms = None
        decode_ms = None
        server_ttft_ms = None
        server_inter_token_ms = None
        server_max_inter_token_ms = None
        prompt_tokens = None
        completion_tokens = None
        cached_prompt_tokens = None

        saw_done = False
        while not saw_done:
            if reader is not None:
                data = reader.read()
                if not data:
                    break
            else:
                # Length- and close-delimited bodies: consume what has arrived
                # (including bytes that came with the head) before reading
                # more, so each event keeps the time of the read carrying it.
                if remaining == 0:
                    break
                if not connection.buffer and not connection._fill():
                    if remaining is not None:
                        raise ConnectionError(
                            "connection closed inside response body"
                        )
                    break
                take = len(connection.buffer)
                if remaining is not None:
                    take = min(take, remaining)
                    remaining -= take
                data = bytes(connection.buffer[:take])
                del connection.buffer[:take]
            read_index = connection.last_read_index
            read_time = connection.last_read_time
            for event, _, event_time in accumulator.feed(
                data, read_index, read_time
            ):
                event_count += 1
                if first_event is None and event != "[DONE]":
                    first_event = event_time
                useful, chunk = _useful_event(event)
                candidate_usage = _usage_metrics(chunk)
                if candidate_usage is not None:
                    usage = candidate_usage
                if useful:
                    useful_times.append(event_time)
                if event == "[DONE]":
                    finished_at = event_time
                    saw_done = True
                    break
            if not saw_done:
                finished_at = read_time
        # Streams never reuse their connection, so the terminal chunk and
        # trailer after [DONE] are not drained.

        gaps = [
            (current - previous) * 1000.0
            for previous, current in zip(useful_times, useful_times[1:])
        ]
        # Reads that carried only the response head say nothing about how the
        # server's event writes coalesce.
        body_reads = connection.body_read_sizes()
        events_per_read = event_count / len(body_reads) if body_reads else 0.0
        bytes_per_read = statistics.fmean(body_reads) if body_reads else 0.0
        if usage is not None:
            metrics = usage.get("gufo")
            if isinstance(metrics, dict):
                queue_ms = _number(metrics.get("queue_ms"))
                prefill_ms = _number(metrics.get("prefill_ms"))
                decode_ms = _number(metrics.get("decode_ms"))
                server_ttft_ms = _number(metrics.get("ttft_ms"))
                server_inter_token_ms = _number(
                    metrics.get("mean_inter_token_ms")
                )
                server_max_inter_token_ms = _number(
                    metrics.get("max_inter_token_ms")
                )
            prompt_tokens = _integer(usage.get("prompt_tokens"))
            completion_tokens = _integer(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached_prompt_tokens = _integer(details.get("cached_tokens"))
        client_ttft_ms = (
            (useful_times[0] - sent_at) * 1000.0 if useful_times else None
        )
        client_minus_server_ttft = None
        if client_ttft_ms is not None and server_ttft_ms is not None:
            client_minus_server_ttft = client_ttft_ms - server_ttft_ms
        return StreamSample(
            connect_ms=connection.connect_ms,
            ttfb_ms=(first_byte - sent_at) * 1000.0,
            first_event_ms=(
                (first_event - sent_at) * 1000.0
                if first_event is not None else None
            ),
            client_ttft_ms=client_ttft_ms,
            total_ms=(finished_at - sent_at) * 1000.0,
            event_count=event_count,
            gap_ms=tuple(gaps),
            events_per_read=events_per_read,
            bytes_per_read=bytes_per_read,
            server_ttft_ms=server_ttft_ms,
            server_inter_token_ms=server_inter_token_ms,
            server_max_inter_token_ms=server_max_inter_token_ms,
            queue_ms=queue_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            client_minus_server_ttft_ms=client_minus_server_ttft,
        )
    finally:
        connection.close()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _stream_report(samples: list[StreamSample]) -> dict[str, Any]:
    gaps = [gap for sample in samples for gap in sample.gap_ms]
    return {
        "requests": len(samples),
        "connect_ms": distribution(sample.connect_ms for sample in samples),
        "ttfb_ms": distribution(sample.ttfb_ms for sample in samples),
        "first_event_ms": distribution(
            sample.first_event_ms for sample in samples
        ),
        "client_ttft_ms": distribution(
            sample.client_ttft_ms for sample in samples
        ),
        "total_ms": distribution(sample.total_ms for sample in samples),
        "eventCount": distribution(
            sample.event_count for sample in samples
        ),
        "eventsPerRead": distribution(
            sample.events_per_read for sample in samples
        ),
        "bytesPerRead": distribution(
            sample.bytes_per_read for sample in samples
        ),
        "gap_ms": distribution(gaps),
        "gapThresholds": {
            f"over_{int(threshold)}ms": sum(
                1 for gap in gaps if gap > threshold
            )
            for threshold in GAP_THRESHOLDS_MS
        },
        "server_ttft_ms": distribution(
            sample.server_ttft_ms for sample in samples
        ),
        "server_inter_token_ms": distribution(
            sample.server_inter_token_ms for sample in samples
        ),
        "server_max_inter_token_ms": distribution(
            sample.server_max_inter_token_ms for sample in samples
        ),
        "client_minus_server_ttft_ms": distribution(
            sample.client_minus_server_ttft_ms for sample in samples
        ),
        "queue_ms": distribution(sample.queue_ms for sample in samples),
        "prefill_ms": distribution(sample.prefill_ms for sample in samples),
        "decode_ms": distribution(sample.decode_ms for sample in samples),
        "promptTokens": distribution(
            sample.prompt_tokens for sample in samples
        ),
        "completionTokens": distribution(
            sample.completion_tokens for sample in samples
        ),
        "cachedPromptTokens": distribution(
            sample.cached_prompt_tokens for sample in samples
        ),
    }


def _stream_round(
    endpoint: Endpoint,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    requests: int,
    concurrency: int,
    temperature: float,
    endpoint_profile: str,
    cache_prompt: bool | None,
    clock: Callable[[], float],
) -> list[StreamSample]:
    start = threading.Event()
    started_condition = threading.Condition()
    started = 0
    target = min(concurrency, requests)

    def worker(index: int) -> StreamSample:
        nonlocal started
        with started_condition:
            started += 1
            started_condition.notify_all()
        start.wait()
        return measure_stream(
            endpoint,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            request_index=index,
            temperature=temperature,
            endpoint_profile=endpoint_profile,
            cache_prompt=cache_prompt,
            clock=clock,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as pool:
        futures = [pool.submit(worker, index) for index in range(requests)]
        try:
            with started_condition:
                if not started_condition.wait_for(
                    lambda: started >= target, timeout=timeout
                ):
                    raise RuntimeError("stream workers did not start in time")
            start.set()
            return [future.result() for future in futures]
        except BaseException:
            # Queued requests would otherwise each run to completion, up to
            # the timeout, before the failure is reported.
            for future in futures:
                future.cancel()
            raise
        finally:
            # Release blocked workers even on failure, or the pool's exit
            # would wait for them forever.
            start.set()


def run_benchmark(
    *,
    base_url: str,
    model: str | None,
    mode: str,
    overhead_path: str,
    requests: int,
    concurrency_levels: list[int],
    max_tokens: int,
    prompt: str,
    timeout: float,
    temperature: float = 0.0,
    prompt_repeat: int = 1,
    endpoint_profile: str = "gufo",
    cache_prompt: bool | None = None,
    source_revision: str | None = None,
    source_dirty: bool | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if mode not in ("overhead", "stream", "both"):
        raise ValueError("mode must be overhead, stream or both")
    if requests <= 0:
        raise ValueError("requests must be positive")
    if not concurrency_levels or any(level <= 0 for level in concurrency_levels):
        raise ValueError("concurrency levels must be positive")
    streams = mode in ("stream", "both")
    if streams and not model:
        raise ValueError("--model is required for streaming measurements")
    if endpoint_profile not in ENDPOINT_PROFILES:
        raise ValueError(f"unknown endpoint profile: {endpoint_profile}")
    endpoint = Endpoint.parse(base_url)
    workload: dict[str, Any] = {
        "mode": mode,
        "transport": "openai-chat-completions-sse",
        "requests": requests,
    }
    if mode in ("overhead", "both"):
        workload["overheadPath"] = overhead_path
    if streams:
        workload.update({
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "promptRepeat": prompt_repeat,
            # Identifies the prompt without recording its text.
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "concurrency": list(concurrency_levels),
            "endpointProfile": endpoint_profile,
            "cachePrompt": cache_prompt,
        })
    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "artifactType": "interfaceBenchmark",
        "benchmarkSchema": BENCHMARK_SCHEMA,
        "source": {"revision": source_revision, "dirty": source_dirty},
        "model": {"id": model},
        "endpoint": endpoint.public(),
        "workload": workload,
    }
    if mode in ("overhead", "both"):
        report["overhead"] = {
            "newConnection": measure_overhead(
                endpoint,
                path=overhead_path,
                requests=requests,
                reuse=False,
                timeout=timeout,
                clock=clock,
            ),
            "reusedConnection": measure_overhead(
                endpoint,
                path=overhead_path,
                requests=requests,
                reuse=True,
                timeout=timeout,
                clock=clock,
            ),
        }
    if streams:
        assert model is not None
        report["streams"] = [
            {
                "concurrency": concurrency,
                **_stream_report(_stream_round(
                    endpoint,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    requests=requests,
                    concurrency=concurrency,
                    temperature=temperature,
                    endpoint_profile=endpoint_profile,
                    cache_prompt=cache_prompt,
                    clock=clock,
                )),
            }
            for concurrency in concurrency_levels
        ]
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"schema: {report['benchmarkSchema']}")
    if "overhead" in report:
        print("\nHTTP request overhead")
        print("  mode      requests  connect p50  ttfb p50/p95  total p50  reuse")
        for name, result in report["overhead"].items():
            connect = result["connect_ms"]
            ttfb = result["ttfb_ms"]
            total = result["total_ms"]

            def value(item: dict[str, float | None], key: str) -> str:
                number = item[key]
                return "-" if number is None else f"{number:.3f}"

            print(
                f"  {name:<9} {result['requests']:>8}  "
                f"{value(connect, 'median'):>11}  "
                f"{value(ttfb, 'median'):>5}/{value(ttfb, 'p95'):<7}  "
                f"{value(total, 'median'):>9}  "
                f"{str(result['connectionReused']):<5}"
            )
    for stream in report.get("streams", []):
        print(f"\nToken arrival, concurrency {stream['concurrency']}")
        print(f"  requests: {stream['requests']}")
        for key in (
            "connect_ms", "ttfb_ms", "client_ttft_ms", "server_ttft_ms",
            "client_minus_server_ttft_ms", "gap_ms",
            "server_max_inter_token_ms",
        ):
            item = stream[key]
            if item["count"] == 0:
                continue
            print(
                f"  {key:<27} "
                f"p50 {item['median']:>7.2f}  p95 {item['p95']:>7.2f}  "
                f"p99 {item['p99']:>7.2f}  max {item['maximum']:>7.2f}"
            )
        thresholds = ", ".join(
            f"{name}={count}"
            for name, count in stream["gapThresholds"].items()
        )
        print(f"  token gaps: {thresholds}")
        print(
            f"  events/read p50: {stream['eventsPerRead']['median']:.2f}, "
            f"bytes/read p50: {stream['bytesPerRead']['median']:.0f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gufo-interface-bench",
        description=(
            "Measure HTTP connection, first-byte and token-arrival costs "
            "separately from model speed."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Server base URL (never written to the artifact)",
    )
    parser.add_argument(
        "--model",
        help="Served OpenAI model ID (default: the one served text model)",
    )
    parser.add_argument(
        "--mode", choices=("overhead", "stream", "both"), default="both",
        help="Measure request overhead, streaming, or both (default: both)",
    )
    parser.add_argument(
        "--overhead-path", default="/v1/models",
        help="Small response used for connection-overhead measurement",
    )
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument(
        "--concurrency",
        type=parse_concurrency_levels,
        default=parse_concurrency_levels("1,2,4"),
        help="Comma-separated stream concurrency levels (default: 1,2,4)",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Streaming prompt text")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--prompt-repeat", type=int, default=8,
        help="Repeat the canonical/custom prompt N times (default: 8)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--endpoint-profile",
        choices=ENDPOINT_PROFILES,
        default="gufo",
        help=(
            "gufo: pin Gufo sampling extensions; openai: send only OpenAI "
            "sampling fields"
        ),
    )
    parser.add_argument(
        "--no-cache-prompt",
        action="store_true",
        help="Disable prompt reuse for every request",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, help="Write the JSON artifact")
    parser.add_argument("--json", action="store_true")
    return parser


def _source_identity() -> tuple[str | None, bool | None]:
    try:
        return source_identity(Path(__file__).resolve().parents[2])
    except (OSError, subprocess.CalledProcessError):
        # An installed copy outside a checkout has no revision to record.
        return None, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    if args.prompt_repeat <= 0:
        raise SystemExit("--prompt-repeat must be positive")
    prompt = args.prompt or DEFAULT_PROMPT
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    prompt = "\n\n".join(prompt for _ in range(args.prompt_repeat))
    revision, dirty = _source_identity()
    try:
        model = args.model
        if model is None and args.mode != "overhead":
            model = discover_model(args.base_url, args.timeout)
        report = run_benchmark(
            base_url=args.base_url,
            model=model,
            mode=args.mode,
            overhead_path=args.overhead_path,
            requests=args.requests,
            concurrency_levels=args.concurrency,
            max_tokens=args.max_tokens,
            prompt=prompt,
            timeout=args.timeout,
            temperature=args.temperature,
            prompt_repeat=args.prompt_repeat,
            endpoint_profile=args.endpoint_profile,
            cache_prompt=False if args.no_cache_prompt else None,
            source_revision=revision,
            source_dirty=dirty,
        )
        if args.output:
            atomic_json(args.output, report)
    except (OSError, ValueError, RuntimeError) as exception:
        print(f"interface benchmark failed: {exception}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
