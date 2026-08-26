#!/usr/bin/env python3
"""Canonical OpenAI-compatible serving benchmark for Strix Halo.

The harness keeps stage throughput, service latency, and aggregate concurrency
throughput separate. Reports omit endpoint hosts, prompts, generated text,
local paths, raw token IDs, and timestamps.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


BENCHMARK_SCHEMA = "strix.serving-benchmark.v1"
DEFAULT_PROMPT = (
    "Explain how a bounded continuous-serving scheduler can preserve "
    "single-request latency while allowing several independent requests to "
    "make fair progress. Discuss prefill, decode, queueing, and cancellation "
    "in concrete technical terms. Continue until the output limit is reached."
)


@dataclass(frozen=True)
class RequestObservation:
    concurrency: int
    repetition: int
    request_index: int
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    prefill_tokens: int
    wall_ms: float
    client_ttft_ms: float | None
    client_event_inter_token_ms: float | None
    queue_ms: float
    prefill_ms: float
    decode_ms: float
    server_ttft_ms: float
    server_inter_token_ms: float
    server_max_inter_token_ms: float
    requested_logical_concurrency: int
    physical_execution_width: int
    execution_plan: str
    cache_hit: bool
    prefill_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    whole_request_tokens_per_second: float | None
    started_at: float
    finished_at: float

    def public(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("started_at")
        result.pop("finished_at")
        return result


@dataclass(frozen=True)
class RoundObservation:
    concurrency: int
    repetition: int
    span_ms: float
    prefill_tokens_per_second: float
    output_tokens_per_second: float
    total_tokens_per_second: float
    samples: tuple[RequestObservation, ...]

    def public(self) -> dict[str, Any]:
        return {
            "repetition": self.repetition,
            "span_ms": self.span_ms,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "output_tokens_per_second": self.output_tokens_per_second,
            "total_tokens_per_second": self.total_tokens_per_second,
        }


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float | None]) -> dict[str, float | None]:
    finite = sorted(
        value
        for value in values
        if value is not None and math.isfinite(value)
    )
    if not finite:
        return {
            "median": None,
            "p95": None,
            "p99": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "median": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "minimum": finite[0],
        "maximum": finite[-1],
    }


def parse_concurrency_levels(value: str) -> list[int]:
    levels: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            level = int(stripped)
        except ValueError as exception:
            raise argparse.ArgumentTypeError(
                f"invalid concurrency level: {stripped}"
            ) from exception
        if level <= 0:
            raise argparse.ArgumentTypeError(
                "concurrency levels must be positive"
            )
        if level not in levels:
            levels.append(level)
    if not levels:
        raise argparse.ArgumentTypeError(
            "at least one concurrency level is required"
        )
    return levels


def _rate(tokens: int, milliseconds: float) -> float | None:
    if tokens <= 0 or milliseconds <= 0.0:
        return None
    return tokens * 1000.0 / milliseconds


def _required_number(mapping: dict[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"terminal usage is missing numeric strix.{name}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"terminal usage has invalid strix.{name}")
    return value


def _required_integer(mapping: dict[str, Any], name: str) -> int:
    value = _required_number(mapping, name)
    if not value.is_integer():
        raise RuntimeError(f"terminal usage has non-integral strix.{name}")
    return int(value)


def _sse_data(response: Any) -> Iterable[str]:
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def _http_error(exception: urllib.error.HTTPError) -> RuntimeError:
    body = exception.read(4096).decode("utf-8", errors="replace")
    return RuntimeError(
        f"serving request failed with HTTP {exception.code}: {body}"
    )


def run_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    client_id: str,
    concurrency: int,
    repetition: int,
    request_index: int,
    start_gate: threading.Barrier | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RequestObservation:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-ID": client_id,
        },
        method="POST",
    )

    if start_gate is not None:
        start_gate.wait()
    started_at = clock()
    usage: dict[str, Any] | None = None
    useful_event_times: list[float] = []
    saw_done = False
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"serving request returned HTTP {response.status}"
                )
            for event in _sse_data(response):
                if event == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError as exception:
                    raise RuntimeError(
                        "serving response contained invalid SSE JSON"
                    ) from exception
                if "error" in chunk:
                    raise RuntimeError(
                        "serving stream failed: "
                        + json.dumps(chunk["error"], sort_keys=True)
                    )
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta", {})
                    if isinstance(delta, dict) and (
                        bool(delta.get("content"))
                        or bool(delta.get("tool_calls"))
                    ):
                        useful_event_times.append(clock())
                candidate_usage = chunk.get("usage")
                if isinstance(candidate_usage, dict):
                    usage = candidate_usage
    except urllib.error.HTTPError as exception:
        raise _http_error(exception) from exception
    except urllib.error.URLError as exception:
        raise RuntimeError(
            f"cannot reach the serving endpoint: {exception.reason}"
        ) from exception
    finished_at = clock()

    if not saw_done:
        raise RuntimeError("serving stream ended without [DONE]")
    if usage is None:
        raise RuntimeError(
            "serving stream omitted terminal usage; rebuild the server with "
            "the canonical benchmark metrics extension"
        )

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details", {})
    metrics = usage.get("strix")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
        or not isinstance(details, dict)
        or not isinstance(metrics, dict)
    ):
        raise RuntimeError("terminal usage has an incompatible schema")
    cached_prompt_tokens = details.get("cached_tokens", 0)
    if (
        isinstance(cached_prompt_tokens, bool)
        or not isinstance(cached_prompt_tokens, int)
        or cached_prompt_tokens < 0
        or cached_prompt_tokens > prompt_tokens
    ):
        raise RuntimeError("terminal usage has invalid cached token count")

    prefill_tokens = _required_integer(metrics, "prefill_tokens")
    queue_ms = _required_number(metrics, "queue_ms")
    prefill_ms = _required_number(metrics, "prefill_ms")
    decode_ms = _required_number(metrics, "decode_ms")
    server_ttft_ms = _required_number(metrics, "ttft_ms")
    server_inter_token_ms = _required_number(
        metrics, "mean_inter_token_ms"
    )
    server_max_inter_token_ms = _required_number(
        metrics, "max_inter_token_ms"
    )
    requested_logical_concurrency = _required_integer(
        metrics, "requested_logical_concurrency"
    )
    physical_execution_width = _required_integer(
        metrics, "physical_execution_width"
    )
    execution_plan = metrics.get("execution_plan")
    if not isinstance(execution_plan, str) or not execution_plan:
        raise RuntimeError(
            "terminal usage is missing string strix.execution_plan"
        )

    wall_ms = (finished_at - started_at) * 1000.0
    client_ttft_ms = (
        (useful_event_times[0] - started_at) * 1000.0
        if useful_event_times
        else None
    )
    client_event_inter_token_ms = None
    if len(useful_event_times) > 1:
        intervals = [
            (current - previous) * 1000.0
            for previous, current in zip(
                useful_event_times, useful_event_times[1:]
            )
        ]
        client_event_inter_token_ms = statistics.fmean(intervals)
    useful_tokens = prefill_tokens + completion_tokens

    return RequestObservation(
        concurrency=concurrency,
        repetition=repetition,
        request_index=request_index,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
        prefill_tokens=prefill_tokens,
        wall_ms=wall_ms,
        client_ttft_ms=client_ttft_ms,
        client_event_inter_token_ms=client_event_inter_token_ms,
        queue_ms=queue_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        server_ttft_ms=server_ttft_ms,
        server_inter_token_ms=server_inter_token_ms,
        server_max_inter_token_ms=server_max_inter_token_ms,
        requested_logical_concurrency=requested_logical_concurrency,
        physical_execution_width=physical_execution_width,
        execution_plan=execution_plan,
        cache_hit=bool(metrics.get("cache_hit", cached_prompt_tokens > 0)),
        prefill_tokens_per_second=_rate(prefill_tokens, prefill_ms),
        decode_tokens_per_second=_rate(completion_tokens, decode_ms),
        whole_request_tokens_per_second=_rate(useful_tokens, wall_ms),
        started_at=started_at,
        finished_at=finished_at,
    )


def _run_round(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    concurrency: int,
    repetition: int,
    run_nonce: str,
) -> RoundObservation:
    start_gate = threading.Barrier(concurrency + 1)

    def execute(index: int) -> RequestObservation:
        unique_prompt = (
            f"Benchmark nonce {run_nonce}-{concurrency}-{repetition}-{index}. "
            + prompt
        )
        return run_request(
            base_url=base_url,
            model=model,
            prompt=unique_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            client_id=f"strix-bench-c{concurrency}-r{repetition}-i{index}",
            concurrency=concurrency,
            repetition=repetition,
            request_index=index,
            start_gate=start_gate,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [executor.submit(execute, index) for index in range(concurrency)]
        start_gate.wait()
        samples = tuple(future.result() for future in futures)

    started_at = min(sample.started_at for sample in samples)
    finished_at = max(sample.finished_at for sample in samples)
    span_ms = (finished_at - started_at) * 1000.0
    prefill_tokens = sum(sample.prefill_tokens for sample in samples)
    completion_tokens = sum(sample.completion_tokens for sample in samples)
    return RoundObservation(
        concurrency=concurrency,
        repetition=repetition,
        span_ms=span_ms,
        prefill_tokens_per_second=prefill_tokens * 1000.0 / span_ms,
        output_tokens_per_second=completion_tokens * 1000.0 / span_ms,
        total_tokens_per_second=(
            (prefill_tokens + completion_tokens) * 1000.0 / span_ms
        ),
        samples=samples,
    )


def _summarize_rounds(
    rounds: list[RoundObservation],
) -> dict[str, Any]:
    samples = [sample for round_ in rounds for sample in round_.samples]
    return {
        "requestCount": len(samples),
        "roundCount": len(rounds),
        "stage": {
            "prefill_tokens_per_second": distribution(
                sample.prefill_tokens_per_second for sample in samples
            ),
            "decode_tokens_per_second": distribution(
                sample.decode_tokens_per_second for sample in samples
            ),
            "whole_request_tokens_per_second": distribution(
                sample.whole_request_tokens_per_second for sample in samples
            ),
        },
        "latency": {
            "request_ms": distribution(sample.wall_ms for sample in samples),
            "client_ttft_ms": distribution(
                sample.client_ttft_ms for sample in samples
            ),
            "server_ttft_ms": distribution(
                sample.server_ttft_ms for sample in samples
            ),
            "server_inter_token_ms": distribution(
                sample.server_inter_token_ms for sample in samples
            ),
            "server_max_inter_token_ms": distribution(
                sample.server_max_inter_token_ms for sample in samples
            ),
            "client_event_inter_token_ms": distribution(
                sample.client_event_inter_token_ms for sample in samples
            ),
            "queue_ms": distribution(sample.queue_ms for sample in samples),
        },
        "aggregate": {
            "prefill_tokens_per_second": distribution(
                round_.prefill_tokens_per_second for round_ in rounds
            ),
            "output_tokens_per_second": distribution(
                round_.output_tokens_per_second for round_ in rounds
            ),
            "total_tokens_per_second": distribution(
                round_.total_tokens_per_second for round_ in rounds
            ),
        },
        "executionPlans": sorted(
            {sample.execution_plan for sample in samples}
        ),
        "logicalConcurrency": sorted(
            {sample.requested_logical_concurrency for sample in samples}
        ),
        "physicalExecutionWidths": sorted(
            {sample.physical_execution_width for sample in samples}
        ),
        "rounds": [round_.public() for round_ in rounds],
        "samples": [sample.public() for sample in samples],
    }


def run_benchmark(
    *,
    base_url: str,
    model: str,
    prompt: str,
    workload_id: str,
    max_tokens: int,
    temperature: float,
    concurrency_levels: list[int],
    warmup_rounds: int,
    repetitions: int,
    timeout_seconds: float,
    fingerprint: dict[str, Any],
    source_revision: str,
    source_dirty: bool,
    prompt_repeat: int = 1,
) -> dict[str, Any]:
    if not model:
        raise ValueError("model must not be empty")
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_tokens <= 0 or repetitions <= 0 or warmup_rounds < 0:
        raise ValueError("token and repetition counts are invalid")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between zero and two")

    run_nonce = uuid.uuid4().hex
    results: dict[str, Any] = {}
    warnings: list[str] = []
    for concurrency in concurrency_levels:
        for warmup in range(warmup_rounds):
            _run_round(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                concurrency=concurrency,
                repetition=-(warmup + 1),
                run_nonce=run_nonce,
            )
        rounds = [
            _run_round(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                concurrency=concurrency,
                repetition=repetition,
                run_nonce=run_nonce,
            )
            for repetition in range(repetitions)
        ]
        summary = _summarize_rounds(rounds)
        results[f"c{concurrency}"] = summary
        observed_capacity = max(summary["logicalConcurrency"])
        if observed_capacity < concurrency:
            warnings.append(
                f"C={concurrency} exceeded the server's reported logical "
                f"capacity {observed_capacity}; queueing is included"
            )

    fingerprint_id = fingerprint.get("fingerprintId")
    canonical = fingerprint.get("canonical")
    if (
        not isinstance(fingerprint_id, str)
        or len(fingerprint_id) != 64
        or not isinstance(canonical, dict)
    ):
        raise ValueError("machine fingerprint has an incompatible schema")

    return {
        "schemaVersion": "1.0.0",
        "artifactType": "servingBenchmark",
        "benchmarkSchema": BENCHMARK_SCHEMA,
        "fingerprintId": fingerprint_id,
        "canonical": canonical,
        "source": {
            "revision": source_revision,
            "dirty": source_dirty,
            "buildMode": "nix-release",
        },
        "model": {"id": model},
        "workload": {
            "id": workload_id,
            "transport": "openai-chat-completions-sse",
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "promptRepeat": prompt_repeat,
            "warmupRounds": warmup_rounds,
            "repetitions": repetitions,
            "concurrency": concurrency_levels,
        },
        "metricDefinitions": {
            "prefill_tokens_per_second": (
                "actual server prefill tokens divided by server prefill time"
            ),
            "decode_tokens_per_second": (
                "completion tokens divided by server decode time"
            ),
            "server_inter_token_ms": (
                "mean scheduler-observed interval between generated tokens"
            ),
            "whole_request_tokens_per_second": (
                "actual prefill plus completion tokens divided by "
                "client-observed request wall time"
            ),
            "aggregate_total_tokens_per_second": (
                "sum of actual prefill plus completion tokens divided by "
                "the synchronized concurrency-round span"
            ),
        },
        "results": results,
        "warnings": warnings,
    }


def discover_model(base_url: str, timeout_seconds: float) -> str:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/models",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds
        ) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exception:
        raise _http_error(exception) from exception
    except urllib.error.URLError as exception:
        raise RuntimeError(
            f"cannot discover the served model: {exception.reason}"
        ) from exception
    models = body.get("data") if isinstance(body, dict) else None
    identifiers = [
        item.get("id")
        for item in models or []
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item.get("capability") in (None, "text")
    ]
    if len(identifiers) != 1:
        raise RuntimeError(
            "model discovery requires exactly one served text model; "
            "pass --model explicitly"
        )
    return identifiers[0]


def load_fingerprint(path: Path | None, strix_binary: Path) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [str(strix_binary), "diagnose", "--fingerprint", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def source_identity(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, bool(status.strip())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _median(summary: dict[str, float | None]) -> str:
    value = summary["median"]
    return "-" if value is None else f"{value:.2f}"


def print_human(report: dict[str, Any]) -> None:
    print(
        "C  requests  request p50/p95 ms  TTFT p50/p95 ms  "
        "ITL p50/p95 ms  prefill tok/s  decode tg  whole tok/s  "
        "aggregate tok/s"
    )
    for key, result in report["results"].items():
        concurrency = key[1:]
        latency = result["latency"]
        stage = result["stage"]
        aggregate = result["aggregate"]
        request_p95 = latency["request_ms"]["p95"]
        ttft_p95 = latency["client_ttft_ms"]["p95"]
        itl_p95 = latency["server_inter_token_ms"]["p95"]
        print(
            f"{concurrency:>1}  {result['requestCount']:>8}  "
            f"{_median(latency['request_ms']):>7}/"
            f"{'-' if request_p95 is None else f'{request_p95:.2f}'}  "
            f"{_median(latency['client_ttft_ms']):>7}/"
            f"{'-' if ttft_p95 is None else f'{ttft_p95:.2f}'}  "
            f"{_median(latency['server_inter_token_ms']):>7}/"
            f"{'-' if itl_p95 is None else f'{itl_p95:.2f}'}  "
            f"{_median(stage['prefill_tokens_per_second']):>13}  "
            f"{_median(stage['decode_tokens_per_second']):>9}  "
            f"{_median(stage['whole_request_tokens_per_second']):>11}  "
            f"{_median(aggregate['total_tokens_per_second']):>15}"
        )
    for warning in report["warnings"]:
        print(f"warning: {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strix-serving-bench",
        description=(
            "Report separate prefill, decode, whole-request, and concurrent "
            "serving metrics."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Server base URL (never written to the artifact)",
    )
    parser.add_argument("--model", help="Served OpenAI model ID")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Benchmark prompt text")
    prompt.add_argument("--prompt-file", type=Path, help="Benchmark prompt file")
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=8,
        help="Repeat the canonical/custom prompt N times (default: 8)",
    )
    parser.add_argument(
        "--workload-id",
        default="canonical-serving-v1",
        help="Public workload identity stored in the artifact",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--concurrency",
        type=parse_concurrency_levels,
        default=parse_concurrency_levels("1,2,4"),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--fingerprint",
        type=Path,
        help="Existing `strix diagnose --fingerprint --json` output",
    )
    parser.add_argument(
        "--strix",
        type=Path,
        default=Path("./result/bin/strix"),
        help="Release strix binary used for automatic fingerprinting",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON artifact")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prompt_repeat <= 0:
        raise SystemExit("--prompt-repeat must be positive")
    prompt = args.prompt or DEFAULT_PROMPT
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    prompt = "\n\n".join(prompt for _ in range(args.prompt_repeat))

    root = Path(__file__).resolve().parents[2]
    model = args.model or discover_model(args.base_url, args.timeout)
    fingerprint = load_fingerprint(args.fingerprint, args.strix)
    revision, dirty = source_identity(root)
    report = run_benchmark(
        base_url=args.base_url,
        model=model,
        prompt=prompt,
        workload_id=args.workload_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency_levels=args.concurrency,
        warmup_rounds=args.warmup,
        repetitions=args.repetitions,
        timeout_seconds=args.timeout,
        fingerprint=fingerprint,
        source_revision=revision,
        source_dirty=dirty,
        prompt_repeat=args.prompt_repeat,
    )
    if args.output is not None:
        atomic_json(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
        if args.output is not None:
            print(f"artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
