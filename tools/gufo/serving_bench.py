#!/usr/bin/env python3
"""Canonical OpenAI-compatible serving benchmark for Strix Halo.

The harness keeps stage throughput, service latency, and aggregate concurrency
throughput separate. Reports omit endpoint hosts, prompts, generated text,
local paths, raw token IDs, and timestamps.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
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


BENCHMARK_SCHEMA = "gufo.serving-benchmark.v1"
DEFAULT_PROMPT = (
    "Explain how a bounded continuous-serving scheduler can preserve "
    "single-request latency while allowing several independent requests to "
    "make fair progress. Discuss prefill, decode, queueing, and cancellation "
    "in concrete technical terms. Continue until the output limit is reached."
)


@dataclass(frozen=True)
class PromptCase:
    identifier: str
    category: str
    text: str


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
    draft_tokens: int
    draft_accepted_tokens: int
    draft_acceptance: float | None
    case_id: str | None
    category: str | None
    completion_sha256: str
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


def load_prompt_suite(
    path: Path,
    *,
    quick: bool = False,
    limit: int | None = None,
    selected: set[str] | None = None,
) -> list[PromptCase]:
    document = json.loads(path.read_text(encoding="utf-8"))
    prompts = document.get("prompts") if isinstance(document, dict) else None
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("suite must contain a non-empty prompts array")

    result: list[PromptCase] = []
    seen: set[str] = set()
    for entry in prompts:
        if not isinstance(entry, dict):
            raise ValueError("suite prompt entries must be objects")
        identifier = entry.get("id")
        category = entry.get("category")
        text = entry.get("text")
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(category, str)
            or not category
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("suite prompts require non-empty id/category/text")
        if identifier in seen:
            raise ValueError(f"suite contains duplicate case id: {identifier}")
        seen.add(identifier)
        if selected is not None and identifier not in selected:
            continue
        if quick and not bool(entry.get("quick")):
            continue
        result.append(PromptCase(identifier, category, text))

    if selected is not None:
        missing = selected - seen
        if missing:
            raise ValueError(
                f"unknown suite case(s): {', '.join(sorted(missing))}"
            )
    if not result:
        raise ValueError("suite selection produced no prompts")
    return result if limit is None else result[:limit]


def _rate(tokens: int, milliseconds: float) -> float | None:
    if tokens <= 0 or milliseconds <= 0.0:
        return None
    return tokens * 1000.0 / milliseconds


def _required_number(mapping: dict[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"terminal usage is missing numeric gufo.{name}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"terminal usage has invalid gufo.{name}")
    return value


def _required_integer(mapping: dict[str, Any], name: str) -> int:
    value = _required_number(mapping, name)
    if not value.is_integer():
        raise RuntimeError(f"terminal usage has non-integral gufo.{name}")
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
    case_id: str | None = None,
    category: str | None = None,
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
    completion_parts: list[str] = []
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
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        tool_calls = delta.get("tool_calls")
                        if isinstance(content, str) and content:
                            completion_parts.append(content)
                        if content or tool_calls:
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
    metrics = usage.get("gufo")
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
    draft_tokens = usage.get("draft_tokens", 0)
    draft_accepted_tokens = usage.get("draft_tokens_accepted", 0)
    if (
        isinstance(draft_tokens, bool)
        or not isinstance(draft_tokens, int)
        or draft_tokens < 0
        or isinstance(draft_accepted_tokens, bool)
        or not isinstance(draft_accepted_tokens, int)
        or draft_accepted_tokens < 0
        or draft_accepted_tokens > draft_tokens
    ):
        raise RuntimeError("terminal usage has invalid draft token counts")

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
            "terminal usage is missing string gufo.execution_plan"
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
        draft_tokens=draft_tokens,
        draft_accepted_tokens=draft_accepted_tokens,
        draft_acceptance=(
            draft_accepted_tokens / draft_tokens if draft_tokens else None
        ),
        case_id=case_id,
        category=category,
        completion_sha256=hashlib.sha256(
            "".join(completion_parts).encode("utf-8")
        ).hexdigest(),
        prefill_tokens_per_second=_rate(prefill_tokens, prefill_ms),
        decode_tokens_per_second=_rate(completion_tokens, decode_ms),
        whole_request_tokens_per_second=_rate(useful_tokens, wall_ms),
        started_at=started_at,
        finished_at=finished_at,
    )


def _round_from_samples(
    concurrency: int,
    repetition: int,
    samples: tuple[RequestObservation, ...],
) -> RoundObservation:
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
            client_id=f"gufo-bench-c{concurrency}-r{repetition}-i{index}",
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
    return _round_from_samples(concurrency, repetition, samples)


def _run_corpus_round(
    *,
    base_url: str,
    model: str,
    cases: list[PromptCase],
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    concurrency: int,
    repetition: int,
    group_index: int,
) -> RoundObservation:
    if len(cases) != concurrency:
        raise ValueError("corpus round must contain exactly C prompt cases")
    start_gate = threading.Barrier(concurrency + 1)

    def execute(index: int) -> RequestObservation:
        case = cases[index]
        return run_request(
            base_url=base_url,
            model=model,
            prompt=case.text,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            client_id=(
                f"gufo-corpus-c{concurrency}-r{repetition}-"
                f"g{group_index}-i{index}"
            ),
            concurrency=concurrency,
            repetition=repetition,
            request_index=index,
            case_id=case.identifier,
            category=case.category,
            start_gate=start_gate,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [executor.submit(execute, index) for index in range(concurrency)]
        start_gate.wait()
        samples = tuple(future.result() for future in futures)
    return _round_from_samples(concurrency, repetition, samples)


def _speculative_summary(
    samples: list[RequestObservation],
) -> dict[str, Any]:
    drafted = sum(sample.draft_tokens for sample in samples)
    accepted = sum(sample.draft_accepted_tokens for sample in samples)
    completion_tokens = sum(sample.completion_tokens for sample in samples)
    return {
        "draftedTokens": drafted,
        "acceptedTokens": accepted,
        "acceptance": accepted / drafted if drafted else None,
        "draftedPerOutputToken": (
            drafted / completion_tokens if completion_tokens else None
        ),
        "acceptedPerOutputToken": (
            accepted / completion_tokens if completion_tokens else None
        ),
        "requestAcceptance": distribution(
            sample.draft_acceptance for sample in samples
        ),
        "requestsWithDrafts": sum(sample.draft_tokens > 0 for sample in samples),
        "cacheHits": sum(sample.cache_hit for sample in samples),
    }


def _grouped_speculative_summary(
    samples: list[RequestObservation],
    attribute: str,
) -> dict[str, Any]:
    grouped: dict[str, list[RequestObservation]] = {}
    for sample in samples:
        value = getattr(sample, attribute)
        if value is not None:
            grouped.setdefault(value, []).append(sample)
    return {
        key: {
            **_speculative_summary(group),
            "decode_tokens_per_second": distribution(
                sample.decode_tokens_per_second for sample in group
            ),
            "completionHashes": sorted(
                {sample.completion_sha256 for sample in group}
            ),
            "completionHashCounts": dict(
                sorted(
                    collections.Counter(
                        sample.completion_sha256 for sample in group
                    ).items()
                )
            ),
        }
        for key, group in sorted(grouped.items())
    }


def _annotate_corpus_exactness(results: dict[str, Any]) -> None:
    baseline = results.get("c1")
    if not isinstance(baseline, dict):
        return
    baseline_cases = baseline.get("cases")
    if not isinstance(baseline_cases, dict):
        return
    references = {
        case_id: summary["completionHashes"][0]
        for case_id, summary in baseline_cases.items()
        if isinstance(summary, dict)
        and len(summary.get("completionHashes", [])) == 1
    }
    for result in results.values():
        if not isinstance(result, dict):
            continue
        cases = result.get("cases")
        if isinstance(cases, dict):
            for case_id, summary in cases.items():
                reference = references.get(case_id)
                hashes = summary.get("completionHashes", [])
                summary["matchesC1"] = (
                    reference is not None
                    and bool(hashes)
                    and all(value == reference for value in hashes)
                )
        compared = 0
        exact = 0
        for sample in result.get("samples", []):
            reference = references.get(sample.get("case_id"))
            if reference is None:
                continue
            compared += 1
            exact += sample.get("completion_sha256") == reference
        result["completionExactness"] = {
            "comparedRequests": compared,
            "exactRequests": exact,
            "exactRate": exact / compared if compared else None,
        }


def _summarize_rounds(
    rounds: list[RoundObservation],
) -> dict[str, Any]:
    samples = [sample for round_ in rounds for sample in round_.samples]
    span_ms = sum(round_.span_ms for round_ in rounds)
    prefill_tokens = sum(sample.prefill_tokens for sample in samples)
    output_tokens = sum(sample.completion_tokens for sample in samples)
    aggregate = {}
    for metric, tokens in (
        ("prefill_tokens_per_second", prefill_tokens),
        ("output_tokens_per_second", output_tokens),
        ("total_tokens_per_second", prefill_tokens + output_tokens),
    ):
        aggregate[metric] = distribution(
            getattr(round_, metric) for round_ in rounds
        )
        # A median of per-group rates gives fast and slow prompt groups equal
        # weight. Overall throughput accounts for all delivered tokens and time.
        aggregate[metric]["overall"] = (
            tokens * 1000.0 / span_ms if span_ms > 0.0 else None
        )
    result = {
        "requestCount": len(samples),
        "roundCount": len(rounds),
        "measuredSpanMs": span_ms,
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
        "aggregate": aggregate,
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
    result["speculative"] = _speculative_summary(samples)
    if any(sample.category is not None for sample in samples):
        result["categories"] = _grouped_speculative_summary(
            samples, "category"
        )
        result["cases"] = _grouped_speculative_summary(samples, "case_id")
    return result


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
                "overall: actual prefill plus completion tokens divided by "
                "the sum of measured round spans; other statistics describe "
                "individual rounds"
            ),
            "draft_acceptance": (
                "accepted support-model tokens divided by drafted "
                "support-model tokens"
            ),
        },
        "results": results,
        "warnings": warnings,
    }


def run_corpus_benchmark(
    *,
    base_url: str,
    model: str,
    cases: list[PromptCase],
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
    suite_bytes: bytes,
    corpus_layout: str = "distinct",
) -> dict[str, Any]:
    if not model:
        raise ValueError("model must not be empty")
    if not cases:
        raise ValueError("corpus must not be empty")
    if max_tokens <= 0 or repetitions <= 0 or warmup_rounds < 0:
        raise ValueError("token and repetition counts are invalid")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between zero and two")
    if corpus_layout not in {"distinct", "homogeneous"}:
        raise ValueError("corpus layout must be distinct or homogeneous")

    results: dict[str, Any] = {}
    warnings: list[str] = []
    for concurrency in concurrency_levels:
        if corpus_layout == "homogeneous":
            groups = [[case] * concurrency for case in cases]
            padded_count = len(cases) * concurrency
        else:
            padded_count = (
                (len(cases) + concurrency - 1) // concurrency
            ) * concurrency
            scheduled = [
                cases[index % len(cases)] for index in range(padded_count)
            ]
            groups = [
                scheduled[index : index + concurrency]
                for index in range(0, padded_count, concurrency)
            ]
        for warmup in range(warmup_rounds):
            _run_corpus_round(
                base_url=base_url,
                model=model,
                cases=groups[warmup % len(groups)],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                concurrency=concurrency,
                repetition=-(warmup + 1),
                group_index=warmup % len(groups),
            )

        rounds: list[RoundObservation] = []
        for repetition in range(repetitions):
            for group_index, group in enumerate(groups):
                rounds.append(
                    _run_corpus_round(
                        base_url=base_url,
                        model=model,
                        cases=group,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout_seconds=timeout_seconds,
                        concurrency=concurrency,
                        repetition=repetition,
                        group_index=group_index,
                    )
                )
        summary = _summarize_rounds(rounds)
        summary["corpusPromptCount"] = len(cases)
        summary["scheduledPromptCount"] = padded_count
        results[f"c{concurrency}"] = summary

        observed_capacity = max(summary["logicalConcurrency"])
        if observed_capacity < concurrency:
            warnings.append(
                f"C={concurrency} exceeded the server's reported logical "
                f"capacity {observed_capacity}; queueing is included"
            )
        cache_hits = summary["speculative"]["cacheHits"]
        if cache_hits:
            warnings.append(
                f"C={concurrency} observed {cache_hits} continuation-cache "
                "hits; use a fresh server for acceptance qualification"
            )
    _annotate_corpus_exactness(results)

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
            "mode": "corpus",
            "transport": "openai-chat-completions-sse",
            "suiteSha256": hashlib.sha256(suite_bytes).hexdigest(),
            "promptCount": len(cases),
            "caseIds": [case.identifier for case in cases],
            "categories": sorted({case.category for case in cases}),
            "corpusLayout": corpus_layout,
            "paddingPolicy": (
                "repeat-each-case"
                if corpus_layout == "homogeneous"
                else "cycle-from-start"
            ),
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
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
            "draft_acceptance": (
                "accepted support-model tokens divided by drafted "
                "support-model tokens"
            ),
            "aggregate_total_tokens_per_second": (
                "overall: actual prefill plus completion tokens divided by "
                "the sum of measured round spans; other statistics describe "
                "individual rounds"
            ),
            "completion_sha256": (
                "SHA-256 of generated text for cross-concurrency exactness "
                "checks; generated text itself is omitted"
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
        "aggregate tok/s  draft accepted"
    )
    for key, result in report["results"].items():
        concurrency = key[1:]
        latency = result["latency"]
        stage = result["stage"]
        aggregate = result["aggregate"]
        aggregate_rate = aggregate["total_tokens_per_second"]["overall"]
        speculative = result["speculative"]
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
            f"{'-' if aggregate_rate is None else f'{aggregate_rate:.2f}':>15}  "
            f"{speculative['acceptedTokens']:>5}/"
            f"{speculative['draftedTokens']:<5}"
        )
        categories = result.get("categories")
        if isinstance(categories, dict):
            for category, summary in categories.items():
                acceptance = summary["acceptance"]
                rendered = "-" if acceptance is None else f"{acceptance * 100:.1f}%"
                print(
                    f"   {category}: {rendered} "
                    f"({summary['acceptedTokens']}/"
                    f"{summary['draftedTokens']}), "
                    f"draft/output="
                    f"{summary['draftedPerOutputToken']:.2f}, "
                    f"decode p50={_median(summary['decode_tokens_per_second'])}"
                )
    for warning in report["warnings"]:
        print(f"warning: {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gufo-serving-bench",
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
    prompt.add_argument(
        "--suite",
        type=Path,
        help="Speculative corpus JSON; runs categorized synchronized requests",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        help="Repeat the canonical/custom prompt N times (default: 8)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Select a corpus case by id (repeatable)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Select only corpus cases marked quick",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of selected corpus cases",
    )
    parser.add_argument(
        "--corpus-layout",
        choices=("distinct", "homogeneous"),
        default="distinct",
        help=(
            "Pair different corpus cases or run C copies of each case "
            "(default: distinct)"
        ),
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
        help="Existing `gufo diagnose --fingerprint --json` output",
    )
    parser.add_argument(
        "--gufo",
        type=Path,
        default=Path("./result/bin/gufo"),
        help="Release gufo binary used for automatic fingerprinting",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON artifact")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt_repeat = args.prompt_repeat if args.prompt_repeat is not None else 8
    if prompt_repeat <= 0:
        raise SystemExit("--prompt-repeat must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.suite is None and (args.case or args.quick or args.limit is not None):
        raise SystemExit("--case/--quick/--limit require --suite")
    if args.suite is not None and args.prompt_repeat not in (None, 1):
        raise SystemExit("--prompt-repeat is not used with --suite")
    prompt = args.prompt or DEFAULT_PROMPT
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    prompt = "\n\n".join(prompt for _ in range(prompt_repeat))

    root = Path(__file__).resolve().parents[2]
    model = args.model or discover_model(args.base_url, args.timeout)
    fingerprint = load_fingerprint(args.fingerprint, args.gufo)
    revision, dirty = source_identity(root)
    if args.suite is None:
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
            prompt_repeat=prompt_repeat,
        )
    else:
        suite_bytes = args.suite.read_bytes()
        cases = load_prompt_suite(
            args.suite,
            quick=args.quick,
            limit=args.limit,
            selected=set(args.case) if args.case else None,
        )
        report = run_corpus_benchmark(
            base_url=args.base_url,
            model=model,
            cases=cases,
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
            suite_bytes=suite_bytes,
            corpus_layout=args.corpus_layout,
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
