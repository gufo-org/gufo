"""Read the pinned antirez server's existing stage timers; no wall-time estimates."""

from __future__ import annotations

from dataclasses import replace
import math
import re

from gufo.serving_bench import RequestObservation

PREFILL = re.compile(
    r"ds4-server: chat ctx=(\d+)\.\.(\d+):(\d+).*? prompt done ([\d.]+)s$")
DECODE = re.compile(
    r"ds4-server: chat ctx=\d+\.\.\d+:\d+ gen=(\d+).*? "
    r"decoding chunk=[\d.]+ t/s avg=[\d.]+ t/s ([\d.]+)s$")


def decode_durations(log: str, tokens: int) -> list[float]:
    values = [float(match[2]) * 1000 for line in log.splitlines()
              if (match := DECODE.search(line)) and int(match[1]) == tokens]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("antirez reported an invalid decode duration")
    return values


def request_metrics(log: str, sample: RequestObservation) -> RequestObservation:
    prefill = [match for line in log.splitlines() if (match := PREFILL.search(line))]
    decode = decode_durations(log, sample.completion_tokens)
    if len(prefill) != 1 or len(decode) != 1:
        raise RuntimeError("cannot identify exactly one antirez prefill/decode timer")
    start, end, count = map(int, prefill[0].groups()[:3])
    if (start, end, count) != (sample.cached_prompt_tokens, sample.prompt_tokens, sample.prefill_tokens):
        raise RuntimeError("antirez log frontier disagrees with HTTP token/cache counts")
    pp_ms = float(prefill[0][4]) * 1000
    if not math.isfinite(pp_ms) or pp_ms < 0 or (sample.prefill_tokens and pp_ms == 0):
        raise RuntimeError("antirez reported an invalid prefill duration")
    return replace(
        sample, prefill_ms=pp_ms, decode_ms=decode[0], metrics_source="antirez-server-log",
        prefill_tokens_per_second=sample.prefill_tokens * 1000 / pp_ms if pp_ms else None,
        decode_tokens_per_second=sample.completion_tokens * 1000 / decode[0],
    )


def cohort_metrics(log: str, result: dict, tokens: int) -> None:
    """Keep each request's timer without inventing a log-to-client ID mapping."""
    durations = decode_durations(log, tokens)
    if len(durations) != len(result["samples"]):
        raise RuntimeError("antirez decode timer count does not match the completed cohort")
    result["loggedDecodeTimers"] = {
        "source": "antirez-server-log", "tokens_per_request": tokens,
        "durations_ms": durations,
    }
