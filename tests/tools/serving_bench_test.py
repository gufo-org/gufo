#!/usr/bin/env python3

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gufo import serving_bench


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeResponse:
    status = 200

    def __init__(self):
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "one"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " two"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 2,
                    "total_tokens": 14,
                    "draft_tokens": 8,
                    "draft_tokens_accepted": 4,
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "gufo": {
                        "cache_hit": True,
                        "prefill_tokens": 10,
                        "prefill_ms": 20.0,
                        "decode_ms": 10.0,
                        "queue_ms": 1.0,
                        "ttft_ms": 21.0,
                        "mean_inter_token_ms": 5.0,
                        "max_inter_token_ms": 6.0,
                        "requested_logical_concurrency": 4,
                        "physical_execution_width": 1,
                        "execution_plan": "serial-fallback",
                    },
                },
            },
        ]
        self.lines = []
        for chunk in chunks:
            self.lines.extend(
                [f"data: {json.dumps(chunk)}\n".encode(), b"\n"]
            )
        self.lines.extend([b"data: [DONE]\n", b"\n"])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


def fake_urlopen(request, timeout):
    del timeout
    body = json.loads(request.data)
    check(body["stream"], "benchmark requests must stream")
    check(
        body["stream_options"]["include_usage"],
        "benchmark requests must request terminal usage",
    )
    return FakeResponse()


original_urlopen = serving_bench.urllib.request.urlopen
serving_bench.urllib.request.urlopen = fake_urlopen
try:
    fingerprint = {
        "schemaVersion": "1.0.0",
        "fingerprintId": "0" * 64,
        "canonical": {},
    }
    report = serving_bench.run_benchmark(
        base_url="https://private.example",
        model="test-model",
        prompt="public benchmark prompt",
        workload_id="test-v1",
        max_tokens=2,
        temperature=0.0,
        concurrency_levels=[1, 2, 4],
        warmup_rounds=0,
        repetitions=1,
        timeout_seconds=5.0,
        fingerprint=fingerprint,
        source_revision="a" * 40,
        source_dirty=False,
    )
finally:
    serving_bench.urllib.request.urlopen = original_urlopen

check(report["artifactType"] == "servingBenchmark", "artifact type")
check(report["benchmarkSchema"] == "gufo.serving-benchmark.v1", "schema")
check(
    sorted(report["results"]) == ["c1", "c2", "c4"],
    "C=1/C=2/C=4 summaries",
)

c4 = report["results"]["c4"]
check(
    c4["stage"]["prefill_tokens_per_second"]["median"] == 500.0,
    "prefill throughput is reported independently",
)
check(
    c4["stage"]["decode_tokens_per_second"]["median"] == 200.0,
    "decode tg is reported independently",
)
check(
    c4["latency"]["server_inter_token_ms"]["median"] == 5.0,
    "server token ITL is retained",
)
check(
    c4["aggregate"]["total_tokens_per_second"]["median"] > 0.0,
    "aggregate whole-request throughput is reported",
)
check(len(c4["samples"]) == 4, "raw per-request samples are retained")
check(c4["speculative"]["draftedTokens"] == 32, "draft tokens aggregate")
check(c4["speculative"]["acceptedTokens"] == 16, "accepted tokens aggregate")
check(c4["speculative"]["acceptance"] == 0.5, "aggregate acceptance")
check(
    c4["speculative"]["draftedPerOutputToken"] == 4.0,
    "drafted tokens per output token",
)
check(
    c4["speculative"]["acceptedPerOutputToken"] == 2.0,
    "accepted tokens per output token",
)
check(
    c4["samples"][0]["completion_sha256"]
    == serving_bench.hashlib.sha256(b"one two").hexdigest(),
    "completion text is retained only as a hash",
)

serving_bench.urllib.request.urlopen = fake_urlopen
try:
    corpus_report = serving_bench.run_corpus_benchmark(
        base_url="https://private.example",
        model="test-model",
        cases=[
            serving_bench.PromptCase("repeat", "repetitive", "red blue blue"),
            serving_bench.PromptCase("json", "structured", "return JSON"),
            serving_bench.PromptCase("code", "code", "write C++"),
        ],
        workload_id="test-corpus-v1",
        max_tokens=2,
        temperature=0.0,
        concurrency_levels=[1, 2],
        warmup_rounds=0,
        repetitions=1,
        timeout_seconds=5.0,
        fingerprint=fingerprint,
        source_revision="a" * 40,
        source_dirty=False,
        suite_bytes=b"test suite",
        corpus_layout="distinct",
    )
finally:
    serving_bench.urllib.request.urlopen = original_urlopen
c2_corpus = corpus_report["results"]["c2"]
check(c2_corpus["requestCount"] == 4, "corpus tail is cycle-padded")
check(
    sorted(c2_corpus["categories"]) == ["code", "repetitive", "structured"],
    "corpus categories are summarized independently",
)
check(
    c2_corpus["cases"]["repeat"]["draftedTokens"] == 16,
    "cycle-padded cases retain per-case draft totals",
)
check(
    corpus_report["workload"]["suiteSha256"]
    == serving_bench.hashlib.sha256(b"test suite").hexdigest(),
    "corpus identity is content-addressed",
)
check(
    c2_corpus["completionExactness"]["exactRate"] == 1.0,
    "concurrent completion hashes match the C1 references",
)
check(
    c2_corpus["cases"]["repeat"]["completionHashCounts"]
    == {
        serving_bench.hashlib.sha256(b"one two").hexdigest(): 2,
    },
    "case summaries retain completion-hash multiplicity",
)
check(
    corpus_report["workload"]["corpusLayout"] == "distinct",
    "corpus layout is recorded",
)

serialized = json.dumps(report)
check("public benchmark prompt" not in serialized, "prompt text is redacted")
check("private.example" not in serialized, "endpoint host is redacted")
check("/home/" not in serialized, "local paths are absent")
check("one two" not in serialized, "generated text is omitted")

check(serving_bench.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5, "p50")
check(
    serving_bench.parse_concurrency_levels("1,2,4") == [1, 2, 4],
    "concurrency parser",
)

print("Serving benchmark harness tests passed.")
