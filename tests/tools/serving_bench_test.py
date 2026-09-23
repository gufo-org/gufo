#!/usr/bin/env python3

import json
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gufo import serving_bench
from gufo.model_bench.render import _serving_rate


def check(condition, message):
    if not condition:
        raise AssertionError(message)


GUFO_USAGE = {
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
}
OPENAI_USAGE = {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}
LLAMA_TIMINGS = {
    "prompt_n": 10,
    "prompt_ms": 20.0,
    "cache_n": 2,
    "predicted_n": 2,
    "predicted_ms": 10.0,
    "draft_n": 8,
    "draft_n_accepted": 4,
}


class FakeResponse:
    status = 200

    def __init__(self, usage=None, timings=None, content=("one", " two")):
        usage = GUFO_USAGE if usage is None else usage
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
        ]
        for part in content:
            chunks.append(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": part},
                            "finish_reason": None,
                        }
                    ]
                }
            )
        final = {"choices": [], "usage": usage}
        if timings is not None:
            final["timings"] = timings
        chunks.append(final)
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
sample = serving_bench.RequestObservation(
    **c4["samples"][0], started_at=0.0, finished_at=1.0
)
mixed_rounds = [
    serving_bench.RoundObservation(
        concurrency=1,
        repetition=0,
        span_ms=span_ms,
        prefill_tokens_per_second=sample.prefill_tokens * 1000.0 / span_ms,
        output_tokens_per_second=sample.completion_tokens * 1000.0 / span_ms,
        total_tokens_per_second=(
            sample.prefill_tokens + sample.completion_tokens
        ) * 1000.0 / span_ms,
        samples=(sample,),
    )
    for span_ms in (1000.0, 1000.0, 10000.0)
]
mixed = serving_bench._summarize_rounds(mixed_rounds)
check(mixed["measuredSpanMs"] == 12000.0, "all measured group spans count")
check(
    mixed["aggregate"]["output_tokens_per_second"]["overall"] == 0.5
    and mixed["aggregate"]["output_tokens_per_second"]["median"] == 2.0,
    "mixed-corpus throughput must account for slow groups",
)
check(
    mixed["aggregate"]["total_tokens_per_second"]["overall"] == 3.0,
    "overall throughput counts uncached prefill plus generated tokens",
)
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

corpus_rounds = []
original_corpus_round = serving_bench._run_corpus_round


def record_corpus_round(**kwargs):
    corpus_rounds.append(
        (
            kwargs["concurrency"],
            kwargs["repetition"],
            [case.identifier for case in kwargs["cases"]],
        )
    )
    return original_corpus_round(**kwargs)


serving_bench._run_corpus_round = record_corpus_round
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
        warmup_rounds=1,
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
    serving_bench._run_corpus_round = original_corpus_round
for concurrency, expected in [
    (1, ["repeat", "json", "code"]),
    (2, ["repeat", "json", "code", "repeat"]),
]:
    observed_rounds = [row for row in corpus_rounds if row[0] == concurrency]
    check(
        [
            case
            for _, repetition, cases in observed_rounds
            if repetition < 0
            for case in cases
        ]
        == expected,
        "warmup visits every scheduled prompt, including the padded tail",
    )
    phases = [repetition for _, repetition, _ in observed_rounds]
    check(phases == sorted(phases), "all warmups precede measurement")
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

# Endpoint profiles: plain OpenAI usage, llama-server timings, strict gufo.
def fake_urlopen_openai(request, timeout):
    del timeout
    body = json.loads(request.data)
    check(body.get("cache_prompt") is False, "cache_prompt=false is sent")
    return FakeResponse(usage=OPENAI_USAGE)


def fake_urlopen_llama(request, timeout):
    del timeout
    body = json.loads(request.data)
    check("cache_prompt" not in body, "cache_prompt is omitted by default")
    return FakeResponse(usage=OPENAI_USAGE, timings=LLAMA_TIMINGS)


def request_with(fake, **overrides):
    serving_bench.urllib.request.urlopen = fake
    try:
        return serving_bench.run_request(
            base_url="https://private.example",
            model="test-model",
            prompt="p",
            max_tokens=2,
            temperature=0.0,
            timeout_seconds=5.0,
            client_id="t",
            concurrency=1,
            repetition=0,
            request_index=0,
            **overrides,
        )
    finally:
        serving_bench.urllib.request.urlopen = original_urlopen


for api_key in ("", "benchmark-test-secret"):
    with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}):
        expected = f"Bearer {api_key}" if api_key else None

        def authenticated_completion(request, timeout):
            check(request.get_header("Authorization") == expected,
                  "generation uses the configured bearer credential")
            return fake_urlopen(request, timeout)

        observation = request_with(authenticated_completion)
        if api_key:
            check(api_key not in json.dumps(observation.public()),
                  "credentials are not included in benchmark observations")

        def authenticated_discovery(request, timeout):
            check(request.get_header("Authorization") == expected,
                  "model discovery uses the configured bearer credential")
            return io.BytesIO(b'{"data":[{"id":"test-model"}]}')

        with patch.object(serving_bench.urllib.request, "urlopen",
                          authenticated_discovery):
            check(serving_bench.discover_model("https://private.example", 5)
                  == "test-model", "authenticated model discovery succeeds")


plain = request_with(
    fake_urlopen_openai, endpoint_profile="openai", cache_prompt=False
)
check(plain.metrics_source == "none", "plain OpenAI usage has no metrics")
check(
    plain.decode_ms is None
    and plain.server_ttft_ms is None
    and plain.execution_plan is None
    and plain.decode_tokens_per_second is None,
    "server-side fields are None without endpoint metrics",
)
check(plain.prefill_tokens == 12, "prefill tokens fall back to prompt tokens")
check(plain.completion_tokens == 2 and plain.wall_ms >= 0.0, "client fields")

llama = request_with(fake_urlopen_llama, endpoint_profile="openai")
check(llama.metrics_source == "llama_timings", "llama timings are detected")
check(llama.decode_tokens_per_second == 200.0, "llama decode tg from timings")
check(llama.prefill_tokens_per_second == 500.0, "llama prefill from timings")
check(
    llama.draft_tokens == 8 and llama.draft_acceptance == 0.5,
    "llama draft acceptance from timings",
)
check(
    llama.cached_prompt_tokens == 2 and llama.cache_hit,
    "llama cache_n marks a prompt-cache hit",
)
check(llama.server_inter_token_ms == 10.0, "llama ITL derived from timings")

try:
    request_with(fake_urlopen_llama, endpoint_profile="gufo")
except RuntimeError as exception:
    check("incompatible schema" in str(exception), "gufo profile is strict")
else:
    raise AssertionError("gufo profile must reject a missing gufo block")

# Aggregate throughput: total completion tokens over the slowest request.
fast = serving_bench.RequestObservation(
    **{**plain.public(), "completion_tokens": 3}, started_at=0.0, finished_at=1.0
)
slow = serving_bench.RequestObservation(
    **{**plain.public(), "completion_tokens": 5}, started_at=0.0, finished_at=4.0
)
wave = serving_bench._round_from_samples(2, 0, (fast, slow))
check(wave.span_ms == 4000.0, "wave span is the slowest request")
check(wave.output_tokens_per_second == 2.0, "8 tokens over 4 s")
two_waves = serving_bench._summarize_rounds([wave, wave])
check(
    two_waves["aggregate"]["output_tokens_per_second"]["overall"] == 2.0,
    "aggregate output tok/s = total tokens / summed wave time",
)
check(
    two_waves["executionPlans"] == []
    and two_waves["logicalConcurrency"] == []
    and two_waves["metricsSources"] == ["none"],
    "summaries tolerate missing server metrics",
)
warnings = []
serving_bench._warn_capacity(warnings, 8, two_waves)
check(warnings == [], "no capacity warning without server metrics")

# Corpus run against a plain OpenAI endpoint works end to end.
serving_bench.urllib.request.urlopen = fake_urlopen_openai
try:
    openai_report = serving_bench.run_corpus_benchmark(
        base_url="https://private.example",
        model="test-model",
        cases=[
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
        endpoint_profile="openai",
        cache_prompt=False,
        notes=["llama-server -np 8"],
    )
finally:
    serving_bench.urllib.request.urlopen = original_urlopen
check(
    openai_report["workload"]["endpointProfile"] == "openai"
    and openai_report["workload"]["cachePrompt"] is False
    and openai_report["notes"] == ["llama-server -np 8"]
    and openai_report["reference"] == {"source": "self-c1"},
    "openai profile, cache flag, notes, and reference are recorded",
)
check(openai_report["warnings"] == [], "no spurious warnings without metrics")
with patch.object(serving_bench.urllib.request, "urlopen", fake_urlopen):
    try:
        serving_bench.run_corpus_benchmark(
            base_url="https://private.example", model="test-model",
            cases=[serving_bench.PromptCase("code", "code", "write C++")],
            workload_id="no-cache-contract", max_tokens=2, temperature=0.0,
            concurrency_levels=[1], warmup_rounds=0, repetitions=1,
            timeout_seconds=5, fingerprint=fingerprint,
            source_revision="a" * 40, source_dirty=False,
            suite_bytes=b"test", cache_prompt=False,
        )
        raise AssertionError("cache hits must reject a no-cache benchmark")
    except RuntimeError as error:
        check("ignored cache_prompt=false" in str(error),
              "cache-policy violation is explicit")
check(
    openai_report["results"]["c2"]["completionExactness"]["matchedRounds"] == 1,
    "fully matching rounds are counted",
)

# Reference hashes from another report flag differing outputs.
reference_hashes = serving_bench.reference_hashes_from_report(corpus_report)
check(
    reference_hashes["code"] == serving_bench.hashlib.sha256(b"one two").hexdigest(),
    "reference hashes come from the report's C=1 cases",
)


def fake_urlopen_diverging(request, timeout):
    del timeout
    body = json.loads(request.data)
    content = ("one", " two")
    if body["messages"][0]["content"] == "write C++":
        content = ("three",)
    return FakeResponse(usage=OPENAI_USAGE, content=content)


serving_bench.urllib.request.urlopen = fake_urlopen_diverging
try:
    diverging_report = serving_bench.run_corpus_benchmark(
        base_url="https://private.example",
        model="test-model",
        cases=[
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
        endpoint_profile="openai",
        reference={
            "source": "report",
            "workloadId": "test-corpus-v1",
            "suiteSha256": None,
            "modelId": "test-model",
            "hashes": reference_hashes,
        },
    )
finally:
    serving_bench.urllib.request.urlopen = original_urlopen
exactness = diverging_report["results"]["c2"]["completionExactness"]
check(
    exactness["exactRequests"] == 1
    and exactness["comparedRequests"] == 2
    and exactness["matchedCases"] == 1
    and exactness["comparedCases"] == 2,
    "differing outputs are flagged against the reference report",
)
check(
    exactness["matchedRounds"] == 0
    and exactness["matchedRoundOutputTokensPerSecond"] is None,
    "a round with any mismatch is excluded from the matched-subset rate",
)
check(
    diverging_report["results"]["c1"]["completionExactness"]["exactRequests"]
    == 1,
    "C=1 is compared against the external reference too",
)
check(
    diverging_report["results"]["c2"]["cases"]["code"]["matchesReference"]
    is False,
    "per-case reference flag",
)
check(
    diverging_report["reference"]["source"] == "report"
    and "hashes" not in diverging_report["reference"],
    "external reference identity is recorded without hashes",
)

# Suite category filters.
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
    json.dump(
        {
            "prompts": [
                {"id": "a", "category": "code", "text": "x"},
                {"id": "b", "category": "repetitive", "text": "y"},
                {"id": "c", "category": "reasoning", "text": "z"},
            ]
        },
        handle,
    )
    suite_path = Path(handle.name)
try:
    only = serving_bench.load_prompt_suite(
        suite_path, categories={"repetitive"}
    )
    check([case.identifier for case in only] == ["b"], "--category filter")
    rest = serving_bench.load_prompt_suite(
        suite_path, excluded_categories={"repetitive"}
    )
    check(
        [case.identifier for case in rest] == ["a", "c"],
        "--exclude-category filter",
    )
    try:
        serving_bench.load_prompt_suite(suite_path, categories={"missing"})
    except ValueError as exception:
        check("unknown suite category" in str(exception), "unknown category")
    else:
        raise AssertionError("unknown categories must be rejected")
finally:
    suite_path.unlink()

rate_report = {"results": {"c2": {
    "rounds": [{"sampleCount": 2}, {"sampleCount": 2}],
    "samples": [{"decode_tokens_per_second": value} for value in (10, 20, 30, 40)],
    "aggregate": {"output_tokens_per_second": {"overall": 1}},
}}}
check(_serving_rate(rate_report, 2) == 50,
      "model tables average the sum of individual decode rates per group")
rate_report["results"]["c2"]["samples"][0]["decode_tokens_per_second"] = None
check(_serving_rate(rate_report, 2) is None,
      "missing decode timings must not fall back to whole-request throughput")
for invalid in (float("nan"), float("inf"), -1):
    rate_report["results"]["c2"]["samples"][0]["decode_tokens_per_second"] = invalid
    check(_serving_rate(rate_report, 2) is None,
          "invalid decode timings must not produce a table rate")
rate_report["results"]["c2"]["samples"][0]["decode_tokens_per_second"] = 10
rate_report["results"]["c2"]["rounds"].pop()
check(_serving_rate(rate_report, 2) is None,
      "missing round metadata must not inflate summed decode rates")

print("Serving benchmark harness tests passed.")
