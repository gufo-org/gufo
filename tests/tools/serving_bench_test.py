#!/usr/bin/env python3

import json
import io
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gufo import serving_bench
from gufo.model_bench.charts import render_charts
from gufo.model_bench.config import BenchConfig, load_config
from gufo.model_bench.llm import Session, _measure_depth, run_loading, run_multi, run_single, run_table
from gufo.model_bench.render import _serving_rate, layout_for, parse_table, render_table
from gufo.model_bench.servers import Server


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
        == expected[:concurrency],
        "warmup uses only the first cohort",
    )
    check(
        [case for _, repetition, cases in observed_rounds if repetition >= 0 for case in cases]
        == expected,
        "measurement still visits every scheduled prompt, including the padded tail",
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

# Prepared cohorts finish every prompt before the measured phase. Explicit
# reference slots prevent two quick preparation requests from warming one slot.
prepared_slots = set()
preparation_lock = threading.Lock()


def prepared_response(request, timeout):
    del timeout
    body = json.loads(request.data)
    slot = body["id_slot"]
    check(body["cache_prompt"] is True, "both phases allow exact prefix reuse")
    if body["max_tokens"] in (0, 1):
        generated = body["max_tokens"]
        with preparation_lock:
            prepared_slots.add(slot)
        return FakeResponse(
            usage={**OPENAI_USAGE, "completion_tokens": generated},
            timings={**LLAMA_TIMINGS, "prompt_n": 12, "cache_n": 0, "predicted_n": generated},
            content=("one",) if generated else (),
        )
    with preparation_lock:
        check(prepared_slots == {0, 1}, "all slots are prepared before any tg request")
    return FakeResponse(usage=OPENAI_USAGE,
                        timings={**LLAMA_TIMINGS, "prompt_n": 4, "cache_n": 8})


prepared_args = dict(
    base_url="http://unused", model="test-model",
    cases=[serving_bench.PromptCase("same", "prose", "matching pp2048 prompt")],
    workload_id="prepared-decode", max_tokens=2, temperature=0.0,
    concurrency_levels=[2], warmup_rounds=1, repetitions=1,
    timeout_seconds=5, fingerprint=fingerprint,
    source_revision="a" * 40, source_dirty=False, suite_bytes=b"test",
    corpus_layout="homogeneous", endpoint_profile="openai",
    cache_prompt=True, prefill_first=True, pin_slots=True,
)
with patch.object(serving_bench.urllib.request, "urlopen", prepared_response):
    prepared_report = serving_bench.run_corpus_benchmark(**prepared_args)
check(len(prepared_report["results"]["c2"]["samples"]) == 2,
      "preparation is excluded from timed samples")
check(len(prepared_report["results"]["c2"]["preparations"][0]["samples"]) == 2,
      "preparation evidence is retained")
check(not prepared_report["warnings"], "intentional prepared cache hits are not warnings")
prepared_slots.clear()
with patch.object(serving_bench.urllib.request, "urlopen", prepared_response):
    zero_prepared = serving_bench.run_corpus_benchmark(**prepared_args, preparation_tokens=0)
check(all(s["completion_tokens"] == 0
          for s in zero_prepared["results"]["c2"]["preparations"][0]["samples"]),
      "zero-token preparation retains the prompt frontier without generating a token")
check(zero_prepared["workload"]["warmupMaxOutputTokens"] == 0,
      "the actual preparation policy is recorded")
with patch.object(serving_bench.urllib.request, "urlopen",
                  lambda request, timeout: FakeResponse(
                      usage={**OPENAI_USAGE,
                             "completion_tokens": json.loads(request.data)["max_tokens"]},
                      timings=LLAMA_TIMINGS)):
    try:
        serving_bench.run_corpus_benchmark(**prepared_args)
    except RuntimeError as error:
        check("prepared session was not reused" in str(error),
              "a cold replay cannot be labeled prepared decoding")
    else:
        raise AssertionError("prepared decoding must fail if the prefix is lost")

server = Server([], "/health", Path("/unused"))
server.process = MagicMock()
server.process.poll.return_value = None
server.stop = MagicMock()
server.__exit__(RuntimeError, RuntimeError("wrong cache frontier"), None)
check(server.failure_exit_code is None and server.stop.called,
      "benchmark failure must not become a server crash after intentional cleanup")
server.process.poll.return_value = -9
server.__exit__(RuntimeError, RuntimeError("connection closed"), None)
check(server.failure_exit_code == -9, "an actual process failure remains distinguishable")


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

chart_config = BenchConfig("test", ROOT, {"tables": {
    "single-ar": {"title": "AR"}, "multi-mixed": {"title": "Mixed"},
}})
chart_document = """<!-- bench:single-ar -->
| Depth | Gufo tg |
| ---: | ---: |
| 0 | 12 |
<!-- /bench -->

![AR](artifacts/charts/single-ar.svg)

<!-- bench:multi-mixed -->
| Users | Gufo AR |
| ---: | ---: |
| 2 | 24 |
<!-- /bench -->

![Mixed](artifacts/charts/multi-mixed.svg)
"""
with patch("gufo.model_bench.charts.chart_for", return_value=True) as draw:
    updated, written = render_charts(chart_config, chart_document, {"multi-mixed"})
    check(updated == chart_document, "partial rendering preserves untouched chart links")
    check(written == ["multi-mixed"], "partial rendering updates only the selected chart")
    repeated, _ = render_charts(chart_config, updated, {"multi-mixed"})
    check(repeated == updated, "repeated chart rendering is idempotent")
    check(all(call.args[1].id == "multi-mixed" for call in draw.call_args_list),
          "partial rendering does not redraw unrelated charts")

# Grouped comparisons retain separate workloads and never rerun AR performance.
with tempfile.TemporaryDirectory() as directory:
    config = load_config(ROOT, "qwen3.8-27b")
    config.artifacts_override = Path(directory)
    config.files = {"gguf": {quant: Path(__file__) for quant in ("q4", "q8")}}
    for quant in ("q4", "q8"):
        ar = config.table(f"multi-ar-{quant}")
        check(ar.spec["modes"] == ["ar"], "one dedicated AR concurrency sweep")
        check([c.header for c in layout_for(config, ar).columns[1:]] ==
              ["Gufo AR", "llama.cpp AR", "Gain"], "AR table has only AR columns")
        table = config.table(f"multi-dflash2-{quant}")
        workloads = table.workload_tables()
        check([w.id for w in workloads] == [f"multi-mixed-{quant}", f"multi-repetition-{quant}"],
              "merged presentation retains the existing artifact identities")
        layout = layout_for(config, table)
        check(f"Qwen27B {quant.upper()} DFlash2" in layout.columns[0].header,
              "first column identifies model, quantization and execution mode")
        check(all(c.unit == "tok/s" for c in layout.columns if c.owner in ("gufo", "reference")),
              "every throughput column states its unit")
        for workload in workloads:
            check(workload.spec["modes"] == ["dflash2"], "speculative tables never schedule AR")
            path = config.artifacts_dir / f"{workload.id}-gufo-ar.json"
            path.write_text(json.dumps({"results": {"c1": {"cases": {
                case: {"completionHashes": ["a" * 64]} for case in workload.spec["cases"]
            }}}}))
        # Only Gufo repetitive C4 and reference mixed C6 need new measurements.
        body = "| " + " | ".join(c.label for c in layout.columns) + " |\n"
        body += "| " + " | ".join(c.align for c in layout.columns) + " |\n"
        for users in table.spec["concurrency"]:
            cells = [str(users), "20", "TODO" if users == 6 else "10", "100%",
                     "TODO" if users == 4 else "40", "30", "33%"]
            body += "| " + " | ".join(cells) + " |\n"
        document = f"<!-- bench:{table.id} -->\n{body}<!-- /bench -->"
        for target in ("gufo", "reference"):
            session = Session(
                config, target, gufo_binary=Path("gufo"), reference_binary="llama-server",
                source={"revision": "a" * 40, "dirty": False}, fingerprint={},
                log_dir=Path(directory), document=document, todo_only=False, fresh=True,
            )
            session.server = MagicMock(return_value=MagicMock())
            session.reference_version = lambda mode: None
            with patch("gufo.model_bench.llm.run_corpus_benchmark") as bench, \
                    patch("gufo.model_bench.llm.wait_process_exit"), \
                    patch("gufo.model_bench.llm.save_artifact"), \
                    patch("sys.stdout", new=io.StringIO()):
                bench.side_effect = lambda **kw: {
                    "artifactType": "servingBenchmark",
                    "workload": {},
                    "results": {f"c{kw['concurrency_levels'][0]}": {}},
                }
                run_multi(session, table)
                check(bench.call_count == 2 * len(table.spec["concurrency"]),
                      "each workload/concurrency runs once without an AR sweep")
                check(all(call.kwargs["mode"] == "dflash2" for call in session.server.call_args_list),
                      "both engines start only DFlash2 servers")
                check(all(set(call.kwargs["reference"]["hashes"]) ==
                          {case.identifier for case in call.kwargs["cases"]}
                          for call in bench.call_args_list), "each workload keeps its own AR reference")
                session.todo_only = True
                session.server.reset_mock()
                bench.reset_mock()
                run_multi(session, table)
                check(bench.call_count == 1, "TODO refresh measures only the missing workload cell")
                expected = (f"multi-repetition-{quant}", 4) if target == "gufo" else (f"multi-mixed-{quant}", 6)
                call = session.server.call_args
                check((call.args[0].id, call.kwargs["sessions"]) == expected,
                      "merged TODO columns select the correct workload and concurrency")
            session.todo_only = False
            session.server.reset_mock()
            with patch("gufo.model_bench.llm.load_reference_report", return_value={"hashes": {}}):
                try:
                    run_multi(session, table)
                except RuntimeError as exception:
                    check("missing isolated AR completion hashes" in str(exception),
                          "missing quality references fail explicitly")
                else:
                    raise AssertionError("speculative runs require AR quality references")
            session.server.assert_not_called()
        with patch("gufo.model_bench.render.load_artifact", return_value=None) as load:
            rendered = render_table(config, table, None)
            check("Gufo AR" not in rendered and "llama.cpp AR" not in rendered,
                  "merged rendering cannot reintroduce AR performance columns")
            check(all(call.args[0].name.endswith("-dflash2.json") for call in load.call_args_list),
                  "performance rendering reads only the configured mode")
        single = render_table(config, config.table(f"single-dflash2-{quant}"), None)
        check("accepted/step" not in single and "tok/s" in single,
              "single-user tables show throughput units without acceptance columns")
        single_table = config.table(f"single-dflash2-{quant}")
        workloads = single_table.workload_tables()
        check([w.id for w in workloads] ==
              [f"single-dflash2-{quant}", f"single-dflash2-repetition-{quant}"],
              "single-user grouping preserves workload artifact identities")
        data = [
            {"pp": 100, "pp_sd": 2, "tg": 10},
            {"pp": 80, "pp_sd": 1, "tg": 8},
            {"pp": 90, "pp_sd": 7, "tg": 30},
            {"pp": 120, "pp_sd": 4, "tg": 20},
        ]
        for index, row in enumerate(data):
            target = ("gufo", "reference")[index % 2]
            path = config.artifacts_dir / f"{workloads[index // 2].id}-{target}.json"
            path.write_text(json.dumps({"rows": {"0": row}}))
        rendered = render_table(config, single_table, None)
        row = parse_table(rendered)["0"]
        check((row["Gufo pp"], row["llama.cpp pp"], row["Gain pp"]) ==
              ("100.00 ± 2.00", "120.00 ± 4.00", "-16.7%"),
              "pp selects each engine's maximum and its own deviation before computing gain")
        check((row["Gufo tg mixed"], row["Gain mixed"], row["Gufo tg repetitive"],
               row["Gain repetitive"]) == ("10.00", "+25.0%", "30.00", "+50.0%"),
              "generation and gains stay independent by text type")
        exclusions = config.artifacts_dir / "unavailable.json"
        exclusions.write_text(json.dumps({workloads[1].id: {"0": "unqualified comparison"}}))
        excluded = parse_table(render_table(config, single_table, None))["0"]
        check(excluded["llama.cpp tg repetitive"] == excluded["Gain repetitive"] == "N/A"
              and excluded["llama.cpp pp"] == "80.00 ± 1.00",
              "explicit exclusions override retained rates and cannot supply the shared pp maximum")
        exclusions.unlink()
        for target, metric, expected in [
            ("gufo", "Gufo tg repetitive", workloads[1]),
            ("reference", "llama.cpp tg mixed", workloads[0]),
        ]:
            layout = layout_for(config, single_table)
            cells = [c.label for c in layout.columns]
            document = "| " + " | ".join(cells) + " |\n"
            document += "| " + " | ".join(c.align for c in layout.columns) + " |\n"
            document += "| 0 | " + " | ".join(
                "TODO" if c.header == metric else row[c.header] for c in layout.columns[1:]
            ) + " |\n"
            document = f"<!-- bench:{single_table.id} -->\n{document}<!-- /bench -->"
            session = Session(
                config, target, gufo_binary=Path("gufo"), reference_binary="llama-server",
                source={}, fingerprint={}, log_dir=Path(directory), document=document,
                todo_only=True, depths=[0],
            )
            session.server = MagicMock(return_value=MagicMock())
            session.request = MagicMock()
            session.artifact = MagicMock(return_value={})
            session.store = MagicMock()
            observation = MagicMock(
                prefill_tokens_per_second=100.0, decode_tokens_per_second=30.0,
                draft_acceptance=None, draft_tokens=0, draft_accepted_tokens=0,
                completion_tokens=128, cached_prompt_tokens=0, prefill_tokens=2048,
            )
            with patch("gufo.model_bench.llm.Tokenizer") as tokenizer, \
                    patch("gufo.model_bench.llm._measure_depth", return_value=observation) as measure, \
                    patch("gufo.model_bench.llm.wait_process_exit"), \
                    patch("sys.stdout", new=io.StringIO()):
                tokenizer.return_value.words_for.return_value = 64
                tokenizer.return_value.ratio = 1.0
                tokenizer.return_value.overhead = 0
                run_single(session, single_table)
            check(measure.call_count == 1 and measure.call_args.kwargs["task"] == expected.spec["workload"],
                  "single-user TODO refresh measures only the missing workload")
            check(session.server.call_count == 1 and session.server.call_args.args[0].id == expected.id,
                  "single-user grouping starts no redundant server")
            check(session.store.call_args.args[0].name == f"{expected.id}-{target}.json",
                  "single-user refresh writes the original workload artifact")

# Loading provenance must identify the actual speculative reference binary.
with tempfile.TemporaryDirectory() as directory:
    config = load_config(ROOT, "qwen3.8-flash-next")
    config.artifacts_override = Path(directory)
    config.files = {"gguf": {"default": Path(__file__)}}
    session = Session(
        config, "reference", gufo_binary=Path("gufo"), reference_binary="llama-server",
        source={"revision": "a" * 40, "dirty": False}, fingerprint={},
        log_dir=Path(directory), document="", todo_only=False,
    )
    server = MagicMock(ready_seconds=0.25)
    server.command = ["llama-server-mtp"]
    session.server = MagicMock(return_value=server)
    session.reference_version = MagicMock(return_value="reference MTP version")
    session.store = MagicMock()
    with patch("gufo.model_bench.llm.drop_file_cache"), \
            patch("gufo.model_bench.llm.wait_process_exit"), \
            patch("sys.stdout", new=io.StringIO()):
        run_loading(session, config.table("loading"))
    check(session.store.call_args.args[1]["mode"] == "mtp",
          "loading records its actual speculative execution mode")
    session.reference_version.assert_called_once_with("mtp")

# Single-user d0 and concurrency C1 must send the same pp2048 prompt.
with tempfile.TemporaryDirectory() as directory:
    config = load_config(ROOT, "qwen3.8-flash-next")
    config.artifacts_override = Path(directory)
    config.files = {"gguf": {"default": Path(__file__)}}
    table = config.table("multi-ar")
    table.spec["concurrency"] = [1]
    for task in ("prose", "repetition"):
        table.spec["workload"] = task
        session = Session(
            config, "gufo", gufo_binary=Path("gufo"), reference_binary="llama-server",
            source={"revision": "a" * 40, "dirty": False}, fingerprint={},
            log_dir=Path(directory), document="", todo_only=False, fresh=True,
        )
        session.server = MagicMock(return_value=MagicMock())
        session.reference_version = lambda mode: None
        session.request = MagicMock(return_value=MagicMock(
            prefill_tokens=2048, cached_prompt_tokens=0, completion_tokens=128))
        tokenizer = MagicMock(overhead=12, ratio=1.3)
        _measure_depth(session, "http://unused", tokenizer, depth=0,
                       prompt_tokens=2048, output_tokens=128, fraction=0.005,
                       seed=1, repetition=0, task=task)
        single_prompt = session.request.call_args.args[1]
        report = {"artifactType": "servingBenchmark", "workload": {},
                  "results": {"c1": {"samples": [{"prompt_tokens": 2048, "prefill_tokens": 0,
                                                 "completion_tokens": 128}]}}}
        with patch("gufo.model_bench.llm.Tokenizer", return_value=tokenizer), \
                patch("gufo.model_bench.llm.run_corpus_benchmark", return_value=report) as bench, \
                patch("gufo.model_bench.llm.wait_process_exit"), \
                patch("gufo.model_bench.llm.save_artifact") as save, \
                patch("sys.stdout", new=io.StringIO()):
            run_multi(session, table)
            check(bench.call_args.kwargs["cases"][0].text == single_prompt,
                  "both benchmark paths send byte-identical mixed/repetitive prompts")
            check(bench.call_args.kwargs["cache_prompt"] is True
                  and bench.call_args.kwargs["prefill_first"],
                  "matched concurrency prepares all sessions before timing decoding")
            report["results"]["c1"]["samples"][0]["completion_tokens"] = 127
            save.reset_mock()
            try:
                run_multi(session, table)
            except RuntimeError as failure:
                check("incomplete generated output" in str(failure),
                      "short generations cannot become throughput results")
            else:
                raise AssertionError("incomplete benchmark output must fail")
            check(not save.called, "invalid measurements are not published")
            table.spec["modes"] = ["mtp"]
            ar_reference = Path(directory) / "multi-ar-gufo-ar.json"
            ar_reference.write_text(json.dumps({"results": {"c1": {"cases": {
                f"synthetic_{task}_pp2048": {"completionHashes": ["a" * 64]}
            }}}}))
            report["results"]["c1"]["samples"][0]["completion_tokens"] = 128
            report["results"]["c1"]["completionExactness"] = {"exactRate": 0.0}
            try:
                run_multi(session, table)
            except RuntimeError as failure:
                check("differs from its isolated AR reference" in str(failure),
                      "a target-output mismatch is a failed benchmark")
            else:
                raise AssertionError("wrong speculative output must fail")
            check(not save.called, "quality failures cannot become speed results")
            ar_reference.unlink()
            table.spec["modes"] = ["ar"]

# Optional reference builds use their native flags and reject unsupported metrics
# before spending time loading the model.
with tempfile.TemporaryDirectory() as directory:
    config = load_config(ROOT, "deepseek-v4-flash")
    config.files = {role: {"default": Path(directory) / role} for role in ("gguf", "dspark")}
    session = Session(
        config, "reference", gufo_binary=Path("gufo"), reference_binary="ds4-server",
        source={"revision": "a" * 40, "dirty": False}, fingerprint={},
        log_dir=Path(directory), document="", todo_only=False,
    )
    command = session.reference_command(config.table("multi-ar"), mode="ar",
                                        context=4 * 4096, parallel=4, port=8081)
    check(command[command.index("--ctx") + 1] == "4096"
          and command[command.index("--batched-session") + 1] == "4",
          "ds4 context is per session, not the total llama.cpp allocation")
    command = session.reference_command(config.table("single-dspark"), mode="dspark",
                                        context=4096, parallel=1, port=8081)
    check("--dspark" in command and "--mtp-model" in command and "--batched-session" not in command
          and "--alias" not in command and "-np" not in command,
          "C1 DSpark omits the upstream flag that silently disables speculation")
    try:
        session.reference_command(config.table("multi-dspark"), mode="dspark",
                                  context=8192, parallel=2, port=8081)
    except RuntimeError as failure:
        check("disables DSpark" in str(failure), "unsupported speculation must not become an AR baseline")
    else:
        raise AssertionError("the pinned reference cannot batch DSpark")
    session.active_log = Path(directory) / "server.log"
    session.active_log.write_text("")
    with patch("gufo.model_bench.llm.run_request") as request, \
            patch("ds4.server_metrics.request_metrics"):
        session.request("http://unused", "prompt", 1)
        check(request.call_args.kwargs["extra_body"]["thinking"] == {"type": "disabled"},
              "reference HTTP requests disable DeepSeek's default thinking")
    session.server = MagicMock()
    config.data["reference"]["stage_timings"] = False
    try:
        run_table(session, config.table("single-ar"))
    except RuntimeError as failure:
        check("stage timings" in str(failure), "missing reference metrics fail explicitly")
    else:
        raise AssertionError("missing reference stage timing must not become a benchmark")
    check(not session.server.called, "unsupported timing fails before model loading")

    config.artifacts_override = Path(directory)
    config.files = {"gguf": {"default": Path(__file__)}}
    table = config.table("multi-ar")
    table.spec["concurrency"] = [2]
    session.target = "gufo"
    session.server = MagicMock(return_value=MagicMock())
    session.reference_version = lambda mode: None
    phases = []
    session.request = lambda *args, **kwargs: phases.append("C1 checkpoint")
    report = {"artifactType": "servingBenchmark", "workload": {}, "results": {
        "c2": {"samples": [{"prompt_tokens": 2048, "completion_tokens": 128}] * 2}}}

    def prepared_cohort(**kwargs):
        phases.append("prepared cohort")
        check(kwargs["prefill_first"] and kwargs["preparation_tokens"] == 1,
              "Gufo DS4 still prepares every session before timing decoding")
        return report

    with patch("gufo.model_bench.llm.Tokenizer", return_value=MagicMock(overhead=4, ratio=1.143)), \
            patch("gufo.model_bench.llm.run_corpus_benchmark", side_effect=prepared_cohort), \
            patch("gufo.model_bench.llm.wait_process_exit"), \
            patch("gufo.model_bench.llm.save_artifact"), \
            patch("sys.stdout", new=io.StringIO()):
        run_multi(session, table)
    check(phases == ["C1 checkpoint", "prepared cohort"],
          "DS4 builds the scalar prompt checkpoint before concurrent preparation")
    check(report["results"]["c2"]["promptPreparation"] ==
          "C1 checkpoint followed by concurrent preparation",
          "the artifact records the prefill arithmetic control")

# An AR-only benchmark must not require the optional MTP build on PATH.
config = load_config(ROOT, "qwen3.8-flash-next")
config.files = {"gguf": {"default": Path("target.gguf")}}
session = Session(
    config, "reference", gufo_binary=Path("gufo"), reference_binary="llama-server",
    source={"revision": "a" * 40, "dirty": False}, fingerprint={},
    log_dir=Path("/tmp"), document="", todo_only=False,
)
with patch("gufo.model_bench.llm.shutil.which", return_value="/bin/llama-server") as which, \
        patch.object(session, "reference_version"):
    session.server(config.table("single-ar"), mode="ar", context=4096, sessions=1, tag="test")
    which.assert_called_once_with("llama-server")

# Existing server log timers must match HTTP counts; never infer stage time from wall time.
from ds4.server_metrics import request_metrics, cohort_metrics
log = (
    "0924 01:00:00 ds4-server: chat ctx=2..12:10 prompt done 0.020s\n"
    "0924 01:00:00 ds4-server: chat ctx=12..14:2 gen=2 decoding chunk=200.00 t/s avg=200.00 t/s 0.010s\n"
)
parsed = request_metrics(log, sample)
check(parsed.prefill_ms == 20 and parsed.decode_ms == 10, "native stage timers are milliseconds")
check(parsed.prefill_tokens_per_second == 500 and parsed.decode_tokens_per_second == 200,
      "native pp/tg use HTTP counts and independent timers")
for invalid in (log + log, log.replace("2..12:10", "0..12:12"), log.replace("0.010s", "0.000s")):
    try:
        request_metrics(invalid, sample)
    except RuntimeError:
        pass
    else:
        raise AssertionError("ambiguous, mismatched or invalid native timers must fail")
cohort = {"samples": [sample.public(), sample.public()], "rounds": [{"sampleCount": 2}]}
cohort_metrics(log + log.replace("0.010s", "0.020s"), cohort, 2)
check(_serving_rate({"results": {"c2": cohort}}, 2) == 300,
      "concurrency sums logged per-request rates without inventing client/log ID mapping")

print("Serving benchmark harness tests passed.")
