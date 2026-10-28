"""EmbeddingGemma 2 end-to-end checks over HTTP with the OpenAI SDK.

Runs against an already running `gufo serve embedding` and replays the probe
texts of tests/fixtures/embedding_gemma2/embedding_golden.json, which hold
sentence-transformers float32 reference vectors:

    python3 tests/functional/embeddings.py --base-url http://127.0.0.1:8080/v1 \
        --model embeddinggemma-2

It checks cosine agreement with the reference, input order in a batched
request, Matryoshka truncation, concurrent identical requests, and the error
shapes. The SDK requests base64 by default, so the agreement check covers that
encoding. With `"truncate": true` an over-length input must embed exactly like
its longest prefix that fits the 8192-token limit unchanged. A
`prompt_name` must embed like the same prompt written by the client.
"""

import argparse
import base64
import concurrent.futures
import json
import math
import struct
import sys
import threading
import time
from pathlib import Path

FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures" /
           "embedding_gemma2" / "embedding_golden.json")
MINIMUM_COSINE = 0.9999
MAX_INPUT_TOKENS = 8192
LONG_SENTENCE = ("The northern lights, or aurora borealis, are produced when "
                 "charged particles collide with gases. ")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def truncated(vector, size):
    head = vector[:size]
    norm = math.sqrt(sum(x * x for x in head))
    return [x / norm for x in head]


def check(name):
    print(f"CHECK {name}", file=sys.stderr, flush=True)


def expect_error(openai, call, status, code, name):
    try:
        call()
    except openai.APIStatusError as error:
        assert error.status_code == status, (name, error.status_code)
        assert error.body["code"] == code, (name, error.body)
    else:
        raise AssertionError(f"{name} unexpectedly succeeded")


def check_truncate(openai, client, model):
    """#726: `truncate` cuts an over-length input to <bos>, its first tokens
    and <eos>; the default and `false` keep refusing it."""
    long_text = LONG_SENTENCE * 700

    def embed(text, **body):
        return client.embeddings.create(model=model, input=text,
                                        encoding_format="float",
                                        extra_body=body)

    for name, body in (("by default", {}), ("with false", {"truncate": False})):
        check(f"truncate: an over-length input is refused {name}")
        expect_error(openai, lambda: embed(long_text, **body), 400,
                     "context_length_exceeded", f"over-length {name}")

    check("truncate: an over-length input embeds the limit's tokens")
    cut = embed(long_text, truncate=True)
    assert cut.usage.prompt_tokens == MAX_INPUT_TOKENS, cut.usage

    # The longest character prefix that the server accepts without truncation
    # holds the same tokens, so it must give the same vector. A refused prefix
    # returns quickly, which keeps the search cheap.
    check("truncate: the vector equals the longest prefix that fits")
    low, high = 0, len(long_text)
    while low < high:
        middle = (low + high + 1) // 2
        try:
            embed(long_text[:middle])
            low = middle
        except openai.BadRequestError:
            high = middle - 1
    prefix = embed(long_text[:low])
    assert prefix.usage.prompt_tokens == MAX_INPUT_TOKENS, prefix.usage
    assert cosine(cut.data[0].embedding,
                  prefix.data[0].embedding) > 0.99999

    check("truncate: a batch truncates only the over-length input")
    short = embed("hello")
    batch = embed(["hello", long_text], truncate=True)
    assert [item.index for item in batch.data] == [0, 1]
    assert batch.usage.prompt_tokens == (short.usage.prompt_tokens +
                                         MAX_INPUT_TOKENS), batch.usage
    assert cosine(batch.data[0].embedding, short.data[0].embedding) > 0.99999
    assert cosine(batch.data[1].embedding, cut.data[0].embedding) > 0.99999

    check("truncate: a non-boolean value is refused")
    expect_error(openai, lambda: embed("hello", truncate="yes"), 400,
                 "invalid_parameter_type", "string truncate")


def check_prompt_names(openai, client, model):
    """#723: `prompt_name` adds a prompt from the checkpoint's
    config_sentence_transformers.json, which embeds exactly like the same
    prefix written by the client."""
    text = "what causes the northern lights"

    def embed(value, **body):
        return client.embeddings.create(model=model, input=value,
                                        encoding_format="float",
                                        extra_body=body)

    for name, prefix in (("query", "task: search result | query: "),
                         ("document", "title: none | text: ")):
        check(f"prompt_name: {name} equals the client-written prefix")
        named = embed(text, prompt_name=name)
        written = embed(prefix + text)
        assert named.usage.prompt_tokens == written.usage.prompt_tokens, (
            named.usage, written.usage)
        assert cosine(named.data[0].embedding,
                      written.data[0].embedding) > 0.99999

    check("prompt_name: an empty name applies no prompt")
    plain = embed(text)
    empty = embed(text, prompt_name="")
    assert empty.usage.prompt_tokens == plain.usage.prompt_tokens
    assert cosine(empty.data[0].embedding, plain.data[0].embedding) > 0.99999

    check("prompt_name: an unknown name is refused")
    expect_error(openai, lambda: embed(text, prompt_name="no-such-prompt"), 400,
                 "invalid_prompt_name", "unknown prompt_name")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--output", type=Path, help="Retain every response and request timing")
    args = parser.parse_args()

    import openai

    import httpx

    requests = []
    lock = threading.Lock()

    def started(request):
        request.extensions["functional_started"] = time.perf_counter()

    def completed(response):
        response.read()
        record = {"status": response.status_code,
                  "elapsed_ms": (time.perf_counter() - response.request.extensions[
                      "functional_started"]) * 1000,
                  "request": json.loads(response.request.content) if response.request.content else None,
                  "response": response.json()}
        with lock:
            requests.append(record)
            if args.output:
                args.output.write_text(json.dumps({"requests": requests}, indent=2) + "\n")

    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0,
                          http_client=httpx.Client(timeout=120, trust_env=False,
                              event_hooks={"request": [started], "response": [completed]}))
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    texts = [case["text"] for case in cases]
    reference = [case["embedding"] for case in cases]

    check("batched request matches the reference in input order")
    response = client.embeddings.create(model=args.model, input=texts)
    assert [item.index for item in response.data] == list(range(len(texts)))
    for case, item, expected in zip(cases, response.data, reference):
        assert len(item.embedding) == 768, case["name"]
        score = cosine(item.embedding, expected)
        assert score >= MINIMUM_COSINE, (case["name"], score)
    assert response.usage.prompt_tokens == sum(
        len(case["token_ids"]) for case in cases), response.usage
    assert response.model == args.model, response.model

    check("single string input equals the batched result")
    single = client.embeddings.create(model=args.model, input=texts[0])
    assert cosine(single.data[0].embedding,
                  response.data[0].embedding) >= MINIMUM_COSINE

    check("float encoding equals base64 encoding")
    as_float = client.embeddings.create(model=args.model, input=texts[:2],
                                        encoding_format="float")
    for floats, decoded in zip(as_float.data, response.data[:2]):
        assert cosine(floats.embedding, decoded.embedding) > 0.99999

    check("raw base64 payload is little-endian float32")
    raw = client.embeddings.with_raw_response.create(
        model=args.model, input=texts[2], encoding_format="base64").http_response
    payload = raw.json()["data"][0]["embedding"]
    values = struct.unpack("<768f", base64.b64decode(payload))
    assert cosine(values, response.data[2].embedding) > 0.99999

    for size in (128, 256, 512):
        check(f"dimensions={size} truncates and renormalizes")
        shorter = client.embeddings.create(model=args.model, input=texts[:3],
                                           dimensions=size)
        for item, expected in zip(shorter.data, reference):
            assert len(item.embedding) == size
            assert math.isclose(sum(x * x for x in item.embedding), 1.0,
                                rel_tol=1e-4)
            assert cosine(item.embedding, truncated(expected, size)) \
                >= MINIMUM_COSINE

    check("concurrent identical requests agree")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: client.embeddings.create(model=args.model, input=texts),
            range(8)))
    # Concurrent requests share device batches, and a different batch shape
    # can pick a different GEMM kernel, so the vectors match only to rounding.
    worst = min(cosine(item.embedding, expected.embedding)
                for result in results
                for item, expected in zip(result.data, response.data))
    assert worst > MINIMUM_COSINE, worst
    check(f"concurrent requests match a lone request, worst cosine {worst:.8f}")

    check("model listing names only the embedding model")
    models = client.models.list()
    assert [m.id for m in models.data] == [args.model], models

    for name, kwargs, status, code in (
        ("empty input", {"input": ""}, 400, "empty_input"),
        ("token-array input", {"input": [1, 2, 3]}, 400, "unsupported_input"),
        ("bad dimensions", {"input": "x", "dimensions": 100}, 400,
         "invalid_dimensions"),
        ("unknown model", {"input": "x", "model": "no-such-model"}, 404,
         "model_not_found"),
    ):
        check(f"error: {name}")
        kwargs.setdefault("model", args.model)
        expect_error(openai, lambda: client.embeddings.create(**kwargs),
                     status, code, name)

    check_truncate(openai, client, args.model)
    check_prompt_names(openai, client, args.model)

    print("PASS embeddings")


if __name__ == "__main__":
    main()
