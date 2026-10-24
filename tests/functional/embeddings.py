"""EmbeddingGemma 2 end-to-end checks over HTTP with the OpenAI SDK.

Runs against an already running `gufo serve embedding` and replays the probe
texts of tests/fixtures/embedding_gemma2/embedding_golden.json, which hold
sentence-transformers float32 reference vectors:

    python3 tests/functional/embeddings.py --base-url http://127.0.0.1:8080/v1 \
        --model embeddinggemma-2

It checks cosine agreement with the reference, input order in a batched
request, Matryoshka truncation, concurrent identical requests, and the error
shapes. The SDK requests base64 by default, so the agreement check covers that
encoding.
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


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def truncated(vector, size):
    head = vector[:size]
    norm = math.sqrt(sum(x * x for x in head))
    return [x / norm for x in head]


def check(name):
    print(f"CHECK {name}", file=sys.stderr, flush=True)


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
    for result in results:
        for item, expected in zip(result.data, response.data):
            assert cosine(item.embedding, expected.embedding) > 0.99999

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
        try:
            client.embeddings.create(**kwargs)
        except openai.APIStatusError as error:
            assert error.status_code == status, (name, error.status_code)
            assert error.body["code"] == code, (name, error.body)
        else:
            raise AssertionError(f"{name} unexpectedly succeeded")

    print("PASS embeddings")


if __name__ == "__main__":
    main()
