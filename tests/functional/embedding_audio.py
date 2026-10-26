#!/usr/bin/env python3
"""Focused real HTTP audio embedding parity, bounds and recovery.

Uses the pinned official float32 goldens. Keeps every request/response and its
latency, including failures. No model or recording downloads.
"""
import argparse
import base64
import concurrent.futures
import io
import json
import math
from pathlib import Path
import struct
import socket
import time
import urllib.error
import urllib.request
import urllib.parse
import wave


def audio(spec, transport="data"):
    if spec["kind"] == "speech":
        data = base64.b64encode((Path(__file__).resolve().parent / "fixtures/audio/clear-english-voice.wav").read_bytes()).decode()
        return {"type": "input_audio", "input_audio": {transport:
            "data:audio/wav;base64," + data if transport == "url" else data, "format": "wav"}}
    samples = []
    for i in range(spec["samples"]):
        value = (min(i % 80, 80 - i % 80) - 20) * 512 if spec["kind"] == "triangle" else (
            ((i * 73 + i * i * 19) % 1024 - 512) * 32 if spec["kind"] == "noise" else 0)
        if spec["kind"] == "music":
            period = 64 if (i // 2000) % 2 == 0 else 80
            phase = i % period
            value = (min(phase, period - phase) - period // 4) * 512
        elif spec["kind"] == "environment":
            value = ((i * 73 + i * i * 19) % 1024 - 512) * 32 if i % 4000 < 800 else 0
        samples.append(value)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    data = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "input_audio", "input_audio": {
        transport: "data:audio/wav;base64," + data if transport == "url" else data,
        "format": "wav"}}


def content(case, transport="data"):
    clips = iter(case["clips"])
    pictures = iter(case["images"])
    import re
    parts = []
    for text in re.split(r"(<\|audio\|>|<\|image\|>)", case["text"]):
        if text == "<|audio|>":
            parts.append(audio(next(clips), transport))
        elif text == "<|image|>":
            from embedding_images import picture
            parts.append({"type": "image_url", "image_url": {"url": picture(next(pictures))}})
        elif text:
            parts.append({"type": "text", "text": text})
    return {"content": parts}


def check_vector(actual, expected):
    assert len(actual) == len(expected) and all(math.isfinite(x) for x in actual)
    norm = sum(x * x for x in actual)
    assert abs(norm - 1) < 1e-4, norm
    similarity = sum(x * y for x, y in zip(actual, expected)) / math.sqrt(norm * sum(y * y for y in expected))
    assert similarity > 0.999, similarity
    return similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/embedding_gemma2/audio_embedding_golden.json").read_text())
    cases = fixture["cases"]
    results = []
    report = {"status": "running", "requests": results,
              "reference": {key: fixture[key] for key in ("revision", "transformers_revision", "dtype")}}

    def post(inputs, **options):
        payload = {"model": args.model, "input": inputs, **options}
        request = urllib.request.Request(args.base_url.rstrip("/") + "/embeddings",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body, status, headers = json.load(response), response.status, dict(response.headers)
        except urllib.error.HTTPError as error:
            body, status, headers = json.load(error), error.code, dict(error.headers)
        results.append({"request": payload, "response": body, "status": status,
                        "headers": headers, "elapsed_ms": (time.perf_counter() - start) * 1000})
        return status, body

    try:
        # Cold audio load, warm reuse, short frames, silence, multiple clips,
        # text and audio/image interleaving through one model.
        for transport in ("data", "url"):
            for case in cases:
                status, result = post([content(case, transport)])
                assert status == 200, result
                assert len(result["data"]) == 1 and result["data"][0]["index"] == 0, result
                check_vector(result["data"][0]["embedding"], case["embedding"])
                assert result["usage"]["prompt_tokens"] == len(case["token_ids"]), result["usage"]
        # Mixed batching and input ordering, with repeated clips and requests.
        selected = [cases[2], cases[0], next(case for case in cases if case["name"] == "multiple_audio"), cases[3]]
        status, batch = post([content(case) for case in selected])
        assert status == 200, batch
        for i, case in enumerate(selected):
            assert batch["data"][i]["index"] == i
            check_vector(batch["data"][i]["embedding"], case["embedding"])
        text_fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/embedding_gemma2/embedding_golden.json").read_text())
        text_case = next(case for case in text_fixture["cases"] if case["name"] == "plain")
        status, mixed = post([text_case["text"], content(cases[2]), content(cases[0])])
        assert status == 200 and len(mixed["data"]) == 3, mixed
        for index, expected in enumerate((text_case["embedding"], cases[2]["embedding"], cases[0]["embedding"])):
            assert mixed["data"][index]["index"] == index
            check_vector(mixed["data"][index]["embedding"], expected)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda _: post([content(cases[2])]), range(3)))
        for status, result in responses:
            assert status == 200, result
            check_vector(result["data"][0]["embedding"], cases[2]["embedding"])
        status, encoded = post([content(cases[2])], dimensions=128, encoding_format="base64")
        assert status == 200, encoded
        vector = struct.unpack("<128f", base64.b64decode(encoded["data"][0]["embedding"]))
        check_vector(vector, cases[2]["embedding"][:128])
        bad_sources = ["!!!!", "data:audio/wav;base64,AAAA", "https://127.0.0.1/test.wav",
                       "file:///etc/passwd", "/etc/passwd", "http://example.com/test.wav"]
        for source in bad_sources:
            status, result = post([{"content": [{"type": "input_audio", "input_audio": {"url": source}}]}])
            assert status == 400 and result.get("error"), result
        status, result = post([{"content": [audio({"samples": 160, "kind": "silence"})]}])
        assert status == 400 and result.get("error"), result
        for truncate in (False, True):
            status, result = post([{"content": [{"type": "text", "text": " word" * 8190},
                                                 audio(cases[2])]}], truncate=truncate)
            assert status == 400 and result.get("error"), result
        # Truncation keeps complete audio features and discards trailing text.
        # An unrelated literal image token beyond the cut is ordinary text in
        # an audio-only input and must not prevent this valid truncation.
        prefix = [audio(cases[2]), {"type": "text", "text": " word" * 8190}]
        status, truncated = post([{"content": prefix}], truncate=True)
        assert status == 200 and truncated["usage"]["prompt_tokens"] == 8192, truncated
        status, extended = post([{"content": prefix + [{"type": "text", "text": "<|image|>"}]}], truncate=True)
        assert status == 200 and extended["usage"] == truncated["usage"], extended
        assert extended["data"] == truncated["data"], "discarded text cannot change the retained audio vector"
        # A literal placeholder split across adjacent text parts must not steal
        # any of the following clip's projected features.
        status, result = post([{"content": [{"type": "text", "text": "<|au"},
                                             {"type": "text", "text": "dio|>"}, audio(cases[2])]}])
        assert status == 400 and result.get("error"), result
        # Correlate disconnect with the server's actual request ID and terminal
        # cancellation event. A closed client socket alone cannot pass.
        import re
        before = len(args.server_log.read_text().splitlines())
        endpoint = urllib.parse.urlsplit(args.base_url)
        payload = json.dumps({"model": args.model, "input": [
            {"content": [audio(cases[2]) for _ in range(32)]}]}).encode()
        with socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=30) as client:
            client.sendall((f"POST {endpoint.path.rstrip('/')}/embeddings HTTP/1.1\r\n"
                            f"Host: {endpoint.netloc}\r\nContent-Type: application/json\r\n"
                            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode() + payload)
            time.sleep(0.02)
            client.shutdown(socket.SHUT_RDWR)
        deadline = time.monotonic() + 15
        cancellation = None
        while time.monotonic() < deadline:
            lines = args.server_log.read_text().splitlines()[before:]
            received = [re.search(r"request=(\S+)", line).group(1) for line in lines
                        if "event=received " in line and "path=/v1/embeddings " in line]
            if received:
                assert len(received) == 1, received
                cancellation = next((line for line in lines if f"request={received[0]} " in line
                                     and "event=completed " in line and "error_code=cancelled" in line), None)
                if cancellation:
                    break
            time.sleep(0.02)
        assert cancellation, "server must observe cancellation of the disconnected request"
        report["cancelled_request"] = {"request": json.loads(payload), "completion": cancellation}
        status, result = post([content(cases[2])])
        assert status == 200, result
        check_vector(result["data"][0]["embedding"], cases[2]["embedding"])
        report["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(report, indent=2))
    print(f"PASS audio embeddings: {len(results)} requests")


if __name__ == "__main__":
    main()
