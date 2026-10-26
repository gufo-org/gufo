"""One quick real-model check per model family against an owned gufo process.

Standard library only, so a runner without pip packages (the Windows Devlab
host) can run it next to a downloaded gufo bundle:

    python tests/functional/family_smoke.py embedding \
        --binary gufo.exe --model <embeddinggemma-2 snapshot>

Each family runs gufo once and checks one property against a committed
reference: EmbeddingGemma 2 vectors against sentence-transformers goldens,
Qwen3-ASR against a reviewed transcript, Qwen3-TTS speech back through
Qwen3-ASR, and a Qwen LLM on one greedy arithmetic answer through
`gufo prompt`. The broader suites (embeddings.py, asr_service.py, ...) stay
the quality gate.
"""

import argparse
import io
import json
import math
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"
AUDIO = HERE / "fixtures" / "audio"
MINIMUM_EMBEDDING_COSINE = 0.9999
MAXIMUM_ASR_WER = 0.15
MAXIMUM_TTS_WER = 0.25
TTS_TEXT = "The little brown dog ran across the garden."


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url, payload=None, *, body=None, content_type=None, timeout=300, raw=False):
    if payload is not None:
        body, content_type = json.dumps(payload).encode(), "application/json"
    headers = {"Content-Type": content_type} if content_type else {}
    with urllib.request.urlopen(urllib.request.Request(url, body, headers),
                                timeout=timeout) as response:
        return response.read() if raw else json.load(response)


def multipart(fields, name, filename, data, mime):
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
             .encode() for key, value in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def word_error_rate(actual, expected):
    words = lambda text: re.findall(r"[^\W_]+", text.casefold())
    reference, hypothesis = words(expected), words(actual)
    previous = list(range(len(reference) + 1))
    for i, word in enumerate(hypothesis, 1):
        current = [i]
        for j, target in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != target)))
        previous = current
    return previous[-1] / len(reference)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


@contextmanager
def server(binary, service, model, extra, log_path, timeout):
    """Starts `gufo serve <service>` on a free port; yields its URL once ready."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = [str(binary), "serve", service, "--model", str(model), *extra,
               "--host", "127.0.0.1", "--port", str(port)]
    print("Starting:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{service} server exited with {process.returncode} "
                                       f"before /ready; see {log_path}")
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"{service} server not ready after {timeout} s")
                try:
                    request(url + "/ready", timeout=5)
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    time.sleep(1)
            print(f"{service} ready in {time.monotonic() - started:.1f} s", flush=True)
            yield url
            if process.poll() is not None:
                raise RuntimeError(f"{service} server exited during the check")
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


def check_embedding(args):
    golden = json.loads((FIXTURES / "embedding_gemma2" / "embedding_golden.json")
                        .read_text(encoding="utf-8"))
    cases = golden["cases"]
    with server(args.binary, "embedding", args.model, ["--served-model-name", "embeddinggemma-2"],
                args.log_dir / "embedding-server.log", args.startup_timeout) as url:
        result = request(url + "/v1/embeddings", {
            "model": "embeddinggemma-2", "input": [case["text"] for case in cases],
            "encoding_format": "float"})
    items = sorted(result["data"], key=lambda item: item["index"])
    assert len(items) == len(cases), f"{len(items)} embeddings for {len(cases)} inputs"
    worst = 1.0
    for case, item in zip(cases, items):
        vector = item["embedding"]
        assert len(vector) == len(case["embedding"]), f"{case['name']}: {len(vector)} dimensions"
        similarity = cosine(vector, case["embedding"])
        print(f"  {case['name']}: cosine {similarity:.6f}")
        worst = min(worst, similarity)
    assert worst >= MINIMUM_EMBEDDING_COSINE, f"worst cosine {worst:.6f} < {MINIMUM_EMBEDDING_COSINE}"
    return f"{len(cases)} golden texts, worst cosine {worst:.6f}"


def check_asr(args):
    audio = AUDIO / "clear-english-voice.wav"
    expected = audio.with_suffix(".txt").read_text(encoding="utf-8").strip()
    body, content_type = multipart({"model": "qwen3-asr", "language": "English",
                                    "max_tokens": "128"},
                                   "file", "speech.wav", audio.read_bytes(), "audio/wav")
    with server(args.binary, "asr", args.model, ["--served-model-name", "qwen3-asr"],
                args.log_dir / "asr-server.log", args.startup_timeout) as url:
        result = request(url + "/v1/audio/transcriptions", body=body, content_type=content_type)
    wer = word_error_rate(result["text"], expected)
    print(f"  transcript: {result['text']!r}")
    assert wer <= MAXIMUM_ASR_WER, f"WER {wer:.3f} > {MAXIMUM_ASR_WER}"
    return f"WER {wer:.3f} against the reviewed transcript"


def check_tts(args):
    reference = AUDIO / "clear-english-voice.wav"
    voice = ["--served-model-name", "qwen3-tts", "--voice", f"narrator={reference}",
             "--voice-text", f"narrator={reference.with_suffix('.txt')}"]
    with server(args.binary, "tts", args.model, voice, args.log_dir / "tts-server.log",
                args.startup_timeout) as url:
        data = request(url + "/v1/audio/speech", {
            "model": "qwen3-tts", "voice": "narrator", "input": TTS_TEXT, "seed": 42,
            "language": "English", "max_new_tokens": 180}, raw=True)
    with wave.open(io.BytesIO(data)) as audio:
        shape = audio.getnchannels(), audio.getsampwidth(), audio.getframerate()
        assert shape == (1, 2, 24000), f"WAV format {shape}"
        pcm = audio.readframes(audio.getnframes())
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    duration = len(samples) / 24000
    rms = math.sqrt(sum((sample / 32768) ** 2 for sample in samples) / max(len(samples), 1))
    assert 0.3 <= duration <= 20, f"duration {duration:.2f} s"
    assert rms > 0.001, f"silent audio (RMS {rms:.5f})"
    summary = f"{duration:.2f} s of 24 kHz speech, RMS {rms:.4f}"
    if args.asr_model:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "speech.wav"
            path.write_bytes(data)
            result = subprocess.run([str(args.binary), "transcribe", "--model", str(args.asr_model),
                                     "--audio", str(path), "--format", "json"],
                                    capture_output=True, timeout=300)
        assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
        transcript = json.loads(result.stdout)["text"]
        wer = word_error_rate(transcript, TTS_TEXT)
        print(f"  transcript: {transcript!r}")
        assert wer <= MAXIMUM_TTS_WER, f"round-trip WER {wer:.3f} > {MAXIMUM_TTS_WER}"
        summary += f", round-trip WER {wer:.3f}"
    return summary


def check_llm(args):
    # gufo prompt streams the weights once; it fits hosts whose OS memory
    # pool is smaller than the serve path's host copy.
    command = [str(args.binary), "prompt", "--model", str(args.model),
               "--prompt", "What is 6 times 7? Answer with the number only.",
               "--temperature", "0", "--max-tokens", "32", "--think", "off",
               "--speculative", "off", "--no-display-prompt"]
    print("Running:", " ".join(command), flush=True)
    log_path = args.log_dir / "llm-prompt.log"
    result = subprocess.run(command, capture_output=True, timeout=args.startup_timeout)
    log_path.write_bytes(result.stdout + b"\n--- stderr ---\n" + result.stderr)
    assert result.returncode == 0, f"gufo prompt exited with {result.returncode}; see {log_path}"
    answer = result.stdout.decode("utf-8", errors="replace").strip()
    print(f"  answer: {answer!r}")
    assert re.search(r"\b42\b", answer), "the greedy answer does not contain 42"
    return "greedy 6 x 7 = 42"


FAMILIES = {"embedding": check_embedding, "asr": check_asr, "tts": check_tts, "llm": check_llm}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="Model directory, or the GGUF file for llm")
    parser.add_argument("--asr-model", type=Path, help="tts: Qwen3-ASR directory for a round trip")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--log-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    try:
        summary = FAMILIES[args.family](args)
    except Exception as error:
        print(f"FAIL {args.family}: {error}", file=sys.stderr)
        for log in sorted(args.log_dir.glob(f"{args.family}-*.log")):
            print(f"--- {log.name} (tail) ---", file=sys.stderr)
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
        return 1
    print(f"PASS {args.family}: {summary} ({time.monotonic() - started:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
