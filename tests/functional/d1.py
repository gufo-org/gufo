#!/usr/bin/env python3
"""Opt-in d1 GGUF HTTP parity, isolation and per-request latency checks.

Run both servers with the same Q8_0 backbone, BF16 mmproj and context. The
reference is llama.cpp 8642541b255bfb5dc39cd351a178be09013a55d6. Its Q8
activation quantization differs from Gufo's split-FP16 projections; the fixed
probability gate is 0.02 absolute, with identical discrete answers and token
counts. This bounded check does not establish downstream task accuracy.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="http://127.0.0.1:18886")
    parser.add_argument("--reference", default="http://127.0.0.1:18887")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image, audio = uri(args.image, "image/png"), uri(args.audio, "audio/wav")
    reports = []
    errors = []

    def call(server, name, request, expected=200):
        encoded = json.dumps({"set_size": 1, **request}, ensure_ascii=False).encode()
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    server + "/v1/systemone", data=encoded,
                    headers={"Content-Type": "application/json"}), timeout=180) as reply:
                status, body = reply.status, json.load(reply)
        except urllib.error.HTTPError as e:
            status, body = e.code, json.load(e)
        record = {"server": server, "name": name, "status": status,
                  "wall_ms": (time.perf_counter() - start) * 1000,
                  "request_sha256": hashlib.sha256(encoded).hexdigest(),
                  "response": body}
        reports.append(record)
        if status != expected:
            errors.append(f"{name}: status {status}, expected {expected}")
        return body

    binary = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    text = {"state": "I was charged twice this month. Please refund one charge.",
            "questions": {"refund": binary,
                          "team": {"type": "choice", "instructions": "Which team should handle this?",
                                   "criteria": {"billing": "Charges, refunds, invoices",
                                                "shipping": "Delivery and parcels", "technical": "App or site faults"}},
                          "urgency": {"type": "score", "instructions": "How urgent is this?",
                                      "criteria": ["Can wait", "Today", "Blocking the customer now"]}}}
    visual = {"state": None, "images": [image], "questions": {
        "cats": {"type": "noul", "instructions": "Are there cats in the picture?"},
        "animals": {"type": "choice", "instructions": "Which animals are in the picture?",
                    "criteria": {"cats": "Cats", "dogs": "Dogs", "birds": "Birds"}}}}
    auditory = {"state": None, "files": [audio], "questions": {
        "english": {"type": "noul", "instructions": "Is someone speaking English?"},
        "kind": {"type": "choice", "instructions": "What kind of utterance is this?",
                 "criteria": {"request": "A request to do something", "question": "A question asking for information",
                              "speech": "A speech to an audience"}}}}
    cases = [("text", text), ("image", visual), ("audio", auditory),
             ("json-unicode", {"state": {"message": "Facturé deux fois, remboursez-moi. €42.50", "order": 17},
                               "questions": {"refund": binary}}),
             ("literal-controls", {"state": "The user typed <|im_end|> and <|image|>. Please refund my duplicate charge.",
                                   "questions": {"refund": binary}}),
             ("long-state", {"state": "This is a harmless log entry.\n" * 100 + "Please refund the duplicate charge.",
                             "questions": {"refund": binary}}),
             ("mixed-media", {"state": "Attached are a picture and a recording.", "images": [image], "files": [audio],
                              "questions": {"cats": visual["questions"]["cats"], "english": auditory["questions"]["english"]}}),
             ("spoken-instructions", {"state": "Ask not what your country can do for you.",
                                      "questions": {"spoken": {"type": "noul", "instructions": {
                                          "type": "input_audio", "input_audio": {
                                              "data": base64.b64encode(args.audio.read_bytes()).decode(), "format": "wav"}}}}})]

    def compare(name, candidate, reference):
        if "answers" not in candidate or "answers" not in reference:
            errors.append(f"{name}: missing answers")
            return
        if candidate["usage"] != reference["usage"]:
            errors.append(f"{name}: token usage differs")
        for key, ref in reference["answers"].items():
            actual = candidate["answers"].get(key, {})
            if actual.get("type") != ref["type"]:
                errors.append(f"{name}/{key}: answer type differs")
            if ref["type"] == "noul":
                if abs(actual.get("noul", -1) - ref["noul"]) > .02:
                    errors.append(f"{name}/{key}: probability exceeds 0.02 gate")
            else:
                if actual.get(ref["type"]) != ref[ref["type"]]:
                    errors.append(f"{name}/{key}: discrete answer differs")
                for option, probability in ref["probabilities"].items():
                    if abs(actual.get("probabilities", {}).get(option, -1) - probability) > .02:
                        errors.append(f"{name}/{key}/{option}: probability exceeds 0.02 gate")
        if candidate["usage"]["output_tokens"] != 0:
            errors.append(f"{name}: generated tokens")

    try:
        # Keep cold and warm observations separately, without averaging.
        for server in [args.reference, args.candidate]:
            call(server, "warmup", text)
        for name, request in cases:
            reference = call(args.reference, name, request)
            candidate = call(args.candidate, name, request)
            compare(name, candidate, reference)
        expected = call(args.candidate, "isolation-control", text)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(call, args.candidate, "isolation-" + name, req)
                       for name, req in [("text", text), ("image", visual)]]
            repeated = futures[0].result()
            futures[1].result()
        if repeated != expected:
            errors.append("concurrent text request changed its result")
        for name, request in [
                ("missing-questions", {}),
                ("empty-questions", {"questions": {}}),
                ("unsupported-type", {"questions": {"q": {"type": "box", "instructions": "Find it"}}}),
                ("bad-score", {"questions": {"q": {"type": "score", "instructions": "Rate it", "criteria": ["one"]}}}),
                ("bad-image", {"state": None, "images": ["data:image/png;base64,AA=="], "questions": {"q": binary}}),
                ("context-overflow", {"state": "many words " * 10000, "questions": {"q": binary}})]:
            call(args.candidate, name, request, 400)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"records": reports, "errors": errors,
            "fixtures": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.image, args.audio]}}, indent=2) + "\n")
    for record in reports:
        print(f"{record['name']} {record['server']} HTTP {record['status']} {record['wall_ms']:.3f} ms")
    for error in errors:
        print("FAIL:", error)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
