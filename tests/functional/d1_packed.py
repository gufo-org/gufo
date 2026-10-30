#!/usr/bin/env python3
"""Opt-in box, packed-set, audio placement, isolation and validation checks."""
import argparse
import base64
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18886")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    image = "data:image/png;base64," + base64.b64encode(a.image.read_bytes()).decode()
    audio_data = base64.b64encode(a.audio.read_bytes()).decode()
    audio = "data:audio/wav;base64," + audio_data
    rows, errors = [], []

    def call(name, payload, expected=200):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(urllib.request.Request(a.url + "/v1/systemone", data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"}), timeout=180) as r:
                status, result = r.status, json.load(r)
        except urllib.error.HTTPError as e:
            status, result = e.code, json.load(e)
        rows.append({"name": name, "wall_ms": (time.perf_counter() - start) * 1000, "status": status, "response": result})
        if status != expected:
            errors.append(f"{name}: {status}, expected {expected}")
        return result

    binary = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    state = "I was charged twice. Please refund one charge."
    request = {"state": state, "questions": {"q" + str(i): binary for i in range(17)}}
    box_request = {"state": None, "images": [image], "questions": {
        "remote": {"type": "box", "instructions": "Where is the remote control on the left?"},
        "cats": {"type": "boxes", "instructions": "Every cat.", "max": 4}}}
    try:
        packed = call("default-17", request)
        explicit = call("explicit-16", {**request, "set_size": 16})
        if packed != explicit:
            errors.append("default packing differs from set_size=16")
        first = call("first-16", {**request, "questions": dict(list(request["questions"].items())[:16])})
        last = call("last-1", {**request, "questions": {"q16": binary}})
        if packed["usage"]["input_tokens"] != first["usage"]["input_tokens"] + last["usage"]["input_tokens"]:
            errors.append("17-question tail did not account for two real prompts")
        if packed["answers"] != {**first["answers"], **last["answers"]}:
            errors.append("ragged tail changed its independent group response")
        split = call("unpacked-17", {**request, "set_size": 1})
        if packed["usage"]["input_tokens"] >= split["usage"]["input_tokens"]:
            errors.append("packing did not reduce repeated state work")
        boxes = call("boxes", box_request)
        if boxes["answers"]["cats"]["count"] != 2:
            errors.append("two-cat fixture did not return two boxes")
        for answer in boxes["answers"].values():
            own = answer["boxes"] if answer["type"] == "boxes" else [answer]
            for item in own:
                r = item["box"]
                if not (0 <= r["top"] < r["bottom"] <= 1 and 0 <= r["left"] < r["right"] <= 1):
                    errors.append("invalid rectangle bounds")
                if not (0 <= item["found"] <= 1 and 0 <= item["confidence"] <= 1):
                    errors.append("invalid box calibration")
        voice = {"state": None, "files": [audio], "questions": {
            "english": {"type": "noul", "instructions": "Is someone speaking English?"},
            "speech": {"type": "choice", "instructions": "Which language is spoken?", "criteria": {"en": "English", "fr": "French", "de": "German"}}}}
        spoken = {"state": "Ask not what your country can do for you.", "questions": {
            "spoken": {"type": "noul", "instructions": {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}},
            "words": {"type": "noul", "instructions": "Does the text mention a country?"}}}
        for name, own in [("audio", voice), ("spoken", spoken), ("mixed-media", {**box_request, "files": [audio]})]:
            control = call(name, own)
            repeated = call(name + "-repeat", own)
            if control != repeated:
                errors.append(name + ": packed replay differs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            text_future = pool.submit(call, "concurrent-text", request)
            box_future = pool.submit(call, "concurrent-boxes", box_request)
            if text_future.result() != packed or box_future.result() != boxes:
                errors.append("concurrent packed requests changed a response")
        for size in [0, -1, 1.5, True, 129]:
            call("bad-set-" + str(size), {**request, "set_size": size}, 400)
        for size in [0, 17, True, 2.5]:
            call("bad-max-" + str(size), {**box_request, "questions": {"q": {"type": "boxes", "instructions": "Cats", "max": size}}}, 400)
        for name, images in [("no-image", []), ("two-images", [image, image])]:
            call(name, {**box_request, "images": images}, 400)
        call("too-long-pack", {"state": "words " * 7000,
                             "questions": {"q" + str(i): binary for i in range(100)}, "set_size": 128}, 400)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({"records": rows, "errors": errors}, indent=2) + "\n")
    for error in errors: print("FAIL", error)
    print("PASS" if not errors else "FAIL", len(rows), "HTTP checks")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
