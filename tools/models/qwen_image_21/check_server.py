#!/usr/bin/env python3
"""Check a local Gufo or llama-swap image endpoint, including seeded replay."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import io
import json
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8094")
    parser.add_argument("--model", default="Qwen-Image-2.1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--edit", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--isolation", action="store_true",
                        help="Check n=2, concurrent seeds and disconnect recovery (use --size 64)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def encode(fields):
        if not args.edit:
            return json.dumps(fields).encode(), "application/json", "/v1/images/generations"
        boundary = "gufo-" + uuid.uuid4().hex
        parts = []
        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n".encode()
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image[]\"; "
            f"filename=\"input.png\"\r\nContent-Type: image/png\r\n\r\n".encode()
            + args.edit.read_bytes() + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        return b"".join(parts), f"multipart/form-data; boundary={boundary}", "/v1/images/edits"

    def send(fields):
        body, content_type, path = encode(fields)
        request = urllib.request.Request(
            args.url.rstrip("/") + path, data=body,
            headers={"Content-Type": content_type}, method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - start
        images = []
        expected_size = tuple(map(int, fields["size"].split("x")))
        for item in result["data"]:
            raw = base64.b64decode(item["b64_json"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                assert image.format == "PNG" and image.size == expected_size
            images.append(raw)
        assert len(images) == fields.get("n", 1)
        return images, elapsed

    fields = {
        "model": args.model,
        "prompt": "Change the cube to blue." if args.edit else "A red cube on a white table.",
        "size": f"{args.size}x{args.size}",
        "seed": 42,
        "steps": args.steps,
    }
    records = []
    previous = None
    for run in range(2 if args.replay else 1):
        images, elapsed = send(fields)
        raw = images[0]
        (args.output / f"image-{run}.png").write_bytes(raw)
        records.append({"seconds": elapsed, "sha256": hashlib.sha256(raw).hexdigest()})
        if previous is not None:
            assert raw == previous, "seeded image replay changed"
        previous = raw
        print(json.dumps(records[-1]), flush=True)
    report = {"request": fields, "mode": "edit" if args.edit else "generate", "runs": records}
    if args.isolation:
        peer = dict(fields, seed=43)
        expected = [previous, send(peer)[0][0]]
        assert send(dict(fields, n=2))[0] == expected, "n=2 changed independent seeds"
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(send, [fields, peer]))
        assert [value[0][0] for value in actual] == expected, "concurrency changed images"
        side = args.size + 32
        send(dict(fields, size=f"{side}x{side}"))
        assert send(fields)[0][0] == previous, "changing image size contaminated replay"
        # Send an expensive request, disconnect, then reuse the same process.
        url = urllib.parse.urlsplit(args.url)
        connection_class = (http.client.HTTPSConnection if url.scheme == "https"
                            else http.client.HTTPConnection)
        connection = connection_class(url.hostname, url.port, timeout=10)
        body, content_type, path = encode(dict(fields, steps=100, size="1024x1024"))
        connection.request("POST", url.path.rstrip("/") + path, body,
                           {"Content-Type": content_type})
        time.sleep(0.1)
        connection.close()
        recovery, elapsed = send(fields)
        assert recovery[0] == previous, "disconnect contaminated subsequent generation"
        assert elapsed < 30, "disconnected request kept the model busy"
        report["isolation"] = {"n2": True, "concurrency": True, "resized_request": True,
                               "disconnect_recovery_seconds": elapsed}
        print(json.dumps(report["isolation"]), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
