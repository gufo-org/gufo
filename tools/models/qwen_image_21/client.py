#!/usr/bin/env python3
"""Local Gufo Images client; retain native pixels and center-crop exact sizes."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid

from PIL import Image, ImageStat


def telemetry():
    result = {"time": time.time()}
    patterns = (
        "/sys/class/drm/card*/device/gpu_busy_percent",
        "/sys/class/drm/card*/device/mem_info_vram_used",
        "/sys/class/drm/card*/device/hwmon/hwmon*/temp1_input",
        "/sys/class/drm/card*/device/hwmon/hwmon*/power1_average",
        "/sys/class/infiniband/usb4_rdma*/ports/1/hw_counters/*",
    )
    for pattern in patterns:
        for path in Path("/").glob(pattern.lstrip("/")):
            try:
                result[str(path)] = int(path.read_text().strip())
            except (OSError, ValueError):
                pass
    return result


def multipart(fields, references):
    boundary = "gufo-" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode())
    for reference in references:
        path = Path(reference)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n").encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def generate(args):
    width, height = map(int, args.size.lower().split("x"))
    native_width, native_height = math.ceil(width / 32) * 32, math.ceil(height / 32) * 32
    fields = {"model": args.model, "prompt": args.prompt,
              "size": f"{native_width}x{native_height}", "seed": args.seed,
              "background": args.background}
    if args.steps is not None:
        fields["steps"] = args.steps
    if args.reference:
        body, content_type = multipart(fields, args.reference)
        route = "/v1/images/edits"
    else:
        body, content_type = json.dumps(fields).encode(), "application/json"
        route = "/v1/images/generations"
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    before = telemetry()
    samples, stop = [], threading.Event()

    def sample():
        while not stop.wait(1):
            samples.append(telemetry())

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    started = time.perf_counter()
    try:
        request = urllib.request.Request(args.url.rstrip("/") + route, data=body,
                                         headers={"Content-Type": content_type})
        with urllib.request.urlopen(request, timeout=3600) as response:
            raw = response.read()
        http_seconds = time.perf_counter() - started
        document = json.loads(raw)
        png = base64.b64decode(document["data"][0]["b64_json"], validate=True)
        image = Image.open(io.BytesIO(png))
        image.load()
        assert image.size == (native_width, native_height), image.size
        native = output.with_name(output.stem + "-native.png")
        native.write_bytes(png)
        left, top = (native_width - width) // 2, (native_height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        image.save(output, compress_level=1)
        total_seconds = time.perf_counter() - started
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read(8192).decode()}") from error
    finally:
        stop.set()
        observer.join(timeout=2)
    after = telemetry()
    stats = ImageStat.Stat(image.convert("RGB"))
    result = {"host": socket.gethostname(), "url": args.url, "route": route,
              "request": fields, "references": args.reference,
              "requested_size": args.size, "native_size": fields["size"],
              "output_size": list(image.size), "crop": [left, top, left + width, top + height],
              "http_seconds": http_seconds, "total_seconds": total_seconds,
              "postprocess_seconds": total_seconds - http_seconds,
              "output": str(output), "native_output": str(native),
              "native_png_sha256": hashlib.sha256(png).hexdigest(),
              "output_png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "pixel_mean_rgb": stats.mean, "pixel_stddev_rgb": stats.stddev,
              "telemetry_before": before, "telemetry_after": after,
              "telemetry_samples": samples}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "telemetry_samples"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28180")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="Qwen-Image-2.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, help="Omit to use the checkpoint default")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--background", choices=["auto", "transparent", "opaque"], default="opaque")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output", required=True)
    generate(parser.parse_args())
