#!/usr/bin/env python3
"""Focused image preprocessing and HTTP checks for both Qwen3.8 runtimes.

Run in nix develop. Fixtures and results are written only to --directory.
For HTTP checks use a fresh server: --served-model-name vision-test --sessions 2.
The native qwen_vision_serving_test adds token-ID, sampler and disk-restart checks.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import struct
import subprocess
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageOps, PngImagePlugin
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf


def fixtures(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for color, size in (("red", (256, 256)), ("blue", (256, 256)),
                        ("blue-wide", (512, 128))):
        Image.new("RGB", size, color.split("-")[0]).save(directory / f"{color}.png")
    shapes = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(shapes)
    draw.rectangle((20, 20, 110, 110), fill="red")
    draw.ellipse((145, 20, 235, 110), fill="blue")
    draw.polygon(((65, 140), (20, 230), (110, 230)), fill="green")
    draw.text((155, 170), "HELLO", fill="black")
    shapes.save(directory / "shapes.png")
    rng = np.random.default_rng(37)
    rgb = Image.fromarray(rng.integers(0, 256, (193, 371, 3), dtype=np.uint8))
    rgb.save(directory / "resize.png")
    rgb.save(directory / "resize.jpg", quality=93)
    rgb.resize((48, 32)).save(directory / "small.png")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(rng.integers(0, 256, (193, 371), dtype=np.uint8)))
    rgba.save(directory / "alpha.png")
    rgb.convert("L").save(directory / "gray.png")
    rgb.convert("L").save(directory / "gray.jpg")
    rgb.convert("1").save(directory / "mono.png")
    rgb.quantize(colors=32).save(directory / "palette.png", transparency=0)
    rgb.convert("CMYK").save(directory / "cmyk.jpg", quality=93)
    Image.fromarray(rng.integers(0, 1024, (193, 371), dtype=np.uint16)).save(directory / "gray16.png")
    gamma = PngImagePlugin.PngInfo()
    gamma.add(b"gAMA", struct.pack(">I", 100000))
    rgb.save(directory / "gamma.png", pnginfo=gamma)
    for orientation in range(2, 9):
        exif = Image.Exif()
        exif[274] = orientation
        rgb.save(directory / f"orientation{orientation}.jpg", quality=93, exif=exif)
    rgb.save(directory / "orientation.png", exif=exif)
    jpeg = (directory / "orientation6.jpg").read_bytes()
    marker = jpeg.index(b"\xff\xe1")
    (directory / "orientation-fill.jpg").write_bytes(jpeg[:marker] + b"\xff" + jpeg[marker:])


def check_preprocess(directory, probe):
    reports = []
    for path in sorted([*directory.glob("*.png"), *directory.glob("*.jpg")]):
        output = directory / (path.name + ".pixels")
        subprocess.run([str(probe.resolve()), "--preprocess", str(path), str(output)], check=True)
        width, height = map(int, (output / "shape.txt").read_text().split())
        decoded = np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        expected = tvf.resize(
            torch.from_numpy(decoded).permute(2, 0, 1), [height, width],
            interpolation=InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
        actual = np.fromfile(output / "resized.rgb", dtype=np.uint8).reshape(height, width, 3)
        delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        report = dict(image=path.name, width=width, height=height,
                      max_pixel_error=int(delta.max()), changed_pixels=int(np.count_nonzero(delta)))
        reports.append(report)
        print(json.dumps(report), flush=True)
    (directory / "preprocessing.json").write_text(json.dumps(reports, indent=2) + "\n")
    if any(report["changed_pixels"] for report in reports):
        raise SystemExit("Pillow/Torchvision preprocessing differs")


def image_part(path):
    mime = "jpeg" if path.suffix == ".jpg" else "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{encoded}"}}


def check_http(directory, url, model):
    def body(color, stream=False):
        return dict(model=model, temperature=0, max_tokens=20, stream=stream,
                    messages=[dict(role="user", content=[
                        image_part(directory / f"{color}.png"),
                        {"type": "text", "text": "Name the dominant color in the image. Explain in a full sentence."}])])

    def call(payload):
        request = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode()
        if payload.get("stream"):
            content = []
            for line in raw.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    raise AssertionError(event)
                content.extend(choice.get("delta", {}).get("content", "")
                               for choice in event.get("choices", []))
            return dict(text="".join(content), stream=True)
        response = json.loads(raw)
        return dict(text=response["choices"][0]["message"].get("content", ""),
                    usage=response["usage"])

    cold = []
    for color in ("red", "blue"):
        result = call(body(color))
        assert color in result["text"].lower(), result
        assert not result["usage"]["gufo"]["cache_hit"], "use a fresh server"
        cold.append(result)
    with ThreadPoolExecutor(2) as executor:
        cached = list(executor.map(call, [body("red"), body("blue")]))
    for expected, result in zip(cold, cached):
        assert result["text"] == expected["text"], result
        assert result["usage"]["gufo"]["cache_hit"], result
        assert result["usage"]["gufo"]["prefill_tokens"] == 0, result
    streamed = call(body("red", True))
    assert streamed["text"] == cold[0]["text"], streamed
    shapes_body = body("red")
    shapes_body["max_tokens"] = 96
    shapes_body["messages"][0]["content"] = [
        image_part(directory / "shapes.png"),
        {"type": "text", "text": "List the three colored shapes as color + shape, ordered top left, top right, bottom left."}]
    shapes = call(shapes_body)
    described = shapes["text"].lower()
    assert ("red" in described and "blue" in described and "green" in described
            and described.index("red") < described.index("blue") < described.index("green")
            and ("square" in described or "rectangle" in described)
            and "circle" in described and "triangle" in described), shapes
    invalid = body("red")
    invalid["messages"][0]["content"][0]["image_url"]["url"] = "data:image/png;base64,AQID"
    try:
        call(invalid)
    except urllib.error.HTTPError as error:
        assert error.code == 400, error.read().decode()
    else:
        raise AssertionError("invalid image was accepted")
    report = dict(cold=cold, concurrent_cached=cached, streamed=streamed, shapes=shapes,
                  invalid_image_status=400)
    (directory / "http.json").write_text(json.dumps(report, indent=2) + "\n")
    print("HTTP image identity, C2 prefix replay, streaming and validation: passed", flush=True)


def check_cli(directory, binary, target, draft, speculative):
    prompt = "Name the dominant color in the image. Explain in a full sentence."
    common = [str(binary.resolve()), "--model", str(target),
              "--image", str(directory / "red.png"),
              "--max-tokens", "16", "--temperature", "0", "--seed", "47", "--verbose"]
    trace = re.compile(r"^\[TokenTrace\]: count=(\d+) sha256=([0-9a-f]{64})$", re.MULTILINE)
    reports = {}

    def run(command, extra, input_text=None):
        result = subprocess.run(
            [common[0], command, *common[1:], *extra], input=input_text,
            capture_output=True, text=True, timeout=240, check=True)
        fingerprints = trace.findall(result.stderr)
        if not fingerprints:
            raise AssertionError("CLI did not report generated token IDs")
        return dict(text=result.stdout, tokens=fingerprints)

    reports["ar"] = run("prompt", ["--prompt", prompt, "--no-display-prompt"])
    assert "red" in reports["ar"]["text"].lower(), reports["ar"]
    extra = []
    if draft:
        extra = ["--speculative", speculative,
                 "--dflash-model" if speculative == "dflash2" else "--mtp-model", str(draft)]
        reports["speculative"] = run(
            "prompt", ["--prompt", prompt, "--no-display-prompt", *extra])
        assert reports["speculative"]["tokens"] == reports["ar"]["tokens"], reports
    reports["chat"] = run("chat", extra, prompt + "\nWhat color was it? Answer in one sentence.\nexit\n")
    assert len(reports["chat"]["tokens"]) == 2, reports["chat"]
    assert reports["chat"]["tokens"][0] == reports["ar"]["tokens"][0], reports
    turns = reports["chat"]["text"].split("<<< Assistant: ")[1:]
    assert len(turns) == 2 and all("red" in turn.lower() for turn in turns), reports["chat"]
    (directory / "cli.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("CLI image prompt/AR/speculation token parity and two-turn chat: passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--probe", type=Path, help="qwen27b_vision_test, for CPU pixel qualification")
    parser.add_argument("--http-url", help="fresh running gufo server")
    parser.add_argument("--model", default="vision-test")
    parser.add_argument("--binary", type=Path, help="gufo binary for prompt/chat checks")
    parser.add_argument("--target", type=Path, help="target GGUF for CLI checks")
    parser.add_argument("--draft", type=Path, help="draft GGUF for CLI checks")
    parser.add_argument("--speculative", choices=("dflash2",))
    args = parser.parse_args()
    if bool(args.binary) != bool(args.target):
        parser.error("--binary and --target must be supplied together")
    if bool(args.draft) != bool(args.speculative) or (args.draft and not args.binary):
        parser.error("--draft and --speculative require CLI checks and each other")
    torch.set_num_threads(4)
    fixtures(args.directory)
    if args.probe:
        check_preprocess(args.directory, args.probe)
    if args.http_url:
        check_http(args.directory, args.http_url, args.model)
    if args.binary:
        check_cli(args.directory, args.binary, args.target, args.draft, args.speculative)


if __name__ == "__main__":
    main()
