#!/usr/bin/env python3
"""Capture deterministic Qwen3-ASR-1.7B reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--language")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention", default="eager", choices=("eager", "sdpa")
    )
    return parser.parse_args()


def tensor_value(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return tensor_value(output[0])
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"unsupported hook output: {type(output)!r}")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).cpu().numpy()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    sys.path.insert(0, str(args.reference_root))
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import (
        normalize_audio_input,
        parse_asr_output,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(
        args.audio, dtype="float32", always_2d=False
    )
    waveform = normalize_audio_input((audio, sample_rate))
    np.save(args.out / "waveform.npy", waveform.astype("<f4", copy=False))

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    wrapper = Qwen3ASRModel.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation=args.attention,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    wrapper.model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    prompt = wrapper._build_text_prompt(
        context=args.context, force_language=args.language
    )
    host_inputs = wrapper.processor(
        text=[prompt],
        audio=[waveform],
        return_tensors="pt",
        padding=True,
    )
    np.save(
        args.out / "prompt_ids.npy",
        host_inputs["input_ids"].cpu().numpy().astype("<i4", copy=False),
    )
    np.save(
        args.out / "input_features.npy",
        host_inputs["input_features"].cpu().numpy().astype("<f4", copy=False),
    )
    np.save(
        args.out / "feature_attention_mask.npy",
        host_inputs["feature_attention_mask"]
        .cpu()
        .numpy()
        .astype("u1", copy=False),
    )

    captured: dict[str, np.ndarray] = {}
    text_layer0_steps: list[np.ndarray] = []
    text_final_steps: list[np.ndarray] = []
    generation_logits: list[np.ndarray] = []

    def capture_once(name: str):
        def hook(_module, _inputs, output):
            if name not in captured:
                captured[name] = to_numpy(tensor_value(output))

        return hook

    def capture_input_once(name: str):
        def hook(_module, inputs):
            if name not in captured:
                captured[name] = to_numpy(tensor_value(inputs[0]))

        return hook

    def capture_text_layer0(_module, _inputs, output):
        value = tensor_value(output)
        if "text_layer0_prefill" not in captured:
            captured["text_layer0_prefill"] = to_numpy(value)
        text_layer0_steps.append(to_numpy(value[0, -1]))

    def capture_lm_head(_module, inputs, output):
        text_final_steps.append(to_numpy(tensor_value(inputs[0])[0, -1]))
        logits = to_numpy(output[0, -1])
        generation_logits.append(logits)
        if "prefill_logits" not in captured:
            captured["prefill_logits"] = logits

    thinker = wrapper.model.thinker
    hooks = [
        thinker.audio_tower.layers[0].register_forward_pre_hook(
            capture_input_once("audio_encoder_input")
        ),
        thinker.audio_tower.layers[0].register_forward_hook(
            capture_once("audio_layer0")
        ),
        thinker.audio_tower.register_forward_hook(capture_once("audio_final")),
        thinker.model.layers[0].register_forward_hook(capture_text_layer0),
        thinker.lm_head.register_forward_hook(capture_lm_head),
    ]

    inputs = host_inputs.to(wrapper.model.device).to(wrapper.model.dtype)
    run_start = time.perf_counter()
    try:
        generated = wrapper.model.generate(
            **inputs, max_new_tokens=args.max_new_tokens
        )
        torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    generate_seconds = time.perf_counter() - run_start

    generated_ids = generated.sequences[:, inputs["input_ids"].shape[1] :]
    generated_array = generated_ids.cpu().numpy().astype("<i4", copy=False)
    np.save(args.out / "generated_ids.npy", generated_array)
    np.save(
        args.out / "text_layer0_steps.npy",
        np.stack(text_layer0_steps).astype("<f4", copy=False),
    )
    np.save(
        args.out / "text_final_steps.npy",
        np.stack(text_final_steps).astype("<f4", copy=False),
    )
    np.save(
        args.out / "generation_logits.npy",
        np.stack(generation_logits).astype("<f4", copy=False),
    )
    raw_text = wrapper.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    language, text = parse_asr_output(raw_text, user_language=args.language)

    required_captures = {
        "audio_encoder_input",
        "audio_layer0",
        "audio_final",
        "text_layer0_prefill",
        "prefill_logits",
    }
    missing = sorted(required_captures.difference(captured))
    if missing:
        raise RuntimeError(f"missing official captures: {missing}")
    for name, value in captured.items():
        np.save(args.out / f"{name}.npy", value.astype("<f4", copy=False))

    artifact_names = [
        "waveform.npy",
        "prompt_ids.npy",
        "input_features.npy",
        "feature_attention_mask.npy",
        "audio_encoder_input.npy",
        "audio_layer0.npy",
        "audio_final.npy",
        "text_layer0_prefill.npy",
        "prefill_logits.npy",
        "text_layer0_steps.npy",
        "text_final_steps.npy",
        "generation_logits.npy",
        "generated_ids.npy",
    ]
    manifest = {
        "schema": "gufo.qwen3-asr-reference.v2",
        "source_revision": source_revision(args.reference_root),
        "model_revision": args.model.name,
        "model": "Qwen/Qwen3-ASR-1.7B",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "attention": args.attention,
        "audio_file": args.audio.name,
        "audio_sample_rate": 16000,
        "audio_samples": int(waveform.size),
        "context": args.context,
        "forced_language": args.language,
        "prompt": prompt,
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "language": language,
        "raw_text": raw_text,
        "text": text,
        "generated_tokens": int(generated_array.size),
        "artifacts": {
            name: {
                "sha256": sha256(args.out / name),
                "shape": list(np.load(args.out / name, mmap_mode="r").shape),
            }
            for name in artifact_names
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
