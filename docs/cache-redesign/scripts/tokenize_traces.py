#!/usr/bin/env python3
"""Turn E2 content traces into token sequences.

Reads every `generation` record of results/e2/<model>/<workload>/trace-*.jsonl
in file order, tokenizes the rendered prompt and the generated text with the
model's own vocabulary (llama-tokenize, special tokens parsed), and writes
requests.npz (token arrays) plus requests.json (per-request metadata). The
tokenized prompt length is checked against the server's prompt_tokens.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

from paths import FN_MODEL, Q27_VOCAB, TOKENIZE, WORK as HERE

VOCAB = {"fn": FN_MODEL, "q27": Q27_VOCAB}


def tokenize(model, text):
    if not text:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=True) as handle:
        handle.write(text)
        handle.flush()
        result = subprocess.run(
            [TOKENIZE, "-m", VOCAB[model], "-f", handle.name, "--ids",
             "--log-disable", "--no-bos"],
            capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(model, workload):
    base = HERE / "results" / "e2" / model / workload
    records = []
    for index, trace in enumerate(sorted(base.glob("trace-*.jsonl"),
                                         key=lambda p: int(p.stem.split("-")[1]))):
        for line in trace.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "generation" and "prompt" in record:
                record["server_lifetime"] = index
                records.append(record)
    arrays, meta = {}, []
    mismatches = 0
    for i, record in enumerate(records):
        prompt = tokenize(model, record["prompt"])
        output = tokenize(model, record.get("output", ""))
        if len(prompt) != record["prompt_tokens"]:
            mismatches += 1
        arrays[f"p{i}"] = np.asarray(prompt, dtype=np.int32)
        arrays[f"o{i}"] = np.asarray(output, dtype=np.int32)
        meta.append({k: record.get(k) for k in (
            "time", "request", "server_lifetime", "prompt_tokens", "cache",
            "cached_tokens", "cache_miss_reason", "common_prefix_tokens",
            "nearest_checkpoint_tokens", "generated_tokens", "finish")}
            | {"tokenized_prompt": len(prompt)})
    np.savez_compressed(base / "requests.npz", **arrays)
    (base / "requests.json").write_text(json.dumps(meta, indent=1))
    print(f"{model}/{workload}: {len(records)} requests, "
          f"{mismatches} prompt length mismatches")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
