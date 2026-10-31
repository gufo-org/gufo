#!/usr/bin/env python3
"""Tokenizer goldens for Gemma 4 from two independent implementations.

Every corpus entry is tokenized by the pinned llama.cpp reference
(`llama-tokenize`, special tokens parsed, no BOS) and, for valid UTF-8, by
Hugging Face `tokenizers` with the model's tokenizer.json. Gufo mirrors
llama.cpp on the GGUF vocabulary; entries where the two references disagree
are recorded, not hidden.

    nix develop -c nix shell .#llama-cpp-reference -c \\
      python3 tools/gemma4/tokenizer_goldens.py --model TARGET.gguf \\
        --hf-tokenizer DIR_WITH_tokenizer.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests/models/gemma4/fixtures/tokenizer_goldens.json"
HF_REPO = "unsloth/gemma-4-31B-it"
HF_REVISION = "51c9f6265ea38755c708a6e356dfc1af4a436e35"

CORPUS: list[tuple[str, bytes]] = [
    ("ascii", b"Hello world, this is a test."),
    ("spaces", b"hello  world   return x;    indented\t\tword"),
    ("leading_trailing_space", b"  padded text  "),
    ("newlines", b"line one\nline two\n\nline four\n\n\n\nend\n"),
    ("crlf", b"windows\r\nline\r\n"),
    ("only_newlines", b"\n\n\n\n\n"),
    ("code", b"def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n"),
    ("numbers", b"3.14159 2026-09-26 1,000,000 0x1F 1e-5"),
    ("punctuation", "Wait... what?! (Really) — “quoted” ‘single’".encode()),
    ("italian", "Perché l'àncora è così pesante? Città, università.".encode()),
    ("cjk", "今日はいい天気です。你好，世界！한국어".encode("utf-8", "surrogatepass").decode("utf-8", "replace").encode()),
    ("emoji", "Party \U0001F389\U0001F44D\U0001F3FD family \U0001F468‍\U0001F469‍\U0001F467".encode()),
    ("rare_unicode", "ᚠᚢ \U0001d49c\U0001d4b7 ༀ ☃".encode()),
    ("special_markup", b"<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Hello<turn|>"),
    ("tool_markup", b"<|tool_call>call:f{a:<|\"|>x y<|\"|>,n:1}<tool_call|><|tool_response>"),
    ("near_special", b"<|turn <turn| <|\"| |> <eos <bos>"),
    ("byte_like", b"<0x41> <0xFF> literal"),
    ("invalid_utf8", b"ok \xff\xfe bad \xe2\x82 trunc \xc3"),
    ("zero_width", "a​b‌c﻿d".encode()),
    ("long_line", ("The quick brown fox jumps over the lazy dog. " * 460).encode()),
]


def llama_ids(binary: str, model: Path, text: bytes) -> list[int]:
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(text)
        path = handle.name
    try:
        result = subprocess.run(
            [binary, "-m", str(model), "-f", path, "--ids", "--no-bos", "--log-disable"],
            check=True, capture_output=True, text=True)
    finally:
        Path(path).unlink()
    line = result.stdout.strip().splitlines()[-1]
    return json.loads(line)


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hf-tokenizer", type=Path, required=True,
                        help=f"directory with tokenizer.json from {HF_REPO}@{HF_REVISION}")
    parser.add_argument("--llama-tokenize", default=shutil.which("llama-tokenize"))
    args = parser.parse_args()
    if not args.llama_tokenize:
        raise SystemExit("llama-tokenize not found; run inside nix shell .#llama-cpp-reference")

    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(str(args.hf_tokenizer / "tokenizer.json"))
    tokenizer_sha = hashlib.sha256((args.hf_tokenizer / "tokenizer.json").read_bytes()).hexdigest()
    version = subprocess.run([args.llama_tokenize, "--version"], capture_output=True, text=True)
    llama_version = (version.stdout + version.stderr).strip().splitlines()[0]

    entries = []
    disagreements = 0
    for name, text in CORPUS:
        ref = llama_ids(args.llama_tokenize, args.model, text)
        try:
            decoded = text.decode("utf-8")
            hf_ids = hf.encode(decoded, add_special_tokens=False).ids
        except UnicodeDecodeError:
            hf_ids = None
        entry = {"name": name, "text_hex": text.hex()}
        if len(ref) > 64:
            entry.update({"llama_cpp_count": len(ref), "llama_cpp_ids_sha256": ids_sha256(ref)})
        else:
            entry["llama_cpp_ids"] = ref
        if hf_ids is None:
            entry["hf"] = "not applicable: invalid UTF-8"
        elif hf_ids == ref:
            entry["hf"] = "agrees"
        else:
            disagreements += 1
            entry["hf"] = "differs"
            entry["hf_ids"] = hf_ids if len(hf_ids) <= 64 else ids_sha256(hf_ids)
        entries.append(entry)

    OUTPUT.write_text(json.dumps({
        "schema": "gufo.gemma4-tokenizer-goldens.v1",
        "parse_special": True,
        "add_bos": False,
        "llama_cpp": llama_version,
        "hf_tokenizer": {"repo": HF_REPO, "revision": HF_REVISION, "tokenizer_sha256": tokenizer_sha},
        "entries": entries,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries ({disagreements} HF/llama.cpp disagreements) to "
          f"{OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
