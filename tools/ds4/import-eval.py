#!/usr/bin/env python3
"""Import pinned DS4 capability cases or official 0731 continuations."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


DS4_REVISION = "84cc882352757baf628a1776badf7cc54d584e28"
DS4_EVAL_BLOB = "7aed5d5c5b5cdc74d1b3aa0310161e2b437c1d08"
OFFICIAL_REVISION = "6289c516273979173abbc062209a81dd3706b804"
SYSTEM_PROMPT = (
    "You are solving a hard benchmark question. Reason carefully. "
    "The final answer must follow the requested format exactly."
)

DATASETS = {
    "GPQA Diamond": {
        "dataset": "GPQA Diamond",
        "dataset_url": "https://huggingface.co/datasets/Wanfq/gpqa",
        "license": "CC BY 4.0",
        "audit_note": "Preserved from the audited DS4 capability subset.",
    },
    "GPQA Diamond (modified)": {
        "dataset": "GPQA Diamond",
        "dataset_url": "https://huggingface.co/datasets/Wanfq/gpqa",
        "license": "CC BY 4.0",
        "audit_note": (
            "DS4 marks this row as modified; Gufo preserves the DS4 prompt and "
            "key without further local repair."
        ),
    },
    "SuperGPQA": {
        "dataset": "SuperGPQA",
        "dataset_url": "https://huggingface.co/datasets/m-a-p/SuperGPQA",
        "license": "ODC-BY",
        "audit_note": (
            "Preserved from DS4's audited subset after DS4 replaced wrong-key, "
            "missing-figure, or underspecified source rows."
        ),
    },
    "AIME2025": {
        "dataset": "AIME 2025",
        "dataset_url": "https://huggingface.co/datasets/test-time-compute/aime_2025",
        "license": "MIT",
        "audit_note": "Preserved from the MIT-licensed mirror used by DS4.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("capability", "official"),
                        default="capability")
    parser.add_argument(
        "--ds4-repository",
        type=Path,
        required=True,
        help="Checkout of antirez/ds4 containing the pinned revision",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; stdout is used when omitted",
    )
    return parser.parse_args()


def git_output(repository: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.stdout.strip()


def initializer_blocks(source: str) -> Iterator[str]:
    marker = "static const eval_case eval_cases[] = {"
    start = source.index(marker)
    index = source.index("{", start) + 1
    depth = 0
    block_start: int | None = None
    in_string = False
    escaped = False
    line_comment = False
    block_comment = False

    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if line_comment:
            line_comment = char != "\n"
            index += 1
            continue
        if block_comment:
            if char == "*" and following == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char == "/" and following == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and following == "*":
            block_comment = True
            index += 2
            continue
        if char == "{":
            if depth == 0:
                block_start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                return
            depth -= 1
            if depth == 0 and block_start is not None:
                yield source[block_start : index + 1]
                block_start = None
        index += 1


def decode_c_strings(expression: str) -> str:
    tokens = re.findall(r'"(?:\\.|[^"\\])*"', expression, re.DOTALL)
    return "".join(ast.literal_eval(token) for token in tokens)


def parse_field(block: str, field: str) -> str:
    pattern = rf"\.{field}\s*=\s*((?:\"(?:\\.|[^\"\\])*\"\s*)+),"
    match = re.search(pattern, block, re.DOTALL)
    if match is None:
        raise ValueError(f"missing .{field} in DS4 eval initializer")
    return decode_c_strings(match.group(1))


def parse_choices(block: str) -> list[str]:
    choices: list[str | None] = []
    pattern = r'\.choice\[(\d+)\]\s*=\s*((?:"(?:\\.|[^"\\])*"\s*)+),'
    for match in re.finditer(pattern, block, re.DOTALL):
        choice_index = int(match.group(1))
        while len(choices) <= choice_index:
            choices.append(None)
        choices[choice_index] = decode_c_strings(match.group(2))
    if any(choice is None for choice in choices):
        raise ValueError("DS4 eval choices are not contiguous")
    return [choice for choice in choices if choice is not None]


def canonical_record_hash(record: dict[str, Any]) -> str:
    serialized = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def parse_case(block: str, index: int) -> dict[str, Any]:
    source = parse_field(block, "source")
    choices = parse_choices(block)
    record: dict[str, Any] = {
        "ds4_index": index,
        "source": source,
        "id": parse_field(block, "id"),
        "domain": parse_field(block, "domain"),
        "title": parse_field(block, "title"),
        "kind": (
            "mcq" if choices else ("linespec" if source == "COMPSEC" else "integer")
        ),
        "question": parse_field(block, "question"),
        "choices": choices,
        "answer": parse_field(block, "answer"),
    }
    record["source_record_sha256"] = canonical_record_hash(record)
    return record


def build_document(repository: Path) -> dict[str, Any]:
    revision = git_output(repository, "rev-parse", DS4_REVISION)
    if revision != DS4_REVISION:
        raise ValueError(f"unexpected DS4 revision: {revision}")
    blob = git_output(repository, "rev-parse", f"{DS4_REVISION}:ds4_eval.c")
    if blob != DS4_EVAL_BLOB:
        raise ValueError(f"unexpected ds4_eval.c blob: {blob}")
    source = git_output(repository, "show", f"{DS4_REVISION}:ds4_eval.c")
    rows = [
        parse_case(block, index)
        for index, block in enumerate(initializer_blocks(source))
    ]
    if len(rows) != 92:
        raise ValueError(f"expected 92 DS4 rows, found {len(rows)}")

    cases = rows[:75]
    for case in cases:
        metadata = DATASETS.get(case["source"])
        if metadata is None:
            raise ValueError(f"unexpected committed source: {case['source']}")
        case.update(metadata)

    excluded_cases = [
        {
            "ds4_index": case["ds4_index"],
            "source": case["source"],
            "id": case["id"],
            "domain": case["domain"],
            "kind": case["kind"],
            "answer": case["answer"],
            "source_record_sha256": case["source_record_sha256"],
        }
        for case in rows[75:]
    ]
    return {
        "schema": "gufo.eval-cases.v2",
        "benchmark": "Antirez DS4 capability subset",
        "source_repository": "https://github.com/antirez/ds4",
        "source_revision": DS4_REVISION,
        "source_path": "ds4_eval.c",
        "source_blob": DS4_EVAL_BLOB,
        "system_prompt": SYSTEM_PROMPT,
        "cases": cases,
        "excluded": {
            "source": "COMPSEC",
            "count": 17,
            "commit_policy": "metadata-only",
            "reason": (
                "Question text remains opt-in and uncommitted until its "
                "redistribution provenance is reviewed."
            ),
            "cases": excluded_cases,
        },
    }


def build_official_document(repository: Path) -> dict[str, Any]:
    revision = git_output(repository, "rev-parse", OFFICIAL_REVISION)
    if revision != OFFICIAL_REVISION:
        raise ValueError(f"unexpected official fixture revision: {revision}")
    source_files: dict[str, str] = {}

    def read(path: str) -> bytes:
        content = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{revision}:{path}"])
        source_files[path] = hashlib.sha256(content).hexdigest()
        return content

    cases = []
    base = "gguf-tools/quality-testing/data/flash"
    for line in read(f"{base}/manifest.tsv").decode().splitlines():
        if not line or line.startswith("#"):
            continue
        name, prompt_path, continuation_path, response_path = line.split("\t")
        response = json.loads(read(response_path))
        choice = response["choices"][0]
        tokens = [bytes(token["bytes"])
                  for token in choice["logprobs"]["content"]]
        continuation = read(continuation_path)
        if b"".join(tokens) != continuation:
            raise ValueError(f"official token bytes do not reconstruct {name}")
        cases.append({
            "id": name, "group": "continuation-100",
            "prompt": read(prompt_path).decode("utf-8"),
            "continuation": continuation.decode("utf-8"),
            "token_bytes_hex": [token.hex() for token in tokens],
        })
    if len(cases) != 100 or sum(len(c["token_bytes_hex"]) for c in cases) != 2313:
        raise ValueError("official continuation fixture coverage changed")

    base = "tests/test-vectors/flash-0731"
    manifest = json.loads(read(f"{base}/manifest.json"))
    if manifest["checkpoint"] != "0731" or len(manifest["prompts"]) != 5:
        raise ValueError("official smoke fixture changed")
    for item in manifest["prompts"]:
        response = json.loads(read(f"{base}/{item['official_file']}"))
        tokens = [bytes(step["token"]["bytes"]) for step in response["steps"]]
        cases.append({
            "id": item["id"], "group": "smoke-5",
            "prompt": read(f"{base}/{item['prompt_file']}").decode("utf-8"),
            "continuation": b"".join(tokens).decode("utf-8"),
            "token_bytes_hex": [token.hex() for token in tokens],
        })
    return {
        "schema": "gufo.ds4-official-continuations.v1",
        "source_repository": "https://github.com/antirez/ds4",
        "source_revision": revision,
        "checkpoint": "0731",
        "license": read("LICENSE").decode("utf-8"),
        "source_files_sha256": source_files,
        "measurement": (
            "Teacher-forced target-token NLL, greedy agreement, and matching "
            "prefix. Saturated API logprobs are not full-distribution logits."
        ),
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    try:
        document = (build_official_document(args.ds4_repository)
                    if args.suite == "official"
                    else build_document(args.ds4_repository))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"import-ds4-eval: {error}", file=sys.stderr)
        return 1

    output = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(output)
    else:
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
