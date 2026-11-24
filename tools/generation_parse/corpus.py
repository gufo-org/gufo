#!/usr/bin/env python3
"""Builds tests/fixtures/generation_parse corpus files for generation_parse_test.

Two sources, neither of which keeps a prompt:

  corpus.py trace TRACE.jsonl OUT.json --source NAME
      Joins each `--trace` generation record (dialect, reasoning markup,
      initial output state, piece boundaries, output) with its request
      record, from which only the declared tools, tool_choice, response
      format and stream flag are read.

  corpus.py capture CAPTURE.jsonl OUT.json --source NAME
      Reads parser setups and pieces recorded from openai_chat_test with a
      temporary hook in src/cli/serve/generation_parse.cpp (see
      tests/fixtures/generation_parse/README.md).

  corpus.py normalize IN.json OUT.json
      Rewrites an existing corpus file in the current layout.

  corpus.py long OUT.json --bytes 13000 --bytes 52000
      Writes synthetic Qwen `write` calls of about the given sizes, split
      into four-byte pieces, for `generation_parse_test --bench` on long
      arguments. Not part of the CTest corpus.

A case whose text is a prefix of another case with the same setup is
dropped: the test truncates every case at every byte anyway. Each distinct
tool list is stored once in `tool_sets`, and cases name it by index.
"""

import argparse
import json
import sys

INITIAL = ["auto", "reasoning", "content", "content_after_reasoning"]
CHOICE = ["auto", "none", "required"]


def piece(data: bytes):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return {"hex": data.hex()}


def setup_key(case):
    return json.dumps({k: v for k, v in case.items() if k not in ("name", "pieces")}, sort_keys=True)


def text_of(case):
    return b"".join(p.encode("utf-8") if isinstance(p, str) else bytes.fromhex(p["hex"]) for p in case["pieces"])


def reduce(cases):
    """Drops duplicates and prefixes, keeping the most finely split copy."""
    by_setup = {}
    for case in cases:
        texts = by_setup.setdefault(setup_key(case), {})
        text = text_of(case)
        previous = texts.get(text)
        if previous is None or len(case["pieces"]) > len(previous["pieces"]):
            texts[text] = case
    kept = []
    for texts in by_setup.values():
        chosen = []
        for text in sorted(texts, key=len, reverse=True):
            if any(other.startswith(text) for other in chosen):
                continue
            chosen.append(text)
            kept.append(texts[text])
    kept.sort(key=lambda case: (setup_key(case), text_of(case)))
    return kept


def dialect_from_capture(record):
    if record["qwen_calls"] and not record["native"]:
        return None if record["closers"] == 9 else "json"
    if record["qwen_calls"]:
        return "qwen"
    return record["dialect_ptr"] or "unregistered"


def from_capture(path):
    cases = []
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        cases.append({
            "dialect": dialect_from_capture(record),
            "reasoning": record["reasoning"],
            "initial": INITIAL[int(record["initial"])],
            "response_format": record["response_format"],
            "constrained_tools": record["constrained_tools"],
            "tool_choice": CHOICE[int(record["tool_choice"])],
            "tools": record["tools"],
            "track_spans": record["track_spans"],
            "buffered": record["buffered"],
            "interrupted": record["interrupted"],
            "pieces": [piece(bytes.fromhex(p)) for p in record["pieces_hex"]],
        })
    return cases


def request_tools(body):
    """The declared tools and choice as the endpoints read them."""
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not isinstance(name, str):
            continue
        parameters = function.get("parameters", function.get("input_schema", {}))
        tools.append({"name": name, "parameters_json": json.dumps(parameters, separators=(",", ":"), ensure_ascii=False)})
    choice = body.get("tool_choice")
    forced = None
    if isinstance(choice, str):
        choice = {"any": "required"}.get(choice, choice)
    elif isinstance(choice, dict):
        kind = choice.get("type")
        if kind in ("function", "tool"):
            function = choice.get("function") if isinstance(choice.get("function"), dict) else choice
            forced = function.get("name")
            choice = "required"
        else:
            choice = {"any": "required", "auto": "auto", "none": "none"}.get(kind, "auto")
    else:
        choice = "auto"
    if forced is not None:
        tools = [tool for tool in tools if tool["name"] == forced][:1]
    return tools, choice if choice in CHOICE else "auto"


def response_format(path, body):
    if path == "/v1/responses":
        text = body.get("text")
        return isinstance(text, dict) and isinstance(text.get("format"), dict) and text["format"].get("type") not in (None, "text")
    if path == "/v1/messages":
        return isinstance(body.get("output_format"), dict)
    fmt = body.get("response_format")
    return isinstance(fmt, dict) and fmt.get("type") not in (None, "text")


def from_trace(path):
    requests = {}
    generations = []
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if record.get("event") == "request":
            requests[record.get("request")] = record
        elif record.get("event") == "generation" and "pieces" in record:
            generations.append(record)
    cases = []
    for generation in generations:
        request = requests.get(generation.get("request"))
        if request is None or generation.get("finish") == "cancelled" or "error" in generation:
            continue
        if "initial_output_state" not in generation:
            continue  # A raw completion, which the chat parsers never read.
        body = json.loads(request["body"])
        tools, choice = request_tools(body)
        output = generation["output"].encode("utf-8")
        ends = [end for end in generation["pieces"] if end <= len(output)]
        if generation["pieces"] and generation["pieces"][-1] != len(output):
            continue  # Malformed UTF-8 was replaced, so the offsets no longer fit.
        pieces = []
        begin = 0
        for end in ends:
            pieces.append(output[begin:end])
            begin = end
        if begin < len(output):
            pieces.append(output[begin:])
        stream = bool(body.get("stream"))
        cases.append({
            "dialect": generation.get("tool_dialect"),
            "reasoning": [generation["reasoning"]["open"], generation["reasoning"]["close"]],
            "initial": generation["initial_output_state"],
            "response_format": response_format(request["path"], body),
            "constrained_tools": bool(tools) and choice != "none",
            "tool_choice": choice,
            "tools": tools,
            "track_spans": False,
            "buffered": request["path"] != "/v1/chat/completions" and not stream,
            "interrupted": generation.get("finish") != "stop",
            "pieces": [piece(p) for p in pieces],
        })
    return cases


def from_corpus(path):
    corpus = json.load(open(path, encoding="utf-8"))
    sets = corpus.get("tool_sets", [])
    cases = []
    for case in corpus["cases"]:
        case = dict(case)
        if isinstance(case["tools"], int):
            case["tools"] = sets[case["tools"]]
        cases.append(case)
    return cases, corpus["source"]


def long_calls(sizes):
    write = {"name": "write", "parameters_json": json.dumps({
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}, separators=(",", ":"))}
    cases = []
    for size in sizes:
        lines = []
        while sum(len(line) + 1 for line in lines) < size:
            index = len(lines)
            lines.append(f"    total += values[{index}] * {index % 7}  # step {index}")
        text = ("<tool_call>\n<function=write>\n<parameter=path>\nsrc/long.py\n</parameter>\n"
                "<parameter=content>\n" + "\n".join(lines) + "\n</parameter>\n</function>\n</tool_call>")
        data = text.encode("utf-8")
        cases.append({
            "dialect": "qwen", "reasoning": ["<think>", "</think>"], "initial": "content",
            "response_format": False, "constrained_tools": True, "tool_choice": "auto",
            "tools": [write], "track_spans": False, "buffered": False, "interrupted": False,
            "pieces": [piece(data[i:i + 4]) for i in range(0, len(data), 4)],
        })
    return cases


def write_corpus(path, source, cases):
    sets = []
    keys = {}
    for case in cases:
        key = json.dumps(case["tools"], sort_keys=True)
        if key not in keys:
            keys[key] = len(sets)
            sets.append(case["tools"])
    with open(path, "w", encoding="utf-8") as out:
        out.write('{"source": ' + json.dumps(source) + ',\n "tool_sets": [\n')
        out.write(",\n".join("  " + json.dumps(tools, ensure_ascii=False) for tools in sets))
        out.write('\n ],\n "cases": [\n')
        rows = []
        for index, case in enumerate(cases):
            row = {**case, "name": case.get("name", f"{index:04d}"),
                   "tools": keys[json.dumps(case["tools"], sort_keys=True)]}
            row = {"name": row.pop("name"), **row}
            rows.append("  " + json.dumps(row, ensure_ascii=False))
        out.write(",\n".join(rows))
        out.write("\n]}\n")
    print(f"{path}: {len(cases)} cases, {len(sets)} tool sets, "
          f"{sum(len(text_of(c)) for c in cases)} bytes", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["trace", "capture", "normalize", "long"])
    parser.add_argument("input", nargs="?")
    parser.add_argument("output")
    parser.add_argument("--source")
    parser.add_argument("--max-bytes", type=int, default=0, help="drop cases longer than this")
    parser.add_argument("--bytes", type=int, action="append", default=[], help="long: a call size")
    args = parser.parse_args()
    if args.kind == "long":
        write_corpus(args.output, args.source or "synthetic long Qwen write calls", long_calls(args.bytes))
        return
    if args.kind == "normalize":
        cases, source = from_corpus(args.input)
        source = args.source or source
    else:
        cases = from_trace(args.input) if args.kind == "trace" else from_capture(args.input)
        source = args.source
    if not source:
        parser.error("--source is required")
    if args.max_bytes:
        cases = [case for case in cases if len(text_of(case)) <= args.max_bytes]
    write_corpus(args.output, source, cases if args.kind == "normalize" else reduce(cases))


if __name__ == "__main__":
    main()
