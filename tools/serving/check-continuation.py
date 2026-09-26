#!/usr/bin/env python3
"""Short HTTP cancellation/replay checks; restart the same server for --restore.

Use a private server with --served-model-name cache-test. Add --cache-disk
only when checking restart persistence; in-memory reuse needs no disk cache.
Run once per AR/speculative backend. Reports contain synthetic requests and
completion hashes, never logits. --image adds an image to each conversation.
"""

import argparse
import base64
import hashlib
import http.client
import json
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit

CASES = tuple(f"{field}-preserve{preserve}-sampled{sampled}"
              for sampled in (0, 1)
              for field in ("reasoning_content", "content")
              for preserve in (0, 1))


def call(url, body, stop_field=None):
    target = urlsplit(url)
    cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = cls(target.hostname, target.port, timeout=180)
    started = time.monotonic()
    connection.request("POST", target.path.rstrip("/") + "/v1/chat/completions",
                       json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read().decode()}")
    if stop_field is None:
        result = json.loads(response.read())
        connection.close()
        return result
    message = {"role": "assistant", "content": "", "reasoning_content": ""}
    pieces = 0
    for line in response:
        if not line.startswith(b"data: "):
            continue
        raw = line[6:].strip()
        if raw == b"[DONE]":
            break
        for choice in json.loads(raw).get("choices", []):
            delta = choice.get("delta", {})
            for field in ("content", "reasoning_content"):
                if delta.get(field):
                    message[field] += delta[field]
            pieces += bool(delta.get(stop_field))
        if pieces >= 8:
            if connection.sock is not None:
                connection.sock.shutdown(socket.SHUT_RDWR)
            response.close()
            connection.close()
            return message, time.monotonic() - started
    connection.close()
    raise RuntimeError(f"fixture never reached eight {stop_field} deltas")


def digest(result):
    message = result["choices"][0]["message"]
    return hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()


def metrics(result):
    usage = result["usage"]
    detail = usage["gufo"]
    return {"cached": usage["cached_tokens"], "prefill": detail["prefill_tokens"],
            "ttft_ms": detail["ttft_ms"], "disk": detail["cache_disk_hit"],
            "accepted": usage.get("draft_tokens_accepted", 0),
            "proposed": usage.get("draft_tokens", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5815")
    parser.add_argument("--model", default="cache-test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--tools", action="store_true",
                        help="Resume after a completed tool response")
    parser.add_argument("--discard-assistant", action="store_true",
                        help="Drop the interrupted assistant and send '.' like an agent client")
    parser.add_argument("--drop-reasoning", action="store_true",
                        help="Omit reasoning_content from replayed assistants like OpenAI clients")
    parser.add_argument("--prefix-repetitions", type=int, default=32,
                        help="Length of the synthetic shared system prefix")
    parser.add_argument("--case", action="append", choices=CASES,
                        help="Run only this case (repeatable for focused checks)")
    args = parser.parse_args()
    reports = []
    if args.restore:
        previous = json.loads(args.restore.read_text())
        for item in previous:
            if args.case and item["case"] not in args.case:
                continue
            result = call(args.url, item["request"])
            measured = metrics(result)
            if not measured["disk"] or measured["cached"] == 0:
                raise RuntimeError(f"{item['case']}: restart did not restore disk state")
            if digest(result) != item["sha256"]:
                raise RuntimeError(f"{item['case']}: disk restore changed seeded output")
            if "followup" in item:
                followup = call(args.url, item["followup"]["request"])
                if not metrics(followup)["cached"] or digest(followup) != item["followup"]["sha256"]:
                    raise RuntimeError(f"{item['case']}: disk-restored third turn differs")
            report = {"case": item["case"], **measured, "exact": True}
            reports.append(report)
            print(json.dumps(report), flush=True)
    else:
        for sampled in (False, True):
            for field, preserve in (("reasoning_content", False),
                                    ("reasoning_content", True),
                                    ("content", False), ("content", True)):
                name = f"{field}-preserve{int(preserve)}-sampled{int(sampled)}"
                if args.case and name not in args.case:
                    continue
                thinking = field == "reasoning_content"
                content = ("Derive the sum of the first 1000 squares step by step."
                           if thinking else "Count from one to one hundred, separated by commas.")
                if args.image:
                    encoded = base64.b64encode(args.image.read_bytes()).decode()
                    content = [{"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{encoded}"}},
                        {"type": "text", "text": content}]
                messages = [
                    {"role": "system", "content": name + ". " +
                     "Follow the user instruction carefully and answer accurately. " *
                     args.prefix_repetitions},
                    {"role": "user", "content": content}]
                if args.tools:
                    messages.extend([
                        {"role": "assistant", "content": "", "reasoning_content": "Read the fixture.",
                         "tool_calls": [{"id": "fixture-call", "type": "function", "function": {
                             "name": "read_fixture", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "fixture-call",
                         "content": "The fixture is ready. Answer the user's request directly."}])
                body = {"model": args.model, "messages": messages, "max_tokens": 256,
                        "temperature": 0.8 if sampled else 0, "seed": 1234,
                        "top_k": 20, "top_p": 0.95,
                        "chat_template_kwargs": {"enable_thinking": thinking,
                                                 "preserve_thinking": preserve},
                        "stream": True}
                if args.tools:
                    body["tools"] = [{"type": "function", "function": {
                        "name": "read_fixture", "description": "Read the fixture.",
                        "parameters": {"type": "object", "properties": {}}}}]
                initial = {**body, "messages": list(messages), "cache_prompt": False}
                assistant, elapsed = call(args.url, initial, field)
                if not preserve or args.drop_reasoning:
                    assistant.pop("reasoning_content", None)
                if not args.discard_assistant:
                    messages.append(assistant)
                messages.append({"role": "user", "content":
                    "." if args.discard_assistant else "Now reply with only the number 7."})
                body.update(stream=False, max_tokens=12)
                resumed = call(args.url, body)
                measured = metrics(resumed)
                if measured["cached"] < 128:
                    raise RuntimeError(f"{name}: interrupted conversation lost its prefix: {measured}")
                if args.discard_assistant and measured["prefill"] > 16:
                    raise RuntimeError(f"{name}: discarded assistant caused re-prefill: {measured}")
                repeated = call(args.url, body)
                cold = call(args.url, {**body, "cache_prompt": False})
                if metrics(cold)["cached"] or cold["usage"]["gufo"]["cache_hit"]:
                    raise RuntimeError(f"{name}: cache_prompt=false was ignored")
                if digest(resumed) != digest(repeated):
                    raise RuntimeError(f"{name}: prompt snapshot changed seeded output")
                # Recreate the interrupted history from a cold first turn.
                # A one-shot full prefill has different matrix shapes; report
                # that comparison separately from exact cache/history replay.
                replayed_assistant, _ = call(args.url, initial, field)
                if not preserve or args.drop_reasoning:
                    replayed_assistant.pop("reasoning_content", None)
                if replayed_assistant != assistant:
                    raise RuntimeError(f"{name}: cold interrupted stream did not replay")
                replayed = call(args.url, body)
                if digest(resumed) != digest(replayed):
                    raise RuntimeError(f"{name}: cold conversation replay changed output")
                if measured["accepted"] > measured["proposed"]:
                    raise RuntimeError(f"{name}: invalid speculative accounting")
                followup_body = {**body, "max_tokens": 8, "messages": [
                    *body["messages"], resumed["choices"][0]["message"],
                    {"role": "user", "content": "What number did I ask you to reply with?"}]}
                followup = call(args.url, followup_body)
                if metrics(followup)["cached"] < measured["cached"]:
                    raise RuntimeError(f"{name}: third turn lost the conversation prefix")
                if digest(followup) != digest(call(args.url, followup_body)):
                    raise RuntimeError(f"{name}: third-turn snapshot changed seeded output")
                report = {"case": name, "request": body, "sha256": digest(resumed),
                          "discard_assistant": args.discard_assistant,
                          "drop_reasoning": args.drop_reasoning,
                          "interrupt_seconds": elapsed, **measured, "exact": True,
                          "full_prefill_equal": digest(resumed) == digest(cold),
                          "followup": {"request": followup_body,
                                       "sha256": digest(followup), **metrics(followup)}}
                reports.append(report)
                print(json.dumps({k: v for k, v in report.items()
                                  if k not in ("request", "followup")}), flush=True)
                args.output.write_text(json.dumps(reports, indent=2) + "\n")
    if not reports:
        raise RuntimeError("no matching continuation cases")
    args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
