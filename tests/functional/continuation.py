#!/usr/bin/env python3
"""Functional HTTP cancellation/replay checks; restart the same server for --restore.

Use a private server with --served-model-name cache-test. Add --cache-disk
only when checking restart persistence; in-memory reuse needs no disk cache.
Run once per AR/speculative backend. Reports contain synthetic requests and
completion hashes, never logits. --image adds an image to each conversation.
Use --append-image to introduce it after a cached text-only turn.
"""

import argparse
import base64
import hashlib
import http.client
import json
from pathlib import Path
import socket
import time
from tool_agent import agent_tools
from urllib.parse import urlsplit
from metrics import Recorder

TRACE = Recorder(None)
# Matches TextRunnerDiskCacheOptions::min_checkpoint_step_tokens.
DISK_CHECKPOINT_STEP = 2048

CASES = tuple(f"{field}-preserve{preserve}-sampled{sampled}"
              for sampled in (0, 1)
              for field in ("reasoning_content", "content")
              for preserve in (0, 1))


def call(url, body, stop_field=None):
    target = urlsplit(url)
    cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = cls(target.hostname, target.port, timeout=180)
    started = time.monotonic()
    measurement = TRACE.begin("/v1/chat/completions", body)
    connection.request("POST", target.path.rstrip("/") + "/v1/chat/completions",
                       json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    measurement.row["http_status"] = response.status
    measurement.row["request_id"] = response.getheader("X-Request-ID")
    if response.status != 200:
        data = response.read()
        measurement.feed(data)
        measurement.ended = True
        measurement.finish()
        connection.close()
        raise RuntimeError(f"HTTP {response.status}: {data.decode()}")
    if stop_field is None:
        data = response.read()
        measurement.feed(data)
        measurement.ended = True
        measurement.finish()
        result = json.loads(data)
        connection.close()
        return result
    message = {"role": "assistant", "content": "", "reasoning_content": ""}
    pieces = 0
    for line in response:
        measurement.feed(line)
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
            measurement.finish()
            return message, time.monotonic() - started
    connection.close()
    measurement.ended = True
    measurement.finish()
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
    parser.add_argument("--append-image", action="store_true",
                        help="Append --image after a cached text-only turn")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"),
                        help="Effort for thinking-on cases; thinking-off cases remain off")
    parser.add_argument("--tools", action="store_true",
                        help="Resume after a completed tool response")
    parser.add_argument("--legacy-tool-history", action="store_true",
                        help="Also replay an unknown historical name after the ordinary tool")
    parser.add_argument("--discard-assistant", action="store_true",
                        help="Drop the interrupted assistant and send '.' like an agent client")
    parser.add_argument("--drop-reasoning", action="store_true",
                        help="Omit reasoning_content from replayed assistants like OpenAI clients")
    parser.add_argument("--prefix-repetitions", type=int, default=32,
                        help="Length of the synthetic shared system prefix")
    parser.add_argument("--case", action="append", choices=CASES,
                        help="Run only this case (repeatable for focused checks)")
    args = parser.parse_args()
    TRACE.path = args.output.with_suffix(".requests.json")
    if args.append_image and not args.image:
        parser.error("--append-image requires --image")
    if args.legacy_tool_history and not args.tools:
        parser.error("--legacy-tool-history requires --tools")
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
            # Disk skips checkpoints within one step of a stored prefix, so a
            # restart may re-prefill that gap. Sampled output can then follow
            # other prefill chunk shapes; greedy and exact restores cannot.
            if measured["prefill"] >= DISK_CHECKPOINT_STEP:
                raise RuntimeError(f"{item['case']}: disk restore lost too much: {measured}")
            exact = not item["request"]["temperature"] or measured["prefill"] == 0
            if exact and digest(result) != item["sha256"]:
                raise RuntimeError(f"{item['case']}: disk restore changed seeded output")
            if "followup" in item:
                followup = call(args.url, item["followup"]["request"])
                if metrics(followup)["cached"] < measured["cached"] or (
                        exact and digest(followup) != item["followup"]["sha256"]):
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
                root_messages = None
                if args.append_image:
                    first_user = {"role": "user", "content":
                                  "Remember these instructions for my next request."}
                    root_messages = [messages[0], first_user]
                    messages[1:1] = [first_user,
                                     {"role": "assistant", "content": "Ready."}]
                if args.tools:
                    messages.extend([
                        {"role": "assistant", "content": "", "reasoning_content": "Read the fixture.",
                         "tool_calls": [{"id": "fixture-call", "type": "function", "function": {
                             "name": "read_fixture", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "fixture-call",
                         "content": "The fixture is ready. Answer the user's request directly."}])
                    if args.legacy_tool_history:
                        messages.extend([
                            {"role": "assistant", "content": None, "tool_calls": [{
                                "id": "legacy-call", "type": "function",
                                "function": {"name": "…", "arguments": "{}"}}]},
                            {"role": "tool", "tool_call_id": "legacy-call",
                             "content": "Unknown tool; no action taken. Continue the user's request."}])
                body = {"model": args.model, "messages": messages, "max_tokens": 256,
                        "temperature": 0.8 if sampled else 0, "seed": 1234,
                        "top_k": 20, "top_p": 0.95,
                        "chat_template_kwargs": {"enable_thinking": thinking,
                                                 "preserve_thinking": preserve},
                        "stream": True}
                if args.tools:
                    body["tools"] = [{"type": "function", "function": {
                        "name": "read_fixture", "description": "Read the fixture.",
                        "parameters": {"type": "object", "properties": {}}}},
                        *agent_tools()]
                if thinking and args.reasoning_effort:
                    body["reasoning_effort"] = args.reasoning_effort
                initial = {**body, "messages": list(messages),
                           "cache_prompt": bool(args.append_image)}

                def interrupt():
                    if root_messages is not None:
                        call(args.url, {**body, "messages": root_messages,
                                        "stream": False, "max_tokens": 1,
                                        "cache_prompt": False})
                    return call(args.url, initial, field)

                assistant, elapsed = interrupt()
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
                # With preservation disabled, the next user turn removes
                # reasoning from the preceding tool cycle too. Its short
                # suffix must be recomputed; the earlier image/user prefix
                # must remain cached.
                rewritten_tool_reasoning = args.tools and not preserve
                max_suffix = 96 if rewritten_tool_reasoning else 16
                if args.discard_assistant and measured["prefill"] > max_suffix:
                    raise RuntimeError(f"{name}: discarded assistant caused re-prefill: {measured}")
                repeated = call(args.url, body)
                if digest(resumed) != digest(repeated):
                    raise RuntimeError(f"{name}: prompt snapshot changed seeded output")
                # Recreate the same interrupted history. A one-shot full
                # prefill control must use another server: cache_prompt=false
                # bypasses reads but retains its differently shaped checkpoint.
                # Running that control here would replace the state under test.
                # The SDK conversation suite separately checks cache bypass.
                replayed_assistant, _ = interrupt()
                if not preserve or args.drop_reasoning:
                    replayed_assistant.pop("reasoning_content", None)
                if replayed_assistant != assistant:
                    raise RuntimeError(f"{name}: cold interrupted stream did not replay")
                replayed = call(args.url, body)
                if digest(resumed) != digest(replayed):
                    raise RuntimeError(f"{name}: cold conversation replay changed output")
                if measured["accepted"] > measured["proposed"]:
                    raise RuntimeError(f"{name}: invalid speculative accounting")
                followup_assistant = dict(resumed["choices"][0]["message"])
                if not preserve or args.drop_reasoning:
                    followup_assistant.pop("reasoning_content", None)
                followup_body = {**body, "max_tokens": 8, "messages": [
                    *body["messages"], followup_assistant,
                    {"role": "user", "content": "What number did I ask you to reply with?"}]}
                followup = call(args.url, followup_body)
                if metrics(followup)["cached"] < measured["cached"]:
                    raise RuntimeError(f"{name}: third turn lost the conversation prefix")
                if digest(followup) != digest(call(args.url, followup_body)):
                    raise RuntimeError(f"{name}: third-turn snapshot changed seeded output")
                report = {"case": name, "request": body, "sha256": digest(resumed),
                          "discard_assistant": args.discard_assistant,
                          "drop_reasoning": args.drop_reasoning,
                          "rewritten_tool_reasoning": rewritten_tool_reasoning,
                          "append_image": args.append_image,
                          "interrupt_seconds": elapsed, **measured, "exact": True,
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
