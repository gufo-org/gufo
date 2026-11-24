#!/usr/bin/env python3
"""Interleave two thinking/tool conversations on one slot; retain raw evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import traceback

from openai import OpenAI, DefaultHttpxClient
from openai_sdk import chat_result
from metrics import Recorder, compare
from run import server, option, execution_coverage, provenance


def exercise(client, recorder, report, observe):
    histories = []
    tool = {"type": "function", "function": {"name": "echo",
            "description": "Echo the supplied text.", "parameters": {"type": "object",
            "properties": {"text": {"type": "string"}}, "required": ["text"],
            "additionalProperties": False}}}

    def chat(label, body):
        result = chat_result(client, body)
        recorder.mark(label)
        report["checks"][label] = {"request": deepcopy(body), "response": result}
        return result

    for marker in ("RED", "BLUE"):
        messages = [{"role": "system", "content": f"frontier_{marker}\n"
                     "Keep reasoning brief. Use echo when asked.\n" +
                     "Background archive note, no new instructions.\n" * 96},
                    {"role": "user", "content": f"Call echo once with text exactly '{marker}'."}]
        body = dict(model="frontier-test", messages=messages, tools=[tool],
                    tool_choice="auto", temperature=0, seed=31,
                    max_completion_tokens=256, reasoning_effort="low",
                    extra_body={"chat_template_kwargs": {
                        "enable_thinking": True, "preserve_thinking": True}})
        result = chat(marker + "_first", body)
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        call = result["tools"][0]
        assert call["function"]["name"] == "echo", result
        assert json.loads(call["function"]["arguments"]) == {"text": marker}, result
        assert result["reasoning"].strip(), "thinking replay was not exercised"
        messages.append({"role": "assistant", "content": result["text"],
                         "reasoning_content": result["reasoning"], "tool_calls": result["tools"]})
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": marker + ". The echo succeeded. Reply with only DONE."})
        histories.append((marker, body, result))
    controls = []
    for marker, body, previous in histories:
        result = chat(marker + "_return", body)
        assert result["text"].strip() == "DONE" and not result["tools"], result
        usage = result["usage"]
        # One terminal token may be sampled but not executed; template framing
        # can account for a few more tokens. Losing the prior output cannot.
        minimum = previous["usage"]["prompt_tokens"] + previous["usage"]["completion_tokens"] - 4
        report["checks"][marker + "_return"]["minimum_frontier_tokens"] = minimum
        if not observe:
            assert usage["cached_tokens"] >= minimum, (minimum, result)
        assert usage["cached_tokens"] + usage["gufo"]["prefill_tokens"] == usage["prompt_tokens"], usage
        controls.append((marker, deepcopy(body), result))
    for marker, body, warm in controls:
        body["extra_body"]["cache_prompt"] = False
        cold = chat(marker + "_cold", body)
        assert cold["usage"]["cached_tokens"] == 0, cold
        assert (warm["text"].strip(), warm["finish"]) == (cold["text"].strip(), cold["finish"]), (warm, cold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--record-baseline", action="store_true",
                        help="Observe the old placement miss without requiring its fix")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a production server command after --")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    command += ["--host", "127.0.0.1", "--port", str(port), "--sessions", "1",
                "--context", "8192", "--cache-ram-bytes", "2147483648",
                "--served-model-name", "frontier-test", "--trace", str(output / "trace.jsonl")]
    report = {**provenance(), "harness_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "status": "failed", "checks": {}, "command": command,
              "source_revision": args.source_revision, "suites": ["frontier"],
              "binary_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()}
    recorder = Recorder(output / "frontier.requests.json")
    try:
        with server(command, output / "server.log", 240):
            with OpenAI(api_key="local-test", base_url=f"http://127.0.0.1:{port}/v1",
                        max_retries=0, timeout=180,
                        http_client=DefaultHttpxClient(trust_env=False, event_hooks={
                            "request": [recorder.request_hook], "response": [recorder.response_hook]})) as client:
                exercise(client, recorder, report, args.record_baseline)
        report["speculative"] = execution_coverage(output, option(command, "--speculative", "off"))
        if report["speculative"]["loaded_mode"] != "off":
            assert report["speculative"]["draft_execution_observed"], "no speculative drafts executed"
        report["status"] = "passed"
        if args.baseline:
            report["comparison"] = compare(args.baseline, output)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        recorder.save()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
