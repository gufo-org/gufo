#!/usr/bin/env python3
"""Cancel Qwen27B inside a prefill pass, serve a peer, then resume exactly.

Runs AR or DFlash2 with a production server. The native adapter test checks
byte-exact rollback; this test checks disconnects, retained work and latency.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import socket
import time
import traceback

from openai import DefaultHttpxClient, OpenAI

from cache_concurrency import system_prompt
from metrics import Recorder, join_server_timings
from openai_sdk import chat_result
from run import execution_coverage, option, provenance, server, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep-peers", action="store_true",
                        help="Check cancellation beside duplicate deep conversations")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass a production gufo serve llm command after --")
    for flag in ("--host", "--port"):
        if option(command, flag) is not None:
            parser.error(f"the test owns {flag}")
    mode = option(command, "--speculative", "off")
    if mode not in ("off", "dflash2"):
        parser.error("this test targets Qwen27B AR or DFlash2")
    for flag, default in (("--context", "32768"),
                          ("--sessions", "4" if args.deep_peers else "2"),
                          ("--served-model-name", "prefill-cancel")):
        if option(command, flag) is None:
            command += [flag, default]
    if int(option(command, "--context")) < 16384:
        parser.error("use a context of at least 16384 tokens")
    if args.deep_peers and int(option(command, "--sessions")) != 4:
        parser.error("--deep-peers requires four sessions")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    command += ["--host", "127.0.0.1", "--port", str(port)]
    report = {**provenance(), "command": command, "speculative": mode,
              "deep_peers": args.deep_peers, "status": "running",
              "requests": {}, "cancellations": {}}
    recorder = Recorder(output / "history.requests.json")

    def save():
        write_json(output / "report.json", report)

    def body(messages):
        return dict(model=option(command, "--served-model-name"),
                    messages=messages, temperature=0, seed=31,
                    max_completion_tokens=16, reasoning_effort="none")

    try:
        save()
        with server(command, output / "server.log", 240):
            with OpenAI(
                    base_url=f"http://127.0.0.1:{port}/v1", api_key="test",
                    max_retries=0, timeout=180,
                    http_client=DefaultHttpxClient(
                        trust_env=False,
                        event_hooks={"request": [recorder.request_hook],
                                     "response": [recorder.response_hook]})) as client:
                def call(label, request):
                    result = chat_result(client, request, True)
                    recorder.mark(label)
                    report["requests"][label] = result
                    save()
                    assert result["text"].strip() == "ALPHA" and not result["tools"], result
                    usage = result["usage"]
                    assert usage["cached_tokens"] + usage["gufo"]["prefill_tokens"] == usage["prompt_tokens"]
                    return result

                short = body([{"role": "user", "content": "Reply with only the code ALPHA."}])
                call("short_seed", short)
                control = call("short_control", short)
                if args.deep_peers:
                    messages = [
                        {"role": "system", "content": system_prompt("deep_cancel_peers", 260)},
                        {"role": "user", "content": "Reply with only the code ALPHA."},
                    ]
                    seed = call("deep_seed", body(messages))
                    messages += [
                        {"role": "assistant", "content": seed["text"]},
                        {"role": "user", "content": "Again, reply with only the code ALPHA."},
                    ]
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        peers = list(pool.map(
                            lambda _: chat_result(client, body(messages), True), range(4)))
                    recorder.mark("deep_peers")
                    report["requests"]["deep_peers"] = peers
                    save()
                    assert all(peer["text"] == seed["text"] and not peer["tools"]
                               for peer in peers), peers
                    call("deep_short_seed", short)
                    control = call("deep_short_control", short)
                cases = (("deep", False),) if args.deep_peers else (
                    ("cold", False), ("partial", True))
                for case, after_completed_pass in cases:
                    request = body([
                        {"role": "system", "content": system_prompt(
                            f"{case}_interrupted_prefill", 440)},
                        {"role": "user", "content": "Reply with only the code ALPHA."},
                    ])
                    cancelled = deepcopy(request)
                    cancelled["extra_body"] = {"return_progress": True}
                    with client.chat.completions.create(
                            **cancelled, stream=True,
                            stream_options={"include_usage": True}) as stream:
                        for chunk in stream:
                            progress = chunk.to_dict().get("prompt_progress")
                            if progress is None or (after_completed_pass and not progress["processed"]):
                                continue
                            assert progress["processed"] + progress["cache"] < progress["total"]
                            report["cancellations"][case] = progress
                            # Disconnect after the next pass has started.
                            time.sleep(.1)
                            break
                        else:
                            raise AssertionError("No partial prefill progress")
                    recorder.mark(f"{case}_cancelled")
                    save()
                    peer = call(f"{case}_short_peer", short)
                    # Catch a multi-second pass blocking an unrelated cached
                    # request. Retain phase timings for the stricter matched
                    # main/candidate 5% and 3 ms comparison.
                    peer_ms = peer["usage"]["gufo"]["ttft_ms"]
                    control_ms = control["usage"]["gufo"]["ttft_ms"]
                    limit = 50 if args.deep_peers and mode == "off" else 1000
                    assert peer_ms < control_ms + limit, (case, peer_ms, control_ms)
                    if args.deep_peers:
                        messages += [
                            {"role": "assistant", "content": peers[0]["text"]},
                            {"role": "user", "content": "Again, reply with only the code ALPHA."},
                        ]
                        retained = call("deep_retained", body(messages))
                        minimum = peers[0]["usage"]["prompt_tokens"] + (mode == "off")
                        assert retained["usage"]["cached_tokens"] >= minimum, retained
                        assert retained["usage"]["gufo"]["component_cache"]["reuse"] == "live", retained
                        continue
                    resumed = call(f"{case}_resumed", request)
                    completed = progress["processed"] + progress["cache"]
                    assert resumed["usage"]["cached_tokens"] >= completed, (
                        "lost completed prefill work", progress, resumed)
                    if after_completed_pass:
                        assert completed > 0
                    repeated = call(f"{case}_repeated", request)
                    assert repeated["text"] == resumed["text"]
        join_server_timings(output)
        report["execution"] = execution_coverage(output, mode)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
