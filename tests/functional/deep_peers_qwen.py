#!/usr/bin/env python3
"""#542/#573 deep conversation plus short same-system peers, with restarts.

Starts the server itself so it can restart it. Prints one JSON line per request
with prompt/cached/prefill tokens and TTFT, and the cache events of interest.
"""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

from qwen_deep_options import setup, retain_response, verify_loaded, stop_process, OwnedRunCleanup, ServerResourceSampler, assert_checkpoint_selection, assert_cold_control, wait_for_published, verify_execution_log
args, out, disk, PORT, CMD = setup('deep-peers')
binary = str(args.binary)
server = None
resource_sampler = None
cleanup = OwnedRunCleanup(out, disk, lambda: server)
MODEL = None
log_index = 0
last_deep_selected = 140000


def start():
    global server, log_index, resource_sampler
    log_index += 1
    log = open(os.path.join(out, f"server-{log_index}.log"), "w")
    server = subprocess.Popen(CMD, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    resource_sampler = ServerResourceSampler(out, f"server-{log_index}", server)
    cleanup.track_sampler(resource_sampler)
    for _ in range(600):
        if server.poll() is not None:
            raise RuntimeError(f"Server exited during startup: {server.returncode}")
        try:
            models = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2))
            if models.get("data"):
                global MODEL
                MODEL = models["data"][0]["id"]
                verify_loaded(out, f"server-{log_index}.log", args.mode)
                resource_sampler.mark_loaded()
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            time.sleep(1)
    raise SystemExit("server did not start")


def stop(kill=False):
    try:
        stop_process(server, kill=kill)
        verify_execution_log(out, f"server-{log_index}.log", args.mode)
    finally:
        if resource_sampler is not None:
            resource_sampler.stop()


LINE = ("Archive record {i}: the harbour ledger lists cargo, tide tables and the "
        "names of ships that called at the northern quay during the season.\n")
SYSTEM = "You are a careful assistant. Answer briefly.\n" + "".join(
    LINE.format(i=i) for i in range(160))


def chat(label, messages, max_tokens=16):
    global last_deep_selected
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    started = time.time()
    result = json.load(urllib.request.urlopen(request, timeout=3600))
    retain_response(out, label, messages, result, time.time() - started)
    usage = result["usage"]
    row = {"label": label, "seconds": time.time() - started,
           "prompt": usage["prompt_tokens"], "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
           "prefill": usage["gufo"]["prefill_tokens"],
           "reuse": usage["gufo"].get("component_cache", {}).get("reuse"),
           "text": result["choices"][0]["message"]["content"]}
    if label == "deep_cold" and row["prompt"] < 140000:
        raise RuntimeError("Deep fixture did not reach the required140K tokens")
    if args.expect_component and not usage["gufo"].get("component_cache"):
        raise RuntimeError("Candidate did not use component cache")
    if args.expect_component and label != "deep_cold":
        is_deep = label.startswith("deep_")
        selected = assert_checkpoint_selection(
            out, label, result,
            restart=label in {"deep_after_restart", "deep_after_second_restart"},
            # The first short chat learns the system-prompt boundary, which
            # the cold deep prompt has no checkpoint for; later ones reuse it.
            minimum=last_deep_selected if is_deep else int(label != "short_0_0"))
        if is_deep:
            last_deep_selected = selected
    print(json.dumps(row), flush=True)
    return result["choices"][0]["message"]["content"]



def cold_control(label, messages, max_tokens=16):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    Path(out, label + "_cold.request.json").write_text(json.dumps(body, indent=2) + "\n")
    raw = json.dumps(body).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     raw, {"Content-Type": "application/json"})
    started = time.time()
    result = json.load(urllib.request.urlopen(request, timeout=3600))
    retain_response(out, label + "_cold", messages, result, time.time() - started)
    cached = json.load(open(os.path.join(out, label + ".response.json")))["response"]
    assert_cold_control(out, label, cached, result)


deep = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "".join(LINE.format(i=1000 + i) for i in range(4300)) +
         "Reply with only the word READY."}]
start()
answer = chat("deep_cold", deep)
deep += [{"role": "assistant", "content": answer}]
# Crash only after observing an actually published deep checkpoint. Shutdown
# must not turn an unfinished writer into a falsely qualified crash recovery.
wait_for_published(out, f"server-{log_index}.log", 140000, server)
stop(kill=True)
if server.returncode != -signal.SIGKILL:
    raise RuntimeError("Controlled crash was not SIGKILL")
start()
deep += [{"role": "user", "content": "Reply with only the word ONE."}]
restored_answer = chat("deep_after_restart", deep)
cold_control("deep_after_restart", deep)
deep += [{"role": "assistant", "content": restored_answer}]
for round_ in range(3):
    for peer in range(4):
        chat(f"short_{round_}_{peer}", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "".join(LINE.format(i=50000 + 1000 * round_ + 100 * peer + i)
                                                for i in range(40)) + "Reply with only OK."}])
    deep += [{"role": "user", "content": "".join(LINE.format(i=9000 + 100 * round_ + i)
                                                 for i in range(60)) +
              f"Reply with only the number {round_ + 2}."}]
    deep += [{"role": "assistant", "content": chat(f"deep_round_{round_}", deep)}]
stop()
start()
deep += [{"role": "user", "content": "Reply with only the word LAST."}]
chat("deep_after_second_restart", deep)
cold_control("deep_after_second_restart", deep)
stop()
cleanup.finish(workload_completed=True)
