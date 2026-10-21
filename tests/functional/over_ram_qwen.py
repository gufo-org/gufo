#!/usr/bin/env python3
"""dblaber's #562 steps C, D and E2 with a conversation larger than the RAM budget.

C: deep turn with many new tokens, disconnect after a completed pass, then resend.
D: three short same-system chats (sessions=3), then a deep append.
E: deep append, then the same history with the final user message rewritten.
"""
import json, os, subprocess, time, urllib.request
import http.client
from pathlib import Path

from qwen_deep_options import setup, retain_response, verify_loaded, stop_process, OwnedRunCleanup, ServerResourceSampler, completed_partial_prefill, assert_cancelled_frontier, assert_checkpoint_selection, assert_cold_control, assert_publication_wait, verify_execution_log
args, out, disk, PORT, CMD = setup('over-ram')
binary = str(args.binary)
server = None
cleanup = OwnedRunCleanup(out, disk, lambda: server)
log = open(os.path.join(out, "server.log"), "w")
server = subprocess.Popen(CMD, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
resource_sampler = ServerResourceSampler(out, "server", server)
cleanup.track_sampler(resource_sampler)
MODEL = None
cancel_observation = None
last_deep_selected = 1
for _ in range(600):
    if server.poll() is not None:
        raise RuntimeError(f"Server exited during startup: {server.returncode}")
    try:
        data = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2))
        if data.get("data"):
            MODEL = data["data"][0]["id"]
            verify_loaded(out, "server.log", args.mode)
            resource_sampler.mark_loaded()
            break
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        pass
    time.sleep(1)

if MODEL is None:
    raise SystemExit("server did not start")

LINE = ("Archive record {i}: the harbour ledger lists cargo, tide tables and the "
        "names of ships that called at the northern quay during the season.\n")
SYSTEM = "You are a careful assistant. Answer briefly.\n" + "".join(
    LINE.format(i=i) for i in range(140))


def body(messages):
    return json.dumps({"model": MODEL, "messages": messages, "max_tokens": 16,
                       "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()


def chat(label, messages):
    global last_deep_selected
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     body(messages), {"Content-Type": "application/json"})
    offset = Path(out, "server.log").stat().st_size
    started = time.time()
    result = json.load(urllib.request.urlopen(request, timeout=3600))
    retain_response(out, label, messages, result, time.time() - started)
    usage = result["usage"]
    if args.expect_component and not usage["gufo"].get("component_cache"):
        raise RuntimeError("Candidate did not use component cache")
    if label == "A1_cold" and args.ram_bytes != "auto" and usage["prompt_tokens"] * 65536 <= int(args.ram_bytes):
        raise RuntimeError("Qwen FP16 KV alone did not exceed the RAM cache budget")
    if label == "C_resend":
        assert_cancelled_frontier(out, label, result, cancel_observation,
                                  component=args.expect_component)
    if args.expect_component and label != "A1_cold":
        # A cold deep prompt is checkpointed at its stable boundary and grid
        # points, not at the system prompt. The first short chat learns that
        # shared boundary; the later ones must reuse it.
        if label.startswith("D_short_"):
            minimum = 0 if label == "D_short_0" else 1
        elif label == "E2_rewritten":
            # Editing the final message forks the live execution. Its previous
            # selected boundary need not exist as a restorable checkpoint.
            # The publication assertion below requires the actual pending
            # checkpoint to publish and become this request's selected frontier.
            minimum = 1
        else:
            minimum = last_deep_selected
        selected = assert_checkpoint_selection(out, label, result, minimum=minimum)
        if not label.startswith("D_short_"):
            last_deep_selected = selected
    if label == "E2_rewritten":
        with Path(out, "server.log").open("rb") as stream:
            stream.seek(offset)
            segment = stream.read().decode(errors="replace")
        assert_publication_wait(out, label, segment, result)
    print(json.dumps({"label": label, "seconds": time.time() - started,
                      "prompt": usage["prompt_tokens"], "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                      "reuse": usage["gufo"].get("component_cache", {}).get("reuse")}),
          flush=True)
    return result["choices"][0]["message"]["content"]


def cold_control(label, messages):
    payload = json.loads(body(messages))
    payload["cache_prompt"] = False
    Path(out, label + "_cold.request.json").write_text(json.dumps(payload, indent=2) + "\n")
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     json.dumps(payload).encode(), {"Content-Type": "application/json"})
    started = time.time()
    result = json.load(urllib.request.urlopen(request, timeout=3600))
    retain_response(out, label + "_cold", messages, result, time.time() - started)
    cached = json.loads(Path(out, label + ".response.json").read_text())["response"]
    assert_cold_control(out, label, cached, result)


def disconnect(messages, after):
    global cancel_observation
    log_path = Path(out, "server.log")
    offset = log_path.stat().st_size
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    sent = time.monotonic()
    try:
        connection.request("POST", "/v1/chat/completions", body(messages),
                           {"Content-Type": "application/json"})
        sent = time.monotonic()
        # Cancellation can now roll back an interrupted pass. Observe actual
        # completed work before disconnecting, instead of relying on the server
        # finishing its first pass after the client has already gone away.
        deadline = sent + 120
        while True:
            with log_path.open("rb") as stream:
                stream.seek(offset)
                segment = stream.read().decode(errors="replace")
            try:
                completed_partial_prefill(segment)
                break
            except ValueError as error:
                reason = str(error)
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"No completed pass before cancellation: {reason}")
            time.sleep(0.01)
        time.sleep(after)
        # Closing the connection releases both its socket and HTTP state.
        connection.close()
        closed = time.monotonic()
    finally:
        connection.close()
    deadline = time.monotonic() + 120
    while True:
        with log_path.open("rb") as stream:
            stream.seek(offset)
            segment = stream.read().decode(errors="replace")
        try:
            cancel_observation = completed_partial_prefill(segment)
            break
        except ValueError as error:
            reason = str(error)
        if server.poll() is not None or time.monotonic() >= deadline:
            Path(out, "C_disconnect.observation.json").write_text(json.dumps({
                "qualified": False, "reason": reason, "raw_log": segment,
                "disconnect_seconds": closed - sent,
                "server_returncode": server.poll()}, indent=2) + "\n")
            raise RuntimeError("No completed partial-prefill step observed after disconnect")
        time.sleep(0.05)
    Path(out, "C_disconnect.observation.json").write_text(json.dumps({
        "qualified": False, "completed_step": cancel_observation,
        "disconnect_seconds": closed - sent, "raw_log": segment,
        "reason": "Resend frontier and cold/main numerical comparison still required"}, indent=2) + "\n")
    print(json.dumps({"label": "disconnect", "requested_after": after,
                      "actual_after": closed - sent, "completed_step": cancel_observation}), flush=True)


def lines(base, count):
    return "".join(LINE.format(i=base + i) for i in range(count))


deep = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": lines(1000, 2600) + "Reply with only READY."}]
deep.append({"role": "assistant", "content": chat("A1_cold", deep)})
deep += [{"role": "user", "content": "Reply with only ONE."}]
deep.append({"role": "assistant", "content": chat("A2_append", deep)})
# C: many new tokens, disconnect 20 ms after a completed pass, resend.
c = deep + [{"role": "user", "content": lines(20000, 300) + "Reply with only TWO."}]
disconnect(c, 0.02)
answer = chat("C_resend", c)
# Run cold comparison after the cold-slot scenario so it cannot itself evict
# the retained cancelled frontier whose survival D is intended to check.
deep = c + [{"role": "assistant", "content": answer}]
# D: three short chats with the same system prompt, then a deep append.
for n in range(3):
    chat(f"D_short_{n}", [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": lines(50000 + 100 * n, 20) + "Reply OK."}])
deep += [{"role": "user", "content": "Reply with only THREE."}]
deep.append({"role": "assistant", "content": chat("D_deep_append", deep)})
# E: append, then the same history with the final user message rewritten.
# A substantial shared user block ends before the volatile final user message.
# Its stable checkpoint is compatible with E2 and deeper than the old frontier.
shared = deep + [{"role": "user", "content": lines(70000, 350)}]
e1 = shared + [{"role": "user", "content": "Reply with only FOUR."}]
chat("E1_append", e1)
e2 = shared + [{"role": "user", "content": "Reply with only the word FIVE."}]
chat("E2_rewritten", e2)
cold_control("C_resend", c)
cold_control("E2_rewritten", e2)
stop_process(server)
verify_execution_log(out, "server.log", args.mode)
cleanup.finish(workload_completed=True)
