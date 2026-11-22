#!/usr/bin/env python3
"""Gufo feature conformance for one served model (gufo serve llm --model <loom package>).

Usage: conformance.py BASE_URL OUT.json [--logits]
Each check records pass/fail/gap and evidence. "gap" marks a Gufo feature the
Loom path cannot honor without the package's logits (the package was not built
with llm.logits); the checks then only
look for graceful behavior. With --logits (the server runs a logits-capable
package with the capability on) the same checks must really hold: a miss is a
"fail", not a "gap".
"""
import concurrent.futures as cf
import http.client
import json, socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE, OUT = sys.argv[1], sys.argv[2]
LOGITS = "--logits" in sys.argv[3:]
URL = urllib.parse.urlparse(BASE)
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
results = []


def call(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


def sse(path, body, timeout=600):
    conn = http.client.HTTPConnection(URL.hostname, URL.port, timeout=timeout)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    events = []
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                events.append(json.loads(line[5:]))
            except Exception:
                pass
    conn.close()
    return resp.status, events


def check(name, status, detail):
    results.append({"check": name, "status": status, "detail": str(detail)[:300]})
    print(f"{status.upper():5} {name}: {str(detail)[:160]}", flush=True)


def needs_logits(ok, detail):
    """pass, or - without logits - the documented gap; with --logits a miss is a failure."""
    return "pass" if ok else ("fail" if LOGITS else "gap")


def guarded(name):
    def wrap(fn):
        try:
            fn()
        except Exception as e:
            check(name, "fail", f"{type(e).__name__}: {e}")
        return fn
    return wrap


def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "temperature": 0, **kw}
    return call("POST", "/v1/chat/completions", body)


def user(text):
    return [{"role": "user", "content": text}]


NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
status, models = call("GET", "/v1/models")
MODEL = models["data"][0]["id"]
_s, _probe = call("POST", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "Say hi."}],
                                                    "temperature": 0, "max_tokens": 400, **NOTHINK})
# Some models reason whatever the request says (gpt-oss, LFM2.5-24B, ...); loom-serve agrees. Give them room.
ALWAYS_REASONS = _s == 200 and bool(_probe["choices"][0]["message"].get("reasoning_content"))
ROOM = 2000 if ALWAYS_REASONS else 0
check("models", "pass" if status == 200 and MODEL else "fail", MODEL)


@guarded("health/ready/props/slots/metrics")
def _():
    codes = {p: call("GET", p)[0] for p in ["/health", "/ready", "/props", "/slots", "/metrics"]}
    check("health/ready/props/slots/metrics", "pass" if all(c == 200 for c in codes.values()) else "fail", codes)


@guarded("chat basic")
def _():
    s, r = chat(user("What is the capital of France? Answer with one word."), max_tokens=256 + ROOM, **NOTHINK)
    text = (r["choices"][0]["message"].get("content") or "") if s == 200 else r
    check("chat basic", "pass" if s == 200 and "paris" in text.lower() else "fail", text)


@guarded("system/developer anywhere")
def _():
    msgs = [{"role": "system", "content": "Always answer in uppercase."}, {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "HELLO."}, {"role": "developer", "content": "Keep answers to one word."},
            {"role": "user", "content": "Say goodbye."}]
    s, r = chat(msgs, max_tokens=256 + ROOM, **NOTHINK)
    check("system/developer anywhere", "pass" if s == 200 and r["choices"][0]["message"].get("content") else "fail",
          r["choices"][0]["message"].get("content") if s == 200 else r)


@guarded("streaming + usage + timings")
def _():
    s, ev = sse("/v1/chat/completions", {"model": MODEL, "messages": user("Count from 1 to 5."), "max_tokens": 64 + ROOM,
                                         "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                                         **NOTHINK})
    text = "".join((e["choices"][0]["delta"].get("content") or "") for e in ev if e.get("choices"))
    usage = [e["usage"] for e in ev if e.get("usage")]
    timings = [e["timings"] for e in ev if e.get("timings")]
    ok = s == 200 and "3" in text and usage and usage[-1].get("completion_tokens") and timings
    check("streaming + usage + timings", "pass" if ok else "fail", f"{text[:40]!r} usage={usage[-1] if usage else None}")


@guarded("reasoning on/off")
def _():
    s1, on = chat(user("What is 23*17? Think step by step, then give the number."), max_tokens=3000,
                  chat_template_kwargs={"enable_thinking": True})
    s2, off = chat(user("What is 23*17? Give only the number."), max_tokens=300, reasoning_effort="none")
    r_on = on["choices"][0]["message"].get("reasoning_content") or "" if s1 == 200 else ""
    c_on = on["choices"][0]["message"].get("content") or "" if s1 == 200 else ""
    r_off = off["choices"][0]["message"].get("reasoning_content") or "" if s2 == 200 else ""
    c_off = off["choices"][0]["message"].get("content") or "" if s2 == 200 else ""
    if s1 != 200 or s2 != 200:
        check("reasoning on/off", "fail", (on, off))
        return
    clean = "<think>" not in c_on and "</think>" not in c_on and "<think>" not in c_off
    if ALWAYS_REASONS:
        check("reasoning on/off", "pass" if clean and "391" in c_on + c_off else "fail",
              f"model always reasons (off not possible); on={c_on[-40:]!r} off={c_off!r}")
    elif not r_on:
        # Models without a thinking mode answer directly.
        check("reasoning on/off", "pass" if clean and "391" in c_on + c_off else "fail",
              f"no reasoning mode; on={c_on[-40:]!r} off={c_off!r}")
    else:
        check("reasoning on/off", "pass" if clean and not r_off and "391" in c_on + c_off else "fail",
              f"reasoning {len(r_on)} chars, on={c_on[-40:]!r}, off reasoning={len(r_off)} {c_off!r}")


@guarded("tool call (auto) + round trip")
def _():
    msgs = user("What's the weather in Paris right now? Use the get_weather tool.")
    s, r = chat(msgs, tools=TOOLS, max_tokens=2000)
    msg = r["choices"][0]["message"] if s == 200 else {}
    calls = msg.get("tool_calls") or []
    ok = calls and calls[0]["function"]["name"] == "get_weather" and "paris" in calls[0]["function"]["arguments"].lower()
    check("tool call (auto)", "pass" if ok else "fail", f"finish={r['choices'][0]['finish_reason'] if s == 200 else r} calls={calls} content={msg.get('content')!r:.80}")
    if not ok:
        return
    msgs += [msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temp_c": 21, "sky": "sunny"}'}]
    s, r = chat(msgs, tools=TOOLS, max_tokens=2000)
    text = r["choices"][0]["message"].get("content") or "" if s == 200 else str(r)
    check("tool round trip", "pass" if "21" in text else "fail", text)


@guarded("streamed tool call")
def _():
    s, ev = sse("/v1/chat/completions", {"model": MODEL, "tools": TOOLS, "max_tokens": 2000, "temperature": 0,
                                         "stream": True, "messages": user("What's the weather in Tokyo? Use the tool.")})
    deltas = [tc for e in ev if e.get("choices") for tc in (e["choices"][0]["delta"].get("tool_calls") or [])]
    ok = any((tc.get("function") or {}).get("name") == "get_weather" for tc in deltas)
    check("streamed tool call", "pass" if ok else "fail", f"{len(deltas)} deltas")


@guarded("tool_choice required")
def _():
    s, r = chat(user("Hello there!"), tools=TOOLS, tool_choice="required", max_tokens=2000 + ROOM)
    calls = r["choices"][0]["message"].get("tool_calls") if s == 200 else None
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    check("tool_choice required", needs_logits(ok, ""),
          f"call made: {bool(calls)}" + ("" if ok else f"; constrained decoding needs logits (HTTP {s}: {str(r)[:200]})"))


@guarded("tool_choice named")
def _():
    tools = TOOLS + [{"type": "function", "function": {
        "name": "get_time", "description": "Get the current time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    choice = {"type": "function", "function": {"name": "get_time"}}
    s, r = chat(user("What is the weather in Paris?"), tools=tools, tool_choice=choice, max_tokens=2000 + ROOM)
    calls = r["choices"][0]["message"].get("tool_calls") if s == 200 else None
    ok = bool(calls) and all(c["function"]["name"] == "get_time" for c in calls)
    check("tool_choice named", needs_logits(ok, ""),
          f"calls={[c['function']['name'] for c in calls] if calls else None}" + ("" if ok else f"; needs logits (HTTP {s}: {str(r)[:200]})"))


@guarded("strict tool schema")
def _():
    tool = {"type": "function", "function": {
        "name": "set_mode", "description": "Set the mode of the device.", "strict": True,
        "parameters": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["eco", "boost", "idle"]}},
                       "required": ["mode"], "additionalProperties": False}}}
    s, r = chat(user("Set the device to turbo mode, nothing else is valid but turbo."), tools=[tool],
                tool_choice="required", max_tokens=2000 + ROOM)
    calls = r["choices"][0]["message"].get("tool_calls") if s == 200 else None
    try:
        mode = json.loads(calls[0]["function"]["arguments"]).get("mode") if calls else None
    except Exception:
        mode = None
    check("strict tool schema", needs_logits(mode in ("eco", "boost", "idle"), ""),
          f"mode={mode!r}" + ("" if mode in ("eco", "boost", "idle") else f"; enum not enforced (needs logits) (HTTP {s}: {str(r)[:200]})"))


@guarded("response_format json_schema")
def _():
    schema = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
        "type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                                         "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city", "country", "unit"], "additionalProperties": False}}}
    s, r = chat(user("Name the capital of Japan as JSON with keys city and country, and unit set to kelvin."),
                response_format=schema, max_tokens=500 + ROOM, **NOTHINK)
    text = r["choices"][0]["message"].get("content") or "" if s == 200 else str(r)
    try:
        obj = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        valid = isinstance(obj, dict) and set(obj) == {"city", "country", "unit"} and obj["unit"] in ("c", "f")
    except Exception:
        valid = False
    check("response_format json_schema", needs_logits(valid and s == 200, ""),
          f"{'valid' if valid else 'invalid'} against the schema: {text[:80]!r}")


@guarded("stop + max_tokens + finish reasons")
def _():
    s1, r1 = chat(user("Count from 1 to 10, comma separated."), max_tokens=64 + ROOM, stop=["5"], **NOTHINK)
    s2, r2 = chat(user("Write a long essay about the ocean."), max_tokens=10, **NOTHINK)
    t1 = r1["choices"][0]["message"]["content"] or "" if s1 == 200 else ""
    if ALWAYS_REASONS and s1 == 200:  # the stop may land in reasoning, which is output too
        t1 = (r1["choices"][0]["message"].get("reasoning_content") or "") + t1
    ok = s1 == 200 and "5" not in t1 and "4" in t1 and r1["choices"][0]["finish_reason"] == "stop" and s2 == 200 and r2["choices"][0]["finish_reason"] == "length" \
        and r2["usage"]["completion_tokens"] == 10
    check("stop + max_tokens + finish reasons", "pass" if ok else "fail",
          f"{t1!r} / length={r2['choices'][0]['finish_reason'] if s2 == 200 else r2} n={r2['usage']['completion_tokens'] if s2 == 200 else ''}")


@guarded("sampling: seed, temperature")
def _():
    kw = dict(max_tokens=40 + ROOM // 4, temperature=1.0, top_p=0.95, top_k=50, **NOTHINK)
    p = user("Write one creative sentence about a cat.")
    out = lambda r: (r["choices"][0]["message"].get("reasoning_content") or "") + (r["choices"][0]["message"]["content"] or "")
    a = out(chat(p, seed=7, **kw)[1])
    b = out(chat(p, seed=7, **kw)[1])
    c = [out(chat(p, seed=s, **kw)[1]) for s in (8, 9, 10)]
    check("sampling: seed, temperature", "pass" if a == b and any(x != a for x in c) else "fail",
          f"same seed equal={a == b}; other seeds differ={any(x != a for x in c)}")


@guarded("sampling: min_p / penalties")
def _():
    def out(r):
        m = r["choices"][0]["message"]
        return (m.get("reasoning_content") or "") + (m.get("content") or "")
    kw = dict(max_tokens=48 + ROOM // 4, **NOTHINK)
    prompt = user("Write one creative sentence about a cat.")
    # min_p is relative to the top probability of the temperature-scaled distribution (threshold: top logit + T * ln(min_p)). min_p = 1
    # keeps only the top token (ties aside) whatever the temperature or seed; a value like 0.97 keeps every token within 0.05 logits of the
    # top at T=1.5, and a creative sentence of 48+ tokens always has such near-ties (a pass of 0.97 depended on the model's logit gaps).
    sampled = {sd: chat(prompt, temperature=1.5, top_k=0, top_p=1.0, min_p=1.0, seed=sd, **kw) for sd in (1, 2, 3)}
    if any(s != 200 for s, _ in sampled.values()):
        check("sampling: min_p / penalties", "fail", [r for s, r in sampled.values() if s != 200][:1])
        return
    texts = [out(r) for _, r in sampled.values()]
    # The reference is greedy on the host as well (logprobs: undrafted, the logit row's argmax), the path min_p takes: a drafted package's
    # device greedy runs other kernels (batched verification) and may part from the row's argmax at a near-tie of its own.
    greedy = out(chat(prompt, logprobs=True, **kw)[1])
    free = [out(chat(prompt, temperature=1.5, top_k=0, top_p=1.0, seed=sd, **kw)[1]) for sd in (1, 2)]
    same = len(set(texts)) == 1
    min_p_ok = same and texts[0] == greedy and free[0] != free[1]
    if not min_p_ok:   # which part, and where: a failure must say what differed
        d = next((i for i, (x, y) in enumerate(zip(texts[0], greedy)) if x != y), min(len(texts[0]), len(greedy)))
        min_p_why = f"seeds agree={same}; equals host greedy={texts[0] == greedy} (first difference at character {d}); unconstrained seeds differ={free[0] != free[1]}"
    # A strong repetition penalty changes a greedy answer that repeats itself.
    rep = user("Repeat the word 'echo' twenty times, separated by spaces.")
    plain = out(chat(rep, **kw)[1])
    punished = out(chat(rep, repeat_penalty=1.6, frequency_penalty=1.0, presence_penalty=1.0, **kw)[1])
    penalty_ok = plain != punished
    check("sampling: min_p / penalties", needs_logits(min_p_ok and penalty_ok, ""),
          f"min_p holds greedy={min_p_ok}; penalties change the output={penalty_ok}" + ("" if min_p_ok else f" ({min_p_why})"))


@guarded("logprobs")
def _():
    s, r = chat(user("Say hi."), max_tokens=5 + ROOM // 4, logprobs=True, top_logprobs=3, **NOTHINK)
    lp = r["choices"][0].get("logprobs") if s == 200 else None
    content = (lp or {}).get("content") or []
    ok = bool(content)
    for entry in content:
        top = entry.get("top_logprobs") or []
        mass = sum(2.718281828 ** t["logprob"] for t in top)
        # greedy: the sampled token is the most likely; probabilities are consistent
        ok = ok and len(top) == 3 and entry["logprob"] <= 1e-6 and mass <= 1.0001 \
            and abs(top[0]["logprob"] - entry["logprob"]) < 1e-3 and top[0]["token"] == entry["token"] \
            and all(top[i]["logprob"] >= top[i + 1]["logprob"] for i in range(2))
    check("logprobs", needs_logits(ok, ""), f"HTTP {s}: {len(content)} tokens; {str(content[:1])[:120]}")


@guarded("Responses API")
def _():
    s, r = call("POST", "/v1/responses", {"model": MODEL, "input": "Name a primary color. One word.",
                                          "max_output_tokens": 3000, "temperature": 0,
                                          "reasoning": {"effort": "none"}})
    text = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", [])) if s == 200 else str(r)
    ok = s == 200 and any(w in text.lower() for w in ("red", "blue", "yellow"))
    s2, r2 = call("POST", "/v1/responses", {"model": MODEL, "input": "What's the weather in Rome? Use the tool.",
                                            "tools": [{"type": "function", **TOOLS[0]["function"]}],
                                            "max_output_tokens": 2000, "temperature": 0})
    calls = [o for o in r2.get("output", []) if o.get("type") == "function_call"] if s2 == 200 else []
    check("Responses API (text + function call)", "pass" if ok and calls else "fail", f"{text[:40]!r} calls={len(calls)}")


@guarded("Anthropic Messages")
def _():
    s, r = call("POST", "/v1/messages", {"model": MODEL, "max_tokens": 3000, "thinking": {"type": "disabled"},
                                         "messages": [{"role": "user", "content": "Say the word hello."}]})
    text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text") if s == 200 else str(r)
    s2, r2 = call("POST", "/v1/messages", {"model": MODEL, "max_tokens": 2000,
                                           "tools": [{"name": "get_weather", "description": "Get weather.",
                                                      "input_schema": TOOLS[0]["function"]["parameters"]}],
                                           "messages": [{"role": "user", "content": "Weather in Oslo? Use the tool."}]})
    uses = [b for b in r2.get("content", []) if b.get("type") == "tool_use"] if s2 == 200 else []
    check("Anthropic Messages (text + tool_use)", "pass" if "hello" in text.lower() and uses else "fail",
          f"{text[:40]!r} tool_use={len(uses)}")


@guarded("raw completions")
def _():
    s, r = call("POST", "/v1/completions", {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8,
                                            "temperature": 0})
    text = r["choices"][0]["text"] if s == 200 else str(r)
    check("raw completions", "pass" if s == 200 and text.strip() else "fail", repr(text))


@guarded("token counting + tokenize")
def _():
    s1, a = call("POST", "/v1/chat/completions/input_tokens", {"model": MODEL, "messages": user("Hello world")})
    s2, b = call("POST", "/tokenize", {"content": "Hello world"})
    s3, c = call("POST", "/v1/messages/count_tokens", {"model": MODEL, "messages": [{"role": "user", "content": "Hello world"}]})
    check("token counting + tokenize", "pass" if s1 == s2 == s3 == 200 else "fail", (a, b, c))


@guarded("cache reuse (multi-turn)")
def _():
    convo = [{"role": "system", "content": "You are a helpful assistant. " + "Be precise and kind. " * 60},
             {"role": "user", "content": "Give me three tips for learning a language."}]
    s, r1 = chat(convo, max_tokens=120 + ROOM, **NOTHINK)
    convo += [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
              {"role": "user", "content": "Now summarize those tips in one sentence."}]
    s, r2 = chat(convo, max_tokens=80 + ROOM, **NOTHINK)
    cached = r2.get("timings", {}).get("cache_n", 0)
    check("cache reuse (multi-turn)", "pass" if cached > 0 and r2["choices"][0]["message"]["content"] else "fail",
          f"cache_n={cached} of {r2['usage']['prompt_tokens']} ({r2['timings']['component_cache']['reuse']})")


@guarded("cache restore exact")
def _():
    # A restored prompt checkpoint must continue exactly like a fresh prefill.
    convo = [{"role": "system", "content": "You are a careful historian. " + "Cite dates exactly. " * 70},
             {"role": "user", "content": "Describe the history of the printing press in four sentences."}]
    fresh = chat(convo, max_tokens=80, cache_prompt=False, **NOTHINK)[1]
    first = chat(convo, max_tokens=80, **NOTHINK)[1]
    again = chat(convo, max_tokens=80, **NOTHINK)[1]
    texts = [x["choices"][0]["message"]["content"] for x in (fresh, first, again)]
    cached = again.get("timings", {}).get("cache_n", 0)
    check("cache restore exact", "pass" if cached > 0 and len(set(texts)) == 1 else "fail",
          f"cache_n={cached}; fresh==first==restored: {len(set(texts)) == 1}")


@guarded("shared prefix across sessions")
def _():
    sys_prompt = {"role": "system", "content": "You are a geography expert. " + "Answer briefly and precisely. " * 200}
    # Gufo learns the shared boundary when two prompts diverge there; the
    # third conversation reuses it.
    ask = lambda c: user(f"I am planning a long trip to {c} next spring with my family of four, including two "
                         "teenagers who love hiking, food markets, museums and beaches. We have three weeks and a "
                         "moderate budget, and we prefer trains to cars, small family-run hotels to large "
                         "resorts, and early mornings to late nights. We would also like one rest day each week "
                         f"and at least one cooking class. What is the capital city of {c}? Answer in one sentence.")
    chat([sys_prompt, *ask("Italy")], max_tokens=20, **NOTHINK)
    chat([sys_prompt, *ask("Spain")], max_tokens=20, **NOTHINK)
    s, r = chat([sys_prompt, *ask("Peru")], max_tokens=20, **NOTHINK)
    cached = r.get("timings", {}).get("cache_n", 0)
    check("shared prefix across sessions", "pass" if cached > 100 else "fail", f"cache_n={cached}")


@guarded("concurrency (greedy consistency)")
def _():
    prompts = [f"Write two sentences about the number {i}." for i in range(4)]
    serial = [chat(user(p), max_tokens=60, cache_prompt=False, **NOTHINK)[1]["choices"][0]["message"]["content"]
              for p in prompts]
    with cf.ThreadPoolExecutor(4) as pool:
        parallel = list(pool.map(lambda p: chat(user(p), max_tokens=60, cache_prompt=False, **NOTHINK)[1]
                                 ["choices"][0]["message"]["content"], prompts))
    same = sum(a == b for a, b in zip(serial, parallel))
    check("concurrency (greedy consistency)", "pass" if same == 4 else "fail", f"{same}/4 identical")


@guarded("cancellation")
def _():
    conn = http.client.HTTPConnection(URL.hostname, URL.port, timeout=60)
    conn.request("POST", "/v1/chat/completions", json.dumps({"model": MODEL, "stream": True, "max_tokens": 3000,
                 "messages": user("Write a very long essay about rivers."), **NOTHINK}), {"Content-Type": "application/json"})
    sock = conn.sock
    resp = conn.getresponse(); resp.read(1500)
    sock.shutdown(socket.SHUT_RDWR); resp.close(); conn.close()  # really disconnect
    t = time.time()
    s, r = chat(user("Say ok."), max_tokens=8, **NOTHINK)
    check("cancellation", "pass" if s == 200 and time.time() - t < 10 else "fail", f"{time.time() - t:.2f}s")


@guarded("reasoning budget")
def _():
    s, r = chat(user("Is 97 prime? Explain."), max_tokens=3000, reasoning_budget_tokens=32,
                chat_template_kwargs={"enable_thinking": True})
    if s != 200:
        check("reasoning budget", "fail", r)
        return
    m = r["choices"][0]["message"]
    rt = r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    reasoning = m.get("reasoning_content") or ""
    ok = (not reasoning) or (rt and rt <= 40) or len(reasoning) < 300
    check("reasoning budget", "pass" if ok and m.get("content") else "fail",
          f"reasoning_tokens={rt} reasoning_chars={len(reasoning)} content={(m.get('content') or '')[:40]!r}")


@guarded("reasoning token usage")
def _():
    s, r = chat(user("Is 91 prime? Answer briefly."), max_tokens=2500, chat_template_kwargs={"enable_thinking": True})
    if s != 200:
        check("reasoning token usage", "fail", r)
        return
    reasoning = r["choices"][0]["message"].get("reasoning_content") or ""
    rt = r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    ok = bool(reasoning) == (rt > 0) and rt <= r["usage"]["completion_tokens"]
    check("reasoning token usage", "pass" if ok else "fail", f"reasoning_chars={len(reasoning)} reasoning_tokens={rt}")


@guarded("reasoning effort")
def _():
    out = {}
    for effort in ("low", "high"):
        s, r = chat(user("What is 23*17? Give the number."), max_tokens=2500, reasoning_effort=effort)
        if s != 200:
            check("reasoning effort", "fail", f"{effort}: {r}")
            return
        m = r["choices"][0]["message"]
        out[effort] = (len(m.get("reasoning_content") or ""), m.get("content") or "")
    ok = all("391" in c for _, c in out.values())
    check("reasoning effort", "pass" if ok else "fail",
          f"low reasoning {out['low'][0]} chars, high {out['high'][0]} chars")


json.dump({"model": MODEL, "results": results}, open(OUT, "w"), indent=1)
counts = {k: sum(r["status"] == k for r in results) for k in ("pass", "fail", "gap")}
print(json.dumps(counts))
