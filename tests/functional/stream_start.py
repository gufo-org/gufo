"""Plain streams commit headers before prompt processing or a long queue ends."""

import threading
import time

# Server-internal bound after which a queued stream commits its headers.
QUEUE_START_S = 5
ENDPOINTS = ("chat", "completions", "responses")


def cold_prompt(label, context):
    # About a third of one session, so prefill clearly precedes the first token.
    facts = " ".join(f"Entry {i} keeps value {i * 7 % 101}." for i in range(context // 30))
    return f"{label}: {facts}\nWhich value does entry 3 keep? Answer with the number."


def open_stream(client, model, endpoint, prompt, max_tokens):
    common = dict(model=model, temperature=0, stream=True)
    if endpoint == "chat":
        return client.chat.completions.create(
            **common, messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens, stream_options={"include_usage": True},
            extra_body={"cache_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": False}})
    if endpoint == "completions":
        # Legacy stream frames have a nullable finish_reason, as in completion_result.
        lenient = client.copy(_extra_kwargs={"_strict_response_validation": False})
        return lenient.completions.create(
            **common, prompt=prompt, max_tokens=max_tokens,
            stream_options={"include_usage": True},
            extra_body={"cache_prompt": False, "ignore_eos": True})
    return client.responses.create(
        **common, input=prompt, max_output_tokens=max_tokens, reasoning={"effort": "none"},
        extra_body={"cache_prompt": False})


def is_output(endpoint, event):
    if endpoint == "responses":
        return event.type.endswith(".delta")
    for choice in event.choices:
        if endpoint == "completions":
            if choice.text:
                return True
        elif choice.delta.content or getattr(choice.delta, "reasoning_content", None):
            return True
    return False


def stream_timing(client, model, endpoint, prompt, max_tokens, opened_event=None, stop=None):
    """Seconds to response headers and to the first generated output."""
    started = time.monotonic()
    stream = open_stream(client, model, endpoint, prompt, max_tokens)
    opened = time.monotonic() - started
    if opened_event is not None:
        opened_event.set()
    first, usage = None, None
    with stream:
        for event in stream:
            if first is None and is_output(endpoint, event):
                first = time.monotonic() - started
            if getattr(event, "usage", None):
                usage = event.usage.to_dict()
            if stop is not None and first is not None and stop(first):
                break
    return {"opened_s": opened, "first_output_s": first, "usage": usage}


def check_stream_start(client, model, checks, width, context):
    # Admission commits the response, so keepalives cover the whole prefill.
    for endpoint in ENDPOINTS:
        timing = stream_timing(client, model, endpoint, cold_prompt(endpoint, context), 4)
        prefill_s = checks.recorder.rows[-1]["metrics"]["prefill_ms"] / 1000
        checks[f"stream_start_{endpoint}"] = {**timing, "prefill_s": prefill_s}
        assert timing["first_output_s"] is not None, timing
        assert prefill_s >= 0.5, f"prompt too short to separate headers: {prefill_s}"
        assert timing["first_output_s"] - timing["opened_s"] >= 0.5 * prefill_s, (
            "headers waited for prompt processing", timing, prefill_s)

    # Occupy every session, then queue one stream behind them. Its headers
    # must arrive after the bound while it is still queued, not at admission.
    occupied = [threading.Event() for _ in range(width)]
    release = threading.Event()
    max_tokens = min(2048, context // 2)

    def occupy(index):
        def hold(_first):
            occupied[index].set()
            return release.is_set()
        return stream_timing(client, model, "completions", f"Count upward from {index}:",
                             max_tokens, stop=hold)

    threads, results = [], [None] * width
    for index in range(width):
        thread = threading.Thread(target=lambda i=index: results.__setitem__(i, occupy(i)))
        thread.start()
        threads.append(thread)
    try:
        for event in occupied:
            assert event.wait(120), "sessions were not occupied"
        opened = threading.Event()
        queued = {}

        def wait_queued():
            queued.update(stream_timing(client, model, "chat", "Name a color.", 4, opened))
        submitted = time.monotonic()
        waiter = threading.Thread(target=wait_queued)
        waiter.start()
        # A deferred-only server opens after the peers finish; bound the wait.
        # Keep the queue occupied past the bound even if headers came first.
        opened.wait(QUEUE_START_S + 10)
        time.sleep(max(0, submitted + QUEUE_START_S + 1 - time.monotonic()))
    finally:
        release.set()
        for thread in threads:
            thread.join(120)
    waiter.join(120)
    assert not waiter.is_alive() and queued, "queued stream did not complete"
    for result in results:
        assert result is not None and result["first_output_s"] is not None, results
    queue_s = queued["usage"]["gufo"]["queue_ms"] / 1000
    checks["stream_start_queued"] = {**queued, "queue_s": queue_s}
    assert queue_s >= QUEUE_START_S - 0.5, f"stream was not queued past the bound: {queue_s}"
    assert queued["opened_s"] <= QUEUE_START_S + 2, queued
    assert queued["opened_s"] < queue_s, ("headers waited for admission", queued)
