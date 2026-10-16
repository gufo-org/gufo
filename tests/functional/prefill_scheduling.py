"""A small request must make progress beside a longer cold prefill."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading

from progress import ProgressTrace
from stream_start import cold_prompt


def check_prefill_scheduling(client, model, checks, chat_result, width, context):
    assert width >= 2 and context >= 16384, "use --sessions 2 --context 32768 or larger"
    common = dict(model=model, temperature=0, seed=31, top_p=1,
                  max_completion_tokens=8, presence_penalty=0, frequency_penalty=0,
                  reasoning_effort="none", extra_body={
                      "top_k": 0, "min_p": 0, "repeat_penalty": 1,
                      "cache_prompt": False, "return_progress": True,
                      "chat_template_kwargs": {"enable_thinking": False}})
    short = {**deepcopy(common), "messages": [
        {"role": "user", "content": "What is 2 + 2? Reply only the digit."}]}
    control = chat_result(client, short, True)
    checks["prefill_short_control"] = control

    # The first positive progress event aligns arrival by work rather than
    # wall time. Keep a real multi-chunk prefill, without filling the context.
    long = {**deepcopy(common), "max_completion_tokens": 1, "messages": [
        {"role": "user", "content": cold_prompt("prefill_scheduling", min(context, 65536))}]}
    ready = threading.Event()
    lock = threading.Lock()
    trace = ProgressTrace(True)
    at_arrival = None
    at_output = None

    def long_progress(chunk):
        nonlocal at_arrival
        with lock:
            trace(chunk)
            if trace.updates:
                update = trace.updates[-1]
                if 0 < update["processed"] < update["total"] and at_arrival is None:
                    at_arrival = dict(update)
                    ready.set()

    def short_progress(chunk):
        nonlocal at_output
        if any(choice.delta.content for choice in chunk.choices):
            with lock:
                if at_output is None:
                    at_output = dict(trace.updates[-1])

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(chat_result, client, long, True, long_progress)
        assert ready.wait(120), "long prompt did not publish intermediate progress"
        result = chat_result(client, short, True, short_progress)
        completed = pending.result(timeout=180)

    # Mark the cohort on its owner thread; timing records still retain each
    # request and phase separately. Faster peers cannot hide a slow request.
    checks["prefill_overlap"] = {
        "short": result, "long": completed, "arrival": at_arrival, "first_output": at_output}
    for key in ("text", "reasoning", "tools", "finish"):
        assert result[key] == control[key], (key, control, result)
    assert result["text"].strip().rstrip(".") == "4", result
    assert at_output and at_output["processed"] < at_output["total"], (
        "short output waited for the entire peer prefill", at_output)
    for row in (control, result, completed):
        usage = row["usage"]
        assert usage["cached_tokens"] == 0, usage
        assert usage["gufo"]["prefill_tokens"] == usage["prompt_tokens"], usage
    assert completed["usage"]["completion_tokens"] == 1, completed
    assert trace.updates[-1]["processed"] == trace.updates[-1]["total"], trace.updates
