"""Mid-conversation system/developer messages: acceptance, effect and reuse.

Qwen3.8's template accepts one leading system turn, so Qwen hoists later
system/developer messages into it and the injection turn is prefilled again;
DeepSeek renders them in place. Injection-turn reuse is therefore recorded, not
required. The message then stays put, so the following turn must reuse it all.
"""

from copy import deepcopy
import sys


def history(label):
    # A long conversation body after a short system prompt, as in agent
    # sessions, so the evidence shows the cost of rewriting the prompt head.
    return [
        {"role": "system", "content": label + "\nYou are a terse assistant."},
        {"role": "user", "content": "Background notes:\n" +
         "The archive contains routine observations from an earlier session.\n" * 160},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Reply with only ALPHA."},
    ]


REMINDER = ("Reminder: the status word changed. From now on, whenever the user "
            "asks for the status word, reply with only BETA.")
QUESTION = "Reply with only the status word."


def check_system_injection(client, model, checks, chat_result):
    failures = []

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    def chat_work(result):
        usage = result["usage"]
        total, reused = usage["prompt_tokens"], usage["cached_tokens"]
        prefilled = usage["gufo"]["prefill_tokens"]
        assert reused + prefilled == total, usage
        return total, reused, prefilled

    def response_work(response):
        usage = response.usage.to_dict()
        details = usage["input_tokens_details"]
        total, reused = usage["input_tokens"], details["cached_tokens"]
        prefilled = details["cache_write_tokens"]
        assert reused + prefilled == total, usage
        return total, reused, prefilled

    def evidence(label, previous_total, work, required):
        total, reused, prefilled = work
        checks[label + "_evidence"] = {
            "previous_prompt_tokens": previous_total, "prompt_tokens": total,
            "cached_tokens": reused, "prefill_tokens": prefilled,
            "full_reuse_required": required}
        if required and reused < previous_total:
            failures.append(f"{label}: earlier prompt was not reused: "
                            f"{checks[label + '_evidence']}")

    # Chat Completions, for both roles. The reminder persists in history.
    for role in ("system", "developer"):
        label = f"system_injection_chat_{role}"
        request = dict(model=model, messages=history(label), temperature=0, seed=37,
                       max_completion_tokens=16, extra_body={
                           "chat_template_kwargs": {"enable_thinking": False}})
        first = record(label + "_first", chat_result(client, request))
        first_total, _, _ = chat_work(first)
        assert first_total >= 1024 and first["text"].strip() == "ALPHA", first

        injected = deepcopy(request)
        injected["messages"] += [
            {"role": "assistant", "content": first["text"]},
            {"role": role, "content": REMINDER},
            {"role": "user", "content": QUESTION},
        ]
        warm = record(label + "_injected", chat_result(client, injected))
        evidence(label + "_injected", first_total, chat_work(warm), required=False)
        control_request = deepcopy(injected)
        control_request["extra_body"]["cache_prompt"] = False
        control = record(label + "_cold_control", chat_result(client, control_request))
        assert chat_work(control)[1] == 0, control
        assert (warm["text"], warm["usage"]["completion_tokens"]) == \
            (control["text"], control["usage"]["completion_tokens"]), (warm, control)
        assert warm["text"].strip() == "BETA", warm

        followup = deepcopy(injected)
        followup["messages"] += [{"role": "assistant", "content": warm["text"]},
                                 {"role": "user", "content": QUESTION}]
        later = record(label + "_followup", chat_result(client, followup))
        evidence(label + "_followup", chat_work(warm)[0], chat_work(later), required=True)
        assert later["text"].strip() == "BETA", later

    # The exact #339 shape: a reminder inserted before the latest user turn of
    # an already-served request.
    label = "system_injection_chat_before_last_user"
    request = dict(model=model, messages=history(label), temperature=0, seed=37,
                   max_completion_tokens=16, extra_body={
                       "chat_template_kwargs": {"enable_thinking": False}})
    first_total = chat_work(record(label + "_first", chat_result(client, request)))[0]
    request["messages"][-1:-1] = [{"role": "system", "content": REMINDER}]
    request["messages"][-1]["content"] = QUESTION
    before_last = record(label + "_injected", chat_result(client, request))
    evidence(label + "_injected", first_total, chat_work(before_last), required=False)
    assert before_last["text"].strip() == "BETA", before_last

    # Responses: Codex records developer items before later user turns.
    label = "system_injection_responses"
    messages = history(label)
    request = dict(model=model, instructions=messages[0]["content"], input=messages[1:],
                   temperature=0, max_output_tokens=16, reasoning={"effort": "none"},
                   store=False)
    first = client.responses.create(**request)
    record(label + "_first", first.to_dict())
    assert first.status == "completed" and first.output_text.strip() == "ALPHA", first
    injected = {**request, "input": [
        *request["input"], *first.output,
        {"type": "message", "role": "developer",
         "content": [{"type": "input_text", "text": REMINDER}]},
        {"role": "user", "content": QUESTION}]}
    warm = client.responses.create(**injected)
    record(label + "_injected", warm.to_dict())
    evidence(label + "_injected", response_work(first)[0], response_work(warm),
             required=False)
    assert warm.status == "completed" and warm.output_text.strip() == "BETA", warm
    followup = {**injected, "input": [*injected["input"], *warm.output,
                                      {"role": "user", "content": QUESTION}]}
    later = client.responses.create(**followup)
    record(label + "_followup", later.to_dict())
    evidence(label + "_followup", response_work(warm)[0], response_work(later),
             required=True)
    assert later.status == "completed" and later.output_text.strip() == "BETA", later

    # Retain every reproduction before failing.
    assert not failures, "\n".join(failures)
