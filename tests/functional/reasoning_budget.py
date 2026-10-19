"""Reasoning token budgets, as llama.cpp defines them (#266).

`reasoning_budget_tokens` (Chat and Responses), Messages `thinking.budget_tokens`
and `--reasoning-budget` count the tokens before the reasoning end: Qwen's
`\\n</think>` is forced once the budget is spent. This ASCII fixture reaches
the cap without needing an extra token to complete a UTF-8 character.
"""

from copy import deepcopy
import json
import sys

import openai

from messages_tools import messages_stream

PROMPT = "Is 1001 a prime number? Think it through step by step, then answer yes or no."
BUDGETS = (0, 1, 4, 16, 64)


def same(result, expected):
    # Responses keeps the newline before the reasoning end; Chat and Messages
    # trim it.
    return (result[0].strip(), result[1]) == (expected[0].strip(), expected[1])


def check_reasoning_budget(client, model, checks, chat_result):
    def record(label, value):
        checks["reasoning_budget_" + label] = value
        print(f"CHECK reasoning_budget_{label}", file=sys.stderr, flush=True)
        return value

    def responses(label, budget=None, temperature=0, seed=None,
                  budget_field="reasoning_budget_tokens", prompt=PROMPT,
                  max_tokens=600, require_complete=True):
        extra = {} if budget is None else {budget_field: budget}
        if seed is not None:
            extra["seed"] = seed
        response = client.responses.create(
            model=model, input=prompt, max_output_tokens=max_tokens,
            temperature=temperature, extra_body=extra)
        assert not require_complete or response.status == "completed", response
        reasoning = "".join(part.text for item in response.output
                            if item.type == "reasoning" for part in item.summary)
        usage = response.usage.to_dict()
        return record(label, {"text": response.output_text, "reasoning": reasoning,
                              "reasoning_tokens":
                                  usage["output_tokens_details"]["reasoning_tokens"],
                              "usage": usage})

    def chat(label, budget, streaming=False, budget_field="reasoning_budget_tokens"):
        body = {"model": model, "temperature": 0, "max_tokens": 600,
                "messages": [{"role": "user", "content": PROMPT}],
                "extra_body": {budget_field: budget}}
        return record(label, chat_result(client, body, streaming))

    def messages(label, thinking, streaming=False):
        body = {"model": model, "max_tokens": 600, "temperature": 0,
                "thinking": thinking, "messages": [{"role": "user", "content": PROMPT}]}
        if streaming:
            result = messages_stream(client, body)
            blocks = result["blocks"]
        else:
            result = client.post("/messages", body=body, cast_to=object)
            blocks = result["content"]
        assert result["stop_reason" if not streaming else "finish"] == "end_turn", result
        record(label, result)
        return ("".join(b["thinking"] for b in blocks if b["type"] == "thinking"),
                "".join(b["text"] for b in blocks if b["type"] == "text"))

    # The prompt must reason past every budget for the caps to be observable.
    control = responses("control")
    assert control["reasoning_tokens"] > max(BUDGETS) and control["text"].strip(), control
    unlimited = chat("chat_unlimited", -1)
    assert same((unlimited["reasoning"], unlimited["text"]),
                (control["reasoning"], control["text"])), (unlimited, control)

    for budget in BUDGETS:
        capped = responses(f"responses_{budget}", budget)
        assert capped["reasoning_tokens"] == budget, capped
        assert capped["text"].strip(), capped
        # Greedy reasoning follows the uncapped reasoning until the cap.
        assert control["reasoning"].startswith(capped["reasoning"].rstrip()), (capped, control)
        assert budget or not capped["reasoning"].strip(), capped
        for streaming in (False, True):
            result = chat(f"chat_{budget}_{'stream' if streaming else 'buffered'}",
                          budget, streaming)
            assert same((result["reasoning"], result["text"]),
                        (capped["reasoning"], capped["text"])), (result, capped)
            assert result["finish"] == "stop", result
        if budget == 0:
            continue
        for streaming in (False, True):
            thinking, text = messages(
                f"messages_{budget}_{'stream' if streaming else 'buffered'}",
                {"type": "enabled", "budget_tokens": budget}, streaming)
            assert same((thinking, text), (capped["reasoning"], capped["text"])), \
                (thinking, text, capped)

    # As in llama.cpp, budget_tokens applies to enabled thinking only.
    adaptive, _ = messages("messages_adaptive", {"type": "adaptive", "budget_tokens": 4})
    assert adaptive.strip() == control["reasoning"].strip(), (adaptive, control)

    # llama.cpp retains the older spelling for existing clients.
    expected = checks["reasoning_budget_responses_16"]
    legacy_response = responses("legacy_responses", 16,
                                budget_field="thinking_budget_tokens")
    legacy_chat = chat("legacy_chat", 16, budget_field="thinking_budget_tokens")
    for result in (legacy_response, legacy_chat):
        assert same((result["reasoning"], result["text"]),
                    (expected["reasoning"], expected["text"])), (result, expected)

    markup_prompt = (
        "Think through how to safely quote the XML literal <tool_call> inside "
        "JSON and Markdown code fences. Then give one short example.")
    markup = responses("markup_control", prompt=markup_prompt,
                       max_tokens=256, require_complete=False)
    for budget in (64, 192):
        capped = responses(f"markup_{budget}", budget, prompt=markup_prompt,
                           max_tokens=320, require_complete=False)
        assert capped["reasoning_tokens"] == min(budget, markup["reasoning_tokens"]), capped
        assert markup["reasoning"].startswith(capped["reasoning"].rstrip()), (capped, markup)
        assert capped["text"].strip(), capped

    for budget in (1, 2):
        unicode_result = responses(
            f"utf8_{budget}", budget, prompt="请用中文思考并回答：1001是质数吗？",
            max_tokens=128, require_complete=False)
        assert "\ufffd" not in unicode_result["reasoning"] + unicode_result["text"], unicode_result
        assert unicode_result["text"].strip(), unicode_result

    # After reasoning closes, markup inside JSON is data. A budget must not
    # restart on a literal <think> and inject its forced end into that data.
    literal = "<think>alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu</think>"
    for streaming in (False, True):
        body = {
            "model": model, "temperature": 0, "max_tokens": 128,
            "messages": [{"role": "user", "content": "Return the required JSON object."}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "literal_reasoning_tags", "strict": True,
                "schema": {"type": "object", "properties": {
                    "text": {"type": "string", "enum": [literal]}},
                    "required": ["text"], "additionalProperties": False}}},
            "extra_body": {"reasoning_budget_tokens": 4},
        }
        result = record(f"literal_{'stream' if streaming else 'buffered'}",
                        chat_result(client, body, streaming))
        assert result["finish"] == "stop", result
        assert json.loads(result["text"]) == {"text": literal}, result

    for invalid in (-2, 1.5, "8", 2147483648, 1e30):
        try:
            client.chat.completions.create(
                model=model, max_tokens=8, messages=[{"role": "user", "content": PROMPT}],
                extra_body={"reasoning_budget_tokens": invalid})
            raise AssertionError(f"accepted reasoning_budget_tokens={invalid!r}")
        except openai.BadRequestError as error:
            assert "reasoning_budget_tokens" in str(error), error
            record(f"invalid_{invalid}", str(error))

    try:
        messages("invalid_messages_overflow",
                 {"type": "enabled", "budget_tokens": 1e30})
        raise AssertionError("accepted an out-of-range Messages budget")
    except openai.BadRequestError as error:
        assert "thinking.budget_tokens" in str(error), error
        record("invalid_messages_overflow", str(error))

    # Sampled drafts may never propose the forced end; it still comes exactly
    # at the budget.
    for seed in (1, 2, 3):
        sampled = responses(f"sampled_{seed}", 16, temperature=1.0, seed=seed)
        assert sampled["reasoning_tokens"] == 16 and sampled["text"].strip(), sampled

    # A Messages client replays the capped thinking block unchanged; the next
    # turn must reuse the first turn as completely as an uncapped one. Qwen
    # reuses all of it; DeepSeek's template drops replayed thinking.
    reuse = {}
    for label, thinking in (("uncapped", {"type": "enabled"}),
                            ("capped", {"type": "enabled", "budget_tokens": 16})):
        body = {"model": model, "max_tokens": 600, "temperature": 0, "thinking": thinking,
                "system": f"reasoning_budget_replay_{label}",
                "messages": [{"role": "user", "content": PROMPT}]}
        first = record(f"replay_{label}_first", client.post("/messages", body=body,
                                                            cast_to=object))
        body = deepcopy(body)
        body["messages"] += [{"role": "assistant", "content": first["content"]},
                             {"role": "user", "content": "And 1003? Answer yes or no."}]
        second = record(f"replay_{label}_second", client.post("/messages", body=body,
                                                              cast_to=object))
        before = first["usage"]["input_tokens"] + first["usage"]["cache_read_input_tokens"]
        reused = second["usage"]["cache_read_input_tokens"]
        generated = first["usage"]["output_tokens"]
        reuse[label] = (reused >= before + generated - 1, max(before - reused, 0))
    assert reuse["capped"] == reuse["uncapped"], reuse
