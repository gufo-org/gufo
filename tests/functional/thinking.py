"""Thinking off, for models that can't switch it off (PR #1340).

Suites ask for thinking off (`reasoning_effort: "none"`, `enable_thinking:
false`, `reasoning.effort: "none"`) to get a short, fixed answer. Some models
have no off switch: Inkling 2 Mini's lowest effort level still writes a
thought (29 to 345 tokens on short prompts, and on some prompts it doesn't
stop: it counted the 24 repetitions of a schema constant for 4,000 tokens), as
its model card says and as llama.cpp serves it. Gufo serves such a model as its
template does, so the suites state what they need instead of assuming there is
no thought:

* `bounded(body)` gives a thinking-off Chat Completions or Responses body
  without a budget a reasoning budget of BUDGET tokens, as a client of such a
  model would; `bound_thought` applies it to the SDK client's requests, and
  the raw-HTTP scripts call it themselves.
* `answer_room(n)` is the cap of a reply that must finish: `n` tokens for
  the answer, plus the bounded thought. Caps that test the limit itself stay
  exact.
* `no_reasoning(value)` replaces assertions that a thinking-off reply carries
  no reasoning (text or token count): it holds for any value when the model
  may still think.

The run sets this from the model's Devlab entry (`thinking_off_reasons`),
passed as `--thinking-off-reasons`.
"""

import json

# The reasoning budget of a thinking-off request, and the room its cap gets
# for the thought, the forced close and the answer block's framing.
BUDGET = 1024
ROOM = BUDGET + 16

_reasons_when_off = False


def configure(reasons_when_off):
    global _reasons_when_off
    _reasons_when_off = bool(reasons_when_off)


def reasons_when_off():
    """Whether a thinking-off request may still return a thought."""
    return _reasons_when_off


def answer_room(tokens):
    """The cap for a thinking-off reply that must finish within `tokens`."""
    return tokens + ROOM if _reasons_when_off else tokens


def no_reasoning(value):
    """A thinking-off reply's reasoning (text or token count) is absent, or
    the model may still think."""
    return _reasons_when_off or not value


def thinking_off(body):
    """Whether a Chat Completions or Responses body turns thinking off."""
    kwargs = body.get("chat_template_kwargs")
    reasoning = body.get("reasoning")
    thinking = body.get("thinking")
    return (body.get("reasoning_effort") == "none"
            or (isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False)
            or (isinstance(reasoning, dict) and reasoning.get("effort") == "none")
            or (isinstance(thinking, dict) and thinking.get("type") == "disabled"))


def bounded(body):
    """A Chat Completions or Responses body, with a reasoning budget when it
    turns thinking off on a model that may still think (and has none)."""
    if (not _reasons_when_off or not isinstance(body, dict) or not thinking_off(body)
            or "reasoning_budget_tokens" in body):
        return body
    return {**body, "reasoning_budget_tokens": BUDGET}


def bound_thought(request):
    """httpx request hook: `bounded` for the harness's SDK requests."""
    if (not _reasons_when_off or request.method != "POST"
            or not request.url.path.endswith(("/chat/completions", "/responses"))):
        return
    try:
        body = json.loads(request.content)
    except ValueError:
        return
    budgeted = bounded(body)
    if budgeted is body:
        return
    content = json.dumps(budgeted).encode()
    import httpx
    request.stream = httpx.ByteStream(content)
    request._content = content
    request.headers["Content-Length"] = str(len(content))
