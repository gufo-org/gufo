"""Endpoint capability checks for gufo-agent-eval.

#153 specifies consuming the #203 conformance fixture. #203 is closed, but its
closing comment records that the versioned fixtures "remain open", and none
exist in this repository. Rather than block, this is a minimal preflight
covering only what Pi actually exercises: if it passes, a run can proceed; if
a capability is missing the endpoint is marked unsupported and the run aborts.

Per #153, missing behaviour is never emulated.

When the #203 fixture lands, this should converge on it rather than persist as
a parallel contract.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Preflight:
    checks: list[Check] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


def _post(base_url: str, api_key: str, payload: dict, timeout: float = 120):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def run(base_url: str, model: str, api_key: str, timeout: float = 120) -> Preflight:
    """Check the endpoint supports what the agent needs."""
    result = Preflight()

    # 1. Model listing and selection.
    try:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            listed = [m.get("id") for m in json.load(response).get("data", [])]
        result.checks.append(
            Check("models", model in listed, f"served: {', '.join(map(str, listed))}")
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        result.checks.append(Check("models", False, str(exc)))
        return result  # nothing else can work

    base = {"model": model, "max_tokens": 32}

    # 2. Non-streamed chat with a system message.
    try:
        with _post(base_url, api_key, {
            **base,
            "messages": [
                {"role": "system", "content": "Reply with the single word OK."},
                {"role": "user", "content": "Say OK."},
            ],
        }, timeout) as response:
            body = json.load(response)
        choice = body["choices"][0]
        result.checks.append(Check("chat", True, f"finish={choice.get('finish_reason')}"))
        result.checks.append(
            Check("usage", isinstance(body.get("usage"), dict),
                  "usage object present" if body.get("usage") else "no usage reported")
        )
    except Exception as exc:  # noqa: BLE001 - any failure means unsupported
        result.checks.append(Check("chat", False, str(exc)[:160]))
        return result

    # 3. Streaming, which is how Pi drives the endpoint.
    try:
        with _post(base_url, api_key, {
            **base,
            "stream": True,
            "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        }, timeout) as response:
            saw_chunk = any(
                line.startswith(b"data:") for line in response if line.strip()
            )
        result.checks.append(Check("streaming", saw_chunk, "SSE chunks received"))
    except Exception as exc:  # noqa: BLE001
        result.checks.append(Check("streaming", False, str(exc)[:160]))

    # 4. Tool definitions and tool-call emission. The agent is useless without
    #    these, so a failure here is fatal rather than cosmetic.
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    try:
        with _post(base_url, api_key, {
            **base,
            "max_tokens": 128,
            "tools": [tool],
            "tool_choice": "auto",
            "messages": [
                {"role": "user", "content": "Read the file /etc/hostname using the tool."}
            ],
        }, timeout) as response:
            body = json.load(response)
        message = body["choices"][0].get("message", {})
        accepted = "tool_calls" in message or message.get("content") is not None
        result.checks.append(
            Check("tools", accepted,
                  "tool_calls emitted" if message.get("tool_calls")
                  else "tool definitions accepted, no call emitted")
        )
    except Exception as exc:  # noqa: BLE001
        result.checks.append(Check("tools", False, str(exc)[:160]))

    # 5. Tool-result messages, which the agent sends back every turn.
    try:
        with _post(base_url, api_key, {
            **base,
            "messages": [
                {"role": "user", "content": "Read /etc/hostname"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file",
                                     "arguments": '{"path": "/etc/hostname"}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "example-host"},
            ],
            "tools": [tool],
        }, timeout) as response:
            json.load(response)
        result.checks.append(Check("tool_results", True, "accepted"))
    except Exception as exc:  # noqa: BLE001
        result.checks.append(Check("tool_results", False, str(exc)[:160]))

    return result
