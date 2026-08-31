"""Parsing of Pi's ``--mode json`` event stream.

Pi emits JSON lines on stdout: a ``session`` header, then agent, turn,
message, and tool-execution events. This module reduces that stream to the
per-task metrics #153 asks for, without retaining prompt or response text --
the raw stream is kept separately as the trajectory artifact, but the metrics
that reach a public result carry no model output.

The stream is treated as untrusted and possibly truncated: a run killed on
timeout ends mid-line, so parsing never assumes a well-formed tail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Usage:
    """Cumulative provider-reported token usage."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int | None = None
    total: int = 0

    @classmethod
    def from_event(cls, raw: dict) -> "Usage":
        return cls(
            input=int(raw.get("input", 0)),
            output=int(raw.get("output", 0)),
            cache_read=int(raw.get("cacheRead", 0)),
            cache_write=int(raw.get("cacheWrite", 0)),
            # Absent from providers that expose no reasoning breakdown; a
            # subset of `output` when present, not an addition to it.
            reasoning=raw.get("reasoning"),
            total=int(raw.get("totalTokens", 0)),
        )


@dataclass
class Trajectory:
    """What one Pi run did, reduced to metrics."""

    session_id: str | None = None
    turns: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    tool_names: dict[str, int] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    stop_reasons: list[str] = field(default_factory=list)
    completed: bool = False
    malformed_lines: int = 0
    # Requests the endpoint refused or failed to serve.
    endpoint_errors: int = 0
    # Turns that ended because the context window was exhausted.
    context_failures: int = 0

    @property
    def requests(self) -> int:
        """Provider requests, approximated by assistant messages.

        Pi does not label requests directly; each assistant message is one
        completion. Retries inside Pi are not visible here.
        """
        return self.assistant_messages


def parse(stream: str) -> Trajectory:
    """Reduce a Pi JSON event stream to a `Trajectory`.

    Tolerates a truncated final line, which is the normal shape of a run
    killed on timeout.
    """
    trajectory = Trajectory()

    for line in stream.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # A killed run ends mid-line. Count it rather than failing: the
            # metrics accumulated so far are still the truth about the run.
            trajectory.malformed_lines += 1
            continue

        if not isinstance(event, dict):
            trajectory.malformed_lines += 1
            continue

        kind = event.get("type")

        if kind == "session":
            trajectory.session_id = event.get("id")

        elif kind == "turn_start":
            trajectory.turns += 1

        elif kind == "message_start":
            if (event.get("message") or {}).get("role") == "assistant":
                trajectory.assistant_messages += 1

        elif kind == "message_update":
            # Cumulative and latest-wins: providers that only report usage at
            # completion leave this at zero until the end.
            if isinstance(event.get("usage"), dict):
                trajectory.usage = Usage.from_event(event["usage"])

        elif kind == "message_end":
            message = event.get("message") or {}
            if isinstance(message.get("usage"), dict):
                trajectory.usage = Usage.from_event(message["usage"])
            stop = message.get("stopReason")
            if stop:
                trajectory.stop_reasons.append(stop)
                if stop == "error":
                    trajectory.endpoint_errors += 1
                elif stop == "length":
                    # The model ran out of room rather than finishing, which
                    # is a context failure, not a wrong answer.
                    trajectory.context_failures += 1

        elif kind == "tool_execution_start":
            trajectory.tool_calls += 1
            name = event.get("toolName") or "unknown"
            trajectory.tool_names[name] = trajectory.tool_names.get(name, 0) + 1

        elif kind == "tool_execution_end":
            if event.get("isError"):
                trajectory.tool_failures += 1

        elif kind == "agent_end":
            trajectory.completed = True

    return trajectory
