"""Validate llama.cpp-compatible prompt progress without counting it as output."""


class ProgressTrace:
    def __init__(self, enabled, allow_missing=False):
        self.enabled = enabled
        self.allow_missing = allow_missing
        self.updates = []
        self.output_seen = False

    def __call__(self, chunk):
        event = chunk if isinstance(chunk, dict) else chunk.to_dict()
        update = event.get("prompt_progress")
        output = event.get("type", "").endswith(".delta") or any(
            choice.get("text") or any(choice.get("delta", {}).get(field)
                                     for field in ("content", "reasoning_content", "tool_calls"))
            for choice in event.get("choices", []))
        if update is not None:
            assert self.enabled, "unrequested prompt progress"
            assert not self.output_seen and not output, "progress followed generated output"
            assert set(update) == {"total", "cache", "processed", "time_ms"}, update
            assert all(type(value) is int and value >= 0 for value in update.values()), update
            assert 0 <= update["cache"] <= update["processed"] <= update["total"], update
            if self.updates:
                old = self.updates[-1]
                assert (update["total"], update["cache"]) == (old["total"], old["cache"])
                assert update["processed"] >= old["processed"] and update["time_ms"] >= old["time_ms"]
            if "type" in event:
                assert event["type"] == "response.in_progress", event
                assert event["response"]["status"] == "in_progress", event
            self.updates.append(update)
        self.output_seen |= bool(output)

    def finish(self, measured):
        if self.enabled and not self.allow_missing:
            assert self.updates, "requested prompt progress is missing"
        if self.updates:
            final = self.updates[-1]
            assert final["processed"] == final["total"] == measured["prompt_tokens"], final
            assert final["cache"] == measured["cached_tokens"], (final, measured)
