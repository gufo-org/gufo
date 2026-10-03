"""Prometheus accounting checks for the isolated functional-test server."""

import http.client
import json
import math
import re
import time
from urllib.parse import urlsplit

PROMPT = "llamacpp:prompt_tokens_total"
GENERATED = "llamacpp:tokens_predicted_total"
PROCESSING = "llamacpp:requests_processing"
DEFERRED = "llamacpp:requests_deferred"
PROMPT_SPEED = "llamacpp:prompt_tokens_seconds"
GENERATED_SPEED = "llamacpp:predicted_tokens_seconds"
CACHED = "llamacpp:prompt_tokens_cached_total"
PROMPT_SECONDS = "llamacpp:prompt_seconds_total"
GENERATED_SECONDS = "llamacpp:tokens_predicted_seconds_total"
MAX_SEQUENCE = "llamacpp:n_tokens_max"
DRAFT_ROUNDS = "llamacpp:spec_decode_num_drafts_total"
DRAFTS = "llamacpp:spec_decode_num_draft_tokens_total"
ACCEPTED = "llamacpp:spec_decode_num_accepted_tokens_total"
KV_USAGE = "llamacpp:kv_cache_usage_ratio"
INTEGER_COUNTERS = (PROMPT, GENERATED, CACHED, MAX_SEQUENCE, DRAFT_ROUNDS, DRAFTS, ACCEPTED)
COUNTERS = (*INTEGER_COUNTERS, PROMPT_SECONDS, GENERATED_SECONDS)
INTEGERS = (*INTEGER_COUNTERS, PROCESSING, DEFERRED)
TYPES = {name: "counter" if name in COUNTERS else "gauge"
         for name in (*COUNTERS, PROCESSING, DEFERRED, PROMPT_SPEED, GENERATED_SPEED, KV_USAGE)}


def parse_metrics(text):
    values, types, help_text = {}, {}, set()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "#":
            if len(parts) >= 4 and parts[2] in TYPES:
                if parts[1] == "TYPE":
                    if parts[2] in types:
                        raise ValueError(f"duplicate metric type: {parts[2]}")
                    types[parts[2]] = parts[3]
                elif parts[1] == "HELP":
                    help_text.add(parts[2])
            continue
        name = parts[0]
        if name not in TYPES:
            continue
        if name in values or len(parts) != 2:
            raise ValueError(f"invalid/duplicate metric: {name}")
        value = float(parts[1])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid metric value: {name}={value}")
        if name in INTEGERS:
            if not value.is_integer():
                raise ValueError(f"fractional integer metric: {name}")
            # Avoid rounding large process-lifetime counters through float.
            value = int(parts[1])
        values[name] = value
    if set(values) != set(TYPES) or types != TYPES or help_text != set(TYPES):
        raise ValueError("missing metric, HELP, or incorrect TYPE")
    if values[KV_USAGE] > 1:
        raise ValueError("invalid KV ratio")
    return values


def assert_accounting(before, after, requests, *, include_seconds=True):
    """Completed SDK requests must contribute their actual uncached work once."""
    completed = [r for r in requests if r["http_status"] < 400]
    fields = [(PROMPT, "prefill_tokens"), (GENERATED, "completion_tokens"),
              (CACHED, "cached_tokens"), (DRAFT_ROUNDS, "draft_rounds"),
              (DRAFTS, "draft_tokens"),
              (ACCEPTED, "draft_tokens_accepted")]
    for metric, field in fields:
        expected = 0
        for request in requests:
            if request["http_status"] >= 400:
                continue
            value = request.get("metrics", {}).get(field)
            if request["status"] != "complete" or type(value) is not int or value < 0:
                raise AssertionError(f"missing completed request accounting: {request}")
            expected += value
        actual = after[metric] - before[metric]
        if actual != expected:
            raise AssertionError(f"{metric}: expected delta {expected}, got {actual}")
    maximum = max([before[MAX_SEQUENCE], *[
        r["metrics"]["prompt_tokens"] + r["metrics"]["completion_tokens"]
        for r in completed]])
    if after[MAX_SEQUENCE] != maximum:
        raise AssertionError(f"incorrect maximum sequence: {after[MAX_SEQUENCE]} != {maximum}")
    if include_seconds:
        for metric, field in ((PROMPT_SECONDS, "prefill_ms"),
                              (GENERATED_SECONDS, "decode_ms")):
            expected = sum(r["metrics"][field] for r in completed) / 1000
            actual = after[metric] - before[metric]
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
                raise AssertionError(f"{metric}: expected delta {expected}, got {actual}")


def validate_metrics_report(directory):
    """Account for cancelled work too, after terminal server logs have drained."""
    checks = json.loads((directory / "metrics.json").read_text())["checks"]
    requests = json.loads((directory / "metrics.requests.json").read_text())["requests"]
    completed = {}
    for line in (directory / "server.log").read_text().splitlines():
        if "event=completed " in line:
            fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
            if "request" in fields:
                completed[fields["request"]] = fields
    finalized = []
    for request in requests:
        fields = completed.get(request.get("request_id"))
        if fields is None:
            raise ValueError("metrics request has no completed server log")
        if request["http_status"] == 400:
            continue  # The suite's invalid sampling request never reaches admission.
        if fields.get("status") != "200":
            raise ValueError(f"metrics request failed: {fields}")
        finalized.append({"http_status": 200, "status": "complete", "metrics": {
            "prefill_tokens": int(fields["prefill_tokens"]),
            "completion_tokens": int(fields["generated_tokens"]),
            "prompt_tokens": int(fields["prompt_tokens"]),
            "cached_tokens": int(fields["cached_tokens"]),
            "draft_rounds": int(fields["draft_rounds"]),
            "draft_tokens": int(fields["draft_proposed"]),
            "draft_tokens_accepted": int(fields["draft_accepted"])}})
    assert_accounting(checks["metrics_chat_cold"]["before"],
                      checks["metrics_after_cancel_cached"]["after"], finalized,
                      include_seconds=False)
    cancelled = sum(completed[row["request_id"]].get("finish") == "cancelled" for row in requests)
    if cancelled != len(checks["metrics_live_queue_cancel"]["cancelled"]):
        raise AssertionError("live streams were not cancelled by the server")
    return {"requests": len(finalized), "cancelled": cancelled}


def assert_slots(slots, width, model, context, speculative):
    """Check the public slot contract without exposing prompts or assuming order."""
    assert isinstance(slots, list) and len(slots) == width, slots
    assert [slot["id"] for slot in slots] == list(range(width)), slots
    tasks = []
    for slot in slots:
        assert slot["model"] == model and slot["n_ctx"] == context, slot
        assert slot["speculative"] is speculative and slot["prompt"] == "", slot
        assert type(slot["is_processing"]) is bool, slot
        assert type(slot["id_task"]) is int and slot["task_id"] == slot["id_task"], slot
        for name in ("n_prompt_tokens", "n_prompt_tokens_cache", "n_prompt_tokens_processed"):
            assert type(slot[name]) is int and slot[name] >= 0, slot
        assert slot["n_prompt_tokens_cache"] + slot["n_prompt_tokens_processed"] <= slot["n_prompt_tokens"], slot
        assert isinstance(slot["next_token"], list) and len(slot["next_token"]) == 1, slot
        token = slot["next_token"][0]
        assert token["has_next_token"] is slot["is_processing"], slot
        assert token["has_new_line"] is False, slot
        assert type(token["n_decoded"]) is int and token["n_decoded"] >= 0, slot
        assert type(token["n_remain"]) is int, slot
        if slot["is_processing"]:
            assert slot["state"] == 1 and slot["id_task"] > 0, slot
            tasks.append(slot["id_task"])
            assert token["n_remain"] >= 0, slot
            assert slot["n_prompt_tokens"] + token["n_decoded"] + token["n_remain"] <= context, slot
            if token["n_decoded"]:
                assert slot["n_prompt_tokens_cache"] + slot["n_prompt_tokens_processed"] == slot["n_prompt_tokens"], slot
        else:
            assert slot["state"] == 0 and slot["id_task"] == -1, slot
            assert token["n_remain"] == -1 and token["n_decoded"] == 0, slot
            assert all(slot[name] == 0 for name in (
                "n_prompt_tokens", "n_prompt_tokens_cache", "n_prompt_tokens_processed")), slot
    assert len(tasks) == len(set(tasks)), slots


class ServerMetrics:
    def __init__(self, base_url):
        url = urlsplit(str(base_url))
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1"):
            raise ValueError("metrics checks require an isolated loopback server")
        self.host, self.port = url.hostname, url.port or 80
        self.previous = None

    def read(self, path="/metrics"):
        # Scrapes are separate from SDK generation measurements and never issue
        # model work. No background polling is added to other functional suites.
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            if response.status != 200 or not response.getheader(
                    "Content-Type", "").startswith("text/plain"):
                raise AssertionError(f"invalid /metrics response: {response.status}")
            values = parse_metrics(response.read().decode("utf-8"))
        finally:
            connection.close()
        if self.previous is not None:
            for name in COUNTERS:
                if values[name] < self.previous[name]:
                    raise AssertionError(f"counter decreased: {name}")
        self.previous = values
        return values

    def wait(self, predicate, description, timeout=15):
        deadline = time.monotonic() + timeout
        while True:
            values = self.read()
            if predicate(values):
                return values
            if time.monotonic() >= deadline:
                raise AssertionError(f"metrics never reached {description}: {values}")
            time.sleep(.05)

    def slots(self, path="/slots"):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            if response.status != 200 or not response.getheader(
                    "Content-Type", "").startswith("application/json"):
                raise AssertionError(f"invalid /slots response: {response.status}")
            return json.loads(response.read())
        finally:
            connection.close()

    def idle(self):
        return self.wait(lambda m: m[PROCESSING] == m[DEFERRED] == 0, "idle")
