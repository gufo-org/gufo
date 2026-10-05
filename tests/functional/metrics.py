"""Per-request measurements for functional tests; no production instrumentation."""

import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

SLOWDOWN = .05
NOISE_MS = 3
PROBE_ENDPOINTS = ("/health", "/v1/health", "/ready", "/v1/ready")
DISCOVERY_ENDPOINTS = (*PROBE_ENDPOINTS, "/v1/models")
GENERATION_ENDPOINTS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")


class CaseComplete(BaseException):
    """Control flow, not an SDK connection error; stop before the next request."""


def canonical(value):
    """Ignore generated identifiers, never tool arguments or message text."""
    identifiers = {}
    payloads = {"schema", "parameters", "parametersJsonSchema", "input_schema",
                "arguments", "metadata", "tools", "const", "enum", "default", "examples"}

    def visit(item):
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return item
        protocol = item.get("type") in (
            "function", "function_call", "function_call_output",
            "message", "reasoning", "item_reference") or item.get("role") in (
                "assistant", "tool") or item.get("object") in (
                    "response", "chat.completion", "chat.completion.chunk", "text_completion")
        result = {}
        for key, child in sorted(item.items()):
            if (protocol and key in ("id", "call_id", "tool_call_id")
                    or key == "previous_response_id") \
                    and isinstance(child, str):
                child = identifiers.setdefault(child, f"identifier-{len(identifiers)}")
            result[key] = child if key in payloads else visit(child)
        return result

    return json.dumps(visit(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_tool_events(events, output):
    """A correct final response must not hide corrupted streamed arguments."""
    items, completed = {}, {}
    for event in events:
        kind = event.get("type")
        if kind == "response.output_item.added":
            index, item = event.get("output_index"), event.get("item", {})
            if type(index) is not int or index < 0 or index in items or not item.get("id"):
                raise ValueError("invalid or duplicate Responses output item")
            items[index] = {"item": item, "arguments": item.get("arguments", ""),
                            "done": False}
        elif kind in ("response.function_call_arguments.delta",
                      "response.function_call_arguments.done"):
            index = event.get("output_index")
            row = items.get(index)
            if (not row or row["item"].get("type") != "function_call"
                    or event.get("item_id") != row["item"]["id"]
                    or index in completed or row["done"]):
                raise ValueError("tool argument event has no matching active item")
            if kind.endswith(".delta"):
                if not isinstance(event.get("delta"), str):
                    raise ValueError("tool argument delta must be a string")
                row["arguments"] += event["delta"]
            else:
                if event.get("arguments") != row["arguments"]:
                    raise ValueError("streamed tool arguments differ from arguments.done")
                if "name" in event and event["name"] != row["item"].get("name"):
                    raise ValueError("tool argument event changed function name")
                row["done"] = True
        elif kind == "response.output_item.done":
            index, item = event.get("output_index"), event.get("item", {})
            row = items.get(index)
            if not row or index in completed or any(
                    item.get(key) != row["item"].get(key) for key in ("id", "type")):
                raise ValueError("completed Responses item does not match its opener")
            if item.get("type") == "function_call" and (
                    not row["done"] or item.get("arguments") != row["arguments"]
                    or any(item.get(key) != row["item"].get(key)
                           for key in ("name", "call_id"))):
                raise ValueError("completed tool item differs from its streamed arguments")
            completed[index] = item
    if output is not None:
        for index, item in enumerate(output["output"]):
            if item.get("type") == "function_call" and completed.get(index) != item:
                raise ValueError("terminal tool call differs from its streamed item")
        if any(item.get("type") == "function_call"
               and (index >= len(output["output"]) or output["output"][index] != item)
               for index, item in completed.items()):
            raise ValueError("completed streamed tool call is missing from final output")


def validate_response(events, endpoint, body, status, ended, usage, output, choices):
    """Mandatory wire/usage checks, in addition to SDK types and scenario assertions."""
    def require(condition, message):
        if not condition:
            raise ValueError(f"{endpoint}: {message}")

    if status >= 400:
        require(len(events) == 1 and isinstance(events[0].get("error"), dict),
                "error response must contain an error object")
        error = events[0]["error"]
        require(isinstance(error.get("message"), str) and isinstance(error.get("type"), str),
                "error.message and error.type must be strings")
        return
    if endpoint not in GENERATION_ENDPOINTS:
        return  # SDK checks /models separately.
    require(status == 200, "unexpected successful status")
    response_events = endpoint == "/v1/responses" and body.get("stream")
    if response_events:
        require([event.get("sequence_number") for event in events] == list(range(len(events))),
                "Responses stream sequence must be contiguous")
        require(not events or events[0].get("type") == "response.created",
                "Responses stream must start with response.created")
        for event in events:
            if event.get("type", "").endswith(".delta"):
                require(isinstance(event.get("delta"), str), "delta must be a string")
        validate_tool_events(events, output)
    for event in events:
        for choice in event.get("choices", []):
            require(type(choice.get("index")) is int, "choice.index must be an integer")
            if body.get("stream"):
                require(choice.get("finish_reason") in (None, "stop", "length", "tool_calls",
                                                         "content_filter", "function_call"),
                        "invalid streaming finish reason")
            else:
                require(choice.get("finish_reason") in ("stop", "length", "tool_calls",
                                                        "content_filter", "function_call"),
                        "missing/invalid finish reason")
            message = choice.get("message")
            if message is not None:
                require(message.get("role") == "assistant", "completion role must be assistant")
    complete = output is not None or any(choice.get("finish") for choice in choices.values())
    if not complete:
        require(not ended, "completed transport has no terminal model response")
        return  # Deliberate cancellation: no final usage is promised by the API.
    require(bool(usage), "completed generation is missing usage/timings")
    if endpoint == "/v1/responses":
        input_count, output_count = usage.get("input_tokens"), usage.get("output_tokens")
        require(output is not None, "missing terminal Responses output")
        require(output["status"] in ("completed", "incomplete"), "invalid Responses status")
        if response_events:
            require(events[-1].get("type") in ("response.completed", "response.incomplete"),
                    "missing terminal Responses event")
        if output["status"] == "incomplete":
            require(isinstance(output.get("incomplete_details"), dict),
                    "incomplete response is missing details")
    else:
        input_count, output_count = usage.get("prompt_tokens"), usage.get("completion_tokens")
    require(type(input_count) is int and type(output_count) is int
            and input_count >= 0 and output_count >= 0, "invalid token counts")
    require(usage.get("total_tokens") == input_count + output_count, "inconsistent usage total")
    limit = body.get("max_completion_tokens", body.get("max_tokens", body.get("max_output_tokens")))
    if limit is not None:
        require(output_count <= limit, "output exceeds the request token limit")
    timing = usage.get("gufo", {})
    required_timings = ["prefill_ms", "decode_ms", "cache_restore_ms",
                        "cache_snapshot_ms", "cache_disk_enqueue_ms"]
    if endpoint == "/v1/chat/completions":
        required_timings += ["queue_ms", "ttft_ms"]
    for field in required_timings:
        value = timing.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f"missing or invalid {field}")
    require(type(timing.get("prefill_tokens")) is int, "missing prefill token count")
    if output_count:
        require(timing["decode_ms"] > 0, "generated tokens have no decode time")
    if timing["prefill_tokens"]:
        require(timing["prefill_ms"] > 0, "prefilled tokens have no prefill time")
    cached = usage.get("cached_tokens", usage.get(
        "prompt_tokens_details", usage.get("input_tokens_details", {})).get("cached_tokens"))
    require(type(cached) is int and 0 <= cached <= input_count, "invalid cached token count")
    definitions = {tool.get("function", tool).get("name"): tool.get("function", tool)
                   for tool in body.get("tools", []) if tool.get("type") == "function"}
    calls = [call for choice in choices.values() for call in choice["tools"].values()]
    if output is not None:
        calls = [item for item in output["output"] if item.get("type") == "function_call"]
    choice = body.get("tool_choice", "auto" if definitions else "none")
    require(not calls or (definitions and choice != "none"), "tool calls violate tool_choice")
    require(body.get("parallel_tool_calls") is not False or len(calls) <= 1,
            "parallel_tool_calls:false returned multiple calls")
    naturally_finished = (output["status"] == "completed" if output is not None else
                          any(item.get("finish") in ("stop", "tool_calls")
                              for item in choices.values()))
    if (choice == "required" or isinstance(choice, dict)) and naturally_finished \
            and not body.get("stop"):
        require(bool(calls), "required/named tool request completed without a call")
    for call in calls:
        name = call.get("name")
        require(name in definitions, "undeclared generated tool")
        if isinstance(choice, dict) and choice.get("type") == "function":
            require(name == choice.get("function", choice).get("name"), "wrong named tool")
        arguments = json.loads(call["arguments"])
        require(isinstance(arguments, dict), "tool arguments must be a JSON object")
        definition = definitions[name]
        if definition.get("strict"):
            from jsonschema import Draft202012Validator, FormatChecker
            schema = definition.get("parameters")
            if schema is None:
                schema = {"type": "object", "properties": {}, "additionalProperties": False}
            Draft202012Validator(schema,
                                 format_checker=FormatChecker()).validate(arguments)
    # Validate complete structured answers independently from Gufo's grammar.
    specification = body.get("response_format", body.get("text", {}).get("format", {}))
    if specification.get("type") not in ("json_object", "json_schema"):
        return
    answers = [(choice["text"], choice.get("finish") == "stop")
               for choice in choices.values()]
    if output is not None:
        answers = [("".join(part.get("text", "") for item in output["output"]
                            if item.get("type") == "message"
                            for part in item.get("content", [])
                            if part.get("type") == "output_text"),
                    output["status"] == "completed")]
    for text, finished in answers:
        # Explicit stops may legally truncate JSON. Dedicated stop tests compare
        # the exact expected prefix instead of claiming a complete schema object.
        if not finished or body.get("stop") or (not text and calls):
            continue
        value = json.loads(text)
        require(isinstance(value, dict) or specification["type"] == "json_schema",
                "JSON-object mode returned a non-object")
        if specification["type"] == "json_schema":
            from jsonschema import Draft202012Validator, FormatChecker
            schema = specification.get("json_schema", specification)["schema"]
            Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)


def summarize(parts, streaming, ended, contract=None):
    """Parse after timing ends, including SSE split across arbitrary byte boundaries."""
    data = b"".join(chunk for _, chunk in parts)
    events, first_output_ms = [], None
    if streaming:
        consumed = 0
        for line in data.splitlines(keepends=True):
            consumed += len(line)
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                continue
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeError):
                if ended:
                    raise
                continue  # A deliberately disconnected response can end mid-frame.
            events.append(event)
            has_output = bool(event.get("delta")) or any(
                choice.get("text") or any(choice.get("delta", {}).get(key)
                                         for key in ("content", "reasoning_content", "tool_calls"))
                for choice in event.get("choices", []))
            if first_output_ms is None and has_output:
                count = 0
                for elapsed, chunk in parts:
                    count += len(chunk)
                    if count >= consumed:
                        first_output_ms = elapsed
                        break
    elif data:
        events = [json.loads(data)]
    usage, output, choices, timings = {}, None, {}, {}
    message_output = None
    for event in events:
        final = event.get("response", event)
        timings.update(final.get("timings", {}))
        if final.get("usage"):
            usage = dict(final["usage"])
        if "output" in final and final.get("status") in ("completed", "incomplete"):
            output = {"output": final["output"], "status": final["status"],
                      "incomplete_details": final.get("incomplete_details")}
        if not streaming and final.get("type") == "message" and final.get("role") == "assistant":
            # /v1/messages currently supports buffered responses only. Keep its
            # ordered blocks and stop metadata, excluding the generated message ID.
            message_output = {key: final[key] for key in ("content", "stop_reason", "stop_sequence")}
        for choice in event.get("choices", []):
            index = str(choice.get("index", 0))
            target = choices.setdefault(index, {"text": "", "reasoning": "", "tools": {}})
            message = choice.get("message", choice.get("delta", {}))
            target["text"] += message.get("content") or choice.get("text") or ""
            target["reasoning"] += message.get("reasoning_content") or ""
            for i, tool in enumerate(message.get("tool_calls") or []):
                call = target["tools"].setdefault(str(tool.get("index", i)),
                                                  {"name": "", "arguments": ""})
                function = tool.get("function", {})
                call["name"] += function.get("name") or ""
                call["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                target["finish"] = choice["finish_reason"]
    if usage and "gufo" not in usage and timings:
        # Legacy Completions sends timings with its terminal choice, then usage
        # in a separate final chunk. Responses nests both in the final response.
        aliases = {"prompt_n": "prefill_tokens", "prompt_ms": "prefill_ms",
                   "predicted_ms": "decode_ms"}
        usage["gufo"] = {aliases.get(key, key): value for key, value in timings.items()}
    if contract:
        validate_response(events, *contract, ended, usage, output, choices)
        if contract[0] in DISCOVERY_ENDPOINTS:
            assert ended and not streaming and len(events) == 1, "incomplete discovery response"
            value = events[0]
            if contract[0] == "/v1/models" and contract[2] == 200:
                # Model IDs and capabilities are stable; creation time is not.
                value = {**value, "data": [{k: v for k, v in entry.items() if k != "created"}
                                          for entry in value["data"]]}
            return {}, digest(value)
    measured = {key: value for key, value in usage.get("gufo", {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)}
    for field, alias in (("draft_tokens", "draft_n"),
                         ("draft_tokens_accepted", "draft_n_accepted")):
        if field in usage or alias in measured:
            measured[field] = usage.get(field, measured.pop(alias, None))
    for key, alternatives in (
        ("prompt_tokens", ("prompt_tokens", "input_tokens")),
        ("completion_tokens", ("completion_tokens", "output_tokens")),
    ):
        for alternative in alternatives:
            if alternative in usage:
                measured[key] = usage[alternative]
                break
    measured["cached_tokens"] = usage.get("cached_tokens", usage.get("cache_read_input_tokens",
        usage.get("prompt_tokens_details", usage.get("input_tokens_details", {})).get("cached_tokens", 0)))
    if first_output_ms is not None:
        measured["client_ttft_ms"] = first_output_ms
    for phase, tokens in (("prefill", "prefill_tokens"), ("decode", "completion_tokens")):
        if measured.get(tokens, 0) > 0 and measured.get(phase + "_ms", 0) > 0:
            measured[phase + "_ms_per_token"] = measured[phase + "_ms"] / measured[tokens]
    if message_output is not None:
        return measured, digest(message_output) if usage else None
    completed = bool(usage) and (output is not None or bool(choices))
    return measured, digest(output if output is not None else choices) if completed else None


class Recorder:
    def __init__(self, path, through_case=None):
        self.path = path
        self.through_case = through_case
        self.reached_case = False
        self.rows = []
        self.lock = threading.RLock()

    def save(self):
        if self.path:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"version": 1, "requests": self.rows}, indent=2) + "\n")
            temporary.replace(self.path)

    def begin(self, path, body):
        with self.lock:
            if self.reached_case:
                raise CaseComplete()
            row = {"index": len(self.rows), "endpoint": path,
                   "request_sha256": digest({"endpoint": path, "body": body}),
                   "stream": bool(body.get("stream")), "status": "running"}
            self.rows.append(row)
        return Measurement(self, row, body)

    def mark(self, case):
        with self.lock:
            for row in self.rows:
                if row["status"] != "running":
                    row.setdefault("case", case)
            self.save()
            if case == self.through_case:
                self.reached_case = True

    def request_hook(self, request):
        body = json.loads(request.content) if request.content else {}
        request.extensions["functional_measurement"] = self.begin(request.url.path, body)

    def response_hook(self, response):
        import httpx
        measurement = response.request.extensions["functional_measurement"]
        source = response.stream
        measurement.row["http_status"] = response.status_code
        measurement.row["request_id"] = response.headers.get("x-request-id")

        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                for data in source:
                    measurement.feed(data)
                    yield data
                measurement.ended = True

            def close(self):
                try:
                    source.close()
                finally:
                    measurement.finish()

        response.stream = Stream()

    async def async_request_hook(self, request):
        self.request_hook(request)

    async def async_response_hook(self, response):
        import httpx
        measurement = response.request.extensions["functional_measurement"]
        source = response.stream
        measurement.row["http_status"] = response.status_code
        measurement.row["request_id"] = response.headers.get("x-request-id")

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                async for data in source:
                    measurement.feed(data)
                    yield data
                measurement.ended = True

            async def aclose(self):
                try:
                    await source.aclose()
                finally:
                    measurement.finish()

        response.stream = Stream()


class Measurement:
    def __init__(self, recorder, row, body):
        self.recorder, self.row = recorder, row
        self.body = body
        self.started = time.monotonic()
        self.parts = []
        self.ended = False
        self.finished = False

    def feed(self, data):
        self.parts.append(((time.monotonic() - self.started) * 1000, data))

    def finish(self):
        if self.finished:
            return
        self.finished = True
        elapsed = (time.monotonic() - self.started) * 1000
        with self.recorder.lock:
            self.row.update(status="complete" if self.ended else "disconnected",
                            wall_ms=elapsed)
            try:
                measured, output_hash = summarize(
                    self.parts, self.row["stream"] and self.row.get("http_status", 0) < 400,
                    self.ended,
                    (self.row["endpoint"], self.body, self.row.get("http_status", 0)))
                self.row.update(metrics=measured, output_sha256=output_hash)
                if output_hash is not None:
                    self.row["status"] = "complete"
            except Exception as error:
                self.row.update(status="invalid", error=str(error))
                raise
            finally:
                self.parts.clear()
                self.body = None
                self.recorder.save()


def timing_measurement(case, metric, old, new, count=1, request_key=None):
    floor = NOISE_MS / max(1, count) if metric.endswith("_per_token") else NOISE_MS
    exceeded = new - old > max(floor, old * SLOWDOWN)
    return {"case": case, "request_key": request_key or ["server"], "metric": metric,
            "baseline": old, "candidate": new, "noise_floor": floor,
            "change_percent": 100 * (new / old - 1) if old else None,
            "exceeds_margin": exceeded,
            "status": "inconclusive" if exceeded else "passed"}


def comparison_status(rows, issues):
    if issues or any(row["status"] == "failed" for row in rows):
        return "failed"
    return "inconclusive" if any(row["status"] == "inconclusive" for row in rows) else "passed"


def compare(baseline, candidate):
    """One matched observation flags timings; deterministic differences fail immediately."""
    def index(directory):
        result, history = {}, []
        root = Path(directory)
        paths = sorted(root.glob("*.requests.json"))
        if (root / "report.json").is_file():
            report = json.loads((root / "report.json").read_text())
            expected = [root / (suite + ".requests.json") for suite in report["suites"]]
            if set(paths) != set(expected):
                raise ValueError("missing or unexpected suite measurements")
            paths = expected
        for path in paths:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or payload.get("version") != 1 \
                    or not isinstance(payload.get("requests"), list):
                raise ValueError(f"unsupported metrics: {path}")
            occurrences = {}
            history.append([path.name])
            # A concurrent case can submit its requests in a different order.
            # Preserve case order and group membership, without ordering peers.
            groups = []
            for row in payload["requests"]:
                label = row.get("case", row.get("index")) if isinstance(row, dict) else None
                if not groups or groups[-1][0] != label:
                    groups.append((label, []))
                groups[-1][1].append(row)
            histories, ordered = {}, []
            for label, group in groups:
                history.append([label, sorted(row.get("request_sha256", "") for row in group
                                             if isinstance(row, dict))])
                for row in group:
                    histories[id(row)] = digest(history)
                # Identical simultaneous bodies can exchange leader/follower
                # roles. Match their cache work, never their observed timings.
                # Different bodies and different cohorts retain exact identity.
                peers = {}
                for row in group:
                    peers.setdefault(row.get("request_sha256", ""), []).append(row)
                for requests in peers.values():
                    ordered.extend(sorted(requests, key=lambda row: (
                        -row.get("metrics", {}).get("prefill_tokens", 0),
                        row.get("metrics", {}).get("cached_tokens", 0))))
            for row in ordered:
                if not isinstance(row, dict) or row.get("status") not in ("complete", "disconnected"):
                    raise ValueError(f"unfinished/invalid request in {path}")
                measured = row.get("metrics", {})
                if not isinstance(measured, dict):
                    raise ValueError(f"invalid stage measurements in {path}")
                for value in (row.get("wall_ms"), *measured.values()):
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid/nonfinite timing or counter in {path}")
                fingerprint = row["request_sha256"]
                occurrence = occurrences.get(fingerprint, 0)
                occurrences[fingerprint] = occurrence + 1
                result[(path.name, fingerprint, occurrence)] = {
                    **row, "history_sha256": histories[id(row)]}
        return result

    before, after = index(baseline), index(candidate)
    rows, issues = [], []
    if not before or before.keys() != after.keys():
        issues.append("request coverage differs or no per-request measurements exist")
    for key in sorted(before.keys() & after.keys()):
        a, b = before[key], after[key]
        label = f"{key[0]}:{b.get('case', b['index'])}:{key[2]}"
        for field in ("http_status", "status", "output_sha256", "case", "history_sha256"):
            if a.get(field) != b.get(field) or b.get("status") in ("running", "invalid"):
                issues.append(f"{label}: {field} changed")
        am, bm = a.get("metrics", {}), b.get("metrics", {})
        for field in ("prompt_tokens", "completion_tokens", "prefill_tokens", "cached_tokens"):
            if am.get(field) != bm.get(field):
                issues.append(f"{label}: {field} changed ({am.get(field)} -> {bm.get(field)})")
        if am.keys() != bm.keys():
            issues.append(f"{label}: metric coverage changed")
        for field in ("wall_ms", *sorted(am.keys() & bm.keys())):
            if field != "wall_ms" and not field.endswith(("_ms", "_ms_per_token")):
                continue
            old = a["wall_ms"] if field == "wall_ms" else am[field]
            new = b["wall_ms"] if field == "wall_ms" else bm[field]
            count = bm.get("prefill_tokens" if field.startswith("prefill") else "completion_tokens", 1)
            measurement = timing_measurement(label, field, old, new, count, list(key))
            measurement["history_sha256"] = b["history_sha256"]
            rows.append(measurement)
    return {"status": comparison_status(rows, issues),
            "slowdown_tolerance": SLOWDOWN, "noise_ms": NOISE_MS,
            "quality_or_coverage_changes": issues, "measurements": rows}


def qualify(comparisons, controls=()):
    """Retain every observation; overlapping timing ranges stay unqualified.

    Deliberately conservative, not a confidence interval: confirm a slowdown only
    when two independent candidate observations exceed every stable main/control
    observation. Sparse or overlapping stall evidence needs more investigation.
    """
    if not comparisons:
        raise ValueError("at least one matched main/PR comparison is required")
    rows, issues = {}, []
    for index, comparison in enumerate([*comparisons, *controls]):
        control = index >= len(comparisons)
        if comparison.get("slowdown_tolerance") != SLOWDOWN \
                or comparison.get("noise_ms") != NOISE_MS:
            raise ValueError("timing margins must remain 5% and 3 ms")
        issues.extend(comparison["quality_or_coverage_changes"])
        seen = set()
        for item in comparison["measurements"]:
            key = (*item["request_key"], item["metric"])
            if key in seen:
                raise ValueError("duplicate request/phase measurement")
            seen.add(key)
            if index and key not in rows:
                raise ValueError("follow-up contains a scenario absent from the initial run")
            row = rows.setdefault(key, {
                "case": item["case"], "request_key": item["request_key"], "metric": item["metric"],
                "history_sha256": item.get("history_sha256"), "noise_floor": item["noise_floor"],
                "samples": [], "controls": [],
            })
            if row["history_sha256"] != item.get("history_sha256") \
                    or row["noise_floor"] != item["noise_floor"]:
                raise ValueError(f"unmatched history or token count: {item['case']}")
            row["controls" if control else "samples"].append({
                "comparison": index, "baseline": item["baseline"], "candidate": item["candidate"],
                "exceeds_margin": item["exceeds_margin"]})
    for row in rows.values():
        baseline = [sample["baseline"] for sample in row["samples"]]
        baseline += [value for sample in row["controls"]
                     for value in (sample["baseline"], sample["candidate"])]
        candidate = [sample["candidate"] for sample in row["samples"]]
        row["baseline_range"] = [min(baseline), max(baseline)]
        row["candidate_range"] = [min(candidate), max(candidate)]
        row["flagged_pairs"] = sum(sample["exceeds_margin"] for sample in row["samples"])
        row["control_flags"] = sum(sample["exceeds_margin"] for sample in row["controls"])
        row["baseline_variable"] = max(baseline) - min(baseline) > max(
            row["noise_floor"], min(baseline) * SLOWDOWN)
        if max(candidate) - min(baseline) <= max(row["noise_floor"], min(baseline) * SLOWDOWN):
            row.update(status="passed", reason="all observations within margin")
        elif len(candidate) >= 2 and not row["baseline_variable"] \
                and min(candidate) - max(baseline) > max(
                row["noise_floor"], max(baseline) * SLOWDOWN):
            row.update(status="failed", reason="reproduced slowdown across observed ranges")
        else:
            row.update(status="inconclusive", reason=(
                "confirmation required" if len(candidate) == 1 and not row["controls"]
                else "variable or overlapping observations; inspect stalls"))
    measurements = list(rows.values())
    return {"status": comparison_status(measurements, issues),
            "slowdown_tolerance": SLOWDOWN, "noise_ms": NOISE_MS,
            "quality_or_coverage_changes": issues, "measurements": measurements}


def join_server_timings(directory):
    """Join after shutdown, when cancelled work and disk restoration have drained."""
    root = Path(directory)
    for path in root.glob("*.requests.json"):
        log = root / ("server-restarted.log" if path.name.endswith("-disk.requests.json")
                      else "server.log")
        completed = {}
        for line in log.read_text().splitlines():
            if "event=completed " in line:
                fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
                if "request" in fields:
                    completed[fields["request"]] = fields
        payload = json.loads(path.read_text())
        for row in payload["requests"]:
            fields = completed.get(row.get("request_id"))
            if not fields:
                if row.get("endpoint") in PROBE_ENDPOINTS:
                    continue  # Probes are deliberately quiet; client timing is retained.
                raise ValueError(f"{path.name}:{row['index']}: no completed server log")
            measured = row.setdefault("metrics", {})
            for field in ("duration_ms", "queue_ms", "ttft_ms", "cache_restore_ms"):
                if field in fields:
                    value = float(fields[field])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {field} in {log}")
                    measured.setdefault("server_" + field if field == "duration_ms" else field, value)
            for field, alias in (("draft_rounds", "draft_rounds"),
                                 ("draft_proposed", "draft_tokens"),
                                 ("draft_accepted", "draft_tokens_accepted")):
                if field in fields:
                    value = int(fields[field])
                    if value < 0 or alias in measured and measured[alias] != value:
                        raise ValueError(f"invalid/inconsistent {field} in {log}")
                    measured[alias] = value
            if "server_duration_ms" not in measured:
                raise ValueError(f"missing request duration in {log}")
            if row.get("output_sha256") and row["endpoint"] in GENERATION_ENDPOINTS:
                for field in ("queue_ms", "ttft_ms"):
                    if field not in measured:
                        raise ValueError(f"{path.name}:{row['index']}: missing {field}")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
