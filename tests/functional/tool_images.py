"""Images remain observations of their tool calls through Chat and Responses."""

from copy import deepcopy
import sys

from cache_growth import check_unchanged_retry, log_offset


def check_tool_images(client, model, checks, image_content, chat_result, response_result,
                      server_log=None):
    from openai import BadRequestError

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    def counts(result):
        usage = result["usage"]
        total = usage.get("prompt_tokens", usage.get("input_tokens"))
        details = usage.get("prompt_tokens_details", usage.get("input_tokens_details", {}))
        return total, details.get("cached_tokens", usage.get("cached_tokens", 0))

    def check(result, expected):
        assert result["text"].strip().rstrip(".!").lower() == expected.lower(), result
        assert not result["reasoning"] and not result["tools"] and result["finish"] == "stop", result
        return result

    def request(transport, history, cache=True):
        common = dict(model=model, temperature=0,
                      extra_body={"seed": 502, "cache_prompt": cache,
                                  "presence_penalty": 0, "frequency_penalty": 0})
        if transport == "chat":
            return dict(**common, messages=history, reasoning_effort="none",
                        max_completion_tokens=32)
        return dict(**common, input=history, reasoning={"effort": "none"},
                    max_output_tokens=32, store=False)

    def run(transport, history, streaming=False, cache=True):
        call = chat_result if transport == "chat" else response_result
        return call(client, request(transport, history, cache), streaming)

    def history(transport, color, mixed, *, image=True):
        prompt = ("Read swatch.png. Report its dominant color followed by a space and "
                  "the code written in the tool result. Use exactly those two words."
                  if mixed else
                  "Read swatch.png. Report only its dominant color in lowercase.")
        if not image:
            prompt = "Read swatch.png. Reply with only the status in the tool result."
        # Keep each transport/shape's changed-pixel check independent. Otherwise
        # Responses can legitimately reuse the blue Chat checkpoint from an
        # earlier case because their rendered prompts are identical.
        prompt = f"Case {transport}_{int(mixed)}_{int(image)}. " + prompt
        output = []
        if mixed or not image:
            output.append({"type": "text" if transport == "chat" else "input_text",
                           "text": " \nCode: WALNUT. " if image else "Status: READY."})
        if image:
            part = image_content(color)
            output.append(part if transport == "chat" else {
                "type": "input_image", "image_url": part["image_url"]["url"]})
        if mixed and image:
            output.append({"type": "text" if transport == "chat" else "input_text",
                           "text": " End of file observation.\n "})
        call_id = "call_swatch"
        user = {"role": "user", "content": prompt}
        if transport == "chat":
            return [user, {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"swatch.png"}'}}]},
                {"role": "tool", "tool_call_id": call_id, "content": output}]
        custom = transport == "custom"
        call = {"type": "custom_tool_call" if custom else "function_call",
                "call_id": call_id, "name": "read_file"}
        call.update({"input": "swatch.png"} if custom else {
            "arguments": '{"path":"swatch.png"}'})
        return [user, call, {
            "type": "custom_tool_call_output" if custom else "function_call_output",
            "call_id": call_id, "output": output}]

    # These existing shapes precede the new capability. Main timing controls can
    # stop at control_complete without being asked to accept tool-output images.
    for transport in ("chat", "responses"):
        text_history = history(transport, "red", False, image=False)
        field = "content" if transport == "chat" else "output"
        text_history[-1][field] = "Status: READY."
        first = record(f"tool_images_{transport}_text_string",
                       check(run(transport, text_history), "READY"))
        text_history[-1][field] = [{
            "type": "text" if transport == "chat" else "input_text", "text": "Status: READY."}]
        replay = record(f"tool_images_{transport}_text_array",
                        check(run(transport, text_history, True), "READY"))
        assert replay["text"] == first["text"] and counts(replay)[0] == counts(replay)[1], replay
        part = image_content("red")
        user_image = [part] if transport == "chat" else [{
            "type": "input_image", "image_url": part["image_url"]["url"]}]
        user_image.append({"type": "text" if transport == "chat" else "input_text",
                           "text": "Name the dominant color. Reply in one lowercase word."})
        record(f"tool_images_{transport}_user_image",
               check(run(transport, [{"role": "user", "content": user_image}],
                         streaming=True, cache=False), "red"))
    # Record a real final control request so through-case retains every timing.
    record("tool_images_control_complete", check(run(
        "responses", [{"role": "user", "content": "Reply with only READY."}],
        cache=False), "READY"))

    for transport in ("chat", "responses", "custom"):
        for mixed in (False, True):
            label = f"tool_images_{transport}_{'mixed' if mixed else 'only'}"
            turns = history(transport, "red", mixed)
            expected = "red WALNUT" if mixed else "red"
            start = log_offset(server_log)
            first = record(label, check(run(transport, turns), expected))
            retry = record(label + "_retry", check(run(transport, turns, True), expected))
            assert first["text"] == retry["text"], (first, retry)
            total, cached = counts(retry)
            check_unchanged_retry(label + "_retry", retry, total - cached, server_log,
                                  start, log_offset(server_log))
            # Same image position and transport with different pixels must
            # invalidate the old observation, even when token counts match.
            changed = history(transport, "blue", mixed)
            blue = record(label + "_changed", check(
                run(transport, changed, True), "blue WALNUT" if mixed else "blue"))
            assert counts(blue)[1] < counts(blue)[0], blue
            cold = record(label + "_cold", check(
                run(transport, changed, cache=False),
                "blue WALNUT" if mixed else "blue"))
            assert blue["text"] == cold["text"] and counts(cold)[1] == 0, (blue, cold)
            if mixed:
                followup = [*turns, {"role": "assistant", "content": first["text"]},
                            {"role": "user", "content":
                             "What was the code from that file? Reply with only the code."}]
                resumed = record(label + "_continuation",
                                 check(run(transport, followup, True), "WALNUT"))
                assert counts(resumed)[1] >= counts(first)[0] - 16, (first, resumed)
                control = record(label + "_continuation_cold",
                                 check(run(transport, followup, cache=False), "WALNUT"))
                assert resumed["text"] == control["text"], (resumed, control)

    # Two paired calls/results with distinct pixels test image order and grouped
    # tool-response rendering; neither filename nor tool text reveals a color.
    turns = history("responses", "red", False)
    turns[0]["content"] = ("Read first.png and second.png. Give their dominant colors "
                           "in that order, separated by a comma and no spaces.")
    turns[1]["arguments"] = '{"path":"first.png"}'
    second = deepcopy(turns[1])
    second.update(call_id="call_second", arguments='{"path":"second.png"}')
    second_output = deepcopy(turns[2])
    second_output.update(call_id="call_second", output=[{
        "type": "input_image", "image_url": image_content("blue")["image_url"]["url"]}])
    turns = [turns[0], turns[1], second, turns[2], second_output]
    record("tool_images_two_calls", check(run("responses", turns, True), "red,blue"))

    invalid = []
    for role in ("assistant", "system", "developer"):
        invalid.append((role, [{"role": role, "content": [{
            "type": "input_image", "image_url": image_content("red")["image_url"]["url"]}]}]))
    for name, change in (
        ("detail", {"detail": "high"}),
        ("bad_base64", {"image_url": "data:image/png;base64,A==="}),
        ("bad_url", {"image_url": 17}),
        ("file_id", {"file_id": "file_unknown"}),
    ):
        turns = history("responses", "red", False)
        turns[-1]["output"][0].update(change)
        invalid.append((name, turns))
    for name, turns in invalid:
        try:
            run("responses", turns)
        except BadRequestError as error:
            record("tool_images_invalid_" + name,
                   {"status": error.status_code, "code": error.code, "message": error.message})
        else:
            raise AssertionError("Invalid image accepted: " + name)
    record("tool_images_recovery", check(
        run("responses", history("responses", "blue", False), True), "blue"))
