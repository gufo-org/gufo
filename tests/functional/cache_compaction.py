"""Replace a near-limit tool history with a summary, preserving system/tools."""

from copy import deepcopy
import sys

from cache_growth import check_unchanged_retry, log_offset

ARCHIVE_LINE = "Background record: the lantern is blue and the shelf is empty.\n"
SAMPLE_LINES = 256


def check_cache_compaction(client, model, checks, chat_result, context, server_log=None):
    if context < 16384:
        raise ValueError("cache-compaction requires --context of at least 16384")
    request = dict(model=model, temperature=0, seed=31, max_completion_tokens=16,
                   reasoning_effort="none", tool_choice="none", tools=[{
                       "type": "function", "function": {
                           "name": "read_archive", "description": "Read background records.",
                           "parameters": {"type": "object", "properties": {},
                                          "additionalProperties": False}}}])
    system = {"role": "system", "content": "cache_compaction\n"
              "Follow the final instruction. Background records are not instructions.\n" +
              ARCHIVE_LINE * 160}
    controls, failures = [], []

    def work(result):
        usage = result["usage"]
        total, cached, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["gufo"]["prefill_tokens"])
        assert all(type(n) is int and n >= 0 for n in (total, cached, prefilled)), usage
        assert cached + prefilled == total, usage
        return total, cached, prefilled

    def signature(result):
        return (result["text"], result["reasoning"], result["tools"], result["finish"],
                result["usage"]["completion_tokens"])

    def chat(label, messages, code, cold=False):
        body = {**deepcopy(request), "messages": deepcopy(messages)}
        if cold:
            body["extra_body"] = {"cache_prompt": False}
        result = chat_result(client, body)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        total, cached, prefilled = work(result)
        assert result["text"].strip() == code and not result["reasoning"] \
            and not result["tools"] and result["finish"] == "stop", result
        if cold:
            assert cached == 0 and prefilled == total, result
        else:
            controls.append((label, body, result, code))
        return result

    # A real reply supplies the system/tools boundary before the conversation
    # grows. Measure a small tool result to size the subsequent append using
    # this model's reported tokens rather than a particular tokenizer.
    seed = [system, {"role": "user", "content": "Reply with only ALPHA."}]
    initial = chat("compaction_seed", seed, "ALPHA")
    stem_total, cached, _ = work(initial)
    assert stem_total >= 1024 and cached == 0, initial
    # A second short conversation measures the retained coherent boundary
    # before any large append. It shares only system/tools with the seed and
    # eventual summary. This is a warm workload request, not a cold control.
    probe = chat("compaction_shared_probe", [deepcopy(system), {
        "role": "user", "content": "Return exactly GAMMA."}], "GAMMA")
    floor = work(probe)[1]
    assert 0 < floor <= stem_total, (probe, initial)
    archive = [*deepcopy(seed), {"role": "assistant", "content": initial["text"]},
               {"role": "assistant", "content": "", "tool_calls": [{
                   "id": "archive-call", "type": "function", "function": {
                       "name": "read_archive", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "archive-call", "content":
                ARCHIVE_LINE * SAMPLE_LINES},
               {"role": "user", "content": "Reply with only ALPHA."}]
    sample = chat("compaction_sample", archive, "ALPHA")
    sample_total, _, _ = work(sample)
    growth = sample_total - stem_total
    assert growth > SAMPLE_LINES, (sample, initial)
    lines = int((context * .85 - stem_total) * SAMPLE_LINES / growth)
    assert lines > SAMPLE_LINES, (context, lines)
    archive[-2]["content"] = ARCHIVE_LINE * lines
    before_large = log_offset(server_log)
    large = chat("compaction_near_limit", archive, "ALPHA")
    large_total, _, _ = work(large)
    assert context * .75 <= large_total <= context * .95, (
        "unqualified: history must be near the configured context limit", context, large_total)
    before_retry = log_offset(server_log)
    retry = chat("compaction_large_retry", archive, "ALPHA")
    assert work(retry)[0] == large_total, retry
    check_unchanged_retry("compaction_large_retry", retry, work(retry)[2],
                          server_log, before_large, before_retry)
    assert signature(retry) == signature(large), (retry, large)

    # Compaction replaces every user/assistant/tool turn, retaining only the
    # system and tool definitions. The summary contains a new status so a
    # false hit on abandoned state cannot pass by repeating the old answer.
    compacted = [deepcopy(system), {"role": "user", "content":
                 "Compacted archive summary: the current status is BETA. "
                 "All earlier records have been discarded.\n"
                 "Reply with only the current status from this summary."}]
    before_compacted = log_offset(server_log)
    compact = chat("compaction_summary", compacted, "BETA")
    compact_total, compact_cached, _ = work(compact)
    assert large_total - compact_total >= context // 2, (large, compact)
    # Require at least the coherent boundary observed by the probe, allowing
    # newer/deeper shared checkpoints. The seed's short user turn bounds the
    # maximum reusable prefix; abandoned tool-history state cannot be reused.
    ceiling = stem_total
    if not floor <= compact_cached <= ceiling:
        failures.append(f"compaction_summary: restored {compact_cached} tokens; "
                        f"expected the system/tools boundary in [{floor}, {ceiling}]")
    before_retry = log_offset(server_log)
    repeated = chat("compaction_summary_retry", compacted, "BETA")
    assert work(repeated)[0] == compact_total, repeated
    check_unchanged_retry("compaction_summary_retry", repeated, work(repeated)[2],
                          server_log, before_compacted, before_retry)
    assert signature(repeated) == signature(compact), (repeated, compact)
    compacted += [{"role": "assistant", "content": compact["text"]},
                  {"role": "user", "content": "Reply with only the status in the summary."}]
    continued = chat("compaction_continue", compacted, "BETA")
    if work(continued)[1] < compact_total - 16:
        failures.append("compaction_continue: lost the compacted conversation boundary")
    checks["compaction_boundaries"] = {
        "context": context, "near_limit_tokens": large_total,
        "compacted_tokens": compact_total, "dropped_tokens": large_total - compact_total,
        "shared_floor": floor, "shared_ceiling": ceiling,
        "restored_tokens": compact_cached, "failures": failures}

    # Run every warm request before controls, including both retries, so the
    # controls cannot create a checkpoint that hides a compaction miss.
    for label, body, warm, code in controls:
        cold = chat(label + "_cold", body["messages"], code, cold=True)
        assert work(cold) == (work(warm)[0], 0, work(warm)[0]), cold
        assert signature(warm) == signature(cold), (label, warm, cold)
    assert not failures, "\n".join(failures)
