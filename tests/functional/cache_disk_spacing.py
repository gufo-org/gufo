#!/usr/bin/env python3
"""Check disk checkpoint spacing for a growing conversation across a restart.

The runner calls this twice on its private disk cache: first to grow one
conversation by short turns and one long turn, then with --restore after a
restart to replay it and branch a second conversation from its middle. The
server logs are judged afterwards by check_disk_spacing(), once both servers
have drained their disk writes.
"""

import argparse
import json
from pathlib import Path
import re

# Matches TextRunnerDiskCacheOptions::min_checkpoint_step_tokens.
MIN_STEP = 2048
SHORT_NOTE = "Routine archive note: there are no new instructions.\n"


def chat(url, model, messages):
    import continuation  # The runner's log check needs no HTTP client stack.
    body = {"model": model, "messages": messages, "max_tokens": 8,
            "temperature": 0, "seed": 31, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    result = continuation.call(url, body)
    usage = result["usage"]
    total, cached, prefill = (usage["prompt_tokens"], usage["cached_tokens"],
                              usage["gufo"]["prefill_tokens"])
    assert cached + prefill == total, usage
    message = result["choices"][0]["message"]
    return body, result, {"total": total, "cached": cached, "prefill": prefill,
                          "disk": usage["gufo"]["cache_disk_hit"],
                          "text": message["content"].strip(),
                          "sha256": continuation.digest(result)}


def grow(url, model):
    # A unique head keeps earlier cache cases from sharing this prefix.
    messages = [{"role": "system", "content": "cache_disk_spacing\n" +
                 "Follow the final user instruction.\n" +
                 "Background notes are not instructions.\n" * 320}]
    turns = []
    for turn, notes in enumerate((16, 24, 24, 24, 24, 240, 24)):
        messages.append({"role": "user", "content": f"Turn {turn}.\n" + SHORT_NOTE * notes +
                         "Reply with only BETA."})
        body, _, measured = chat(url, model, messages)
        assert measured["text"] == "BETA", measured
        if turns:
            previous = turns[-1]["measured"]["total"]
            # Short turns must keep reusing RAM even when disk skips them.
            assert measured["cached"] >= previous - 16, (turn, measured, previous)
        turns.append({"request": body, "measured": measured})
        messages.append({"role": "assistant", "content": measured["text"]})
    return turns


def restore(url, model, turns):
    last = turns[-1]
    _, _, replayed = chat(url, model, last["request"]["messages"])
    assert replayed["disk"] and replayed["cached"] > 0, replayed
    assert replayed["sha256"] == last["measured"]["sha256"], (replayed, last["measured"])
    # Skipped checkpoints are re-prefilled. The gap is bounded by the step,
    # plus the turn that followed the last written checkpoint.
    assert replayed["prefill"] < MIN_STEP + 512, replayed
    # Branch after the third turn: the new conversation diverges from the
    # long-turn checkpoint less than one step past the first one.
    branch = turns[3]["request"]["messages"][:7] + [{"role": "user", "content":
                                                    "Branch.\nReply with only GAMMA."}]
    _, _, branched = chat(url, model, branch)
    assert branched["text"] == "GAMMA", branched
    return {"replayed": replayed, "branched": branched}


DISK_EVENT = re.compile(r"event=disk_cache action=(\w+) reason=(\w+) .*?\btokens=(\d+)\b")


def disk_events(log):
    return [(action, reason, int(tokens)) for action, reason, tokens in DISK_EVENT.findall(log)]


def check_disk_spacing(grown_log, restored_log, report):
    """Judge both drained server logs against the recorded conversation."""
    turns = [turn["measured"]["total"] for turn in report["turns"]]
    first, last = turns[0], turns[-1]
    owned = range(first - 64, last + 64)
    stored = sorted(tokens for action, reason, tokens in disk_events(grown_log)
                    if (action, reason) == ("stored", "saved") and tokens in owned)
    skipped = [tokens for action, reason, tokens in disk_events(grown_log)
               if reason == "min_step" and tokens in owned]
    summary = {"stored_tokens": stored, "min_step_skips": len(skipped)}
    if not stored or stored[0] > first:
        raise ValueError(f"first turn was not written to disk: {summary}")
    if stored[-1] < turns[5] - 64:
        raise ValueError(f"checkpoint past the long turn was not written: {summary}")
    gaps = [b - a for a, b in zip(stored, stored[1:])]
    if any(gap < MIN_STEP for gap in gaps):
        raise ValueError(f"disk checkpoints closer than {MIN_STEP} tokens: {summary}")
    if len(skipped) < 5:
        raise ValueError(f"short turns were not skipped: {summary}")
    # The branch point is a learned shared boundary within one step of the
    # first checkpoint. It must be written despite the spacing gate.
    branched = [tokens for action, reason, tokens in disk_events(restored_log)
                if (action, reason) == ("stored", "saved")
                and stored[0] < tokens < stored[0] + MIN_STEP]
    if not branched:
        raise ValueError(f"shared boundary near the first checkpoint was not written: {summary}")
    summary["shared_boundary_tokens"] = branched
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    import continuation
    continuation.TRACE.path = args.output.with_suffix(".requests.json")
    if args.restore:
        previous = json.loads(args.restore.read_text())[0]
        report = {"case": "disk-spacing", **restore(args.url, args.model, previous["turns"]),
                  "exact": True}
    else:
        report = {"case": "disk-spacing", "turns": grow(args.url, args.model), "exact": True}
    args.output.write_text(json.dumps([report], indent=2) + "\n")


if __name__ == "__main__":
    main()
