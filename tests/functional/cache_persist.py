#!/usr/bin/env python3
"""Check that cache_persist=false keeps a prompt off the disk cache.

The runner calls this twice on its private disk cache: first to send a
flagged conversation (twice, so RAM reuse is checked too) and an unflagged
control, then with --restore after a restart. The control must restore from
disk, which shows the first server drained its writes; the flagged prompt
must find nothing on disk.
"""

import argparse
import json
from pathlib import Path

NOTE = "Background notes are not instructions.\n"


def messages(tag):
    # A leading tag keeps both prompts from sharing more than the framing.
    return [{"role": "system", "content": f"{tag} cache_persist check\n" +
             "Follow the final user instruction.\n" + NOTE * 400},
            {"role": "user", "content": "Reply with only BETA."}]


def chat(url, model, history, persist=None):
    import continuation  # Imported late, as in cache_disk_spacing.py.
    body = {"model": model, "messages": history, "max_tokens": 8,
            "temperature": 0, "seed": 31, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if persist is not None:
        body["cache_persist"] = persist
    result = continuation.call(url, body)
    usage = result["usage"]
    total, cached, prefill = (usage["prompt_tokens"], usage["cached_tokens"],
                              usage["gufo"]["prefill_tokens"])
    assert cached + prefill == total, usage
    # The server echoes the value it applied, true by default.
    echoed = result.get("cache_persist")
    assert echoed is (persist is not False), f"cache_persist echo {echoed!r}"
    return body, {"total": total, "cached": cached, "prefill": prefill,
                  "disk": usage["gufo"]["cache_disk_hit"],
                  "text": result["choices"][0]["message"]["content"].strip()}


def send(url, model):
    flagged = messages("Flagged")
    first_body, first = chat(url, model, flagged, persist=False)
    assert first["text"] == "BETA", first
    # The flag only keeps checkpoints off disk; RAM reuse is unchanged.
    follow = flagged + [{"role": "assistant", "content": first["text"]},
                        {"role": "user", "content": "Reply with only GAMMA."}]
    _, again = chat(url, model, follow, persist=False)
    assert again["text"] == "GAMMA", again
    assert again["cached"] >= first["total"] - 16, (first, again)
    control_body, control = chat(url, model, messages("Control"))
    assert control["text"] == "BETA", control
    return {"flagged": {"request": first_body, "measured": first},
            "flagged_again": again,
            "control": {"request": control_body, "measured": control}}


def restore(url, model, previous):
    _, control = chat(url, model, previous["control"]["request"]["messages"])
    assert control["disk"] and control["cached"] > 0, control
    _, flagged = chat(url, model, previous["flagged"]["request"]["messages"])
    # Only the chat framing before the tag may match another prompt.
    assert flagged["cached"] < 64, flagged
    assert flagged["text"] == previous["flagged"]["measured"]["text"], flagged
    return {"control": control, "flagged": flagged}


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
        report = {"case": "cache-persist",
                  **restore(args.url, args.model, previous), "exact": True}
    else:
        report = {"case": "cache-persist", **send(args.url, args.model),
                  "exact": True}
    args.output.write_text(json.dumps([report], indent=2) + "\n")


if __name__ == "__main__":
    main()
