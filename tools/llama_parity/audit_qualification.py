#!/usr/bin/env python3
"""Attribute every main/PR qualification change to the removed JSON envelope.

Usage: audit_qualification.py CANDIDATE_DIR [INSTRUCTION_TOKENS]

CANDIDATE_DIR is a `run.py --baseline` output whose comparison.json compares
main with this change. Main switched some requests to a JSON envelope and added
an instruction to the prompt; this change keeps them native, as llama.cpp
does. The audit fails unless every timing measurement passed or was
inconclusive, and every quality change is one of:

- a request whose prompt lost exactly the instruction's tokens (it hit the
  fallback on main), with its output, completion and prefill following;
- a request whose cache history differs only because an earlier request of
  the run (same or an earlier suite) changed;
- a listed llama.cpp parity case (a value native tags cannot carry).
"""

import collections
import json
from pathlib import Path
import re
import sys

# Values native tags cannot carry, and prose naming the call marker, which
# llama.cpp's grammar also turns into a call that ends the output.
PARITY_CASES = ("responses_tool_delimiter_native", "responses_tool_pattern_native",
                "envelope_closer_explain_tool_call_syntax",
                "envelope_closer_envelope_documented_then_prose",
                "envelope_closer_qwen_envelope_documented_then_prose")
CHANGE = re.compile(r"^(?P<suite>[^:]+)\.requests\.json:(?P<case>.+):(?P<index>\d+): "
                    r"(?P<what>\w+) changed(?: \((?P<before>\d+) -> (?P<after>\d+)\))?$")


def main():
    directory = Path(sys.argv[1])
    instruction = int(sys.argv[2]) if len(sys.argv) > 2 else None
    comparison = json.loads((directory / "comparison.json").read_text())
    statuses = collections.Counter(m["status"] for m in comparison["measurements"].values()
                                   if isinstance(m, dict)) if isinstance(
        comparison["measurements"], dict) else collections.Counter(
        m["status"] for m in comparison["measurements"])
    problems = [f"timing {status}: {count}" for status, count in statuses.items()
                if status not in ("passed", "inconclusive")]
    changes = collections.defaultdict(dict)
    unparsed = []
    for line in comparison.get("quality_or_coverage_changes", []):
        match = CHANGE.match(line)
        if not match:
            if line != "request coverage differs or no per-request measurements exist":
                unparsed.append(line)
            continue
        key = (match["suite"], match["case"], int(match["index"]))
        changes[key][match["what"]] = (match["before"], match["after"])
    fallback, history, parity = set(), set(), set()
    deltas = collections.Counter()
    for key, what in changes.items():
        suite, case, _ = key
        if case.startswith(PARITY_CASES):
            parity.add(key)
            continue
        if "prompt_tokens" in what:
            before, after = map(int, what["prompt_tokens"])
            deltas[before - after] += 1
            fallback.add(key)
            continue
        if set(what) == {"history_sha256"}:
            history.add(key)
            continue
        problems.append(f"unexplained change {key}: {sorted(what)}")
    if instruction is None and deltas:
        instruction = deltas.most_common(1)[0][0]
    for delta, count in deltas.items():
        if delta != instruction:
            problems.append(f"{count} request(s) changed the prompt by {delta} tokens, "
                            f"not the {instruction}-token instruction")
    # History hashes cover the whole server run, so any earlier change explains
    # a later one; suites run in report order.
    order = list(json.loads((directory / "report.json").read_text())["suites"])
    causes = [order.index(key[0]) for key in fallback | parity]
    for key in history:
        if not causes or order.index(key[0]) < min(causes):
            problems.append(f"history changed before any attributed change: {key}")
    problems += [f"unparsed change: {line}" for line in unparsed]
    print(json.dumps({"timing": dict(statuses), "instruction_tokens": instruction,
                      "json_fallback_requests": len(fallback),
                      "downstream_history": len(history), "parity_cases": len(parity),
                      "problems": problems}, indent=1))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
