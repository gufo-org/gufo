"""Functional tier shared by the suites: `full` runs every transport combination.

The `pr` tier keeps every case but trims its transport matrix. The runner sets
GUFO_FUNCTIONAL_TIER for each suite process; manual runs default to `full`.
"""

import os


TIER = os.environ.get("GUFO_FUNCTIONAL_TIER", "full")
if TIER not in ("pr", "full"):
    raise ValueError(f"GUFO_FUNCTIONAL_TIER must be pr or full, not {TIER!r}")


def streaming_modes():
    """Streaming re-parses partial output, then checks it against the same final
    parse a buffered request returns, so the pr tier keeps only that mode."""
    return (False, True) if TIER == "full" else (True,)


def rotate(options, index):
    """Every option in the full tier; one per case in the pr tier, rotating so
    consecutive cases cover each option."""
    options = tuple(options)
    return options if TIER == "full" else (options[index % len(options)],)
