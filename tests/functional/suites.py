"""The one registry of functional suites and their runner requirements.

run.py, openai_sdk.py and the impact manifest check (impact.suite_names) read
these names; a suite added here also needs an impact.core.json entry, and a
dispatch entry in openai_sdk.py unless run.py drives it itself.
"""

SUITES = ("discovery", "openai-count-tokens", "messages-count-tokens", "messages-image-counts", "responses", "stops", "conversation", "image-inputs", "image-count", "tool-images", "image-detail", "structured", "structured-limits",
          "tool-reasoning", "tool-streaming", "reasoning-separator", "reasoning-budget",
          "tools", "auto-tools", "tool-edges", "tool-agent", "tool-agent-loop", "tool-history", "messages-tools", "tool-untyped", "tool-mixed", "tool-native-schemas", "tool-native-types", "tool-choice-cache", "tool-parameter-order", "tool-type-unions", "tool-schema-edges", "sampling-defaults", "sampling-ranges", "logprobs", "responses-logprobs", "batch",
          "long-context", "state-edges", "progress", "stream-start", "prefill-scheduling", "metrics", "cache-edits", "cache-growth", "cache-depth", "cache-rotation", "cache-concurrency", "cache-shared-prefix", "cache-bridge", "cache-compaction", "cache-transforms", "cache-pressure", "cache-messages-loop", "cache-stalls", "system-injection", "cache")
# Suites that already run another suite's checks; selecting both repeats the work.
INCLUDED_IN = {"auto-tools": "tools", "tool-native-types": "tool-native-schemas"}
# Server --context each suite's histories need. Without an explicit --context
# the runner starts the server with the largest one selected (8192 at least).
MIN_CONTEXT = {"cache-stalls": 65536, "cache-depth": 32768, "cache-transforms": 32768,
               "cache-pressure": 32768, "cache-shared-prefix": 16384, "cache-compaction": 16384,
               "cache-messages-loop": 16384, "prefill-scheduling": 16384}
# These fill the RAM budget before measuring; the default budget sized for the
# host takes far more requests than they send for a model with small checkpoints.
NEEDS_CACHE_RAM = ("cache-bridge", "cache-pressure")
# Run by `all` only when named: they need a dedicated server configuration.
NOT_IN_ALL = ("cache-bridge", "cache-compaction", "cache-transforms", "cache-pressure", "cache-messages-loop",
              "cache-stalls", "prefill-scheduling")
# These need the server's vision projector.
VISION = ("image-inputs", "image-count", "tool-images", "image-detail")
# run.py drives these itself (continuation.py and the disk restart); openai_sdk.py runs the rest.
RUNNER_ONLY = ("cache",)
SDK_SUITES = tuple(suite for suite in SUITES if suite not in RUNNER_ONLY)
