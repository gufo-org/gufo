# Target ownership

Every model and support library records which targets depend on it, and two
gates keep that record and the link graph honest. A library can be moved,
split or excluded safely only when everything that reaches it is known.

## The ownership record

Each library directory has a `library.json`:

| Field | Meaning |
| --- | --- |
| `name`, `target`, `kind`, `purpose` | Library identity; names and targets are unique. |
| `consumers` | Shipped libraries and executables that depend on it. |
| `checks` | Test executables that depend on it. |
| `tools` | Developer and benchmark executables that depend on it. |
| `compiled_elsewhere` | `target`, `source` and `reason` for each of its sources another target compiles. |

A target depends on a library when it links it (also through a variable such as
`GUFO_QWEN_LIBRARIES`), compiles a source below its directory, or includes a
header below its directory from its own sources. Every such target declared
outside the library directory must be listed. Targets under
`tests/models/<model>/` belong to the model's test tree and need no claim.
Family libraries under `src/families/` are checked separately by
`--fail-on-family-graph`.

```sh
python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership
python3 tools/ci/check-extension-isolation.py --target-ownership-report
```

The first is CTest `extension_target_ownership_gate` (`cpu;repository`); the
second prints the graph. An unclaimed dependency, a claim naming no real
dependency, a stale `compiled_elsewhere` entry, or a model directory without a
manifest fails the gate.

As of November 2026:

| Directory | Target | Claimed consumers | Tests and tools |
| --- | --- | --- | --- |
| `src/eval` | `gufo_eval` | `gufo` | 1 |
| `src/models/deepseek_v4_flash` | `gufo_deepseek_v4_flash` | none | 1 |
| `src/models/embedding_gemma2` | `gufo_embedding_gemma2` | none | 0 |
| `src/models/minimax_h3` | `gufo_minimax_h3` | none | 16 |
| `src/models/qwen` | `gufo_qwen27b` | none | 2 |
| `src/models/qwen38_flash_next` | `gufo_qwen38_flash_next` | `bench_cli_test` | 0 |
| `src/models/qwen3_asr` | `gufo_qwen3_asr` | none | 8 |
| `src/models/qwen3_tts` | `gufo_qwen3_tts` | none | 11 |
| `src/models/qwen_image_21` | `gufo_qwen_image_21` | none | 5 |
| `src/testing/compare` | `gufo_compare` | `gufo_llm_cli` | 5 |

Models need no production consumer claim: the executable reaches them through
the generated extension registry, not a direct link.

## The link-graph guard

`gufo_serving` (`src/cli/serve/`: the text scheduler, runner, cache facade and
inference backend) and `gufo_http` are the serving tier. Nothing below them may
link them. `cmake/TargetGraph.cmake` writes `build/<dir>/gufo-link-graph.json`
at the end of a configure with `BUILD_TESTING=ON`, listing the transitive
`INTERFACE_LINK_LIBRARIES` of `gufo_core`, `gufo_cache`, `gufo_cache_hip`,
`gufo_ext_runtime`, every manifest library and the fixed support targets.

```sh
python3 tools/ci/check-target-graph.py --report build/<dir>/gufo-link-graph.json
python3 tools/ci/check-target-graph.py --report build/<dir>/gufo-link-graph.json --show
```

This is CTest `extension_link_graph_gate`. It fails when a protected target
reaches `gufo_serving` or `gufo_http`, or when a protected target is missing
from the report. Only the HIP-only targets (`gufo_hip_host`, `gufo_cache_hip`,
`gufo_ext_hip`, `gufo_deepseek_v4_flash`, `gufo_qwen_kernels`) may be absent,
and only in a build without HIP.

`gufo_serving` links `gufo_ext_runtime`, `gufo_logging`, `gufo_core`,
`gufo_cache`, `gufo_sched` and, with HIP, `gufo_cache_hip` and `hip::host`. It
links no model. Some model test executables under `tests/models/` link
`gufo_serving` to drive a real runner through the scheduler; that is the
intended direction. `--fail-on-serving-boundary` also keeps `src/models/**`,
`src/families/**`, `src/cache/**`, `src/ext/**` and `include/gufo/ext/**` free
of `src/cli` includes.

## Build conditions

`library.json` records ownership; CMake decides when a target exists:

| Target | Exists when |
| --- | --- |
| `gufo_deepseek_v4_flash`, `gufo_qwen_kernels`, `gufo_hip_host`, `gufo_cache_hip`, `gufo_ext_hip` | `ENGINE_ENABLE_HIP` |
| `minimax_h3_*_hip_test`, `video_*_test`, `qwen3_tts_*_hip_test` | `BUILD_TESTING` and `ENGINE_ENABLE_HIP` |
| `gufo_serving`, `gufo_eval`, `gufo_compare` | Always |

## Limits

- The CMake scan is lexical: it resolves plain paths,
  `${CMAKE_CURRENT_SOURCE_DIR}`, `${PROJECT_SOURCE_DIR}` and simple list
  variables, and skips generator expressions. A target created inside a helper
  function is unattributable unless the helper is called from
  `tests/models/<model>/`.
- Includes are read from a target's own sources; a header that arrives through
  another header shows in the coupling report (`--mode report`), not as an edge.
- The audit is static. Whether a target builds is shown by the build and CTest.
