# CLI (`src/cli`)

This directory contains the unified `strix` CLI.

## Philosophy

1. **One Binary**: All user-facing capabilities (`serve`, `prompt`, `chat`, `bench`, `video`, `diagnose`, `probe`) live under the single `strix` executable.
2. **Subcommands Over Flag Soup**: Model-specific or modality-specific options belong under explicit subcommands (e.g. `strix serve ... llm ...`, `strix serve ... video ...`), keeping global options clean and self-documenting.
3. **Declarative Parsing**: Flags and arguments are declared and bound to typed struct fields using `ArgParser` (`src/cli/arg_parser.hpp`). Never parse raw `argv` manually.
4. **CLI Layer vs Engine Core**: Code in `src/cli/` handles argument parsing, terminal output, and daemon lifecycle. Heavy computational work and kernel dispatches belong in `src/models/` and `src/core/`.

## Directory Organization

Each top-level command has its own dedicated subdirectory:

```
src/cli/
├── main.cpp              # Root dispatcher & global flags (-h, -v, help)
├── arg_parser.hpp        # Declarative type-safe argument parser
├── bench/                # `strix bench` (model throughput & kernel benchmarks)
├── diagnose/             # `strix diagnose` & `strix probe` (hardware inventory & toolchain probe)
├── prompt/               # `strix prompt` & `strix chat` (text generation CLI & REPL)
├── serve/                # `strix serve` (HTTP API server: `llm`, `video`, `audio`)
└── video/                # `strix video` (MiniMax H3 text-to-video generator CLI)
```

## Adding a Command or Flag

- **Adding a Flag**:
  - Add a field to the command's options struct.
  - Bind it in the command's `ArgParser` definition using `parser.AddOption(...)` or `parser.AddFlag(...)`.
  - **Group Organization**: Place the flag in a logical group (e.g. `Model`, `Sampling`, `Hardware`, `Speculative`, `Diagnostics`). If no existing group fits, create a new meaningful group.
  - **Descriptions**: Keep descriptions concise yet explanatory (include defaults and expected value formats).
  - **Partial Implementation Notation**: If a flag is not yet supported for all models/backends, annotate the description with `(TODO: <model>)` (e.g., `(TODO: deepseek)` or `(TODO: qwen)`).
- **Adding a Command**:
  1. Create `src/cli/<command>/<command>.hpp` and `<command>.cpp` in `namespace strix::cli`.
  2. Implement `Run<Command>(std::span<const char* const> args)`.
  3. Register the routing in `src/cli/main.cpp`.
  4. Add the source files to `CMakeLists.txt` under `add_executable(strix ...)` and write unit tests in `tests/cli/`.
