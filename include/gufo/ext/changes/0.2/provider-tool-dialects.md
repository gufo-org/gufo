# Provider-defined native tool dialects

Text (`src/ext/text.hpp`, `src/core/json_constraint.hpp`): a provider registers
a native tool dialect, its grammar and its parser by identity, without a core
format enum entry.

- `ToolDialect` gains `id` and `syntax`. `id` is the selection key; an empty
  one is `BuiltinToolDialectId(format)`, so existing registrations keep their
  identity. `syntax` (`JsonConstraint::ToolSyntax`) describes the call grammar;
  without it the built-in grammar of `format` applies.
- `TextModelRunner::ToolDialectId()` selects the dialect; its default is the
  built-in identity of `ToolFormat()` (`json`, `qwen`, `deepseek`).
  `BuiltinToolDialectId()` names those identities and `FindToolDialect(id)`
  finds a registration. The host rejects a runner whose identity is neither
  built in nor registered at load.
- `JsonConstraint::ToolSyntax`, `TaggedArguments` and `ValueSyntax` describe a
  call envelope with either tagged key/value parameters or JSON-shaped
  arguments in another value spelling. `JsonConstraint::Syntax(format)` returns
  the descriptors of the built-in Qwen and DeepSeek grammars, which are now
  compiled from them unchanged. New overloads `ToolParameters(schema, strict,
  syntax)`, `OpenToolParameters(syntax)` and `WithTools(..., syntax)` compile
  them.
- The host now uses a registered dialect's `markers`, `reasoning_markers` and
  `closers`, and selects its scanner by identity; only the selected dialect's
  markers start calls. `closers` are removed when echoed directly after an
  accepted call; the DeepSeek registration now declares none, as the host
  already behaved.
- `RegisterToolDialect` validates its input (std::invalid_argument): `json`
  and `qwen` are built into the host, and a dialect needs markers containing
  `<`, a scanner, and a syntax opened by one of its markers or a native
  `format`. Registered dialects keep stable addresses for the process.
- Constrained reasoning uses the runner's `ReasoningDelimiters()`:
  `JsonConstraint::WithReasoning(answer, close, open)` ends the reasoning phase
  at `close` (or a complete call header) instead of `</think>`. Prompt-opened
  reasoning (`kThought`) uses `close` alone; for `kAuto`, unless the request
  disables reasoning, the output may start with the complete `open` (which may
  span tokens and include whitespace) and reason, or answer directly. The
  defaults (`</think>`, already open) build the previous grammar.
- With `partial`, `scan_tool_calls` appends only an unfinished call that later
  text can still complete, and its arguments only grow.

Additive for existing callers, except that constrained output of a `kAuto`
runner with a non-empty opener now admits a leading reasoning block, as its
response parsers already did; a registration that the new validation rejects
must declare a scanner, `<` markers and a grammar.
