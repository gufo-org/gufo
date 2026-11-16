# The grammar's view of a runner's own tokens

`TextModelRunner::ToolGrammarSyntax()` and `TextModelRunner::ReasoningGrammarMarkers(const ChatRequest&)`
(`src/ext/text.hpp`, published through `gufo/ext/text.hpp`) let a runner that rewrites the model's native
tool calls and reasoning markers into the canonical markup Gufo parses describe the model's own spelling to
the grammars. Tool grammars (`tool_choice` required/named, strict tools, lazy tool grammars) then use the
returned `JsonConstraint::ToolSyntax` instead of the one `ToolDialectId()` selects, and reasoning phases of
response-format and tool grammars use the returned open/close markers instead of `ReasoningDelimiters()`.
With a native syntax the host adds no JSON-call instruction to the prompt. Response parsing, streaming and
every other use of `ToolDialectId()` and `ReasoningDelimiters()` are unchanged.

The change is additive: both virtuals default to `std::nullopt`, which keeps the previous behavior for every
runner. Migration: none.
