# Tool choice none bans the call marker

Add `JsonConstraint::WithoutTools(ToolFormat)` and
`JsonConstraint::WithoutTools(const ToolSyntax&)`: free text that never
completes the call marker (`ToolSyntax::open`, or `<tool_call>` for the JSON
envelope), also across tokens. A `ToolSyntax` without an `open` literal is
`std::invalid_argument`.

For a request with tools declared and `tool_choice: "none"`, the host now
constrains the answer with this grammar for the runner's selected dialect
(`ToolGrammarSyntax()` first, as for calls) instead of leaving it
unconstrained. The prompt is unchanged. A runner that already omits tools under
`tool_choice: "none"` needs no change; rebuild against the new headers.
