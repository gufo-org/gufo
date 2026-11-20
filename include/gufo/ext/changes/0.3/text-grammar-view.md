# Grammars in the model's own spelling

Add optional `TextModelRunner::ToolGrammarSyntax(const ChatRequest&)` and
`ReasoningGrammarMarkers(const ChatRequest&)`, for a runner that publishes
`ToolDialectId()` and `ReasoningDelimiters()` markup but whose model writes
another spelling. Tool grammars and the reasoning phase of response-format and
tool grammars then constrain that spelling. Parsing is unchanged. Both default
to `std::nullopt`, which keeps the previous behavior.
