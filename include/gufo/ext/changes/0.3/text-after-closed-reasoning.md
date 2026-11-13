# Text after a reasoning block the prompt closed

`InitialOutputState::kTextAfterThought`, with the alias
`kContentAfterReasoning` (`src/ext/text.hpp`), tells the host that the prompt
closed reasoning, as a template that ends in `<think></think>` does. The host
treats it as `kText`, with one difference: before the answer or tool call it
drops the separator it drops after a `</think>` the model wrote (line breaks,
or any whitespace with the `qwen` tool format), in buffered and streamed
responses.

The change is additive. Runners that return the existing values behave as
before; a runner opts in by returning the new value.

Migration: a `switch` that lists every `InitialOutputState` needs a case for
the new value. A runner that trimmed the separator from its own output can
stop when it returns `kTextAfterThought`.
