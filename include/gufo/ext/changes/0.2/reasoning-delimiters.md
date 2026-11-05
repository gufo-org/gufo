# Reasoning delimiters for output parsing

Text (`text_contracts.hpp`, `src/ext/text.hpp`): `ReasoningMarkup` declares the
literal `open` and `close` delimiters of a provider's reasoning, and
`TextModelRunner::ReasoningDelimiters()` returns it. The OpenAI-compatible
chat, Responses and Messages endpoints use it, buffered and streamed, to split
reasoning from content, and usage accounting counts reasoning tokens with the
same delimiters, including a block a `kAuto` output opens itself.
`ReasoningBudgetTags()` keeps its token-level meaning for budget enforcement,
and a `kAuto` output's budget starts at its leading `start` token.

- `close` must be non-empty, and `open` must not begin with whitespace; the host
  rejects either at load. `open` may be empty (no opener) or span several tokens
  and end in whitespace, such as a channel marker ending in a newline.
- With `InitialOutputState::kThought` the prompt has opened reasoning: an
  `open` that the model repeats is dropped, and output without `close` is
  entirely reasoning. With `kAuto`, reasoning starts only when the output begins
  (after whitespace) with a non-empty `open`. With `kText`, delimiters are data.
- Delimiters not declared by the runner are never interpreted, so another
  provider's tags remain literal text. Tool-call markup still ends an open
  reasoning phase.
- The default derives from the budget tags: `start` opens and the first of
  `ends` closes (`<think>`/`</think>` for Qwen and DeepSeek); a runner that
  declares no budget tags keeps `<think>`/`</think>`. Existing extensions keep
  their behavior; an extension whose reasoning text differs from its budget
  control tokens overrides `ReasoningDelimiters()`, keeping `start` as the control
  token that begins `open` and `ends` covering `close`.
- `TextModelRunner::BuildReasoningBudget` takes an optional
  `InitialOutputState`, and `sampling::ReasoningBudget` gains `opener` and
  `blank`.

Additive: existing callers compile unchanged.
