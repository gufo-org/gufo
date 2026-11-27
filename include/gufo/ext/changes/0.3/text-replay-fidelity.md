# Text replay fidelity

Additive changes to `gufo/ext/text.hpp` and `text_contracts.hpp`. They let a
runner keep a replayed conversation the same tokens it sampled.

- `TextPromptContext::TokenizationIndependent()` (default `false`). A context
  that returns `true` and has no `cache_identity` stays correct for any
  tokenization of the same prompt text. The host then aligns a replayed
  answer to the tokens its live frontier sampled for that text (#1154), as it
  already did for requests without a context. A runner that attaches a
  context to every request, for example to pass tool names, keeps that
  frontier reuse.
- `ReasoningMarkup::repeats` (default `false`). Unconstrained output may open
  further thoughts after content, as a model that writes its turn as a list of
  blocks can. The parsers move each later `open`...`close` span to the
  reasoning, in order, and join the content around it. Streamed deltas keep
  the order, so Responses and Messages outputs open a new reasoning item or
  thinking block for each.
- `ChatMessage::parts`: the text and thought parts in the order the client
  sent them, when that order says more than `thought` followed by `content`
  (several text parts, or a thought after text). Chat accepts assistant
  `{"type": "thinking", "thinking": ...}` content parts, Messages keeps its
  thinking blocks in place, and Responses joins a reasoning item after an
  answer to that turn. `content` and `thought` still hold the parts joined, so
  a renderer may ignore `parts`.

Migration: none.
