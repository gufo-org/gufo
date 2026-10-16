# Qwen control tokens

[`control_tokens.hpp`](control_tokens.hpp) owns shared Qwen framing spellings
in `gufo::tokenization`. Inspect it when reviewing or changing their use.
Centralizing these constants does not change tokenization or EOS handling.

## Rules

- Use the named constants in production framing and tokenizer code.
- Keep fixed prompts compile-time; append constants when building dynamic text.
- Confirm new tokens against pinned model metadata or the official reference,
  and update both language catalogues when shared consumers need them.
- Preserve independent reference fixtures and rendered-byte goldens. Expected
  results must not be derived solely from the constants they validate.
- Keep other models' token families in their model directories.

## Catalogue

| Constant | Meaning |
|---|---|
| `kEndOfText` | End-of-text entry; fallback eos and pad when GGUF metadata does not carry them. |
| `kImStart` | Opens a chat turn, emitted immediately before the role name; fallback BOS. |
| `kImEnd` | Closes a chat turn; the primary end-of-generation stop token. |
| `kObjectRefStart`, `kObjectRefEnd` | Delimit an object reference in grounding output. |
| `kBoxStart`, `kBoxEnd` | Delimit a bounding box in grounding output. |
| `kQuadStart`, `kQuadEnd` | Delimit a quadrilateral in grounding output. |
| `kVisionStart`, `kVisionEnd` | Delimit the image-placeholder run inside a user turn. |
| `kVisionPad` | Padding entry in the pinned MiniMax H3 vocabulary. |
| `kImagePad` | One image-placeholder token, repeated once per visual token. |
| `kVideoPad` | One video-placeholder token. |

## Python

`tools/gufo/control_tokens.py` mirrors this catalogue for the Python tools and
tests, imported as `gufo.control_tokens`. Keep shared values identical across
the two catalogues.
