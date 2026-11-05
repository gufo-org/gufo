# Qwen family rules

## Control tokens

[`control_tokens/control_tokens.hpp`](control_tokens/control_tokens.hpp) owns
shared Qwen framing spellings in `gufo::tokenization`. Inspect it when reviewing
or changing their use. Centralizing these constants does not change tokenization
or EOS handling.

### Rules

- Use the named constants in production framing and tokenizer code.
- Keep fixed prompts compile-time; append constants when building dynamic text.
- Confirm new tokens against pinned model metadata or the official reference,
  and update both language catalogues when shared consumers need them.
- Preserve independent reference fixtures and rendered-byte goldens. Expected
  results must not be derived solely from the constants they validate.
- Keep other models' token families in their model directories.

### Catalogue

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

### Python

`tools/gufo/control_tokens.py` mirrors this catalogue for the Python tools and
tests, imported as `gufo.control_tokens`. Keep shared values identical across
the two catalogues.

## Library boundaries

Every library here is one directory with a `library.json` manifest and a
`CMakeLists.txt`. Keep them small and purpose-named so no family library becomes
a second core, and keep the graph acyclic:

- A family library may use `gufo_core` and the libraries listed in its own
  `depends`; it must never include a header from `src/models/**`, and `src/core`
  must never include one from here.
- `depends` names the libraries whose files this library actually includes —
  no more (an unused declaration fails) and no less (an undeclared edge fails).
- Add the new library to `CMakeLists.txt` in dependency order, lowest layer
  first, and update `README.md`.

`python3 tools/ci/check-extension-isolation.py --fail-on-family-graph`
(CTest `extension_family_graph_gate`) enforces all of it, including that every
header under `src/families/` lives inside a declared library.
