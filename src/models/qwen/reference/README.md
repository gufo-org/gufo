# Qwen3.8 chat-template reference

The compiled formatter is pinned to `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The official `Qwen/Qwen3.8-Flash-Next` template at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` has the same SHA-256.
The two pinned `tokenizer.json` files are also byte-identical.
Both default to `enable_thinking=true`, `reasoning_effort="xhigh"`, and
`preserve_thinking=true`. Native efforts are `low`, `medium`, and `xhigh`;
medium adds no effort instruction. Explicit thinking-off suppresses the
effort instruction and closes the generation prompt's thinking block.

- Official `chat_template.jinja` SHA-256:
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
- Repository copy with its terminating LF SHA-256:
  `514d5304fcac63bda60e6faf860d5657010cf43cbc4c633c8bf678ee184e7a37`
- Official `tokenizer.json` SHA-256:
  `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
- Recognized Unsloth Qwen3.8 GGUF template SHA-256:
  `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`

`chat_template.jinja` is reference data only. Gufo validates recognized
artifact hashes and executes the bounded C++ implementation.

Full rendered-byte goldens cover tools before system instructions, the Unsloth
artifact's leading developer-message extension, tool-loop reasoning retention,
Unicode trimming around images and optional `Picture N:` numbering.
The reference rejects late system/developer messages; Gufo hoists them into
the leading system turn in their original order. Tool/system cases also compare
complete token sequences using the real GGUF tokenizer.

The tokenizer applies NFC normalization and the pinned Unicode split pattern,
including Unicode case-insensitive contractions, letter/mark runs and
trailing-whitespace backtracking. Ten raw-text
goldens check exact token IDs for spacing, code, accents, scripts with combining
marks, Unicode number categories and special-token boundaries on both models.
