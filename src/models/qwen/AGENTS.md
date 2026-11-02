# Qwen3.8-27B model directory

This directory owns the 27B model: target inference, native MTP and DFlash2.
It is the model's home until Card 20 gives it an `extension.json` manifest, so
`CMakeLists.txt` here still defines a plain `gufo_qwen27b` library.

Shared Qwen code is not here. The tokenizer, chat template, control tokens,
GGUF configuration extraction, shared HIP operators, RoPE layout and vision
stack live in [`src/families/qwen/`](../../families/qwen/README.md), and their
rules — including the control-token catalogue and the Python mirror — are in
[`src/families/qwen/AGENTS.md`](../../families/qwen/AGENTS.md).

Model rules:

- Depend on family libraries through the libraries named in
  `src/models/qwen/CMakeLists.txt`; do not reach into another model directory.
- Keep kernel per-source compile options bound to this target
  (`TARGET_DIRECTORY gufo_qwen27b`); a moved kernel must keep them and its
  production code objects must stay identical.
- Run the model quality runners before changing arithmetic, layouts or cache
  boundaries.
