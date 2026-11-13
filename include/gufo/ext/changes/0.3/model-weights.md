# Downloadable model weights

Publish downloadable model weights in extension metadata
(`schemas/extension.schema.json`, `cmake/GufoExtension.cmake`):
- Optional top-level `models`: each model has an `id`, the `provider` that
  runs it, an optional `license` and `description`, and `components`.
- Each component names its `role`, the command `option` its path fills,
  `install` (`required`, `default` or `optional`), a Hugging Face `source`
  (`hf` repository and 40-hex `revision`), optional `include`/`exclude` globs
  and an optional `entry` path, where `"."` is the model directory.
- Exactly one required component fills `--model`.
- `models.lock.json` beside `extension.json` pins every component's files by
  path, size and SHA-256; `tools/models/lock.py` writes and checks it.
  `gufo model` installs only locked files.
- CMake rejects a non-array `models`, `models` in a library and a lock without
  a `models` object, and embeds built-in models for `gufo model`.

Existing manifests remain valid. No C++ declarations change.
