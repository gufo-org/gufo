# llama.cpp Jinja interpreter

Imported from `ggml-org/llama.cpp` revision
`68d9053afd4f4d0752ced6187585f862355a40be`, the reference revision already
pinned by Gufo. The `lexer`, `parser`, `runtime`, `string`, `value`, and `utils`
files come from `common/jinja/`; `unicode.h` and `unicode.cpp` come from
`common/`. No inference, template autodetection, or response parser is imported.

Source: https://github.com/ggml-org/llama.cpp/tree/68d9053afd4f4d0752ced6187585f862355a40be/common/jinja

License: MIT, Copyright (c) 2023-2026 The ggml authors. The upstream notice is
retained in `licenses/llama.cpp.txt` and installed by Gufo's existing license
installation rule.

Local adaptations:

- Replace the `common_json` conversion bridge with Gufo's ordered JSON values.
- Use local include paths for the isolated source import.
- Serialize `strftime_now` access to `std::localtime` for concurrent renders.
- Use the existing ICU dependency for Python-compatible Unicode stripping.
- Retain input provenance through JSON serialization, splitting, joining, and
  string indexing, including JSON object keys; collapsed transformations
  conservatively retain input status.

Keep upstream formatting in this directory. Gufo's wrapper and tests follow
the repository's formatting rules. Compare rendering against the independent
Python Jinja2 fixtures and the existing Hugging Face byte/token goldens when
updating this import.
