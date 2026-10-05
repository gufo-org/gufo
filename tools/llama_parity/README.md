# llama.cpp tool-grammar parity

Regenerates `tests/fixtures/llama_cpp_tool_grammar.json`, which
`json_constraint_test` replays: for 86 tool schemas and 4 candidate outputs per
schema, the verdict of llama.cpp's own Qwen3-Coder and DeepSeek tool grammars
(its GBNF engine, `tool_choice: "required"`) beside Gufo's. Every disagreement
must match a reviewed reason in `golden.py`. This fixture covers its listed
cases, not complete grammar or parser equivalence. Gufo retains string
constraints and exact schema keys; llama.cpp permits repeated optional
parameters and trims surrounding spaces from parsed keys.

```sh
OUT=$(mktemp -d)
python3 tools/llama_parity/cases.py "$OUT"
# In a llama.cpp checkout built with build-rocm (or any build with libllama-common):
g++ -std=c++17 -Icommon -Iinclude -Iggml/include -Ivendor -Isrc \
  /path/to/gufo/tools/llama_parity/llama_probe.cpp -Lbuild-rocm/bin \
  -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-rocm/bin -o "$OUT/llama_probe"
"$OUT/llama_probe" QWEN38_FLASH_NEXT.gguf "$OUT/cases_qwen.tsv" > "$OUT/llama_qwen.tsv" 2>/dev/null
"$OUT/llama_probe" DEEPSEEK_V4_FLASH.gguf "$OUT/cases_deepseek.tsv" > "$OUT/llama_deepseek.tsv" 2>/dev/null
# In this tree, after `cmake --build --preset cpu-test --target gufo_core`:
g++ -std=c++20 -I. tools/llama_parity/gufo_probe.cpp build/cpu-test/libgufo_core.a \
  $(pkg-config --libs icu-uc icu-i18n) -o "$OUT/gufo_probe"
"$OUT/gufo_probe" "$OUT/cases_qwen.tsv" qwen required > "$OUT/gufo_qwen.tsv"
"$OUT/gufo_probe" "$OUT/cases_deepseek.tsv" ds required > "$OUT/gufo_deepseek.tsv"
python3 tools/llama_parity/golden.py "$OUT" LLAMA_CPP_REVISION \
  tests/fixtures/llama_cpp_tool_grammar.json
```

Models load vocabulary-only; only their chat templates matter. The probes also
accept `auto`, which the fixture does not record because llama.cpp's lazy
grammar starts at the call marker.

## Qualifying against main

Main moved some requests to a JSON envelope and added a prompt instruction, so
a `run.py --baseline` comparison of this behavior reports changed prompts and
outputs. `audit_qualification.py CANDIDATE_DIR` passes only when every timing
measurement passed or was inconclusive and each change is the removed
instruction (an exact, constant prompt-token drop), a listed llama.cpp parity
case, or later cache history that follows from them.
