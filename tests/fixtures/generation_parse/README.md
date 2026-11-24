# Generation parse corpus

Cases for `generation_parse_test`, which compares the Rust generation parser
with a frozen copy of the C++ parser it replaced (#1352). Each case is the
parser's setup and the generated pieces, never a prompt:

- `openai_chat_test.json`: the `FakeBackend` cases of
  `tests/cli/openai_chat_test.cpp`, recorded through a temporary hook in
  `GenerationStream` and `ParseGeneratedText` and reduced with
  `tools/generation_parse/corpus.py capture`.
- `mutations.json`: hand-written duplicate and conflicting parameters, Python
  literals, recursive `$ref`, malformed UTF-8, escaped JSON and quoted
  markers.
- `trace-*.json`: model output from the tool and reasoning functional suites,
  converted from `gufo serve llm --trace` with
  `tools/generation_parse/corpus.py trace`.

Run `generation_parse_test tests/fixtures/generation_parse --exhaustive` to
split and truncate long responses at every byte, and `--bench` for the cost
per token of each parser.
