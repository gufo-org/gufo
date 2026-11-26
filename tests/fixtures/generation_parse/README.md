# Generation parse corpus

Cases for `generation_parse_test`, which checks the generation parser
against the expected results in `expected/`. Each case is the parser's setup
and the generated pieces, never a prompt:

- `fake_backend.json`: `FakeBackend` cases of
  `tests/cli/openai_chat_test.cpp`, recorded through a temporary hook in
  `GenerationStream` and `ParseGeneratedText` and reduced with
  `tools/generation_parse/corpus.py capture`.
- `mutations.json`: hand-written duplicate and conflicting parameters, Python
  literals, recursive `$ref`, malformed UTF-8, escaped JSON and quoted
  markers.
- `trace-*.json`: model output from the tool and reasoning functional suites,
  converted from `gufo serve llm --trace` with
  `tools/generation_parse/corpus.py trace`.

The recorded and traced cases are the subset that covers every dialect,
request setup, event shape, mutation result and documented split dependence
of the full recordings; `openai_chat_test` checks the `FakeBackend` cases
through the endpoints. Each distinct tool list is stored once in
`tool_sets`.

`expected/<file>` holds, per case, the events at the recorded boundaries
(`stream`), the same pieces only buffered (`buffered`), the whole-response
parse (`parse`), and how many splits of the case and its mutations assemble
differently from the whole response, by documented reason
(`split_dependence`). Events are `[kind, ...]` arrays, described in the
test. The expected results were written from the Rust parser while it still
matched the frozen C++ parser it replaced (#1352).

CTest runs one test per file. Responses over 96 bytes are split and truncated
at 48 points by default; run `generation_parse_test
tests/fixtures/generation_parse --exhaustive` for every byte. After adding a
case or changing the parser on purpose, run `generation_parse_test FILE
--write` and review the diff of `expected/`.
