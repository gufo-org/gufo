# Deprecated items to remove at the next breaking release

Keep compatibility surfaces below until the next breaking release. Remove each
entry when its removal ships; historical benchmark artifacts stay unchanged.

## `usage.gufo.cache_shared_bytes`

Deprecated: this HTTP response field has always been zero and has no producer.
The internal result member is removed; the serializer retains a literal zero for
client compatibility. Do not interpret it as measured sharing or alias it to
component-cache capacity metrics. See [issue #925](https://github.com/gufo-org/gufo/issues/925).

At the breaking release, remove the compatibility key from
`src/cli/serve/openai_chat.cpp` and its assertion in
`tests/cli/openai_chat_test.cpp`.

## `--prefill-chunk N`

Deprecated and ignored: text runners split prompt passes by layers beside active
decoders. Users should remove the option; no replacement flag is needed.

At the breaking release, remove both option registrations and their unused
storage, plus the `deprecated_option` warning, from `src/cli/serve/serve.cpp`.
Update any remaining CLI fixtures and documentation that mention the option.
