# Tokenization, Sampling, and Output Constraints

Status: design draft, 2026-08-11

## Purpose

Tokenization is part of the model's numerical and API contract. A server can
execute every kernel correctly and still produce incompatible results through
a different tokenizer, chat template, special-token policy, stop condition, or
sampling stream.

`strix` contains compiled-in tokenizer and prompt-formatting
implementations for its curated model kinds. Weight artifacts provide validated
vocabulary and tokenizer data, not executable tokenizer or template code.

## Model Registry Contract

Each compiled-in model descriptor declares:

```cpp
struct ModelTextContract {
    ModelKind model_kind;
    TokenizerKind tokenizer_kind;
    ChatTemplateKind chat_template_kind;
    SamplingPolicyId sampling_policy;
    ConstraintCapabilities constraints;
};
```

The weight manifest records expected hashes for:

- Vocabulary and merge data.
- Tokenizer JSON or equivalent tokenizer assets.
- Added and special tokens.
- Tokenizer configuration.
- Chat-template source and compiled representation.

The server rejects an artifact whose tokenizer kind, vocabulary size, special
token IDs, or hashes do not match the compiled-in model implementation.

## Supported Tokenizers

Only tokenizers required by supported model kinds are implemented. Do not add a
generic runtime capable of executing arbitrary tokenizer plugins.

The tokenizer interface provides:

```cpp
class Tokenizer {
public:
    Result<TokenSequence> encode(const PromptInput&, EncodeOptions);
    Result<TextChunk> decode_incremental(TokenId, DecodeState&);
    Result<std::string> decode(std::span<const TokenId>);
    TokenizerMetadata metadata() const;
};
```

Required behavior:

- Deterministic encoding for identical bytes and options.
- Explicit beginning/end-of-sequence policy.
- Explicit special-token parsing policy.
- No implicit whitespace or Unicode normalization not required by the source
  tokenizer.
- Bounded allocation and input length.
- Thread-safe immutable vocabulary data.
- Request-local encode and decode state.

## Prompt Representation

Transport adapters first create a backend-neutral item sequence:

```text
SYSTEM_TEXT
DEVELOPER_TEXT
USER_TEXT
ASSISTANT_TEXT
TOOL_DEFINITION
TOOL_CALL
TOOL_RESULT
RAW_TEXT
```

The selected chat formatter converts supported items into a deterministic byte
sequence and then tokenizes it once.

The scheduler receives:

- Exact input token IDs.
- Token count.
- Template and tokenizer identity.
- Stop-token IDs and stop-byte sequences.
- Tool and structured-output state.

HTTP, Chat Completions, and Responses adapters must converge on the same item
sequence before formatting. Endpoint-specific JSON must not reach model code.

## Chat Templates

Chat templates are data interpreted by a small validated formatter or compiled
offline into a bounded instruction sequence. The server does not execute
arbitrary Jinja, Python, or code downloaded with a model.

The compiled template may perform only approved operations:

- Iterate input items.
- Emit fixed byte strings.
- Emit escaped item content.
- Branch on known role or item type.
- Insert registered special tokens.
- Render validated tool definitions.

Template compilation validates:

- Supported roles and item types.
- Required generation prompt.
- Tool-call and tool-result pairing.
- Maximum static output expansion.
- Special-token references.
- No recursion or unbounded loops.

The compiled template hash is part of model, prefix-cache, snapshot, and test
identity.

## Special Tokens

Every model implementation declares:

- BOS and EOS behavior.
- End-of-turn and end-of-message tokens.
- Tool-call delimiters.
- Fill, padding, and unknown tokens where applicable.
- Tokens forbidden in ordinary user text.
- Tokens that terminate generation.

Requests cannot override these IDs. User-provided text is not interpreted as a
special token unless the endpoint and model explicitly permit special-token
input.

## Incremental Detokenization

Generated token bytes may not form complete UTF-8 code points individually.
`DecodeState` retains incomplete byte prefixes until they form valid output.

Rules:

- Never emit invalid UTF-8 in JSON or SSE.
- Preserve the tokenizer's exact byte-decoding semantics.
- Do not replace incomplete bytes with the Unicode replacement character while
  more tokens may complete them.
- Flush or report a terminal decoding error at generation end according to the
  model contract.
- Keep a bounded suffix for stop-sequence matching.

SSE chunk boundaries are transport decisions and do not alter decoded text.

## Stop Conditions

Generation can stop because of:

- Model EOS or another registered terminal token.
- Maximum output tokens.
- A request stop-token ID.
- A request stop byte string.
- Completed tool call.
- Completed structured output.
- Cancellation, deadline, or backend failure.

Stop strings are matched against decoded bytes across token and SSE boundaries.
Use a bounded streaming matcher. The server does not emit bytes belonging to a
matched stop string unless the endpoint contract explicitly requests them.

When several conditions occur at one position, use a deterministic precedence
and report the corresponding finish reason.

## Sampling Contract

Sampling state belongs to the logical request, not a physical batch row,
device, or graph slot.

```cpp
struct SamplingState {
    uint64_t seed;
    uint64_t counter;
    SamplerVersion version;
};
```

For the same:

```text
model artifact
input token sequence
sampling configuration
seed
sampler version
```

the selected tokens must not change merely because:

- Other requests joined or left the batch.
- The physical row moved.
- Execution changed between eager and graph replay.
- Unrelated GPU and NPU work overlapped.

Different numerical routes may change logits and therefore sampled tokens.
Route determinism and numerical determinism are reported separately.

## Sampling Pipeline

The ordered pipeline is versioned:

1. Validate finite logits.
2. Apply model-specific logit transformations.
3. Apply presence and frequency penalties.
4. Apply repetition or token suppression rules.
5. Apply structured-output or grammar mask.
6. Apply temperature.
7. Apply top-k.
8. Apply top-p or other supported truncation.
9. Normalize and sample from request-owned RNG state.

Greedy generation bypasses stochastic normalization and selects a deterministic
argmax with a defined token-ID tie break.

Unsupported sampler fields are rejected by the server rather than ignored.

## RNG

Use a counter-based or otherwise reproducible RNG implementation with a
versioned algorithm. Do not depend on:

- Global process RNG state.
- Host thread scheduling.
- Physical batch order.
- Device workgroup execution order.
- An implementation-defined C++ standard-library distribution.

The RNG counter advances only for committed sampling decisions. Cancelled or
rolled-back speculative rows do not consume canonical request RNG state.

## Structured Output

The first implementation supports a bounded subset of JSON Schema sufficient
for common structured responses.

Schema compilation occurs before admission and enforces limits on:

- Input schema bytes.
- Nesting depth.
- Property and enum counts.
- String and numeric constraints.
- Compiled automaton states and transitions.
- Per-token mask computation.

Unsupported schema features return an explicit request error.

The constraint engine operates over token byte strings and incremental parser
state. It must handle tokens containing:

- Multiple JSON characters.
- Partial UTF-8 code points.
- Partial escape sequences.
- Complete or partial literals.

Grammar state is request-local and transactionally copied or selected during
speculative verification.

## Tool Calls

The server formats tool definitions and may constrain generation to the
model-specific tool-call syntax.

The server:

- Validates tool names and bounded JSON schemas.
- Returns generated tool calls to the client.
- Never executes tools, shell commands, URLs, or generated code.
- Does not fetch remote tool definitions.
- Requires clients to return tool results in a subsequent request.

Any future tool-execution service must be a separate, explicitly configured
security boundary.

## Usage Accounting

Usage is computed from committed logical tokens:

- Input tokens after final chat formatting.
- Cached input tokens where the endpoint exposes that distinction.
- Output text tokens.
- Accepted speculative tokens counted once.
- Rejected speculative rows not counted as output.
- Tool-call tokens according to the same tokenizer stream.

Physical padding rows, verifier rows, replayed tokens, and internal support-model
tokens are operational metrics, not client usage.

## Performance

- Load and validate tokenizer assets once per resident model.
- Share immutable vocabulary and merge tables across requests.
- Use bounded per-thread or pooled scratch.
- Tokenize independent requests concurrently within a configured CPU budget.
- Do not delay a ready decode token while performing unrelated long prompt
  tokenization.
- Incrementally decode and stream without reconstructing the complete output.
- Cache formatted and tokenized prefixes only under the complete tokenizer and
  template identity.

## Tests

- Golden token IDs against the source tokenizer.
- Empty, ASCII, Unicode, invalid-byte, and whitespace-sensitive inputs.
- Added, special, BOS, EOS, and tool tokens.
- Chat-template fixtures for every supported item sequence.
- Direct Responses and Chat Completions token parity.
- Incremental detokenization across every byte split of multi-byte UTF-8.
- Stop strings split across tokens and SSE chunks.
- Greedy tie-breaking.
- Seeded isolated versus continuously batched sampling.
- Row movement and graph replay RNG stability.
- Speculative rollback does not advance canonical RNG.
- Structured-output schema limits and malformed schemas.
- JSON grammar tokens containing partial escapes and UTF-8.
- Tool calls are returned and never executed.
- Usage counts exclude physical and rejected speculative work.
- Tokenizer and template hash mismatch rejection.
