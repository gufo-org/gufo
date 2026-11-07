# Text runner hidden-state encoding

Text (`src/ext/text.hpp`): add `TextHiddenStateEncoding`,
`TextHiddenStateRequest`, `TextHiddenStates`, `TextHiddenStateStatus` and the
optional `TextModelRunner::HiddenStateEncoding()` /
`TextModelRunner::EncodeHiddenStates(...)` pair, so another extension can use a
text model created through `FindTextProvider(...)->create` as a frozen encoder.
A call encodes one complete, caller-framed prompt from position zero,
independent of every session state, and returns owned row-major FP32
`[tokens, width]` states in token order: the residual stream after the last
decoder layer, before the final output norm, which consumers apply exactly
once. `HiddenStateEncoding()` returning `std::nullopt` means unsupported, and
`EncodeHiddenStates` then throws `std::logic_error`; execution outcomes are
reported as `kOk`, `kCancelled`, `kInvalidInput` or `kFailed`. The Qwen text
runner implements it on a private executor. Additive; existing runners keep
the unsupported defaults.
