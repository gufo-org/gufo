# Whole prompt end during prefill

Add `TextRunnerState::prefill_prompt_end()` and `SetPrefillPromptEnd`
(`src/ext/text.hpp`). During a prefill step, `prefill_prompt_end()` holds the
whole prompt's token count; a step's own `prompt` span ends at its frontier,
which may be a cache boundary short of it. It is 0 outside a step and when
a caller drives the runner without the cache, meaning the step's own end.

Add the virtual `TextModelRunner::CanContinue(state, prompt_tokens)`. A runner
that keeps only the state longer prompts need returns false for a prompt its
live or restored state cannot continue, and the cache serves that prompt from a
shallower boundary.

Additive: the default `CanContinue` returns true, so existing runners behave
identically.
