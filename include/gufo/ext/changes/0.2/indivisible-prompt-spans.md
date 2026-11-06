# Indivisible prompt spans

Add `TextPromptContext::PrefillStop(position, up)` (`src/ext/text.hpp`), a
virtual hook through which a model-owned prompt context keeps inputs whose
tokens must be evaluated in one forward, such as image rows that attend each
other, out of split prefill steps. It moves a candidate stop to the first row
of the span containing it, or past the span's last token when `up`. The text
runner pool applies it to every position it chooses: step ends, planned cache
captures, the stable snapshot boundary and shared-prefix checkpoints. The cache
never restores a checkpoint inside a span. A step starting at a span consumes it
whole, beyond its token budget, so a runner's scratch must fit its largest span.
A runner stopping inside a span, or a hook moving a stop the wrong way, fails
the request. With `prefill_alignment`, every stop is both aligned and outside
the spans.

The default returns `position`, so existing contexts and models behave
identically. No other declaration changes.
