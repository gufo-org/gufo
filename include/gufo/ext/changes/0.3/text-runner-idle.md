# Text runner idle notification

Additive: `TextModelRunner::Idle()` (`gufo/ext/text.hpp`) is a new virtual
with an empty default. The text scheduler calls it on its own thread when it
goes idle (no request runs, is queued or waits) and about once a second while
it stays idle. A runner releases memory
there that only running work holds. Flash-Next releases its wide prefill
scratch, which a prefill that ends without a decode step would otherwise keep
until the next decode. It must be cheap when nothing is held.

Migration: none.
