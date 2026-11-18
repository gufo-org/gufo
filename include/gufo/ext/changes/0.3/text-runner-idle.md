# Text runner idle notification

Additive: `TextModelRunner::Idle()` (`gufo/ext/text.hpp`) is a new virtual
with an empty default. The text scheduler calls it on its own thread each time
it goes idle: no request runs, is queued or waits. A runner releases memory
there that only running work holds. Flash-Next releases its wide prefill
scratch, which a prefill that ends without a decode step would otherwise keep
until the next decode. The call may come again before more work runs, so it
must be cheap when nothing is held.

Migration: none.
