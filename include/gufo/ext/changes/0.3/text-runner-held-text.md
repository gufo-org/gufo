# Text runner held text

Additive: `TextModelRunner::HeldText(state, unadvanced)` (`gufo/ext/text.hpp`)
is a new virtual with an empty default. It returns text the runner's
selection pieces have held back so far (a runner that lets later tokens decide
what earlier ones mean, such as Inkling's block headers), as of the advanced
tokens plus `unadvanced`, a selected token not advanced yet. When a request
ends on a stop token or the token limit, the text scheduler publishes it as
the reply's last text, through the stop-sequence filter and within the output
byte limit (#1436).

Migration: none.
