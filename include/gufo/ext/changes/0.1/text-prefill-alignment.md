# Text runner prefill alignment

Text (`include/gufo/ext/text.hpp`, `src/ext/text.hpp`):
`TextRunnerCapabilities::prefill_alignment` (default 1) declares the position
granularity from which a runner's prefill continues a prompt exactly. The
serving runner ends budget-limited prefill passes on a multiple of it and
rounds published prompt checkpoints down to one. The default keeps every
position unchanged. Source compatible; extensions that construct
`TextRunnerCapabilities` with designated initializers rebuild.
