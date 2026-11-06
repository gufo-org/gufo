Text (`src/ext/text.hpp`): `TextModelRunner::PrefillThrough` gains an
overload that captures several checkpoints in one prefill pass. It takes the
ascending boundaries the chunk covers and a `TextModelRunner::InPassConsumer`,
which the runner calls with each boundary's index, either while that boundary
is readable as the lease's component state or with the boundary's own native
snapshot. `TextRunnerCapabilities::max_in_pass_checkpoints` (default 1) bounds
the boundaries per call. The single-boundary overload with a snapshot
out-parameter and capture callback is deprecated but still works: the new
overload's default forwards one boundary to it. `PrefillCheckpointBytes` may
claim a boundary anywhere inside the chunk, not only within eight rows of its
end. Existing runners compile and behave unchanged.
