Text (`src/ext/text.hpp`): `TextModelRunner::PrefillThrough` captures several
checkpoints in one prefill pass. It takes the ascending boundaries the chunk
covers and a `TextModelRunner::InPassConsumer`, which the runner calls with
each boundary's index, either while that boundary is readable as the lease's
component state or with the boundary's own native snapshot. It replaces the
single-boundary overload with a snapshot out-parameter and capture callback.
`PrefillCheckpointBytes` may now claim a boundary anywhere inside the chunk,
not only within eight rows of its end. Runners that do not override
`PrefillThrough` are unaffected.
