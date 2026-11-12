# Qwen3.8 27B implementation notes

Developer detail behind the [public model page](../../../models/qwen3.8-27b.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)
· [ROCm 10.1 migration](ROCM.md)

BF16 targets are reference-only. Targets and drafts need not have matching
precision; the Q4_K_M DFlash2 draft serves both targets.

## Wide copy verification

A request decoding alone can verify a prompt-lookup copy wider than the DFlash2
block: greedy requests, and sampled ones without a seed. Copy-only steps start
at the ordinary eight rows; two whole copies in a row at the current width
double it, up to 64 rows, and any rejected copy returns it to eight. The wide
pass shares every projection across its rows and runs attention in eight-row
segments, so each row's logits and DFlash2 target features equal those of
consecutive ordinary verifications (`qwen27b_wide_verification_test` also checks
decoding after cutting back to any kept prefix). Rollback replays the captured
recurrent inputs of every kept row, so the session's SSM replay ring grows from
16 to 64 rows (about 91 MiB) on first use; the resource claim includes it, and
the load narrows the widest pass when the configured sessions would not fit.
When more than eight rows are kept, the draft injects all but the last at once,
so snapshots and batched steps still hold at most eight pending rows. Seeded
sampled requests and batched steps keep the ordinary width.

Greedy speculation must match AR. Sampled modes preserve the target policy,
but AR and speculation need not share the same-seed sequence. See the
[quality contract](QUALITY.md).

## Vision

Vision weights upload lazily. In `chat`, `--image` attaches to the first user
turn. Ordered text/image parts, multiple images and later turns work with
AR/DFlash2 and caches. Official dynamic resizing allows 64–16384 merged image
tokens. Cache identity includes processed pixels, placement, preprocessing
version and projector contents. See [vision checks and limits](QUALITY.md#vision).
