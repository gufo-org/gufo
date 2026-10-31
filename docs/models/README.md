# Models

Linux x86-64 / AMD Strix Halo gfx1151 is the production target. Build with
`nix build`; Python/reference toolchains are development-only. Model weights
are downloaded separately and are not part of the runtime package.

The [project README](../../README.md#models-and-benchmarks) lists every model by
task, with its license and headline benchmarks. Add new models there. Each
model has one folder here, named as the README links it.

Each model folder contains:

- `README.md`: model card, acquisition, usage and modes.
- `BENCHMARKS.md`: current retained measurements, scope/date and TODO cells.
- `QUALITY.md`: measured quality, reference/scope, replay guarantees and unresolved gaps.
- `EXPERIMENTS.md`: short retained/rejected decisions.
- `artifacts/`: only useful machine-readable results/calibration, when present.

Keep executable fixtures with their tests. New logit/trace dumps and local
profiles belong in ignored top-level `artifacts/`; commit only independent
reference evidence needed by a maintained check. Old experiment records remain
in Git history. See [benchmark methodology](../BENCHMARKS.md).

Quality reports distinguish agreement with an independent model implementation
from consistency between Gufo execution modes. Greedy speculative decoding
should reproduce AR; sampled speculation can use different random draws.
Seeded replay is checked within the same execution configuration.
For numerical comparisons, lower KL divergence, total variation, relative L2
and word-error rate are better; higher cosine similarity and image PSNR are
better. These bounded checks are not broad capability or perceptual scores.
