# Model evidence

Per-model developer records behind the public [model pages](../../models/README.md).

Each model folder contains:

- `README.md`: implementation notes that the public page leaves out.
- `BENCHMARKS.md`: current retained measurements, scope/date and TODO cells.
- `QUALITY.md`: measured quality, reference/scope, replay guarantees and unresolved gaps.
- `EXPERIMENTS.md`: short retained/rejected decisions.
- `artifacts/`: only useful machine-readable results/calibration, when present.

Keep executable fixtures with their tests. New logit/trace dumps and local
profiles belong in ignored top-level `artifacts/`; commit only independent
reference evidence needed by a maintained check. Old experiment records remain
in Git history. See [benchmark methodology](../benchmarks.md).

Quality reports distinguish agreement with an independent model implementation
from consistency between Gufo execution modes. Greedy speculative decoding
should reproduce AR; sampled speculation can use different random draws.
Seeded replay is checked within the same execution configuration.
For numerical comparisons, lower KL divergence, total variation, relative L2
and word-error rate are better; higher cosine similarity and image PSNR are
better. These bounded checks are not broad capability or perceptual scores.
