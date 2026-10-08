# Retained measurement inputs

This directory keeps the small measurement inputs and fitted constants used
by the research scripts: E1 JSON samples and fit, GPU copy CSVs, disk benchmark
JSON, capture/write and prefill fits, restore timings, and run metadata.
These 14 data files total about 42 KiB.

The maintained conclusions and tables are in
[Experiments](../experiments.md), [Cost model](../cost-model.md) and
[Decision brief](../decision-brief.md). Generated result tables and per-run
simulation outputs are not maintained as additional documentation copies.

## Archived research data

The original token arrays, request metadata, server logs, generated outputs
and working notes remain available in the immutable research snapshot
`a32fc43bcb9ec66264f166b5162d93a9223557d5`:

- [Browse the original results](https://github.com/gufo-org/gufo/tree/a32fc43bcb9ec66264f166b5162d93a9223557d5/docs/cache-redesign/results).
- [Download the source archive containing those results](https://github.com/gufo-org/gufo/archive/a32fc43bcb9ec66264f166b5162d93a9223557d5.tar.gz).
- [File checksums](archive.sha256) cover all 133 original tracked result files.

The archive includes the whole source snapshot; extract only its
`docs/cache-redesign/results/` subtree into an external replay directory.
Follow the [replay instructions](../scripts/README.md#replaying-the-analysis-from-archived-data)
to download, verify and analyse it without a GPU or models. Keep recovered
data and generated outputs outside the checkout.

Removing these files from the branch's current tree does not rewrite the
published Git history. The archive revision records evidence for the original
experiments; it is not a matching performance baseline for later code changes.
