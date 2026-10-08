# Experiment scripts

Every number in [experiments.md](../experiments.md) and
[cost-model.md](../cost-model.md) comes from these scripts. Run metadata
(binary hash, source revision, models, tool versions) is in
[results/metadata.json](../results/metadata.json).

## Replaying the analysis from archived data

The original research snapshot keeps, for every run, the token arrays
(`requests.npz`), per-request metadata (`requests.json`) and server log
(`server.log`). They are archived rather than tracked in the current tree;
see [results/README.md](../results/README.md). The simulations and analyses use
those files and the measured constants; no GPU, model or server is needed.

Download the immutable snapshot and recover its results into a fresh external
directory. The following commands use the authenticated GitHub CLI and GNU tar:

```sh
export GUFO_REPO=$(git rev-parse --show-toplevel)
export CACHE_EXP_DIR=$(mktemp -d)
gh api repos/gufo-org/gufo/tarball/a32fc43bcb9ec66264f166b5162d93a9223557d5 \
  > "$CACHE_EXP_DIR/research.tar.gz"
tar -xzf "$CACHE_EXP_DIR/research.tar.gz" -C "$CACHE_EXP_DIR" \
  --strip-components=3 --wildcards '*/docs/cache-redesign/results/*'
cd "$CACHE_EXP_DIR"
sha256sum --quiet --check "$GUFO_REPO/docs/cache-redesign/results/archive.sha256"
cd "$GUFO_REPO/docs/cache-redesign"
```

The checksum check covers all original result files before regeneration.
Recovered inputs live in `$CACHE_EXP_DIR/results/`; scripts write new outputs
there too. Run the focused analysis, or regenerate the reference summaries:

```sh
python3 scripts/analyze_e3.py fn w2                # missed reuse
python3 scripts/simulate_e8.py fn w2               # today / Phase 0 / hybrid
python3 scripts/simulate_e8.py q27 w1 --restart-at 20 --abrupt
python3 scripts/analyze_e4.py                      # measured totals for summaries
python3 scripts/fit_prefill.py                     # fitted timing constants
PYTHON=python3 scripts/run_e8.sh && python3 scripts/summarize_e8.py
python3 scripts/cost_model.py
```

Python 3.13 with `numpy` is required, plus `jsonschema` for W1. Workload
names: `w1`–`w4`, and `w2-c4` and `w3-c4` for concurrency 4 (pass
`--sessions 4` to the simulators).

## Rerunning the measurements

The measurements need the GPU, the models and a gufo binary. They start their
own servers; stop other model servers first.

| Variable | Meaning | Default |
| --- | --- | --- |
| `CACHE_EXP_DIR` | Work directory for `results/` and `cache/` | This directory |
| `GUFO_BIN` | gufo binary | `$CACHE_EXP_DIR/bin/gufo` |
| `GUFO_REPO` | gufo checkout (corpora, `tests/functional` drivers) | The checkout containing the scripts |
| `LLAMA_TOKENIZE` | llama.cpp `llama-tokenize`, for re-tokenizing traces | Found on `PATH` |
| `FN_MODEL`, `FN_MTP`, `Q27_MODEL`, `Q27_DFLASH` | Model files | The paths used on the test host |
| `PYTHON` | Interpreter for the runners and for W1's driver | `python3` |

| Step | Script | Notes |
| --- | --- | --- |
| E1 sizes | `run_e1.sh` → `fit_e1.py` | Raises disk staging to 6 GiB so deep checkpoints are written |
| E2 traces | `run_e2.sh` (`w1_agent.py`, `workloads.py`, `serverctl.py`) | W1 drives a real `pi` through `tests/functional/agent_long.py`; each run deletes its cache directory |
| E3 missed reuse | `tokenize_traces.py`, then `analyze_e3.py` | Re-tokenizes `trace-*.jsonl` into `requests.npz` |
| E4 overhead | `analyze_e4.py` | Reads every `server.log` |
| E5, E7, E8 | `simulate_e5.py`, `simulate_e7.py` (`run_e7.sh`), `simulate_e8.py` (`run_e8.sh`, `summarize_e8.py`) | E8 revision 2 is the reference: concurrent event replay, production capture sequence and admission, disk entries restorable only after their write, graceful or abrupt restarts, lineage-aware chunks. E5 and E7 are kept for history |
| E6 concurrency 4 | `run_e6.sh` | `--sessions 4`, 131,072 context, four clients |
| Micro-benchmarks | `copybench.hip`, `chunkcopy.hip` (build with `nix develop -c tools/bench/build.sh <file>`), `diskbench.py` | `diskbench.py` writes up to 8.4 GB in `$CACHE_EXP_DIR/cache` |
| Prefill model, cost model | `fit_prefill.py`, `cost_model.py` | |

Free disk space is checked before each GPU run (minimum 30 GB). A `cache`
directory can reach the disk budget (16 GiB) during a run.

## Measured or predicted

- **Measured:** E1, E2, E3, E4, E6, the micro-benchmarks, and the "Measured"
  or "Server" columns of the simulation tables.
- **Simulated:** all Phase 0, hybrid and dense figures (E5, E7, E8).
- **Model:** the [cost model](../cost-model.md) tables, built from measured
  constants.

The token arrays and logs are research data. Keep recovered datasets and
generated per-run results outside the checkout; retain their archive reference
and checksums alongside the small measurement inputs.
