# Experiment scripts

Every number in [experiments.md](../experiments.md) and
[cost-model.md](../cost-model.md) comes from these scripts. Run metadata
(binary hash, source revision, models, tool versions) is in
[results/metadata.json](../results/metadata.json).

## Replaying the analysis from committed data

`results/e2/<model>/<workload>/` keeps, for every run, the token arrays
(`requests.npz`), per-request metadata (`requests.json`) and the server log
(`server.log`). The simulations and analyses run from those alone; no GPU,
model or server is needed:

```sh
cd docs/cache-redesign
export CACHE_EXP_DIR=$PWD           # read and write results/ here
python3 scripts/analyze_e3.py fn w2                # missed reuse
python3 scripts/simulate_e8.py fn w2               # today / Phase 0 / hybrid
python3 scripts/simulate_e8.py q27 w1 --restart-at 20
python3 scripts/fit_prefill.py && python3 scripts/cost_model.py
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
| E5, E7, E8 | `simulate_e5.py`, `simulate_e7.py` (`run_e7.sh`), `simulate_e8.py` (`run_e8.sh`) | E8 follows production admission and asynchronous disk |
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

The token arrays are research data. If these documents are merged into
`main`, leave `results/e2` out.
