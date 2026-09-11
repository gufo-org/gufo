# Hyperloom Development Integration

Status: experimental, development-only

Hyperloom is available in the Gufo development shell as an agentic profiling,
benchmarking, and optimization tool. It is not linked into `gufo`, installed by
the production package, or required at runtime.

The Nix derivation pins:

```text
repository: AMD-AGI/Hyperloom
revision:   c92784cbf1c62652a75c751063c52ffecce9a909
version:    1.0.0b2-unstable-2026-08-19
```

The package provides the Hyperloom Python libraries and these entry points:

```text
hyperloom
inference_optimizer
framework-agent
fa
robustness-agent
quantization-agent
```

## Enter the Development Shell

From the Gufo repository:

```bash
nix develop
```

Verify the installation without starting an optimization run:

```bash
python -c 'import hyperloom; print("Hyperloom import successful")'

for command in \
  hyperloom \
  inference_optimizer \
  framework-agent \
  fa \
  robustness-agent \
  quantization-agent
do
  printf '%-24s %s\n' "$command" "$(command -v "$command")"
  "$command" --help >/dev/null
done
```

The package can also be built independently of the full development shell:

```bash
nix build --impure --no-link --expr '
let
  flake = builtins.getFlake (toString ./.);
  pkgs = import flake.inputs.nixpkgs {
    system = "x86_64-linux";
  };
  scope = pkgs.callPackage ./.devops/nix/scope.nix {
    version = "validation";
  };
in
  scope.hyperloom
'
```

## Development-Only Boundary

Hyperloom is included through the development Python environment in
`flake.nix`. The production package remains `gufo` with ROCm/HIP and XRT
inputs only.

After building the production package, confirm that Hyperloom is not in its
runtime closure:

```bash
nix path-info --recursive .#default | grep -i hyperloom
```

The command should produce no output.

## Current gfx1151 Limitation

The pinned Hyperloom revision does not currently define a Strix Halo board or a
`gfx1151` runner. Its GPU identity table accepts MI300X, MI308X, MI325X, and
MI355X, mapping them to `gfx942` or `gfx950`.

Consequences:

- package build and CLI inspection work;
- framework, robustness, and operator utilities can be evaluated;
- a normal `inference_optimizer optimize` run cannot honestly identify the
  target as Strix Halo;
- passing an MI-series `--gpu-type` for a `gfx1151` machine would corrupt
  hardware provenance and may select invalid benchmark or kernel policies.

Do not work around this by pretending the machine is MI300X or MI355X. Before
using the complete optimization loop, add upstream or locally maintained
support for:

1. a Strix Halo board identity;
2. the `gfx1151` dispatch architecture;
3. the 40-CU device description;
4. a matching custom benchmark runner label;
5. ROCm and profiler capability checks appropriate for the APU;
6. tests proving that GPU probing and user-supplied identity agree.

Keep any compatibility patch separate from the Nix packaging change so it can
be reviewed and updated independently.

## Safe Workspace Layout

Hyperloom creates sessions, logs, runtime files, dependency checkouts, and
candidate source modifications. Do not point it at the primary checkout.

Create a dedicated worktree and workspace:

```bash
repo_root="$(git rev-parse --show-toplevel)"
worktree=/tmp/gufo-hyperloom-worktree
workspace=/tmp/gufo-hyperloom-workspace

rm -rf "$worktree" "$workspace"
git worktree add --detach "$worktree" HEAD
mkdir -p "$workspace"

export USER_DATA_PATH="$workspace"
```

Run Hyperloom from a shell created by the main repository flake:

```bash
cd "$worktree"
nix develop "$repo_root"
```

The dedicated worktree protects the primary branch from experimental patches
and allows candidate changes to be inspected or discarded independently.

## Credentials

Hyperloom needs an LLM provider for agentic optimization. Export credentials in
the launching shell; do not commit them to `.env`, Nix files, benchmark scripts,
or session artifacts.

Anthropic-compatible example:

```bash
export ANTHROPIC_API_KEY='...'
export ANTHROPIC_BASE_URL='https://api.anthropic.com'
```

OpenAI-compatible example:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api.openai.com/v1'
```

The Nix package includes the Claude agent SDK, OpenAI client, HTTP client,
PyYAML, and web-processing dependencies available in the pinned nixpkgs.
The optional `openai-codex` Python distribution is not currently packaged in
that nixpkgs revision. Codex-session-specific paths require a separately
validated package; ordinary OpenAI-compatible HTTP operation does not.

Use `--help` and deterministic local tools without credentials. Do not launch a
long optimization job until authentication preflight succeeds.

## Avoid Uncontrolled Setup

The `hyperloom` entry point runs Hyperloom setup. Upstream bare-metal setup may
inspect the ROCm installation and install or clone serving frameworks, GEAK,
TraceLens, Magpie, AITER, or other components.

That behavior is intentionally not part of the Nix derivation. The derivation
installs Hyperloom itself; it does not make every optional serving or kernel
backend reproducible.

Before running setup:

- use the dedicated worktree and `USER_DATA_PATH`;
- inspect the planned dependency sources and revisions;
- avoid installing into the host Python environment;
- keep Nix-provided ROCm and Python packages authoritative;
- record any external dependency revision used by a benchmark;
- do not allow setup to modify the production package definition silently.

## Intended Gufo Integration

Gufo is not a shipped Hyperloom framework, so the appropriate integration is
Hyperloom's custom, server-less workload path.

The custom path requires:

| Input | Hyperloom flag | Gufo value |
| --- | --- | --- |
| Candidate checkout | `--framework-path` | Dedicated Gufo worktree |
| Benchmark adapter directory | `--benchmark-scripts-dir` | Gufo-specific scripts |
| Model weights | `--model` | Pinned GGUF/model fixture |
| Hardware identity | `--gpu-type` | Future validated Strix Halo identity |
| Search budget | `--max-hours` | Start with a short bounded run |

The custom benchmark backend is selected with:

```bash
export HYPERLOOM_BENCHMARK_BACKEND=bypass
```

Do not start the complete command until the pinned Hyperloom source recognizes
the Strix Halo identity. Once gfx1151 support exists, the command shape is:

```bash
python -m hyperloom.inference_optimizer.cli -v optimize \
  --framework custom \
  --framework-path "$worktree" \
  --benchmark-scripts-dir /path/to/gufo-hyperloom-adapter \
  --model /path/to/pinned/model \
  --gpu-type <validated-gufo-identity> \
  --tp 1 \
  --max-hours 2
```

Start with a two-hour or shorter experiment. Increase the budget only after the
baseline, correctness gate, and candidate patch lifecycle are trustworthy.

## Benchmark Adapter Contract

Hyperloom searches `--benchmark-scripts-dir` for:

1. `custom_<runner-type>.sh`; or
2. the only `.sh` file in the directory.

It invokes the script with:

```text
RESULT_DIR=<session benchmark directory>
RESULT_FILENAME=inferencex_result
```

The script must write:

```text
$RESULT_DIR/inferencex_result.json
```

Minimal result shape:

```json
{
  "framework": "custom",
  "workload_kind": "scriptable",
  "throughput_unit": "tokens/s",
  "output_throughput": 0.0,
  "quality_gate": {
    "passed": false,
    "reason": "replace with measured Gufo correctness evidence"
  }
}
```

`output_throughput` is maximized. The `quality_gate` is mandatory and
fail-closed: a missing or invalid gate rejects every candidate.

A Gufo adapter should:

1. build through Nix only;
2. add new candidate files to the disposable worktree index before a Nix build,
   because Nix does not see untracked files;
3. run a fixed warmup;
4. measure separate prefill and decode cases;
5. compare logits or exact tokens against a pinned baseline;
6. reject non-finite values and correctness regressions;
7. write one atomic JSON result;
8. retain logs outside the source checkout.

Do not report a combined score that hides a prefill regression behind a decode
improvement. If Hyperloom requires one objective, choose the primary case and
include the other cases in the quality gate or attached evidence.

## Suggested First Adapter

The first adapter should optimize one narrow path rather than the complete
server:

```text
model:          one pinned supported Qwen fixture
prompt suite:   fixed tokenized inputs
prefill cases:  128, 512, 1024 tokens
decode cases:   128, 1024, 4096 context
iterations:     fixed warmup plus repeated timed samples
objective:      median tokens/s for one selected case
quality:        finite logits plus configured logit/token agreement
```

Candidate validation should run the repository's focused tests first, followed
by the selected benchmark. The canonical release gate remains:

```bash
nix build .#checks.x86_64-linux.pr
```

Run gfx1151 kernel tests and real performance measurements on the dedicated
Strix Halo host.

## Monitoring a Run

Hyperloom writes artifacts below `USER_DATA_PATH`. Relevant files include:

```text
<session>/runs/baseline/<id>/benchmark_custom_<stamp>/scriptable_stdout.log
<session>/runs/baseline/<id>/benchmark_custom_<stamp>/inferencex_result.json
<session>/reports/optimization_journal.json
```

Check the baseline `output_throughput` and `quality_gate` before allowing the
search to continue. A fast but incorrect baseline invalidates every later
comparison.

For each accepted candidate, retain:

- source diff;
- build command and exit status;
- test output;
- benchmark samples, not only the aggregate;
- selected backend and dispatch policy;
- ROCm, kernel, firmware, and hardware fingerprint;
- residual risks.

## Useful Non-Optimization Commands

Inspect the available interfaces:

```bash
inference_optimizer --help
inference_optimizer optimize --help
framework-agent --help
robustness-agent --help
quantization-agent --help
```

Verify the host and model preconditions:

```bash
python -m hyperloom.inference_optimizer.tools.preflight_optimizer \
  /path/to/model
```

The preflight may report gfx1151 as unsupported by the pinned Hyperloom board
registry. Treat that as expected until the compatibility work described above
lands; do not suppress it with a false MI-series identity.

## Updating Hyperloom

When changing the pin in `.devops/nix/hyperloom.nix`:

1. inspect the upstream changelog and packaging metadata;
2. update the revision, date, version, and fixed-output hash;
3. rebuild the package on x86-64 Linux;
4. run import and CLI help smoke tests;
5. review new runtime dependencies and setup behavior;
6. rerun gfx1151 compatibility checks;
7. keep the production Gufo closure free of Hyperloom.

The package is intentionally exposed from `.devops/nix/scope.nix` and consumed
only by the development Python environment in `flake.nix`.
