"""Independent AR comparisons used by check.py; all model jobs run sequentially."""
from __future__ import annotations

from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/models/deepseek_v4_flash"
REVISION = "6289c516273979173abbc062209a81dd3706b804"
DEPTHS = (0, 4096, 8192, 12288, 16384)
PREFILL_STEPS = (2048, 4096)
# Upstream prints each case's summed NLL with nine decimal places. Every case
# has at least one token, so this also bounds the mean's rounding uncertainty.
UPSTREAM_NLL_ROUNDING = 0.5e-9


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def tsv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def production_sha256() -> str:
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "src"], cwd=ROOT).decode().split("\0")
    digest = hashlib.sha256()
    for name in sorted(filter(None, paths)):
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        digest.update(name.encode() + b"\0" + str(len(data)).encode() + b"\0")
        digest.update(data + b"\0")
    return digest.hexdigest()


def compare_official(directory: Path) -> dict:
    candidate = load(directory / "gufo-1.json")
    repeat = load(directory / "gufo-2.json")
    require(candidate == repeat, "official AR scoring is not exactly reproducible")
    report = {"exact_repeat": True, "groups": {}}
    for group, count, tokens in (("continuation-100", 100, 2313), ("smoke-5", 5, 14)):
        upstream = tsv(directory / f"upstream-{group}.tsv")
        ours = [case for case in candidate["cases"] if case["group"] == group]
        require(len(upstream) == len(ours) == count, f"incomplete {group}")
        pairs = []
        for before, after in zip(upstream, ours):
            require(before["id"] == after["id"], "different official case order")
            for field in ("prompt_tokens", "target_tokens"):
                require(int(before[field]) == after[field],
                        f'{after["id"]}: different {field}')
            old_nll, new_nll = float(before["nll"]), after["nll"]
            require(math.isfinite(old_nll) and math.isfinite(new_nll),
                    "non-finite official likelihood")
            pairs.append({
                "id": after["id"], "tokens": after["target_tokens"],
                "upstream_nll": old_nll, "gufo_nll": new_nll,
                "delta": new_nll - old_nll,
                "upstream_first": int(before["first_match"]),
                "gufo_first": after["first_match"],
                "upstream_prefix": int(before["greedy_lcp"]),
                "gufo_prefix": after["greedy_lcp"],
            })
        require(sum(pair["tokens"] for pair in pairs) == tokens,
                "official target token coverage changed")
        old_nll = sum(pair["upstream_nll"] for pair in pairs) / tokens
        new_nll = sum(pair["gufo_nll"] for pair in pairs) / tokens
        rng = random.Random(731)
        deltas = []
        for _ in range(10000):
            sample = rng.choices(pairs, k=count)
            deltas.append(sum(pair["delta"] for pair in sample) /
                          sum(pair["tokens"] for pair in sample))
        deltas.sort()
        summary = {
            "cases": count, "tokens": tokens,
            "upstream_nll": old_nll, "gufo_nll": new_nll,
            "relative_nll_change_percent": 100 * (new_nll / old_nll - 1),
            "upstream_first_matches": sum(pair["upstream_first"] for pair in pairs),
            "gufo_first_matches": sum(pair["gufo_first"] for pair in pairs),
            "upstream_mean_prefix": statistics.mean(pair["upstream_prefix"] for pair in pairs),
            "gufo_mean_prefix": statistics.mean(pair["gufo_prefix"] for pair in pairs),
            "nll_case_wins_gufo": sum(pair["delta"] < -1e-8 for pair in pairs),
            "nll_case_wins_upstream": sum(pair["delta"] > 1e-8 for pair in pairs),
            "paired_case_bootstrap_delta_nll_95pct": [deltas[250], deltas[9749]],
        }
        report["groups"][group] = {"summary": summary, "cases": pairs}
    return report


def compare_logits(ours: bytes, old: bytes, vocabulary: int) -> dict:
    require(len(ours) == len(old) == vocabulary * 4,
            "different vocabulary size")
    vectors = []
    for data in (ours, old):
        vector = array("f")
        vector.frombytes(data)
        if sys.byteorder != "little":
            vector.byteswap()
        require(all(map(math.isfinite, vector)), "non-finite full logits")
        vectors.append(vector)
    a, b = vectors
    rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    require(norm > 0, "zero-norm full logits")
    cosine = sum(x * y for x, y in zip(a, b)) / norm
    maximum = max(abs(x - y) for x, y in zip(a, b))
    shift = statistics.mean(x - y for x, y in zip(a, b))
    centered_rmse = math.sqrt(
        sum((x - y - shift) ** 2 for x, y in zip(a, b)) / len(a))
    probabilities = []
    for vector in vectors:
        maximum_logit = max(vector)
        weights = [math.exp(value - maximum_logit) for value in vector]
        total = sum(weights)
        probabilities.append([value / total for value in weights])
    pa, pb = probabilities
    js = 0.0
    for x, y in zip(pa, pb):
        total = x + y
        if x:
            js += x * math.log(2 * x / total) / 2
        if y:
            js += y * math.log(2 * y / total) / 2
    return {
        "rmse": rmse, "cosine": cosine, "max_absolute_error": maximum,
        # Softmax is invariant to a common logit shift. Keep the old gates,
        # but also report differences in the actual probability distribution.
        "mean_logit_shift": shift, "centered_rmse": centered_rmse,
        "jensen_shannon_nats": js,
        "max_probability_difference": max(abs(x - y) for x, y in zip(pa, pb)),
        "same_top1": a.index(max(a)) == b.index(max(b)),
        "within_existing_vector_bounds":
            rmse <= 1.12 and cosine >= 0.979 and maximum <= 5,
    }


def compare_frontiers(directory: Path) -> dict:
    base, repeat, upstream = (directory / name for name in
                              ("frontiers-gufo-1", "frontiers-gufo-2", "frontiers-upstream"))
    report = load(base / "summary.json")
    require(report == load(repeat / "summary.json"),
            "AR frontier scoring is not exactly reproducible")
    require([(point["prefill_step"], point["depth"]) for point in report["points"]]
            == [(step, depth) for depth in DEPTHS for step in PREFILL_STEPS],
            "incomplete frontier prefill/depth matrix")
    decode_tokens = report["decode_tokens"]
    require(decode_tokens in (0, 128), "unsupported frontier decode length")
    phases = ("before", "after") if decode_tokens else ("before",)
    rows = tsv(upstream / "steps.tsv")
    require(len(rows) == 10 * decode_tokens, "incomplete independent decode matrix")
    targets = [int(value) for value in (base / "targets.txt").read_text().split()]
    require(len(targets) == 128, "incomplete forced continuation")
    result = {"exact_repeat": True, "vectors": [], "decode": [],
              "cross_prefill": []}
    for point in report["points"]:
        depth = point["depth"]
        prefill_step = point["prefill_step"]
        selected = [row for row in rows if int(row["depth"]) == depth and
                    int(row["prefill_step"]) == prefill_step]
        require(len(selected) == decode_tokens, f"incomplete decode at {prefill_step}/{depth}")
        errors, matches = [], 0
        for index, row in enumerate(selected):
            require(int(row["step"]) == index and int(row["target"]) == targets[index],
                    "different forced token sequence")
            require(int(row["prompt_tokens"]) == point["prompt_tokens"],
                    "different frontier prompt length")
            require(int(row["prefill_capacity"]) == point["prefill_capacity"],
                    "different allocated prefill capacity")
            before, after = float(row["logprob"]), point["target_logprobs"][index]
            require(math.isfinite(before) and math.isfinite(after),
                    "non-finite frontier likelihood")
            errors.append(after - before)
            matches += int(row["greedy"]) == point["greedy"][index]
        if decode_tokens:
            result["decode"].append({
                "depth": depth, "prefill_step": prefill_step,
                "prefill_capacity": point["prefill_capacity"],
                "greedy_matches": matches, "tokens": 128,
                "mean_signed_target_logprob_difference": statistics.mean(errors),
                "mean_absolute_target_logprob_difference": statistics.mean(map(abs, errors)),
                "max_absolute_target_logprob_difference": max(map(abs, errors)),
            })
        for phase in phases:
            name = f"{prefill_step}-{depth}-{phase}.f32"
            ours, old = (base / name).read_bytes(), (upstream / name).read_bytes()
            require(ours == (repeat / name).read_bytes(),
                    f"AR logits are not exactly reproducible: {name}")
            result["vectors"].append({
                "depth": depth, "phase": phase, "prefill_step": prefill_step,
                **compare_logits(ours, old, report["vocab"]),
            })
    # At depth zero both budgets make the identical single 16-token call.
    first, second = report["points"][:2]
    for field in ("greedy", "target_logprobs", *(phase + "_sha256" for phase in phases)):
        require(first[field] == second[field],
                "identical short-prefill calls changed output")
    for engine, location in (("gufo", base), ("upstream", upstream)):
        for depth in DEPTHS:
            for phase in phases:
                a = (location / f"2048-{depth}-{phase}.f32").read_bytes()
                b = (location / f"4096-{depth}-{phase}.f32").read_bytes()
                result["cross_prefill"].append({
                    "engine": engine, "depth": depth, "phase": phase,
                    **compare_logits(a, b, report["vocab"]),
                })
    return result


def run(model: Path, output: Path, upstream: Path, environment: dict,
        prefill_only: bool = False) -> None:
    """Score both engines, retaining inputs, full logits, scores and provenance."""
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    require(revision == REVISION and not status,
            f"reference requires a clean upstream checkout at {REVISION}")
    require(not output.exists(), "reference output directory already exists")
    require(not any(character in str(output) for character in "\t\r\n"),
            "manifest directory cannot contain tabs or newlines")
    output.mkdir(parents=True)
    source_hash = production_sha256()

    def execute(command: list, name: str, cwd: Path = ROOT) -> None:
        print(f"DS4 reference: {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(list(map(str, command)), cwd=cwd, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    execute(["nix", "build", "--file", TESTS / "reference",
             "--argstr", "upstream", upstream, "--cores", "4",
             "--out-link", output / "upstream-build"], "upstream-build")
    fixture_path = TESTS / "fixtures/official-0731.json"
    fixture = load(fixture_path)
    manifests = {}
    for group in (() if prefill_only else ("continuation-100", "smoke-5")):
        directory = output / group
        directory.mkdir()
        manifest = directory / "manifest.tsv"
        with manifest.open("w") as stream:
            for case in fixture["cases"]:
                if case["group"] != group:
                    continue
                # Fixture IDs are audited by ds4.dataset.
                prompt = directory / f'{case["id"]}.prompt.txt'
                continuation = directory / f'{case["id"]}.continuation.txt'
                prompt.write_text(case["prompt"])
                continuation.write_text(case["continuation"])
                stream.write(f'{case["id"]}\t{prompt}\t{continuation}\n')
        manifests[group] = manifest

    native = ROOT / "build/gpu-test/tests/models/deepseek_v4_flash/ds4_quality_test"
    native_hash = sha256(native)
    control = output / "upstream-build/bin"
    if not prefill_only:
        for repeat in (1, 2):
            execute([native, "--official", output / f"gufo-{repeat}.json"],
                    f"gufo-{repeat}")
        for group, context in (("continuation-100", 4096), ("smoke-5", 16384)):
            execute([control / "score_official", model, manifests[group],
                     output / f"upstream-{group}.tsv", context], f"upstream-{group}")
    frontier_options = ["--prefill-only"] if prefill_only else []
    for repeat in (1, 2):
        execute([native, "--reference-frontiers", output / f"frontiers-gufo-{repeat}", *frontier_options],
                f"frontiers-gufo-{repeat}")
    (output / "frontiers-upstream").mkdir()
    execute([control / "frontier_reference", model, output / "frontiers-gufo-1",
             output / "frontiers-upstream", *frontier_options], "frontiers-upstream")
    require(source_hash == production_sha256(),
            "production sources changed during the reference comparison")
    require(native_hash == sha256(native),
            "native test binary changed during the reference comparison")
    report = {
        "schema": "gufo.ds4-independent-ar-comparison.v1",
        "upstream_revision": revision,
        "upstream_build": str((output / "upstream-build").resolve()),
        "model_file": model.name, "model_sha256": sha256(model),
        "fixture_sha256": sha256(fixture_path),
        "gufo_production_sha256": source_hash,
        "gufo_native_test_sha256": native_hash,
        "gufo_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "flake_lock_sha256": sha256(ROOT / "flake.lock"),
        "gufo_build_preset": "gpu-test",
        "scope": "prefill-only" if prefill_only else "full",
        "frontiers": compare_frontiers(output),
    }
    report["gates"] = {
        "exact_gufo_repeat": True,
        "full_logits_within_existing_bounds": all(
            vector["within_existing_vector_bounds"]
            for vector in report["frontiers"]["vectors"]),
    }
    if not prefill_only:
        report["official"] = compare_official(output)
        continuation = report["official"]["groups"]["continuation-100"]["summary"]
        report["gates"]["no_detected_official_nll_regression"] = (
            continuation["paired_case_bootstrap_delta_nll_95pct"][0]
            <= UPSTREAM_NLL_ROUNDING)
    report["upstream_nll_rounding_bound"] = UPSTREAM_NLL_ROUNDING
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Comparison recorded in {output / 'comparison.json'}.", flush=True)
    require(all(report["gates"].values()),
            f'independent AR comparison flagged differences: {report["gates"]}; '
            'investigate both implementations against official formulas and task scores')
