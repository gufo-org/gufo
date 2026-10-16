"""MiniMax H3 quality-oracle artifacts and deterministic comparison metrics.

Large teacher outputs stay outside Git in directories named by the SHA-256 of
their canonical manifest.  The manifest hashes every payload and records the
complete model, oracle, runtime, dtype, prompt, and generation contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import control_tokens, safetensors

SCHEMA = "gufo.minimax-h3-quality-artifact.v1"
CONTRACT_SCHEMA = "gufo.minimax-h3-quality-contract.v1"
CONTRACT_ID = "minimax-h3-fl2va-bf16-quality-v1"
CONTRACT_SHA256 = (
    "c2b7a8c0bf0113aa736ef94acdde4c2a0a1ceb05d6ef4977795c35a9716c204f"
)
MODEL_KIND = "minimax-h3-fl2va-bf16"
MODEL_REPOSITORY = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REFERENCE_REPOSITORY = "antirez/h3.c"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"

SHA256_LENGTH = 64
CASE_CLASSES = {"component", "rapid", "exact", "independent"}
COMPONENT_ROLES = {
    "text_encoder": {"token_ids", "layer50_hidden"},
    "dit_block": {"block_input", "block_output"},
    "adaln": {"modulation"},
    "dit_velocity": {"video_velocity", "audio_velocity"},
    "video_vae": {"video_latent", "frames"},
    "audio_vae": {"audio_latent", "waveform"},
}
END_TO_END_ROLES = {
    "rapid": {
        "video_initial",
        "audio_initial",
        "video_final",
        "audio_final",
        "selected_frames",
        "metrics",
    },
    "exact": {
        "video_final",
        "audio_final",
        "video",
        "audio",
        "metrics",
        "human_review",
    },
    "independent": {
        "video_final",
        "audio_final",
        "video",
        "audio",
        "metrics",
        "human_review",
    },
}


class H3QualityError(Exception):
    """Raised when an H3 quality contract or artifact is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return safetensors.load_json_file(path)
    except safetensors.SafetensorsError as exc:
        raise H3QualityError(str(exc)) from exc


def _require_mapping(value: Any, description: str) -> dict:
    if not isinstance(value, dict):
        raise H3QualityError(f"{description} must be an object")
    return value


def _require_list(value: Any, description: str) -> list:
    if not isinstance(value, list):
        raise H3QualityError(f"{description} must be an array")
    return value


def _require_text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise H3QualityError(f"{description} must be a non-empty string")
    return value


def _require_exact_keys(
    value: dict, required: set[str], optional: set[str], description: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise H3QualityError(
            f"{description} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise H3QualityError(
            f"{description} has unknown fields: {sorted(unknown)}"
        )


def _validate_sha256(value: Any, description: str) -> str:
    text = _require_text(value, description)
    if len(text) != SHA256_LENGTH:
        raise H3QualityError(f"{description} is not a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise H3QualityError(f"{description} is not hexadecimal") from exc
    if text.lower() != text:
        raise H3QualityError(f"{description} must use lowercase hexadecimal")
    return text


def _safe_relative_path(value: Any, description: str) -> Path:
    text = _require_text(value, description)
    path = Path(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or ".." in path.parts
        or "." in path.parts
        or text == "manifest.json"
    ):
        raise H3QualityError(f"{description} is not a safe relative path")
    return path


def _validate_finite_tree(value: Any, description: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise H3QualityError(f"{description} contains a non-finite value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{description}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise H3QualityError(f"{description} has an invalid key")
            _validate_finite_tree(item, f"{description}.{key}")
        return
    raise H3QualityError(
        f"{description} contains unsupported value type {type(value).__name__}"
    )


def _metric_value(metrics: dict, dotted_name: str) -> float:
    value: Any = metrics
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise H3QualityError(f"required metric {dotted_name!r} is absent")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise H3QualityError(f"metric {dotted_name!r} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise H3QualityError(f"metric {dotted_name!r} is non-finite")
    return result


def evaluate_thresholds(metrics: dict, thresholds: list[dict]) -> dict:
    """Evaluate a flat list of dotted metric floors/ceilings."""

    results = []
    passed = True
    for index, threshold in enumerate(thresholds):
        threshold = _require_mapping(
            threshold, f"thresholds[{index}]"
        )
        _require_exact_keys(
            threshold,
            {"metric", "source"},
            {"maximum", "minimum"},
            f"thresholds[{index}]",
        )
        metric = _require_text(
            threshold["metric"], f"thresholds[{index}].metric"
        )
        source = _require_text(
            threshold["source"], f"thresholds[{index}].source"
        )
        has_maximum = "maximum" in threshold
        has_minimum = "minimum" in threshold
        if has_maximum == has_minimum:
            raise H3QualityError(
                f"thresholds[{index}] must define exactly one bound"
            )
        value = _metric_value(metrics, metric)
        if has_maximum:
            bound = float(threshold["maximum"])
            ok = math.isfinite(bound) and value < bound
            direction = "maximum"
        else:
            bound = float(threshold["minimum"])
            ok = math.isfinite(bound) and value > bound
            direction = "minimum"
        passed &= ok
        results.append(
            {
                "metric": metric,
                "value": value,
                "direction": direction,
                "bound": bound,
                "source": source,
                "passed": ok,
            }
        )
    return {"passed": passed, "results": results}


def manifest_artifact_id(manifest: dict) -> str:
    """Return the content address, excluding the self-referential ID field."""

    identity = copy.deepcopy(manifest)
    identity.pop("artifact_id", None)
    return _sha256_bytes(_canonical_bytes(identity))


def _validate_contract_reference(contract: dict) -> None:
    _require_exact_keys(
        contract,
        {"id", "sha256"},
        set(),
        "contract",
    )
    if contract["id"] != CONTRACT_ID:
        raise H3QualityError("artifact references an unsupported contract ID")
    _validate_sha256(contract["sha256"], "contract.sha256")
    if contract["sha256"] != CONTRACT_SHA256:
        raise H3QualityError("artifact references the wrong contract bytes")


def _validate_model(model: dict) -> None:
    _require_exact_keys(
        model,
        {
            "kind",
            "repository",
            "revision",
            "source_manifest_sha256",
        },
        set(),
        "model",
    )
    if model["kind"] != MODEL_KIND:
        raise H3QualityError(f"unsupported model kind {model['kind']!r}")
    if model["repository"] != MODEL_REPOSITORY:
        raise H3QualityError("model repository does not match the pinned source")
    if model["revision"] != MODEL_REVISION:
        raise H3QualityError("model revision does not match the pinned source")
    _validate_sha256(
        model["source_manifest_sha256"], "model.source_manifest_sha256"
    )


def _validate_oracle(oracle: dict) -> None:
    _require_exact_keys(
        oracle,
        {"implementation", "repository", "revision"},
        set(),
        "oracle",
    )
    _require_text(oracle["implementation"], "oracle.implementation")
    if oracle["repository"] != REFERENCE_REPOSITORY:
        raise H3QualityError("oracle repository does not match the pinned source")
    if oracle["revision"] != REFERENCE_REVISION:
        raise H3QualityError("oracle revision does not match the pinned source")


def _validate_runtime(runtime: dict) -> None:
    _require_exact_keys(
        runtime,
        {
            "name",
            "version",
            "platform",
            "hardware",
            "command",
            "storage_dtype",
            "compute_dtype",
            "accumulation_dtype",
        },
        set(),
        "runtime",
    )
    for key in runtime:
        _require_text(runtime[key], f"runtime.{key}")
    command = runtime["command"]
    if "\n" in command or "\r" in command:
        raise H3QualityError("runtime.command must be a single shell-escaped line")


def _validate_case(case: dict) -> tuple[str, str | None]:
    _require_exact_keys(
        case,
        {
            "id",
            "class",
            "prompt",
            "prompt_sha256",
            "seed",
            "width",
            "height",
            "frames",
            "steps",
            "blocks",
            "reuse",
        },
        {"component", "selected_frames"},
        "case",
    )
    case_id = _require_text(case["id"], "case.id")
    case_class = case["class"]
    if case_class not in CASE_CLASSES:
        raise H3QualityError(f"unsupported case class {case_class!r}")
    prompt = _require_text(case["prompt"], "case.prompt")
    expected_prompt_hash = _sha256_bytes(prompt.encode("utf-8"))
    if case["prompt_sha256"] != expected_prompt_hash:
        raise H3QualityError("case.prompt_sha256 does not match case.prompt")
    for key in ("seed", "width", "height", "frames", "steps", "blocks", "reuse"):
        if not isinstance(case[key], int) or isinstance(case[key], bool):
            raise H3QualityError(f"case.{key} must be an integer")
    if case["seed"] < 0 or case["seed"] > (1 << 64) - 1:
        raise H3QualityError("case.seed is outside uint64")

    component = case.get("component")
    if case_class == "component":
        if component not in COMPONENT_ROLES:
            raise H3QualityError(f"unsupported component {component!r}")
    elif component is not None:
        raise H3QualityError("non-component case must not declare component")

    if case_class != "component":
        if (
            case["width"] < 32
            or case["height"] < 32
            or case["width"] % 32
            or case["height"] % 32
        ):
            raise H3QualityError("end-to-end canvas must use positive multiples of 32")
        if case["frames"] < 22:
            raise H3QualityError(
                "end-to-end case requires at least 22 frames for the "
                "released VisualVAE"
            )
        if (case["frames"] - 5) % 17:
            raise H3QualityError("case.frames violates H3 5+17*n alignment")
        if case["steps"] < 2 or not 1 <= case["blocks"] <= 50:
            raise H3QualityError("invalid end-to-end step or block count")
        if case["reuse"] < 1:
            raise H3QualityError("case.reuse must be positive")
    if case_class == "rapid":
        selected = _require_list(
            case.get("selected_frames"), "case.selected_frames"
        )
        if selected != [0, case["frames"] // 2, case["frames"] - 1]:
            raise H3QualityError(
                "rapid case must retain first, middle, and last frames"
            )
    elif "selected_frames" in case:
        raise H3QualityError("selected_frames is only valid for a rapid case")
    return case_id, component


def _validate_files(files: list, root: Path | None) -> set[str]:
    roles: set[str] = set()
    paths: set[str] = set()
    for index, entry in enumerate(files):
        entry = _require_mapping(entry, f"files[{index}]")
        _require_exact_keys(
            entry,
            {"path", "role", "size", "sha256", "dtype", "shape"},
            {"media_type"},
            f"files[{index}]",
        )
        relative = _safe_relative_path(entry["path"], f"files[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in paths:
            raise H3QualityError(f"duplicate artifact path {relative_text!r}")
        paths.add(relative_text)
        roles.add(_require_text(entry["role"], f"files[{index}].role"))
        if (
            not isinstance(entry["size"], int)
            or isinstance(entry["size"], bool)
            or entry["size"] < 0
        ):
            raise H3QualityError(f"files[{index}].size must be non-negative")
        _validate_sha256(entry["sha256"], f"files[{index}].sha256")
        _require_text(entry["dtype"], f"files[{index}].dtype")
        shape = _require_list(entry["shape"], f"files[{index}].shape")
        if not all(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension >= 0
            for dimension in shape
        ):
            raise H3QualityError(f"files[{index}].shape is invalid")
        if "media_type" in entry:
            _require_text(entry["media_type"], f"files[{index}].media_type")

        if root is not None:
            path = root / relative
            if path.is_symlink():
                raise H3QualityError(f"artifact payload is a symlink: {relative}")
            if not path.is_file():
                raise H3QualityError(f"artifact payload is missing: {relative}")
            size = path.stat().st_size
            if size != entry["size"]:
                raise H3QualityError(
                    f"artifact payload size differs for {relative}: "
                    f"expected {entry['size']}, got {size}"
                )
            digest = _sha256_file(path)
            if digest != entry["sha256"]:
                raise H3QualityError(
                    f"artifact payload hash differs for {relative}"
                )

    if root is not None:
        actual_paths = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if path.is_symlink():
                raise H3QualityError(f"artifact contains a symlink: {relative}")
            if path.is_file() and relative.as_posix() != "manifest.json":
                actual_paths.add(relative.as_posix())
        unreferenced = actual_paths - paths
        missing = paths - actual_paths
        if unreferenced:
            raise H3QualityError(
                f"artifact has unreferenced payloads: {sorted(unreferenced)}"
            )
        if missing:
            raise H3QualityError(
                f"artifact manifest references absent payloads: {sorted(missing)}"
            )
    return roles


def validate_manifest(
    manifest: dict,
    root: Path | None = None,
    *,
    require_content_address: bool = False,
) -> dict:
    """Validate one quality manifest and, when supplied, all payload bytes."""

    manifest = _require_mapping(manifest, "manifest")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "artifact_id",
            "contract",
            "model",
            "oracle",
            "runtime",
            "case",
            "files",
            "metrics",
            "thresholds",
            "determinism",
        },
        {"notes"},
        "manifest",
    )
    if manifest["schema"] != SCHEMA:
        raise H3QualityError(f"unsupported quality schema {manifest['schema']!r}")
    artifact_id = _validate_sha256(manifest["artifact_id"], "artifact_id")
    expected_id = manifest_artifact_id(manifest)
    if artifact_id != expected_id:
        raise H3QualityError("artifact_id does not match the canonical manifest")

    _validate_contract_reference(
        _require_mapping(manifest["contract"], "contract")
    )
    _validate_model(_require_mapping(manifest["model"], "model"))
    _validate_oracle(_require_mapping(manifest["oracle"], "oracle"))
    _validate_runtime(_require_mapping(manifest["runtime"], "runtime"))
    _case_id, component = _validate_case(
        _require_mapping(manifest["case"], "case")
    )
    root_path = Path(root).resolve() if root is not None else None
    roles = _validate_files(
        _require_list(manifest["files"], "files"), root_path
    )

    case_class = manifest["case"]["class"]
    required_roles = (
        COMPONENT_ROLES[component]
        if case_class == "component"
        else END_TO_END_ROLES[case_class]
    )
    missing_roles = required_roles - roles
    if missing_roles:
        raise H3QualityError(
            f"{case_class} artifact is missing roles: {sorted(missing_roles)}"
        )

    metrics = _require_mapping(manifest["metrics"], "metrics")
    _validate_finite_tree(metrics, "metrics")
    thresholds = _require_list(manifest["thresholds"], "thresholds")
    threshold_result = evaluate_thresholds(metrics, thresholds)
    if not threshold_result["passed"]:
        failures = [
            result["metric"]
            for result in threshold_result["results"]
            if not result["passed"]
        ]
        raise H3QualityError(
            f"quality thresholds failed: {sorted(failures)}"
        )

    determinism = _require_mapping(manifest["determinism"], "determinism")
    _require_exact_keys(
        determinism,
        {"capture_count", "byte_identical"},
        set(),
        "determinism",
    )
    capture_count = determinism["capture_count"]
    byte_identical = determinism["byte_identical"]
    if (
        not isinstance(capture_count, int)
        or isinstance(capture_count, bool)
        or capture_count < 1
        or not isinstance(byte_identical, bool)
        or (capture_count == 1 and byte_identical)
        or (capture_count >= 2 and not byte_identical)
    ):
        raise H3QualityError(
            "artifact determinism must declare one non-repeated capture or "
            "at least two byte-identical captures"
        )

    if "notes" in manifest:
        _require_text(manifest["notes"], "notes")
    if root_path is not None and require_content_address:
        if root_path.name != artifact_id:
            raise H3QualityError(
                "artifact directory name does not match artifact_id"
            )
    return {
        "artifact_id": artifact_id,
        "case_id": manifest["case"]["id"],
        "thresholds_passed": threshold_result["passed"],
        "threshold_results": threshold_result["results"],
    }


def build_manifest(staging: Path, template: dict) -> dict:
    """Hash the template's declared payloads and produce a sealed manifest."""

    staging = Path(staging).resolve()
    if not staging.is_dir():
        raise H3QualityError(f"staging directory does not exist: {staging}")
    manifest = copy.deepcopy(_require_mapping(template, "template"))
    if "artifact_id" in manifest:
        raise H3QualityError("manifest template must not set artifact_id")
    files = _require_list(manifest.get("files"), "files")
    for index, entry in enumerate(files):
        entry = _require_mapping(entry, f"files[{index}]")
        relative = _safe_relative_path(entry.get("path"), f"files[{index}].path")
        path = staging / relative
        if path.is_symlink() or not path.is_file():
            raise H3QualityError(f"staged payload is missing: {relative}")
        entry["size"] = path.stat().st_size
        entry["sha256"] = _sha256_file(path)
    manifest["artifact_id"] = manifest_artifact_id(manifest)
    validate_manifest(manifest, staging)
    return manifest


def seal_artifact(staging: Path, template_path: Path, store: Path) -> Path:
    """Validate, hash, and atomically move a staging directory into a store."""

    staging = Path(staging).resolve()
    store = Path(store).resolve()
    template = _load_json(Path(template_path))
    manifest = build_manifest(staging, template)
    destination = store / manifest["artifact_id"]
    if destination.exists():
        raise H3QualityError(f"artifact already exists: {destination}")
    store.mkdir(parents=True, exist_ok=True)
    (staging / "manifest.json").write_bytes(_canonical_bytes(manifest))
    os.replace(staging, destination)
    verify_artifact(destination)
    return destination


def verify_artifact(root: Path) -> dict:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise H3QualityError(f"quality artifact lacks manifest.json: {root}")
    manifest = _load_json(manifest_path)
    return validate_manifest(
        manifest, root, require_content_address=True
    )


def validate_contract(contract: dict) -> dict:
    """Validate the committed H3 oracle plan without reading model weights."""

    contract = _require_mapping(contract, "quality contract")
    _require_exact_keys(
        contract,
        {
            "schema",
            "contract_id",
            "model",
            "reference",
            "source_manifest_sha256",
            "oracle_hierarchy",
            "tokenizer",
            "retained_cases",
            "thresholds",
            "metrics",
        },
        set(),
        "quality contract",
    )
    if contract["schema"] != CONTRACT_SCHEMA:
        raise H3QualityError(
            f"unsupported contract schema {contract['schema']!r}"
        )
    _require_text(contract["contract_id"], "contract_id")
    model = _require_mapping(contract["model"], "model")
    if model != {
        "kind": MODEL_KIND,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
    }:
        raise H3QualityError("quality contract model is not the pinned H3 source")
    reference = _require_mapping(contract["reference"], "reference")
    if reference != {
        "repository": REFERENCE_REPOSITORY,
        "revision": REFERENCE_REVISION,
    }:
        raise H3QualityError("quality contract reference is not pinned h3.c")
    _validate_sha256(
        contract["source_manifest_sha256"], "source_manifest_sha256"
    )

    hierarchy = _require_list(
        contract["oracle_hierarchy"], "oracle_hierarchy"
    )
    if hierarchy != [
        "analytic-host",
        "pinned-component-teacher",
        "pinned-bf16-end-to-end",
    ]:
        raise H3QualityError("quality contract oracle hierarchy changed")

    tokenizer = _require_mapping(contract["tokenizer"], "tokenizer")
    _require_exact_keys(
        tokenizer,
        {"prompts", "special_tokens"},
        set(),
        "tokenizer",
    )
    prompts = _require_list(tokenizer.get("prompts"), "tokenizer.prompts")
    if len(prompts) < 2:
        raise H3QualityError("quality contract needs two tokenizer prompts")
    for index, prompt_case in enumerate(prompts):
        prompt_case = _require_mapping(
            prompt_case, f"tokenizer.prompts[{index}]"
        )
        prompt = _require_text(
            prompt_case.get("prompt"), f"tokenizer.prompts[{index}].prompt"
        )
        ids = _require_list(
            prompt_case.get("ids"), f"tokenizer.prompts[{index}].ids"
        )
        if not ids or not all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token_id in ids
        ):
            raise H3QualityError(
                f"tokenizer.prompts[{index}].ids is invalid"
            )
        expected_hash = _sha256_bytes(prompt.encode("utf-8"))
        if prompt_case.get("prompt_sha256") != expected_hash:
            raise H3QualityError(
                f"tokenizer.prompts[{index}] prompt hash differs"
            )
    special_tokens = _require_mapping(
        tokenizer["special_tokens"], "tokenizer.special_tokens"
    )
    expected_special_tokens = {
        control_tokens.kEndOfText: 151643,
        control_tokens.kImStart: 151644,
        control_tokens.kImEnd: 151645,
        control_tokens.kVisionStart: 151652,
        control_tokens.kVisionEnd: 151653,
        control_tokens.kImagePad: 151655,
        control_tokens.kVideoPad: 151656,
        control_tokens.kLyricsStart: 151672,
    }
    if special_tokens != expected_special_tokens:
        raise H3QualityError("quality contract special-token IDs changed")

    cases = _require_list(contract["retained_cases"], "retained_cases")
    classes = set()
    case_ids = set()
    for index, case in enumerate(cases):
        case = _require_mapping(case, f"retained_cases[{index}]")
        case_id, _component = _validate_case(case)
        if case_id in case_ids:
            raise H3QualityError(f"duplicate retained case {case_id!r}")
        case_ids.add(case_id)
        classes.add(case["class"])
    if not {"rapid", "exact", "independent"}.issubset(classes):
        raise H3QualityError(
            "quality contract must retain rapid, exact, and independent cases"
        )

    thresholds = _require_mapping(contract["thresholds"], "thresholds")
    required_thresholds = {
        "dit_block",
        "prompt",
        "semantic_latent",
        "video_vae",
        "audio_vae",
    }
    if set(thresholds) != required_thresholds:
        raise H3QualityError("quality contract threshold set changed")
    _validate_finite_tree(thresholds, "thresholds")

    metrics = _require_list(contract["metrics"], "metrics")
    required_metrics = {
        "relative_max",
        "relative_l2",
        "nonfinite_count",
        "psnr",
        "ssim",
        "lpips",
        "temporal_delta_relative_l2",
        "waveform_max_abs",
        "spectrogram_relative_l2",
        "channel_integrity",
        "duration",
        "av_sync",
        "human_review",
    }
    if set(metrics) != required_metrics:
        raise H3QualityError("quality contract metric set changed")
    return {
        "contract_id": contract["contract_id"],
        "case_count": len(cases),
        "prompt_count": len(prompts),
    }


def load_contract(path: Path) -> dict:
    contract = _load_json(Path(path))
    validate_contract(contract)
    return contract


def numeric_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Return shape-aware max/L2 error and non-finite counts."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise H3QualityError(
            f"shape mismatch: reference {reference.shape}, "
            f"candidate {candidate.shape}"
        )
    reference_f64 = reference.astype(np.float64, copy=False)
    candidate_f64 = candidate.astype(np.float64, copy=False)
    reference_nonfinite = int((~np.isfinite(reference_f64)).sum())
    candidate_nonfinite = int((~np.isfinite(candidate_f64)).sum())
    if reference_nonfinite or candidate_nonfinite:
        return {
            "valid": False,
            "elements": int(reference.size),
            "nonfinite_reference": reference_nonfinite,
            "nonfinite_candidate": candidate_nonfinite,
        }
    delta = candidate_f64 - reference_f64
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    reference_max = (
        float(np.max(np.abs(reference_f64))) if reference_f64.size else 0.0
    )
    square_error = float(np.sum(delta * delta))
    square_reference = float(np.sum(reference_f64 * reference_f64))
    return {
        "valid": True,
        "elements": int(reference.size),
        "nonfinite_reference": 0,
        "nonfinite_candidate": 0,
        "max_abs": max_abs,
        "relative_max": max_abs / max(reference_max, 1e-12),
        "relative_l2": math.sqrt(
            square_error / max(square_reference, 1e-24)
        ),
    }


def psnr(
    reference: np.ndarray, candidate: np.ndarray, *, data_range: float
) -> float | None:
    """Return PSNR in dB, or None for byte/numerically identical arrays."""

    if not math.isfinite(data_range) or data_range <= 0:
        raise H3QualityError("PSNR data_range must be finite and positive")
    metrics = numeric_metrics(reference, candidate)
    if not metrics["valid"]:
        raise H3QualityError("PSNR inputs contain non-finite values")
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    mse = float(np.mean(difference * difference)) if difference.size else 0.0
    if mse == 0.0:
        return None
    return 10.0 * math.log10((data_range * data_range) / mse)


def global_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    data_range: float,
) -> float:
    """Deterministic global SSIM diagnostic.

    Release reports may additionally record a pinned windowed-SSIM runtime.
    This dependency-free form is used for corruption and CI fixture tests.
    """

    if not math.isfinite(data_range) or data_range <= 0:
        raise H3QualityError("SSIM data_range must be finite and positive")
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise H3QualityError("SSIM inputs have different shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise H3QualityError("SSIM inputs contain non-finite values")
    mean_reference = float(reference.mean()) if reference.size else 0.0
    mean_candidate = float(candidate.mean()) if candidate.size else 0.0
    centered_reference = reference - mean_reference
    centered_candidate = candidate - mean_candidate
    variance_reference = (
        float(np.mean(centered_reference * centered_reference))
        if reference.size
        else 0.0
    )
    variance_candidate = (
        float(np.mean(centered_candidate * centered_candidate))
        if candidate.size
        else 0.0
    )
    covariance = (
        float(np.mean(centered_reference * centered_candidate))
        if reference.size
        else 0.0
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (
        (2.0 * mean_reference * mean_candidate + c1)
        * (2.0 * covariance + c2)
    )
    denominator = (
        (mean_reference * mean_reference + mean_candidate * mean_candidate + c1)
        * (variance_reference + variance_candidate + c2)
    )
    return numerator / denominator


def _gaussian_filter_valid(
    values: np.ndarray, *, window_size: int, sigma: float
) -> np.ndarray:
    coordinates = np.arange(window_size, dtype=np.float64)
    coordinates -= (window_size - 1) / 2.0
    kernel = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    horizontal = np.lib.stride_tricks.sliding_window_view(
        values, window_size, axis=1
    )
    horizontal = np.tensordot(horizontal, kernel, axes=([-1], [0]))
    vertical = np.lib.stride_tricks.sliding_window_view(
        horizontal, window_size, axis=0
    )
    return np.tensordot(vertical, kernel, axes=([-1], [0]))


def windowed_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    data_range: float,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Return deterministic Gaussian-window SSIM averaged over channels.

    The arithmetic follows the population-covariance form of the canonical
    SSIM metric. For fixtures smaller than the release 11x11 window, the
    largest fitting odd window is used so corruption tests remain defined.
    """

    if not math.isfinite(data_range) or data_range <= 0:
        raise H3QualityError("windowed SSIM data_range must be positive")
    if window_size <= 0 or window_size % 2 == 0:
        raise H3QualityError("windowed SSIM window must be positive and odd")
    if not math.isfinite(sigma) or sigma <= 0:
        raise H3QualityError("windowed SSIM sigma must be positive")
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 3:
        raise H3QualityError("windowed SSIM inputs must be [H,W,C]")
    if reference.shape != candidate.shape:
        raise H3QualityError("windowed SSIM inputs have different shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise H3QualityError("windowed SSIM inputs contain non-finite values")
    fitting_window = min(window_size, reference.shape[0], reference.shape[1])
    if fitting_window % 2 == 0:
        fitting_window -= 1
    if fitting_window < 1:
        raise H3QualityError("windowed SSIM inputs have empty spatial axes")
    mean_reference = _gaussian_filter_valid(
        reference, window_size=fitting_window, sigma=sigma
    )
    mean_candidate = _gaussian_filter_valid(
        candidate, window_size=fitting_window, sigma=sigma
    )
    mean_reference_squared = mean_reference * mean_reference
    mean_candidate_squared = mean_candidate * mean_candidate
    mean_product = mean_reference * mean_candidate
    variance_reference = np.maximum(
        0.0,
        _gaussian_filter_valid(
            reference * reference,
            window_size=fitting_window,
            sigma=sigma,
        )
        - mean_reference_squared,
    )
    variance_candidate = np.maximum(
        0.0,
        _gaussian_filter_valid(
            candidate * candidate,
            window_size=fitting_window,
            sigma=sigma,
        )
        - mean_candidate_squared,
    )
    covariance = (
        _gaussian_filter_valid(
            reference * candidate,
            window_size=fitting_window,
            sigma=sigma,
        )
        - mean_product
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mean_product + c1) * (2.0 * covariance + c2)
    denominator = (
        mean_reference_squared + mean_candidate_squared + c1
    ) * (variance_reference + variance_candidate + c2)
    return float(np.mean(numerator / denominator))


def frame_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    data_range: float = 1.0,
) -> dict:
    """Compare [frames, height, width, channels] frame arrays."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim != 4 or reference.shape[-1] not in (1, 3, 4):
        raise H3QualityError("frame arrays must be [T,H,W,C]")
    if reference.shape != candidate.shape:
        raise H3QualityError("frame arrays have different shapes")
    per_frame = []
    for index in range(reference.shape[0]):
        per_frame.append(
            {
                "index": index,
                "numeric": numeric_metrics(reference[index], candidate[index]),
                "psnr_db": psnr(
                    reference[index], candidate[index], data_range=data_range
                ),
                "ssim_global": global_ssim(
                    reference[index], candidate[index], data_range=data_range
                ),
                "ssim_windowed": windowed_ssim(
                    reference[index],
                    candidate[index],
                    data_range=data_range,
                ),
            }
        )
    temporal = numeric_metrics(
        np.diff(reference.astype(np.float64), axis=0),
        np.diff(candidate.astype(np.float64), axis=0),
    )
    finite_psnr = [
        item["psnr_db"] for item in per_frame if item["psnr_db"] is not None
    ]
    return {
        "shape": list(reference.shape),
        "numeric": numeric_metrics(reference, candidate),
        "per_frame": per_frame,
        "psnr_db_mean": (
            float(np.mean(finite_psnr)) if finite_psnr else None
        ),
        "ssim_global_mean": float(
            np.mean([item["ssim_global"] for item in per_frame])
        ),
        "ssim_windowed_mean": float(
            np.mean([item["ssim_windowed"] for item in per_frame])
        ),
        "temporal_delta": temporal,
    }


def _spectrogram(
    waveform: np.ndarray, *, window: int = 1024, hop: int = 256
) -> np.ndarray:
    """Match the pinned torch.stft(center=True, pad_mode="reflect") oracle."""

    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 2:
        raise H3QualityError("waveform must be [channels,samples]")
    if window <= 0 or hop <= 0:
        raise H3QualityError("spectrogram window and hop must be positive")
    padding = window // 2
    if waveform.shape[1] <= padding:
        raise H3QualityError(
            "waveform is too short for centered reflect-padded spectrogram"
        )
    padded = np.pad(
        waveform, ((0, 0), (padding, padding)), mode="reflect"
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window, axis=1
    )[:, ::hop, :]
    coordinates = np.arange(window, dtype=np.float64)
    periodic_hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * coordinates / window)
    spectrum = np.abs(
        np.fft.rfft(windows * periodic_hann[None, None, :], axis=-1)
    )
    return np.transpose(spectrum, (0, 2, 1))


def audio_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_rate: int,
) -> dict:
    """Compare channel-major F32 PCM and its deterministic spectrogram."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim != 2 or reference.shape != candidate.shape:
        raise H3QualityError(
            "audio arrays must have equal [channels,samples] shapes"
        )
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise H3QualityError("sample_rate must be positive")
    channels = [
        numeric_metrics(reference[index], candidate[index])
        for index in range(reference.shape[0])
    ]
    reference_spectrum = _spectrogram(reference)
    candidate_spectrum = _spectrogram(candidate)
    spectrum_metrics = numeric_metrics(
        reference_spectrum, candidate_spectrum
    )
    spectrum_metrics["shape"] = list(reference_spectrum.shape)
    return {
        "shape": list(reference.shape),
        "sample_rate": sample_rate,
        "duration_seconds": reference.shape[1] / sample_rate,
        "waveform": numeric_metrics(reference, candidate),
        "channels": channels,
        "spectrogram": spectrum_metrics,
    }
