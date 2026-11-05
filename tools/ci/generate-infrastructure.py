#!/usr/bin/env python3
"""Generate and drift-check repository infrastructure from model metadata manifests.

Manages:
1. cmake/Checks.cmake (check-pr targets and CTest overrides)
2. tools/ci/check-format.py (vendored paths exclusion list)
3. tools/ci/check-docs.py (required model doc paths)
4. docs/models/README.md (models index table)
5. tests/functional/devlab.json (merged from devlab.core.json and model fragments)
6. tests/functional/impact.json (merged from impact.core.json and model fragments)
"""

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple


GUIDE_LABELS = {
    "deepseek-v4-flash": "Usage and modes",
    "embeddinggemma-2": "Usage",
    "minimax-h3": "Presets and usage",
    "qwen-image-2.1": "Usage",
    "qwen3-asr": "Usage",
    "qwen3-tts": "Voice modes",
    "qwen3.8-27b": "Usage and modes",
    "qwen3.8-flash-next": "Usage and modes",
}


def load_manifests(root: Path) -> List[Dict[str, Any]]:
    """Discover and parse all extension manifests."""
    manifest_paths = sorted((root / "src/models").glob("*/extension.json"))
    manifest_paths.extend(sorted((root / "src/families").glob("*/*/extension.json")))
    manifests = []
    seen_names = set()
    seen_provider_ids = set()
    for path in manifest_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse manifest {path}: {e}")
        name = data.get("name")
        if not name:
            raise ValueError(f"Manifest {path} missing 'name'")
        if name in seen_names:
            raise ValueError(f"Duplicate extension name '{name}' in {path}")
        seen_names.add(name)

        provider_id = (data.get("provider") or {}).get("id")
        if provider_id:
            if provider_id in seen_provider_ids:
                raise ValueError(f"Duplicate provider id '{provider_id}' in {path}")
            seen_provider_ids.add(provider_id)
        data["_manifest_path"] = path
        manifests.append(data)
    manifests.sort(key=lambda m: m["name"])
    return manifests


def replace_marked_block(
    content: str, start_marker: str, end_marker: str, replacement: str
) -> str:
    """Replace content between start_marker and end_marker."""
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL
    )
    if not pattern.search(content):
        raise ValueError(f"Markers '{start_marker}' ... '{end_marker}' not found")
    new_block = f"{start_marker}\n{replacement}\n{end_marker}"
    return pattern.sub(new_block, content)


def generate_checks_cmake(root: Path, manifests: List[Dict[str, Any]]) -> str:
    """Generate model targets and CTest overrides for cmake/Checks.cmake."""
    path = root / "cmake/Checks.cmake"
    content = path.read_text(encoding="utf-8")

    pr_targets: List[str] = []
    overrides: List[Tuple[str, str]] = []  # (target, ctest_regex)

    for manifest in manifests:
        tests = manifest.get("tests", [])
        for test in tests:
            target = test.get("target")
            ctest = test.get("ctest")
            check_pr = test.get("check_pr", False)
            if check_pr and target:
                pr_targets.append(target)
                if ctest and ctest != target:
                    overrides.append((target, ctest))

    pr_targets = sorted(set(pr_targets))
    overrides = sorted(set(overrides), key=lambda x: x[0])

    # Format targets: 2-3 per line
    target_lines = []
    current_line = []
    for t in pr_targets:
        current_line.append(t)
        if len(" ".join(current_line)) > 70:
            target_lines.append("  " + " ".join(current_line))
            current_line = []
    if current_line:
        target_lines.append("  " + " ".join(current_line))
    targets_block = "\n".join(target_lines)

    remove_lines = []
    current_line = []
    for target, _ in overrides:
        current_line.append(target)
        if len(" ".join(current_line)) > 70:
            remove_lines.append("  " + " ".join(current_line))
            current_line = []
    if current_line:
        remove_lines.append("  " + " ".join(current_line))

    append_lines = []
    current_line = []
    for _, ctest in overrides:
        escaped_ctest = ctest.replace("\\", "\\\\")
        current_line.append(f'"{escaped_ctest}"')
        if len(" ".join(current_line)) > 70:
            append_lines.append("  " + " ".join(current_line))
            current_line = []
    if current_line:
        append_lines.append("  " + " ".join(current_line))

    overrides_block = (
        "list(REMOVE_ITEM gufo_pr_tests\n"
        + "\n".join(remove_lines)
        + ")\nlist(APPEND gufo_pr_tests\n"
        + "\n".join(append_lines)
        + ")"
    )

    content = replace_marked_block(
        content,
        "  # BEGIN GENERATED MODEL PR TARGETS",
        "  # END GENERATED MODEL PR TARGETS",
        targets_block,
    )
    content = replace_marked_block(
        content,
        "# BEGIN GENERATED MODEL CTEST OVERRIDES",
        "# END GENERATED MODEL CTEST OVERRIDES",
        overrides_block,
    )
    return content


def generate_check_format(root: Path, manifests: List[Dict[str, Any]]) -> str:
    """Generate vendored_paths in tools/ci/check-format.py."""
    path = root / "tools/ci/check-format.py"
    content = path.read_text(encoding="utf-8")

    vendored_paths = set()
    for manifest in manifests:
        for vp in manifest.get("vendored_paths", []):
            vendored_paths.add(vp)

    sorted_paths = sorted(vendored_paths)
    lines = ["    excluded = ("]
    for vp in sorted_paths:
        lines.append(f'        "{vp}",')
    lines.append("    )")
    block = "\n".join(lines)

    return replace_marked_block(
        content,
        "    # BEGIN GENERATED VENDORED PATHS",
        "    # END GENERATED VENDORED PATHS",
        block,
    )


def generate_check_docs(root: Path, manifests: List[Dict[str, Any]]) -> str:
    """Generate model doc directories in tools/ci/check-docs.py."""
    path = root / "tools/ci/check-docs.py"
    content = path.read_text(encoding="utf-8")

    model_dirs = set()
    for manifest in manifests:
        docs = manifest.get("docs")
        if docs and "guide" in docs:
            guide_path = docs["guide"]
            # e.g. "docs/models/deepseek-v4-flash/README.md" -> "deepseek-v4-flash"
            parts = Path(guide_path).parts
            if len(parts) >= 3 and parts[0] == "docs" and parts[1] == "models":
                model_dirs.add(parts[2])

    sorted_dirs = sorted(model_dirs)
    dir_strings = [f'"{d}"' for d in sorted_dirs]
    chunk_size = 3
    lines = []
    for i in range(0, len(dir_strings), chunk_size):
        chunk = dir_strings[i : i + chunk_size]
        if chunk:
            lines.append(", ".join(chunk))
    tuple_content = ",\n                    ".join(lines)
    if len(dir_strings) == 1:
        tuple_content += ","
    block = (
        '    *[f"docs/models/{model}/{document}.md"\n'
        + f"      for model in ({tuple_content})\n"
        + '      for document in ("README", "BENCHMARKS", "QUALITY", "EXPERIMENTS")],'
    )

    return replace_marked_block(
        content,
        "    # BEGIN GENERATED MODEL DOCS",
        "    # END GENERATED MODEL DOCS",
        block,
    )


def generate_models_readme(root: Path, manifests: List[Dict[str, Any]]) -> str:
    """Generate model table in docs/models/README.md."""
    path = root / "docs/models/README.md"
    content = path.read_text(encoding="utf-8")

    rows = []
    rows.append("| Model | Weights / supported inputs | Guide |")
    rows.append("| --- | --- | --- |")

    # Order by model name or title
    model_docs = []
    for manifest in manifests:
        docs = manifest.get("docs")
        if docs:
            model_docs.append(docs)

    model_docs.sort(key=lambda d: d.get("title", ""))

    for docs in model_docs:
        title = docs["title"]
        inputs = docs["inputs"]
        guide = docs["guide"]
        parts = Path(guide).parts
        if len(parts) >= 3 and parts[0] == "docs" and parts[1] == "models":
            slug = parts[2]
            rel_guide = f"{slug}/{parts[3]}" if len(parts) > 3 else f"{slug}/README.md"
        else:
            slug = Path(guide).stem
            rel_guide = guide
        label = GUIDE_LABELS.get(slug, "Usage")
        rows.append(f"| {title} | {inputs} | [{label}]({rel_guide}) |")

    block = "\n".join(rows)

    return replace_marked_block(
        content,
        "<!-- BEGIN GENERATED MODEL TABLE -->",
        "<!-- END GENERATED MODEL TABLE -->",
        block,
    )


def generate_devlab_registry(root: Path, manifests: List[Dict[str, Any]]) -> str:
    """Merge tests/functional/devlab.core.json and model fragments into devlab.json."""
    core_path = root / "tests/functional/devlab.core.json"
    if not core_path.is_file():
        raise FileNotFoundError(f"Missing core devlab registry: {core_path}")
    merged = json.loads(core_path.read_text(encoding="utf-8"))

    merged_assets: Dict[str, Any] = merged.get("assets", {})
    merged_models: Dict[str, Any] = merged.get("models", {})
    core_rules: List[Dict[str, Any]] = merged.get("rules", [])
    model_rules: List[Dict[str, Any]] = []

    for manifest in manifests:
        manifest_path = manifest["_manifest_path"]
        model_dir = manifest_path.parent
        fragment_path = model_dir / "functional/devlab.json"
        if fragment_path.is_file():
            fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
            for asset_key, asset_val in fragment.get("assets", {}).items():
                if asset_key in merged_assets:
                    raise ValueError(f"Duplicate asset '{asset_key}' in {fragment_path}")
                merged_assets[asset_key] = asset_val
            for model_key, model_val in fragment.get("models", {}).items():
                if model_key in merged_models:
                    raise ValueError(f"Duplicate model '{model_key}' in {fragment_path}")
                merged_models[model_key] = model_val
            model_rules.extend(fragment.get("rules", []))

    # Order assets and models deterministically (as in original devlab.json)
    ASSET_ORDER = [
        "flash-next",
        "qwen27",
        "dflash2",
        "deepseek4",
        "dspark",
        "embedding",
        "asr",
        "tts-base",
        "tts-custom",
        "tts-design",
    ]
    ordered_assets = {}
    for k in ASSET_ORDER:
        if k in merged_assets:
            ordered_assets[k] = merged_assets[k]
    for k in sorted(merged_assets):
        if k not in ordered_assets:
            ordered_assets[k] = merged_assets[k]

    MODEL_ORDER = [
        "flash-next-q4",
        "qwen27-q4",
        "deepseek4",
        "embedding-gemma2",
        "qwen3-asr",
        "qwen3-tts-base",
        "qwen3-tts-custom",
        "qwen3-tts-design",
    ]
    ordered_models = {}
    for k in MODEL_ORDER:
        if k in merged_models:
            ordered_models[k] = merged_models[k]
    for k in sorted(merged_models):
        if k not in ordered_models:
            ordered_models[k] = merged_models[k]

    # Combine rules:
    # 1. documentation (from core)
    # 2. outside-current-scope (from core)
    # 3. model rules (qwen27, flash-next, deepseek, embedding, asr, tts)
    # 4. shared rules (shared-audio, service-harness, shared-qwen, shared-text, functional-text, hosted-checks, shared-infrastructure, devlab-orchestration)
    RULE_ORDER = [
        "documentation",
        "outside-current-scope",
        "private-qwen27",
        "private-flash-next",
        "private-deepseek",
        "embedding",
        "asr",
        "tts",
        "shared-audio",
        "service-harness",
        "shared-qwen",
        "shared-text",
        "functional-text",
        "hosted-checks",
        "shared-infrastructure",
        "devlab-orchestration",
    ]

    all_rules_by_id = {}
    for r in core_rules + model_rules:
        all_rules_by_id[r["id"]] = r

    ordered_rules = []
    for rid in RULE_ORDER:
        if rid in all_rules_by_id:
            ordered_rules.append(all_rules_by_id[rid])
    for rid in sorted(all_rules_by_id):
        if all_rules_by_id[rid] not in ordered_rules:
            ordered_rules.append(all_rules_by_id[rid])

    result = {
        "schema": merged["schema"],
        "disabled": merged["disabled"],
        "representatives": merged["representatives"],
        "assets": ordered_assets,
        "models": ordered_models,
        "rules": ordered_rules,
    }
    return json.dumps(result, indent=2) + "\n"


def generate_impact_registry(root: Path, manifests: List[Dict[str, Any]]) -> str:
    """Merge tests/functional/impact.core.json and model fragments into impact.json."""
    core_path = root / "tests/functional/impact.core.json"
    if not core_path.is_file():
        raise FileNotFoundError(f"Missing core impact registry: {core_path}")
    merged = json.loads(core_path.read_text(encoding="utf-8"))

    merged_profiles: Dict[str, Any] = merged.get("profiles", {})
    core_rules: List[Dict[str, Any]] = merged.get("rules", [])
    model_rules: List[Dict[str, Any]] = []

    for manifest in manifests:
        manifest_path = manifest["_manifest_path"]
        model_dir = manifest_path.parent
        fragment_path = model_dir / "functional/impact.json"
        if fragment_path.is_file():
            fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
            for prof_key, prof_val in fragment.get("profiles", {}).items():
                if prof_key in merged_profiles:
                    raise ValueError(f"Duplicate profile '{prof_key}' in {fragment_path}")
                merged_profiles[prof_key] = prof_val
            model_rules.extend(fragment.get("rules", []))

    PROFILE_ORDER = [
        "qwen27-q4/off",
        "qwen27-q4/dflash2",
        "qwen27-q8/off",
        "qwen27-q8/dflash2",
        "flash-next-q4/off",
        "flash-next-q4/mtp",
        "deepseek4/off",
        "deepseek4/dspark",
    ]
    ordered_profiles = {}
    for p in PROFILE_ORDER:
        if p in merged_profiles:
            ordered_profiles[p] = merged_profiles[p]
    for p in sorted(merged_profiles):
        if p not in ordered_profiles:
            ordered_profiles[p] = merged_profiles[p]

    RULE_ORDER = [
        "documentation",
        "qualification-metadata",
        "shared-runtime",
        "build-and-toolchain",
        "qwen27-runtime",
        "flash-next-runtime",
        "deepseek-runtime",
        "shared-qwen-framing",
        "shared-qwen-vision",
        "dflash2",
        "flash-next-continuation",
        "text-continuation",
        "structured-output",
        "sampling",
        "framing-and-stops",
        "outside-text-models",
    ]
    all_rules_by_id = {}
    for r in core_rules + model_rules:
        all_rules_by_id[r["id"]] = r

    ordered_rules = []
    for rid in RULE_ORDER:
        if rid in all_rules_by_id:
            ordered_rules.append(all_rules_by_id[rid])
    for rid in sorted(all_rules_by_id):
        if all_rules_by_id[rid] not in ordered_rules:
            ordered_rules.append(all_rules_by_id[rid])

    result = {
        "schema": merged["schema"],
        "policy": merged["policy"],
        "behaviors": merged["behaviors"],
        "profiles": ordered_profiles,
        "suites": merged["suites"],
        "rules": ordered_rules,
        "baseline_suites": merged["baseline_suites"],
        "fallback": merged["fallback"],
        "configurations": merged["configurations"],
    }
    return json.dumps(result, indent=2) + "\n"


def get_generated_targets(
    root: Path, manifests: List[Dict[str, Any]] = None
) -> Dict[Path, str]:
    """Calculate the expected content for all generated targets."""
    if manifests is None:
        manifests = load_manifests(root)
    return {
        root / "cmake/Checks.cmake": generate_checks_cmake(root, manifests),
        root / "tools/ci/check-format.py": generate_check_format(root, manifests),
        root / "tools/ci/check-docs.py": generate_check_docs(root, manifests),
        root / "docs/models/README.md": generate_models_readme(root, manifests),
        root / "tests/functional/devlab.json": generate_devlab_registry(root, manifests),
        root / "tests/functional/impact.json": generate_impact_registry(root, manifests),
    }


def check_targets(root: Path) -> Tuple[bool, Dict[str, str]]:
    """Check for drift against committed files. Returns (is_clean, drifts)."""
    manifests = load_manifests(root)
    targets = get_generated_targets(root, manifests)
    drifts: Dict[str, str] = {}

    for path, expected_content in targets.items():
        rel_path = path.relative_to(root).as_posix()
        if not path.is_file():
            drifts[rel_path] = f"File {rel_path} does not exist"
            continue
        current_content = path.read_text(encoding="utf-8")
        if current_content != expected_content:
            diff = "".join(
                difflib.unified_diff(
                    current_content.splitlines(keepends=True),
                    expected_content.splitlines(keepends=True),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                )
            )
            drifts[rel_path] = diff

    return len(drifts) == 0, drifts


def write_targets(root: Path) -> List[str]:
    """Write all generated targets. Returns list of relative paths written."""
    manifests = load_manifests(root)
    targets = get_generated_targets(root, manifests)
    written: List[str] = []

    for path, expected_content in targets.items():
        rel_path = path.relative_to(root).as_posix()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected_content, encoding="utf-8")
        written.append(rel_path)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Check drift against committed files (default)",
    )
    group.add_argument(
        "--write", action="store_true", help="Write generated files in place"
    )
    args = parser.parse_args()

    root = args.root.resolve()

    if args.write:
        written = write_targets(root)
        for rel in written:
            print(f"WROTE: {rel}")
        print("PASS: Updated generated infrastructure files.")
        return 0

    is_clean, drifts = check_targets(root)
    if is_clean:
        print("PASS: Infrastructure files are up to date with model metadata.")
        return 0

    for rel_path, diff in drifts.items():
        print(f"DRIFT: {rel_path} is out of sync with model manifests:", file=sys.stderr)
        if diff:
            sys.stderr.write(diff)

    print("\nFAIL: Generated infrastructure has drifted from model metadata. Run with --write to update.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
