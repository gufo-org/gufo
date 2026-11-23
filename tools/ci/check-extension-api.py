#!/usr/bin/env python3
"""Track the public header closure and gate extension API changes in a PR diff.

The token comparison is deliberately conservative: it also gates inline bodies,
initializers, schema and public CMake contracts. Comments and whitespace alone do
not require a change record. Card10 expands classification; no binary ABI is promised.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

SURFACE = "include/gufo/ext/SURFACE"
# The SURFACE paths reachable only through the no-promise headers. They stay in
# SURFACE so their diffs remain visible, but do not need a record.
NO_PROMISE = "include/gufo/ext/NO_PROMISE"
NO_PROMISE_ROOTS = ("include/gufo/ext/experimental/", "include/gufo/ext/families/")
SERIES = "include/gufo/ext/api-series.json"
RECORDS = "include/gufo/ext/changes"
VERSION = "include/gufo/ext/version.hpp"
# One-time normalization of the original records to descriptive names/headings.
# Bodies are preserved; the baseline no longer hardcodes a revision floor.
LEGACY_RECORD_NAMES = {
    '0': ('baseline', 'Source API baseline'),
    '1': ('extension-provenance', 'Extension provenance and excluded metadata'),
    '2': ('public-forwarding-headers', 'Public forwarding headers'),
    '3': ('embedding-contracts', 'Capability-neutral embedding contracts'),
    '4': ('metadata-only-discovery', 'Metadata-only extension discovery'),
    '5': ('ggml-block-formats', 'GGML block format support'),
    '6': ('media-provider-contracts', 'Media provider contracts'),
    '7': ('sampler-logit-scratch', 'Sampler logit scratch storage'),
    '8': ('text-generation-contracts', 'Text generation contracts'),
    '9': ('embedding-video-token-budget', 'Embedding video token budget'),
    '10': ('external-extension-sources', 'External extension source trees'),
    'automatic-versioning': ('automatic-versioning', 'Automatic API revision derivation'),
    '784-image-default-steps': ('image-default-steps', 'Image checkpoint default steps'),
    'speech-realtime-decision': ('speech-realtime-decision', 'Realtime speech and decision contracts'),
}
GUIDANCE = {f"{RECORDS}/{name}" for name in ("README.md", "AGENTS.md", "CLAUDE.md")}
CONTRACTS = {"cmake/GufoExtension.cmake", "cmake/GufoExtensionDiscovery.cmake", "schemas/extension.schema.json",
             "cmake/GufoExtensionVersion.cmake", "cmake/GufoExtensionVersion.hpp.in"}
TOKEN = re.compile(r'R"(?P<raw>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=raw)"|' + r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\w+|[^\s]')


CMAKE_TOKEN = re.compile(r'#\[(?P<comment>=*)\[[\s\S]*?\](?P=comment)\]|#[^\n]*|'
                         r'\[(?P<literal>=*)\[[\s\S]*?\](?P=literal)\]|'
                         r'"(?:\\.|[^"\\])*"|\w+|[^\s]')


def tokens(text, path=""):
    pattern = CMAKE_TOKEN if path.endswith(".cmake") else TOKEN
    comments = ("#",) if pattern is CMAKE_TOKEN else ("//", "/*")
    return [match[0] for match in pattern.finditer(text) if not match[0].startswith(comments)]


HEADER_SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".inc", ".def")
FAMILIES = "include/gufo/ext/families"
# Public model layers and the one src/models/ directory each may forward to.
# A model layer without an entry may not reach src/models/ at all.
MODEL_SOURCES = {("qwen", "flash_next"): "src/models/qwen38_flash_next"}
COMMENT_OR_STRING = re.compile(r'R"(?P<raw>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=raw)"|'
                               r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
DIRECTIVE = re.compile(r'^[ \t]*#[ \t]*include(?:_next)?\b[ \t]*(.*)$', re.M)
OPERAND = re.compile(r'"([^"\n]+)"|<([^>\n]+)>')


def strip_comments(text):
    """Blank comments, keeping strings and line breaks, so directives keep their lines."""
    def blank(match):
        token = match[0]
        if token.startswith(("//", "/*")):
            return re.sub(r"[^\n]", " ", token)
        return token
    return COMMENT_OR_STRING.sub(blank, text)


def includes(text):
    """Yield (quoted, path) per include directive, or (None, operand) for a macro include."""
    for match in DIRECTIVE.finditer(strip_comments(text)):
        operand = OPERAND.match(match[1].strip())
        if operand is None:
            yield None, match[1].strip()
        else:
            yield operand[1] is not None, operand[1] or operand[2]


def reach(root, starts, errors=None):
    """Return the repository headers starts include, transitively.

    A quoted include that resolves nowhere and a macro include are reported in
    errors; an unresolved angle include is a system header.
    """
    errors = [] if errors is None else errors
    pending = list(starts)
    seen = set()
    while pending:
        path = pending.pop().resolve()
        relative = path.relative_to(root.resolve()).as_posix()
        if relative in seen: continue
        seen.add(relative)
        for quoted, include in includes(path.read_text()):
            if quoted is None:
                errors.append(f"{relative}: macro include {include} cannot be audited; spell the path")
                continue
            bases = (path.parent, root / "include", root) if quoted else (root / "include", root)
            for candidate in (base / include for base in bases):
                if candidate.is_file():
                    pending.append(candidate)
                    break
            else:
                if quoted:
                    errors.append(f"{relative}: unresolved include \"{include}\"")
    return seen


def scan_errors(root):
    errors = []
    reach(root, public_headers(root), errors)
    return sorted(set(errors))


def public_headers(root):
    return [path for path in (root / "include/gufo/ext").rglob("*")
            if path.is_file() and path.suffix in HEADER_SUFFIXES]


def closure(root):
    return sorted(reach(root, public_headers(root)))


def layer_of(path):
    """Classify a path for the family layer rules.

    Returns ("family", family, None), ("model", family, model), ("other",) or
    ("invalid", reason). Public and source model directories are
    families/<family>/models/<model>/ under include/gufo/ext/ and src/.
    """
    for prefix in (FAMILIES + "/", "src/families/"):
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix):].split("/")
        if len(parts) < 2:
            return ("invalid", "a family header belongs in families/<family>/")
        if parts[1] != "models":
            return ("family", parts[0], None)
        if len(parts) < 4:
            return ("invalid", "a model header belongs in families/<family>/models/<model>/")
        return ("model", parts[0], parts[2])
    return ("other",)


def layer_errors(root):
    """Family headers never reach a model, and a model reaches no other model.

    Each public header under families/ is checked through its transitive
    include closure, which also covers the src/ headers it forwards to.
    """
    errors = []
    starts = sorted(p for p in (root / FAMILIES).rglob("*")
                    if p.is_file() and p.suffix in HEADER_SUFFIXES) if (root / FAMILIES).is_dir() else []
    for start in starts:
        source = start.relative_to(root).as_posix()
        layer = layer_of(source)
        if layer[0] == "invalid":
            errors.append(f"{source}: {layer[1]}")
            continue
        _, family, model = layer
        for target in sorted(reach(root, [start], errors) - {source}):
            owner = layer_of(target)
            in_models = owner[0] == "model" or any(
                target.startswith(prefix) and target[len(prefix):].split("/")[1:2] == ["models"]
                for prefix in (FAMILIES + "/", "src/families/"))
            if model is None:
                if in_models or target.startswith("src/models/"):
                    errors.append(f"{source}: the {family} family layer reaches model header {target}")
            elif in_models and owner[1:] != (family, model):
                errors.append(f"{source}: model {family}/{model} reaches another model's header {target}")
            elif target.startswith("src/models/"):
                allowed = MODEL_SOURCES.get((family, model))
                if allowed is None or not target.startswith(allowed.rstrip("/") + "/"):
                    errors.append(f"{source}: model {family}/{model} reaches {target}, outside its "
                                  "declared source directory (MODEL_SOURCES)")
    return sorted(set(errors))


def inventory(root):
    return "# Generated by tools/ci/check-extension-api.py --write; transitive source API paths.\n" + "".join(path + "\n" for path in closure(root))


def no_promise(root):
    """SURFACE paths no versioned public header reaches."""
    versioned = [path for path in public_headers(root)
                 if not path.relative_to(root).as_posix().startswith(NO_PROMISE_ROOTS)]
    return sorted(set(closure(root)) - reach(root, versioned))


def no_promise_inventory(root):
    return ("# Generated by tools/ci/check-extension-api.py --write; SURFACE paths reachable only\n"
            "# through include/gufo/ext/experimental/ or families/. Their changes need no record.\n"
            + "".join(path + "\n" for path in no_promise(root)))


def listed(text):
    return {p for p in text.splitlines() if p and not p.startswith("#")}


def api_version(text):
    match = re.search(r'kApiVersion\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', text)
    if not match: raise ValueError("missing extension API kApiVersion")
    return match[1]


def series_values(text):
    value = json.loads(text)
    fields = ("major", "minor", "revision_base")
    if set(value) != set(fields) or any(type(value[k]) is not int or value[k] < 0 for k in fields):
        raise ValueError("invalid api-series.json: expected nonnegative major, minor, revision_base")
    return tuple(value[k] for k in fields)


def record_errors(records):
    errors = []
    for path, body in records.items():
        if not re.fullmatch(RECORDS + r"/(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)/[a-z0-9][a-z0-9_-]*\.md", path):
            errors.append(f"invalid extension API change record path: {path}")
        if not body.strip(): errors.append(f"empty extension API change record: {path}")
    return errors


def computed_version(series, records):
    major, minor, base = series_values(series)
    count = sum(path.startswith(f"{RECORDS}/{major}.{minor}/") and Path(path).name not in ("baseline.md", f"{base}.md")
                for path in records)
    return f"{major}.{minor}.{base + count}"


def records_for_root(root):
    return {p.relative_to(root).as_posix(): p.read_text()
            for p in (root / RECORDS).rglob("*")
            if p.is_file() and p.relative_to(root).as_posix() not in GUIDANCE}


def version_for_root(root):
    records = records_for_root(root)
    errors = record_errors(records)
    if errors: raise ValueError("\n".join(errors))
    return computed_version((root / SERIES).read_text(), records)


def historical_revision(path, text):
    series = path.split("/")[-2]
    match = re.match(rf"# {re.escape(series)}\.(0|[1-9][0-9]*)\n", text)
    return int(match[1]) if match else None


def changelog(root):
    records = records_for_root(root)
    groups = sorted({p.split("/")[-2] for p in records}, key=lambda s: tuple(map(int, s.split('.'))), reverse=True)
    result = ["# Extension API changes", "", f"Current API: {version_for_root(root)}", "",
              "Record names and headings describe changes; API revisions are derived from the record count.", ""]
    def order(path):
        revision = historical_revision(path, records[path])
        return (1, -revision, path) if revision is not None else (0, 0, path)
    for group in groups:
        result.extend([f"## Series {group}", ""])
        for path in sorted(records, key=order):
            if path.split("/")[-2] != group: continue
            body = records[path].strip()
            if body.startswith("# "):
                title, _, body = body.partition("\n")
                title = title.removeprefix("# ")
            else:
                title = Path(path).stem
            result.extend([f"### {title}", "", body.strip(), ""])
    return "\n".join(result)


def check_diff(read_old, read_new, labels, old_records=None, new_records=None):
    old_records = old_records or {}
    new_records = new_records or {}
    errors = record_errors(new_records)
    normalized_records = set()
    for path, body in old_records.items():
        if new_records.get(path) != body:
            legacy = LEGACY_RECORD_NAMES.get(Path(path).stem)
            if legacy is not None and path == f"{RECORDS}/0.1/{Path(path).stem}.md":
                name, title = legacy
                normalized = f"{RECORDS}/0.1/{name}.md"
                description = re.sub(r"^# [0-9]+\.[0-9]+\.[0-9]+\n\n", "", body, count=1)
                description = description.replace('`api_version: ">= 0.2.0"`',
                    'an `api_version` floor in the active compatibility series')
                if (not body.startswith(f"# {title}\n\n")
                        and new_records.get(normalized) == f"# {title}\n\n{description}"):
                    normalized_records.add(normalized)
                    continue
            errors.append(f"extension API change records are permanent and immutable: {path}")
    old_surface = listed(read_old(SURFACE))
    new_surface = listed(read_new(SURFACE))
    new_free = listed(read_new(NO_PROMISE))
    # Before NO_PROMISE existed, the old surface is classified like the new one.
    old_free = listed(read_old(NO_PROMISE)) if read_old(NO_PROMISE) else new_free & old_surface
    def free(path):
        return path in (new_free if path in new_surface else old_free)
    changed = sorted(path for path in old_surface | new_surface | CONTRACTS
                     if tokens(read_old(path), path) != tokens(read_new(path), path) and not free(path))
    series_changed = read_old(SERIES) != read_new(SERIES)
    added = set(new_records) - set(old_records) - normalized_records
    promised_same = old_surface - old_free == new_surface - new_free
    if not changed and promised_same and not series_changed and not added: return errors
    if "ext-api" not in labels: errors.append("extension API changes require the ext-api PR label")
    if not added: errors.append("extension API changes require a new permanent change record")
    try:
        version = computed_version(read_new(SERIES), new_records)
        # One-time transition from the old manually maintained version header.
        old_version = (computed_version(read_old(SERIES), old_records) if read_old(SERIES)
                       else api_version(read_old(VERSION)))
        if tuple(map(int, version.split('.'))) <= tuple(map(int, old_version.split('.'))):
            errors.append("extension API version must increase")
        major, minor, base = series_values(read_new(SERIES))
        if not read_old(SERIES):
            old_major, old_minor, old_revision = map(int, old_version.split('.'))
            if (major, minor) == (old_major, old_minor) and base < old_revision:
                # A lower baseline is safe only when the preceding revisions
                # have been converted into individually versioned records.
                for revision in range(base, old_revision + 1):
                    name = f"{revision}.md"
                    path = f"{RECORDS}/{major}.{minor}/{name}"
                    if historical_revision(path, new_records.get(path, "")) != revision:
                        errors.append("migration must preserve the existing API revision: "
                                      f"missing historical record for {major}.{minor}.{revision}")
        if not any(path.startswith(f"{RECORDS}/{major}.{minor}/") for path in added):
            errors.append("new extension API change record must belong to the active series")
        if read_old(SERIES):
            old_major, old_minor, old_base = series_values(read_old(SERIES))
            if (major, minor) == (old_major, old_minor) and series_values(read_new(SERIES))[2] != old_base:
                errors.append("revision_base is fixed within an extension API compatibility series")
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    print("Extension API changed paths: " + ", ".join(changed))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--base", help="base Git commit for a PR diff, independent of the working VCS")
    parser.add_argument("--labels", default="[]", help="JSON list of PR label names")
    parser.add_argument("--changelog", type=Path, help="write a combined changelog without editing source records")
    args = parser.parse_args()
    root = args.root.resolve()
    expected = inventory(root)
    expected_free = no_promise_inventory(root)
    if args.write:
        (root / SURFACE).write_text(expected)
        (root / NO_PROMISE).write_text(expected_free)
    for path, text in ((SURFACE, expected), (NO_PROMISE, expected_free)):
        if not (root / path).is_file() or (root / path).read_text() != text:
            parser.exit(1, f"extension {Path(path).name} drift: run tools/ci/check-extension-api.py --write\n")
    try:
        version = version_for_root(root)
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"extension API: {exc}\n")
    if errors := sorted(set(scan_errors(root) + layer_errors(root))):
        parser.exit(1, "\n".join(errors) + "\n")
    if args.changelog:
        args.changelog.parent.mkdir(parents=True, exist_ok=True)
        args.changelog.write_text(changelog(root))
    if args.base:
        def old(path):
            result = subprocess.run(["git", "show", f"{args.base}:{path}"], cwd=root,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return result.stdout if result.returncode == 0 else ""
        # Reject a misspelled base instead of treating every file as newly added.
        subprocess.run(["git", "rev-parse", "--verify", f"{args.base}^{{commit}}"], cwd=root,
                       check=True, stdout=subprocess.DEVNULL)
        def new(path):
            return (root / path).read_text() if (root / path).is_file() else ""
        paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", args.base, "--", RECORDS],
                                        cwd=root, text=True).splitlines()
        old_records = {path: old(path) for path in paths if path not in GUIDANCE}
        errors = check_diff(old, new, json.loads(args.labels), old_records, records_for_root(root))
        if errors: parser.exit(1, "\n".join(errors) + "\n")
    print(f"PASS: extension API {version}; surface ({len(closure(root))} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
