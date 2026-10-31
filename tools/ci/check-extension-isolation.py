#!/usr/bin/env python3
"""Report extension coupling; enforce the migrated core boundary.

This is a lexical inventory, not a C++ name resolver. Includes and qualified
model/family namespace references are separate evidence. Conditional includes
are inventoried even when disabled. JSON retains every location for later
ratchets; no baseline or exception silently hides current dependencies.

Enforcement ratchet (one boundary per migrated layer): with
--fail-on-core-model-includes, no file under src/core/** may include a header
under src/models/** or src/families/**. Card 06 (RFC step 4) retired the last
such include, so a new violation fails the check. Other layers stay report
only until their cards migrate them.
"""

import argparse
import io
import json
import posixpath
import re
import tokenize
from collections import Counter
from pathlib import Path

CODE_SUFFIXES = {".cpp", ".c", ".cc", ".h", ".hpp", ".hip", ".cuh"}
CONFIG_SUFFIXES = {".cmake", ".nix", ".json", ".yaml", ".yml", ".py", ".sh"}
SKIP_DIRS = {"__pycache__", "vendor", ".git", ".jj"}
# Historical names that do not match the owning source directory.
ALIASES = {
    "qwen": {"qwen27b", "qwen_27b", "qwen3.8-27b"},
    "deepseek_v4_flash": {"deepseek", "deepseek4", "ds4", "deepseek-v4-flash"},
    "embedding_gemma2": {"embeddinggemma", "embeddinggemma-2"},
    "qwen38_flash_next": {"qwen3.8-flash-next"},
    "qwen3_asr": {"qwen3-asr"},
    "qwen3_tts": {"qwen3-tts"},
    "qwen_image_21": {"qwen-image-2.1"},
    "minimax_h3": {"minimax-h3"},
}
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*["<]([^">]+)[">]', re.MULTILINE)
CPP_TOKEN = re.compile(r'R"(?P<delim>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delim)"'
                       r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
                       r'|//[^\n]*|/\*.*?\*/', re.DOTALL)


def blank(text):
    return "".join("\n" if char == "\n" else " " for char in text)


def include_matches(text, suffix):
    """Yield literal C/C++ directives whose # lies outside strings/comments.

    Keep the original spelling for include targets and offsets for locations.
    Masking tokens only for the directive-start check also excludes directives
    embedded in multiline raw strings, without losing quoted header names.
    """
    if suffix not in CODE_SUFFIXES:
        return
    directives = CPP_TOKEN.sub(lambda match: blank(match[0]), text)
    for match in INCLUDE.finditer(text):
        hash_offset = text.index("#", match.start(), match.end())
        if directives[hash_offset] == "#":
            yield match


def strip_comments(text, suffix):
    """Keep strings and line numbers, excluding comment-only coupling."""
    if suffix in CODE_SUFFIXES:
        return CPP_TOKEN.sub(lambda m: blank(m[0]) if m[0].startswith(("//", "/*"))
                             else m[0], text)
    if suffix == ".py":
        lines = text.splitlines(keepends=True)
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT:
                    row, col = token.start
                    lines[row - 1] = lines[row - 1][:col] + blank(token.string) + lines[row - 1][token.end[1]:]
        except (tokenize.TokenError, IndentationError):
            # Inventory malformed/incomplete sources too; do not suppress them.
            return text
        return "".join(lines)
    # Tokenize strings before comments so globs/URLs in manifests survive.
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
                        r'|#\[(?P<eq>=*)\[.*?\](?P=eq)\]|/\*.*?\*/|#[^\n]*',
                        re.DOTALL)
    return tokens.sub(lambda m: blank(m[0]) if m[0].startswith(("#", "/*"))
                      else m[0], text)


def owner(path):
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] == "src" and parts[1] in {"models", "families"}:
        return f"{parts[1]}/{parts[2]}"
    if len(parts) >= 3 and parts[:2] == ("tests", "models"):
        return f"tests/models/{parts[2]}"
    if parts[0] in {"src", "tests", "tools"}:
        return "/".join(parts[:2])
    return "configuration"


def source_paths(root):
    for directory in ("src", "tests", "tools", "include", "cmake", ".github", ".devops"):
        for path in sorted((root / directory).rglob("*")):
            relative = path.relative_to(root)
            if (path.is_file() and not SKIP_DIRS.intersection(relative.parts)
                    and (path.suffix in CODE_SUFFIXES | CONFIG_SUFFIXES or path.name == "CMakeLists.txt")):
                yield relative.as_posix()
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.suffix in CONFIG_SUFFIXES or path.name in {"CMakeLists.txt", "extensions.lock"}):
            yield path.name


def is_configuration(path):
    parts = Path(path).parts
    return (parts[0] in {"cmake", ".github", ".devops"} or len(parts) == 1
            or path.startswith("tools/ci/")
            or (parts[0] in {"tests", "tools"} and Path(path).suffix == ".json"))


def inventory(root):
    models = sorted(path.name for path in (root / "src/models").iterdir() if path.is_dir())
    families = sorted(path.name for path in (root / "src/families").glob("*") if path.is_dir())
    names = {f"models/{name}": {name} | ALIASES.get(name, set()) for name in models}
    names.update({f"families/{name}": {name} for name in families})
    mentions = {key: re.compile(r"(?<![a-zA-Z0-9])(?:" + "|".join(
        re.escape(value) for value in sorted(values, key=len, reverse=True)) + r")(?![a-zA-Z0-9])",
        re.IGNORECASE) for key, values in names.items()}
    namespaces = {key: re.compile(r"\b(?:" + "|".join(re.escape(value) for value in sorted(values)
                                  if re.fullmatch(r"\w+", value)) + r")\s*::")
                  for key, values in names.items()}
    texts = {path: strip_comments((root / path).read_text(encoding="utf-8"), Path(path).suffix)
             for path in source_paths(root)}
    includes, symbols, model_mentions, name_references = [], [], [], []
    graph = {}
    for path, text in sorted(texts.items()):
        graph[path] = set()
        for match in include_matches(text, Path(path).suffix):
            target = match[1]
            candidates = (posixpath.normpath(candidate) for candidate in (
                target, (Path(path).parent / target).as_posix(), f"include/{target}")
            )
            resolved = next((candidate for candidate in candidates if candidate in texts), None)
            if resolved:
                graph[path].add(resolved)
            includes.append({"path": path, "line": text[:match.start()].count("\n") + 1,
                             "owner": owner(path), "include": target, "resolved": resolved,
                             "dependency": owner(resolved) if resolved else "external-or-unresolved"})
        # Strings are valid include evidence but not C++ symbol references.
        symbol_text = CPP_TOKEN.sub(lambda m: blank(m[0]), text) if Path(path).suffix in CODE_SUFFIXES else ""
        for key, pattern in namespaces.items():
            for match in pattern.finditer(symbol_text):
                symbols.append({"path": path, "line": symbol_text[:match.start()].count("\n") + 1,
                                "owner": owner(path), "dependency": key, "symbol": match[0]})
        for number, line in enumerate(text.splitlines(), 1):
            for key, pattern in mentions.items():
                if pattern.search(line):
                    reference = {"path": path, "line": number, "model": key,
                                 "owner": owner(path), "text": line.strip()}
                    name_references.append(reference)
                    if is_configuration(path):
                        model_mentions.append(reference)

    qwen_headers = {}
    for header in sorted(path for path in texts if path.startswith("src/models/qwen/")
                         and Path(path).suffix in {".h", ".hpp", ".cuh"}):
        closure, pending = set(), list(graph[header])
        while pending:
            dependency = pending.pop()
            if dependency == header or dependency in closure:
                continue
            closure.add(dependency)
            pending.extend(graph.get(dependency, ()))
        qwen_headers[header] = {
            "direct": sorted(graph[header]), "transitive": sorted(closure),
            "unresolved": sorted(edge["include"] for edge in includes
                                 if edge["path"] == header and edge["resolved"] is None),
            "consumers": sorted(path for path, dependencies in graph.items() if header in dependencies),
        }
    cross = [edge for edge in includes if edge["resolved"] and edge["owner"] != edge["dependency"]]
    return {"mode": "report", "schema_version": 1, "models": models, "families": families,
            "limitations": ["Lexical namespace references are candidates, not compiler-resolved symbols.",
                            "Includes in all conditional branches are counted; macro includes are not resolved.",
                            "Header closure covers literal repository-local includes, not compiler search paths.",
                            "Directories named vendor and Python bytecode caches are excluded."],
            "summary": {"files": len(texts), "includes": len(includes), "cross_owner_includes": len(cross),
                        "namespace_references": len(symbols), "name_references": len(name_references),
                        "configuration_mentions": len(model_mentions),
                        "qwen_headers": len(qwen_headers)},
            "cross_owner_counts": dict(sorted(Counter(f'{edge["owner"]} -> {edge["dependency"]}'
                                                       for edge in cross).items())),
            "includes": includes, "namespace_references": symbols,
            "name_references": name_references, "configuration_mentions": model_mentions,
            "qwen_headers": qwen_headers}


def core_boundary_violations(report):
    """Return src/core -> src/models|src/families include edges.

    Card 06 moved Qwen-specific speculation out of src/core/speculative/, so
    the core layer is now free of model includes and must stay that way.
    """
    violations = []
    for edge in report["includes"]:
        if not edge["path"].startswith("src/core/"):
            continue
        target = edge["resolved"] or posixpath.normpath(
            posixpath.join(posixpath.dirname(edge["path"]), edge["include"]))
        if target.startswith("src/models/") or target.startswith("src/families/"):
            violations.append(f"{edge['path']}:{edge['line']}: {edge['include']}")
    return sorted(violations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--mode", choices=["report"], default="report")
    parser.add_argument("--json-report", type=Path, help="Write complete deterministic inventory")
    parser.add_argument("--details", action="store_true", help="Print the full JSON inventory")
    parser.add_argument("--fail-on-core-model-includes", action="store_true",
                        help="Fail if any src/core file includes a model or family header")
    args = parser.parse_args()
    report = inventory(args.root.resolve())
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.details:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("[check-extension-isolation] REPORT ONLY (coupling does not fail this check)")
        for key, value in report["summary"].items():
            print(f"  {key}: {value}")
        for key, count in report["cross_owner_counts"].items():
            print(f"  {key}: {count}")
        for limitation in report["limitations"]:
            print(f"  Note: {limitation}")
    if args.fail_on_core_model_includes:
        violations = core_boundary_violations(report)
        if violations:
            print("[check-extension-isolation] CORE BOUNDARY VIOLATED "
                  "(src/core must not include model or family headers):")
            for violation in violations:
                print(f"  {violation}")
            return 1
        print("[check-extension-isolation] PASS: core boundary "
              "(no src/core -> src/models|src/families includes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
