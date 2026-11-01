#!/usr/bin/env python3
"""Report extension coupling; enforce the migrated core boundary.

This is a lexical inventory, not a C++ name resolver. Includes and qualified
model/family namespace references are separate evidence. Conditional includes
are inventoried even when disabled. JSON retains every location for later
ratchets; no baseline or exception silently hides current dependencies.

Enforcement ratchet (one boundary per migrated layer): with
--fail-on-core-model-includes, no file under src/core/** may include a header
under src/models/** or src/families/**. Card 06 (RFC step 4) retired the last
such include, so a new violation fails the check. --fail-on-serving-boundary
audits the Card 09 (RFC step 6) split: no src/cli/** include from production
sources under src/models/**, src/families/**, src/cache/**, src/ext/** or
include/gufo/ext/**; sources under a tests/ tree stay exempt until step 7b.
--fail-on-model-core-includes audits the Card 11 (RFC step 7) migration:
no production file under src/models/** may include src/core/** or src/cache/**.
--fail-on-family-graph audits the Card 07 family manifests. --fail-on-target-ownership audits the Card 08
(RFC step 5) target graph: every model directory needs a manifest, and every
target that links, compiles or includes a library must be claimed by it, so
excluding a library cannot leave an unregistered test or tool behind. Other
layers stay report only until their cards migrate them.
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
# The RFC step-1 header inventory followed the Qwen code when Card 07 extracted
# the family libraries, so it now covers both trees.
QWEN_PREFIXES = ("src/models/qwen/", "src/families/qwen/")
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
    for header in sorted(path for path in texts if path.startswith(QWEN_PREFIXES)
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
    libraries, edges, family_violations = family_analysis(root, graph)
    return {"mode": "report", "schema_version": 1, "models": models, "families": families,
            "limitations": ["Lexical namespace references are candidates, not compiler-resolved symbols.",
                            "Includes in all conditional branches are counted; macro includes are not resolved.",
                            "Header closure covers literal repository-local includes, not compiler search paths.",
                            "Directories named vendor and Python bytecode caches are excluded."],
            "summary": {"files": len(texts), "includes": len(includes), "cross_owner_includes": len(cross),
                        "namespace_references": len(symbols), "name_references": len(name_references),
                        "configuration_mentions": len(model_mentions),
                        "qwen_headers": len(qwen_headers),
                        "family_libraries": len(libraries), "family_edges": len(edges)},
            "family_libraries": libraries, "family_edges": edges,
            "family_violations": family_violations,
            "cross_owner_counts": dict(sorted(Counter(f'{edge["owner"]} -> {edge["dependency"]}'
                                                       for edge in cross).items())),
            "includes": includes, "namespace_references": symbols,
            "name_references": name_references, "configuration_mentions": model_mentions,
            "qwen_headers": qwen_headers}


def family_library(path):
    """Return 'family/library' for a path under src/families/, else None."""
    parts = Path(path).parts
    if len(parts) >= 4 and parts[0] == "src" and parts[1] == "families" and (len(parts) >= 5 or Path(path).suffix in CODE_SUFFIXES):
        return f"{parts[2]}/{parts[3]}"
    return None


def family_analysis(root, graph):
    """Read family manifests and audit them against the real include graph.

    Returns (libraries, edges, violations). A library is a directory under
    src/families/<family>/ that owns a library.json manifest; the manifest's
    `depends` must name exactly the sibling libraries whose files it includes.
    """
    libraries = {}
    violations = []
    for path in sorted(p for p in graph if family_library(p) and p.endswith("/library.json")):
        name = family_library(path)
        try:
            manifest = json.loads((root / path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            violations.append(f"{path}: unreadable library manifest ({error})")
            continue
        declared = manifest.get("name")
        target = manifest.get("target")
        kind = manifest.get("kind")
        if declared != name.partition("/")[2]:
            violations.append(f"{path}: manifest name {declared!r} != directory {name}")
        if not isinstance(target, str) or not target:
            violations.append(f"{path}: manifest needs a non-empty CMake target")
        if kind not in ("interface", "static"):
            violations.append(f"{path}: kind must be interface or static, got {kind!r}")
        if not manifest.get("purpose"):
            violations.append(f"{path}: manifest needs a purpose")
        depends = manifest.get("depends")
        if not isinstance(depends, list) or any(not isinstance(d, str) for d in depends or ()):
            violations.append(f"{path}: depends must be a list of library names")
            depends = []
        libraries[name] = {"target": target, "kind": kind,
                           "purpose": manifest.get("purpose", ""),
                           "depends": sorted(set(depends))}
        for dependency in depends:
            if dependency == name.split("/")[1]:
                violations.append(f"{path}: library depends on itself")

    edges = set()
    for source, targets in graph.items():
        library = family_library(source)
        if library is None:
            continue
        if library not in libraries:
            violations.append(f"{source}: file in src/families is not inside a library "
                              "with a library.json manifest")
        for target in targets:
            other = family_library(target)
            if other and other != library:
                edges.add(f"{library} -> {other}")
            if target.startswith("src/models/"):
                violations.append(f"{source}: family code includes model header {target}")

    for name, library in sorted(libraries.items()):
        family, _, _ = name.partition("/")
        declared = {f"{family}/{dependency}" for dependency in library["depends"]}
        real = {edge.split(" -> ")[1] for edge in edges if edge.startswith(f"{name} -> ")}
        for missing in sorted(real - declared):
            violations.append(f"src/families/{name}/library.json: undeclared dependency "
                              f"{missing} (headers include it)")
        for unused in sorted(declared - real):
            violations.append(f"src/families/{name}/library.json: declared dependency "
                              f"{unused} is not used by any header")
        for dependency in library["depends"]:
            if f"{family}/{dependency}" not in libraries:
                violations.append(f"src/families/{name}/library.json: unknown dependency "
                                  f"{dependency}")

    violations.extend(sorted(cycle_violations(libraries)))
    return libraries, sorted(edges), sorted(set(violations))


def cycle_violations(libraries):
    """Report a dependency cycle through the declared family graph, if any."""
    active, done = [], set()

    def visit(name):
        if name in active:
            return active[active.index(name):] + [name]
        if name in done:
            return None
        active.append(name)
        family = name.partition("/")[0]
        for dependency in sorted(libraries[name]["depends"]):
            other = f"{family}/{dependency}"
            if other in libraries:
                cycle = visit(other)
                if cycle:
                    return cycle
        active.pop()
        done.add(name)
        return None

    for name in sorted(libraries):
        cycle = visit(name)
        if cycle:
            return ["family dependency cycle: " + " -> ".join(cycle)]
    return []


# --- target ownership -------------------------------------------------------
#
# Card 08 (RFC step 5) gave the model, comparison and eval tiers their own
# targets. A target depends on a library when it links the library target,
# compiles a file from the library directory, or includes a header from it.
# Every such target outside the library's own directory must be named by that
# library's library.json, so dropping a library cannot leave an unregistered
# test or tool behind. Targets declared under tests/models/<model>/ stay with
# the model test tree (CARDS.md leaves that placement open) and are exempt.

CMAKE_COMMAND = re.compile(r"\b(add_executable|add_library|target_link_libraries|set)\s*\(")
LINK_QUALIFIERS = {"PRIVATE", "PUBLIC", "INTERFACE"}
TARGET_ATTRIBUTES = {"STATIC", "SHARED", "MODULE", "OBJECT", "IMPORTED", "ALIAS",
                     "EXCLUDE_FROM_ALL", "INTERFACE", "WIN32"}
SOURCE_SUFFIXES = {".cpp", ".cc", ".c", ".hip", ".cu"}
SOURCE_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def cmake_body(text, start):
    """Return (body, index after the closing paren) for the paren at `start`."""
    depth, index, end = 0, start, len(text)
    while index < end:
        char = text[index]
        if char == '"':
            index += 1
            while index < end and text[index] != '"':
                index += 2 if text[index] == "\\" else 1
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
        index += 1
    return text[start + 1:], end


def cmake_arguments(body):
    """Split a CMake argument list on whitespace outside quotes and parentheses."""
    args, current, depth, index, end = [], [], 0, 0, len(body)

    def flush():
        token = "".join(current).strip()
        if token:
            args.append(token.strip('"'))
        current.clear()

    while index < end:
        char = body[index]
        if char == '"':
            closing = index + 1
            while closing < end and body[closing] != '"':
                closing += 2 if body[closing] == "\\" else 1
            current.append(body[index:closing + 1])
            index = closing + 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char in " \t\n;" and depth == 0:
            flush()
        else:
            current.append(char)
        index += 1
    flush()
    return args


def cmake_invocations(text):
    """Yield (command, arguments) for the CMake commands this audit needs."""
    text = strip_comments(text, ".cmake")
    index = 0
    while True:
        match = CMAKE_COMMAND.search(text, index)
        if not match:
            return
        body, index = cmake_body(text, text.index("(", match.start()))
        arguments = cmake_arguments(body)
        if arguments:
            yield match.group(1), arguments


def cmake_tables(root):
    """Parse the project CMake files into variables and target tables.

    Returns (variables, targets) where targets maps a target name to its
    creating file, its literal source files and its linked names. Generator
    expressions and unresolved variables are dropped: a target that only a
    toolchain option builds is still inventoried through the manifest claim.
    """
    variables, targets = {}, {}
    for path in source_paths(root):
        if Path(path).name != "CMakeLists.txt":
            continue
        declaring = Path(path).parent.as_posix()
        declaring = "" if declaring == "." else declaring
        text = (root / path).read_text(encoding="utf-8")
        for command, arguments in cmake_invocations(text):
            name = arguments[0]
            if command == "set":
                values, skip = [], 0
                for item in arguments[1:]:
                    if skip:
                        skip -= 1
                    elif item.upper() == "CACHE":
                        skip = 2
                    else:
                        values.append(item)
                variables.setdefault(name, values)
            elif command in ("add_executable", "add_library"):
                entry = targets.setdefault(name, {"file": path, "files": [],
                                                  "sources": [], "links": []})
                entry["file"] = path
                entry["files"].append(path)
                for argument in arguments[1:]:
                    if argument.upper() in TARGET_ATTRIBUTES:
                        continue
                    entry["sources"].extend(
                        source for source in expand_cmake(argument, variables, declaring)
                        if Path(source).suffix in SOURCE_SUFFIXES)
            elif command == "target_link_libraries":
                entry = targets.setdefault(name, {"file": path, "files": [],
                                                  "sources": [], "links": []})
                entry["files"].append(path)
                for argument in arguments[1:]:
                    if argument.upper() in LINK_QUALIFIERS:
                        continue
                    entry["links"].extend(
                        expand_cmake(argument, variables, declaring, as_path=False))
    return variables, targets


def expand_cmake(argument, variables, declaring, as_path=True):
    """Resolve one CMake argument to repository-relative paths, or to names.

    Source arguments are paths and become repository-relative; link arguments
    are target names and stay as written once their variables are expanded.
    """
    values, pending = [], [argument]
    for _ in range(4):
        resolved = []
        for value in pending:
            if "${" not in value:
                resolved.append(value)
                continue
            for name in SOURCE_VARIABLE.findall(value):
                for replacement in variables.get(name, ["${" + name + "}"]):
                    resolved.append(value.replace("${%s}" % name, replacement))
        pending = [value for value in resolved if "${" in value]
        values.extend(value for value in resolved if "${" not in value)
        if not pending:
            break
    # Sources spelled with a built-in directory variable never lose their `${}`
    # above, and that spelling is the common way to compile a file that lives in
    # another directory. Keep them so the resolution below can turn them into
    # paths; an unknown user variable still falls through unresolved. Link names
    # keep the previous behaviour: an unresolved name is not a target.
    if as_path:
        values.extend(pending)
    paths = []
    for value in values:
        for token in value.split(";"):
            for item in token.split():
                if item.startswith("$<"):
                    continue
                if not as_path:
                    paths.append(item)
                    continue
                if item.startswith("${CMAKE_CURRENT_SOURCE_DIR}/"):
                    item = item[len("${CMAKE_CURRENT_SOURCE_DIR}/"):]
                    item = posixpath.normpath(posixpath.join(declaring, item))
                elif item.startswith(("${PROJECT_SOURCE_DIR}/", "${CMAKE_SOURCE_DIR}/")):
                    item = posixpath.normpath(item.split("}/", 1)[1])
                elif "$" in item:
                    continue
                elif not posixpath.isabs(item):
                    item = posixpath.normpath(posixpath.join(declaring, item))
                paths.append(item)
    return paths


def library_manifests(root):
    """Read every library.json under src/, keyed by its CMake target."""
    libraries = {}
    violations = []
    for path in sorted(p.relative_to(root).as_posix()
                       for p in (root / "src").rglob("library.json")):
        try:
            manifest = json.loads((root / path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            violations.append(f"{path}: unreadable library manifest ({error})")
            continue
        directory = Path(path).parent.as_posix()
        name, target, kind = manifest.get("name"), manifest.get("target"), manifest.get("kind")
        if not isinstance(name, str) or not name:
            violations.append(f"{path}: manifest needs a non-empty library name")
            continue
        if not isinstance(target, str) or not target:
            violations.append(f"{path}: manifest needs a non-empty CMake target")
            continue
        if kind not in ("interface", "static"):
            violations.append(f"{path}: kind must be interface or static, got {kind!r}")
        if not manifest.get("purpose"):
            violations.append(f"{path}: manifest needs a purpose")
        if target in libraries:
            violations.append(f"{path}: target {target} is already declared by "
                              f"{libraries[target]['manifest']}")
        claims = {}
        for field in ("consumers", "checks", "tools"):
            entries = manifest.get(field, [])
            if not isinstance(entries, list) or any(not isinstance(e, str) for e in entries):
                violations.append(f"{path}: {field} must be a list of target names")
                entries = []
            for entry in entries:
                claims[entry] = field
        compiled = []
        for entry in manifest.get("compiled_elsewhere", []):
            if not isinstance(entry, dict) or not entry.get("target") or not entry.get("source") \
                    or not entry.get("reason"):
                violations.append(f"{path}: compiled_elsewhere entries need target, "
                                  "source and reason")
                continue
            compiled.append((entry["target"], entry["source"]))
            claims[entry["target"]] = "compiled_elsewhere"
        libraries[target] = {"manifest": path, "name": name, "directory": directory,
                             "family": directory.startswith("src/families/"),
                             "kind": kind, "claims": claims, "compiled_elsewhere": compiled}
    names = Counter(entry["name"] for entry in libraries.values())
    for duplicate in sorted(name for name, count in names.items() if count > 1):
        violations.append(f"library name {duplicate} is declared by "
                          f"{count} manifests (names are repository-wide)")
    return libraries, violations


def included_prefixes(root, source, cache):
    """Return the `src/...` header directories a source includes, cached per file."""
    if source in cache:
        return cache[source]
    found = set()
    path = root / source
    if path.is_file():
        text = strip_comments(path.read_text(encoding="utf-8"), Path(source).suffix)
        for match in INCLUDE.finditer(text):
            target = match[1]
            if target.startswith("src/"):
                found.add(posixpath.dirname(target))
    return cache.setdefault(source, found)


def target_ownership(root):
    """Audit library.json ownership against the real CMake target graph.

    Returns (libraries, edges, violations). A target depends on a library when
    it links the library target, compiles a file below the library directory,
    or includes a header from it. Every such target declared outside that
    directory must be claimed by the manifest, except targets declared under
    tests/models/<model>/, which stay with the model test tree.
    """
    _, targets = cmake_tables(root)
    libraries, violations = library_manifests(root)
    for directory in sorted(p.relative_to(root).as_posix()
                            for p in (root / "src/models").iterdir() if p.is_dir()):
        if not any(library["directory"] == directory for library in libraries.values()):
            violations.append(f"{directory}/library.json: model directory without a manifest")

    includes, edges, dependents = {}, set(), set()
    for target, info in sorted(targets.items()):
        outside_test_trees = [path for path in info["files"]
                              if not Path(path).parent.as_posix().startswith("tests/models/")]
        if "$" in target:
            if outside_test_trees:
                violations.append(f"{outside_test_trees[0]}: {target} is registered through a "
                                  "helper function, so the ownership audit cannot attribute it")
            continue
        declaring = Path(info["file"]).parent.as_posix()
        sources = sorted(set(info["sources"]))
        for name, library in sorted(libraries.items()):
            if library["family"] or name == target:
                continue
            directory = library["directory"]
            if declaring == directory or declaring.startswith(directory + "/"):
                continue
            reasons = []
            if name in info["links"]:
                reasons.append(f"links {name}")
            compiled = [source for source in sources if source.startswith(directory + "/")]
            reasons.extend(f"compiles {source}" for source in compiled)
            if not compiled and any(direct == directory or direct.startswith(directory + "/")
                                    for source in sources
                                    for direct in included_prefixes(root, source, includes)):
                reasons.append(f"includes {directory}")
            if not reasons:
                continue
            edges.update(f"{library['name']} <- {target} ({reason})"
                         for reason in dict.fromkeys(reasons))
            dependents.add((name, target))
            claim = library["claims"].get(target)
            if claim is None and not declaring.startswith("tests/models/"):
                violations.append(f"{info['file']}: {target} depends on {library['name']} "
                                  f"({'; '.join(reasons[:2])}) but {library['manifest']} "
                                  "does not claim it")
            recorded = {source for entry, source in library["compiled_elsewhere"]
                        if entry == target}
            for source in compiled:
                if source not in recorded:
                    violations.append(f"{info['file']}: {target} compiles {source} outside "
                                      f"{directory}/ without a compiled_elsewhere entry in "
                                      f"{library['manifest']}")

    for name, library in sorted(libraries.items(), key=lambda entry: entry[1]["manifest"]):
        if library["family"]:
            continue
        for claim in sorted(library["claims"]):
            if claim not in targets:
                violations.append(f"{library['manifest']}: claimed target {claim} does not exist")
            elif (name, claim) not in dependents:
                violations.append(f"{library['manifest']}: claimed target {claim} "
                                  f"does not depend on {library['name']}")
        for entry, source in library["compiled_elsewhere"]:
            if source not in set(targets.get(entry, {}).get("sources", ())):
                violations.append(f"{library['manifest']}: compiled_elsewhere names {entry} "
                                  f"for {source}, which it does not compile")
    return libraries, sorted(edges), sorted(set(violations))


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


# Layers that reach execution only through neutral contracts (RFC step 6).
BOUNDARY_LAYERS = ("src/models/", "src/families/", "src/cache/", "src/ext/",
                   "include/gufo/ext/")


def in_test_tree(path):
    """True for a source below a `tests/` (or `test/`) directory."""
    return any(part in ("tests", "test") for part in path.split("/")[:-1])


def serving_boundary_violations(report):
    """Return model/runtime/cache/API includes of a src/cli/** header.

    Card 09 separated serving policy from extension execution: models, family
    libraries, the cache tier, the extension runtime and the public extension
    API must compile without a serving or HTTP implementation header. Card 12
    (RFC step 7b) introduced gufo_test_support, eliminating the test-tree
    exemption: all sources under BOUNDARY_LAYERS (including model tests under
    src/models/) must compile without a serving or HTTP implementation header.
    """
    violations = []
    for edge in report["includes"]:
        path = edge["path"]
        if not path.startswith(BOUNDARY_LAYERS):
            continue
        target = edge["resolved"] or posixpath.normpath(
            posixpath.join(posixpath.dirname(path), edge["include"]))
        if target.startswith("src/cli/"):
            violations.append(f"{path}:{edge['line']}: {edge['include']}")
    return sorted(violations)


def model_core_boundary_violations(report):
    """Return production src/models/** includes of src/core/** or src/cache/** headers.

    Card 11 (RFC step 7) migrated all eight model implementations to public
    extension headers under <gufo/ext/...>. Production sources under
    src/models/** must not include src/core/** or src/cache/**.
    """
    violations = []
    for edge in report["includes"]:
        path = edge["path"]
        if not path.startswith("src/models/") or in_test_tree(path):
            continue
        inc = edge["include"]
        target = edge["resolved"] or posixpath.normpath(
            posixpath.join(posixpath.dirname(path), inc))
        if (target.startswith("src/core/") or target.startswith("src/cache/")
                or inc.startswith(("src/core/", "src/cache/"))):
            violations.append(f"{path}:{edge['line']}: {inc}")
    return sorted(violations)


MIGRATED_EXTENSIONS = (
    "src/models/deepseek_v4_flash/",
    "src/models/embedding_gemma2/",
    "src/models/minimax_h3/",
    "src/models/qwen/",
    "src/models/qwen3_asr/",
    "src/models/qwen3_tts/",
    "src/models/qwen38_flash_next/",
    "src/models/qwen_image_21/",
)


def extension_boundary_violations(report):
    """Return src/cli/** includes of migrated model extensions.

    Cards 13 and 15–18 migrated EmbeddingGemma 2, Qwen3-ASR, Qwen3-TTS,
    Qwen-Image-2.1, and MiniMax H3 to the neutral extension runtime.
    The CLI and serving layer must not include headers from migrated extensions,
    interacting only via ext::Registry and neutral extension interfaces.
    """
    violations = []
    for edge in report["includes"]:
        path = edge["path"]
        if not path.startswith("src/cli/"):
            continue
        inc = edge["include"]
        target = edge["resolved"] or posixpath.normpath(
            posixpath.join(posixpath.dirname(path), inc))
        if (target.startswith(MIGRATED_EXTENSIONS)
                or inc.startswith(MIGRATED_EXTENSIONS)):
            violations.append(f"{path}:{edge['line']}: {inc}")
    return sorted(violations)


def model_name_from_path(owner):
    """Extract model directory name from an owner string."""
    if owner.startswith("models/"):
        return owner[len("models/"):]
    if owner.startswith("tests/models/"):
        sub = owner[len("tests/models/"):]
        if sub.startswith("minimax_h3"):
            return "minimax_h3"
        if "/" in sub:
            return sub.split("/")[0]
        return sub
    return None


def classify_edge(owner, dep, root=None):
    """Classify an include edge between owners.

    Returns (status, icon, reason) where status is:
      - 'FAIL': An active architectural boundary violation that violates a CI ratchet.
      - 'WARN': Legacy coupling or cross-model coupling tracked for RFC migration.
      - 'OK':   An architecturally compliant and expected dependency.
    """
    # Detect migrated extensions with manifests and build files
    migrated = set()
    if root is not None:
        try:
            for manifest in (Path(root) / "src/models").glob("*/extension.json"):
                if (manifest.parent / "extension.cmake").exists():
                    migrated.add(manifest.parent.name)
        except Exception:
            pass
    if not migrated:
        migrated.add("embedding_gemma2")

    m_owner = model_name_from_path(owner)
    m_dep = model_name_from_path(dep)

    # 1. STRICT BOUNDARY VIOLATIONS
    if owner == "src/core" and (dep.startswith("models/") or dep.startswith("families/")):
        return ("FAIL", "❌", "core boundary violation (core includes model/family)")
    if (owner.startswith(("models/", "families/", "src/cache", "src/ext")) or owner.startswith("include/gufo/ext")) and dep == "src/cli":
        return ("FAIL", "❌", "serving boundary violation (engine/extension includes CLI)")
    if owner.startswith("models/") and dep in {"src/core", "src/cache"}:
        return ("FAIL", "❌", "model core boundary violation (production model includes core/cache)")
    if owner == "src/cli" and m_dep in migrated:
        return ("FAIL", "❌", f"extension boundary violation (CLI includes migrated extension {m_dep})")

    # 2. LEGACY COUPLING / PENDING RFC MIGRATION
    if owner == "src/cli" and dep.startswith("models/"):
        return ("WARN", "⚠️ ", "legacy model in CLI (pending RFC migration)")
    if owner.startswith("models/") and dep.startswith("models/") and m_owner != m_dep:
        return ("WARN", "⚠️ ", "cross-model include (unisolated model boundary)")
    if owner.startswith("tests/models/") and dep.startswith("models/") and m_owner != m_dep:
        if m_owner == "qwen27b" and m_dep == "qwen":
            return ("WARN", "⚠️ ", "cross-model test coupling (qwen27b tests reuse base qwen)")
        return ("WARN", "⚠️ ", f"cross-model test coupling ({m_owner} tests include {m_dep})")
    if owner.startswith("tests/models/") and dep.startswith("tests/models/") and m_owner != m_dep:
        return ("WARN", "⚠️ ", f"cross-model test coupling ({owner} includes {dep})")
    if owner == "tests/cli" and dep.startswith("models/"):
        return ("WARN", "⚠️ ", "CLI test depends directly on unmigrated model")
    if owner == "tests/support" and dep == "src/cli":
        return ("WARN", "⚠️ ", "test support includes CLI headers")
    if owner == "src/core" and dep == "src/cache":
        return ("WARN", "⚠️ ", "core depends on cache")

    # 3. EXPECTED / COMPLIANT
    if m_owner and m_dep and m_owner == m_dep:
        return ("OK", "✅", "test -> own model")
    if owner.startswith("models/") and dep.startswith("families/"):
        return ("OK", "✅", "model -> shared family library")
    if owner.startswith("tests/") and dep in {"tests/support", "tests/fixtures", "tests/testing"}:
        return ("OK", "✅", "test -> test support")
    if owner.startswith("tests/") and "golden_helpers" in dep:
        return ("OK", "✅", "test -> shared golden helper")
    if owner == "src/cli" and dep in {"src/core", "src/cache", "src/ext", "src/eval", "src/testing", "configuration", "families/qwen"}:
        return ("OK", "✅", "serving -> engine/infra")
    if owner in {"src/cache", "src/ext", "src/eval"} and dep in {"src/core", "configuration"}:
        return ("OK", "✅", "engine layer -> core")
    if owner.startswith("tests/") and dep in {"src/core", "src/cache", "src/cli", "src/ext", "src/eval", "src/testing", "families/qwen"}:
        return ("OK", "✅", "test -> engine component")
    if owner == "configuration" or dep == "configuration":
        return ("OK", "✅", "build/manifest reference")
    if owner.startswith("tools/"):
        return ("OK", "✅", "tooling dependency")
    if owner.startswith("families/") and dep == "src/core":
        return ("OK", "✅", "family library -> core")

    return ("OK", "✅", "internal layer dependency")


def print_cross_owner_report(cross_owner_counts, root=None):
    """Print classified, grouped cross-owner include mappings with status emojis."""
    by_status = {"FAIL": [], "WARN": [], "OK": []}
    for mapping, count in cross_owner_counts.items():
        owner, dep = mapping.split(" -> ", 1)
        status, icon, reason = classify_edge(owner, dep, root)
        by_status[status].append((icon, status, mapping, count, reason))

    print("\n[check-extension-isolation] Cross-Layer Include Mappings:")
    if by_status["FAIL"]:
        print(f"  ─── Strict Boundary Violations ({len(by_status['FAIL'])}) " + "─" * 40)
        for icon, status, mapping, count, reason in by_status["FAIL"]:
            print(f"  {icon} [{status}] {mapping}: {count} ({reason})")
    else:
        print("  ─── Strict Boundary Violations (0) ──────────────────────────────────")
        print("  ✅ [OK] No active boundary violations.")

    print(f"\n  ─── Pending RFC Migrations / Legacy Coupling ({len(by_status['WARN'])}) " + "─" * 28)
    if by_status["WARN"]:
        for icon, status, mapping, count, reason in by_status["WARN"]:
            print(f"  {icon} [{status}] {mapping}: {count} ({reason})")
    else:
        print("  ✅ [OK] Zero legacy couplings remaining.")

    print(f"\n  ─── Expected & Compliant Dependencies ({len(by_status['OK'])}) " + "─" * 33)
    for icon, status, mapping, count, reason in by_status["OK"]:
        print(f"  {icon} [{status}] {mapping}: {count} ({reason})")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--mode", choices=["report"], default="report")
    parser.add_argument("--json-report", type=Path, help="Write complete deterministic inventory")
    parser.add_argument("--details", action="store_true", help="Print the full JSON inventory")
    parser.add_argument("--legacy-counts", action="store_true",
                        help="Print legacy unclassified flat cross-owner counts")
    parser.add_argument("--fail-on-core-model-includes", action="store_true",
                        help="Fail if any src/core file includes a model or family header")
    parser.add_argument("--fail-on-family-graph", action="store_true",
                        help="Fail unless every family library is declared, assigned, "
                             "acyclic and free of model includes")
    parser.add_argument("--fail-on-serving-boundary", action="store_true",
                        help="Fail if a model, family, cache, extension-runtime or "
                             "public-API file includes a src/cli serving or HTTP header")
    parser.add_argument("--fail-on-model-core-includes", action="store_true",
                        help="Fail if any production model file includes a core or cache header")
    parser.add_argument("--fail-on-extension-boundary", "--fail-on-cli-model-includes",
                        dest="fail_on_extension_boundary", action="store_true",
                        help="Fail if any src/cli file includes a migrated extension (src/models/{embedding_gemma2,minimax_h3,qwen3_asr,qwen3_tts,qwen_image_21})")
    parser.add_argument("--fail-on-target-ownership", action="store_true",
                        help="Fail unless every model directory has a manifest and every "
                             "target that links, compiles or includes a library claims it")
    parser.add_argument("--target-ownership-report", action="store_true",
                        help="Print the library/target ownership graph")
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
        if args.legacy_counts:
            for key, count in report["cross_owner_counts"].items():
                print(f"  {key}: {count}")
        else:
            print_cross_owner_report(report["cross_owner_counts"], args.root.resolve())
        for violation in report["family_violations"]:
            print(f"  family: {violation}")
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
    if args.fail_on_serving_boundary:
        violations = serving_boundary_violations(report)
        if violations:
            print("[check-extension-isolation] SERVING BOUNDARY VIOLATED "
                  "(models, families, cache, src/ext and the public API must "
                  "not include a src/cli header):")
            for violation in violations:
                print(f"  {violation}")
            return 1
        print("[check-extension-isolation] PASS: serving boundary "
              "(no src/cli include from models, families, cache, src/ext or "
              "the public extension API)")
    if args.fail_on_model_core_includes:
        violations = model_core_boundary_violations(report)
        if violations:
            print("[check-extension-isolation] MODEL CORE BOUNDARY VIOLATED "
                  "(production models must not include src/core or src/cache headers):")
            for violation in violations:
                print(f"  {violation}")
            return 1
        print("[check-extension-isolation] PASS: model core boundary "
              "(no production src/models -> src/core|src/cache includes)")
    if args.fail_on_extension_boundary:
        violations = extension_boundary_violations(report)
        if violations:
            print("[check-extension-isolation] EXTENSION BOUNDARY VIOLATED "
                  "(src/cli must not include migrated extension headers):")
            for violation in violations:
                print(f"  {violation}")
            return 1
        print("[check-extension-isolation] PASS: extension boundary "
              "(no src/cli -> migrated extension includes)")
    if args.fail_on_target_ownership or args.target_ownership_report:
        libraries, edges, violations = target_ownership(args.root.resolve())
        if args.target_ownership_report:
            print(f"[check-extension-isolation] target ownership: {len(libraries)} libraries, "
                  f"{len(edges)} dependency edges")
            for edge in edges:
                print(f"  {edge}")
        for violation in violations:
            print(f"  ownership: {violation}")
        if args.fail_on_target_ownership:
            if violations:
                print("[check-extension-isolation] TARGET OWNERSHIP INVALID "
                      "(every model and support library must assign its dependent "
                      "tests, tools and consumers):")
                for violation in violations:
                    print(f"  {violation}")
                return 1
            print(f"[check-extension-isolation] PASS: target ownership "
                  f"({len(libraries)} libraries, {len(edges)} dependency edges)")
    if args.fail_on_family_graph:
        if report["family_violations"]:
            print("[check-extension-isolation] FAMILY GRAPH INVALID "
                  "(manifests must assign every header, match the real includes, "
                  "stay acyclic and never reach a model):")
            for violation in report["family_violations"]:
                print(f"  {violation}")
            return 1
        print(f"[check-extension-isolation] PASS: family graph "
              f"({len(report['family_libraries'])} libraries, "
              f"{len(report['family_edges'])} declared edges, acyclic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
