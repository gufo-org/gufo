#!/usr/bin/env python3
"""Generate or check an extension's models.lock.json.

An extension's extension.json declares each model's components: a pinned
Hugging Face revision plus include/exclude globs. This tool resolves those
globs against the repository tree at that revision and records every file's
path, size and SHA-256 in models.lock.json next to extension.json. `gufo model
install` trusts only the lock.

    python3 tools/models/lock.py src/models/<extension>      # write the lock
    python3 tools/models/lock.py --check src/models/*        # offline check

The check needs no network: it verifies that each lock records the same
sources, globs and components as its extension.json, and that every locked
path matches the globs. Large files are hashed by Hugging Face (the LFS
object ID); small files are downloaded and hashed here. Set HF_TOKEN for gated
repositories; HF_ENDPOINT selects a mirror.
"""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

LOCK_NAME = "models.lock.json"
REVISION = re.compile(r"^[0-9a-f]{40}$")


def endpoint():
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def token():
    value = os.environ.get("HF_TOKEN", "")
    if value:
        return value
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = home / "token"
    return path.read_text().strip() if path.is_file() else ""


def request(url):
    headers = {"User-Agent": "gufo-model-lock"}
    if token():
        headers["Authorization"] = "Bearer " + token()
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def tree(repo, revision):
    """Every file of `repo` at `revision`: {path: entry}."""
    url = (f"{endpoint()}/api/models/{repo}/tree/{revision}?recursive=true")
    files = {}
    while url:
        with request(url) as response:
            for entry in json.load(response):
                if entry.get("type") == "file":
                    files[entry["path"]] = entry
            url = None
            for part in response.headers.get("Link", "").split(","):
                match = re.match(r'\s*<([^>]+)>;\s*rel="next"', part)
                if match:
                    url = match.group(1)
    return files


def matches(path, patterns):
    # Hugging Face semantics: fnmatch, where * also matches "/", and a
    # trailing "/" selects a whole directory.
    return any(fnmatch.fnmatchcase(path, p + "*" if p.endswith("/") else p) for p in patterns)


def selected(paths, include, exclude):
    return sorted(p for p in paths if (not include or matches(p, include)) and not matches(p, exclude))


def spec(component):
    """The fields of a component that its lock entry must reproduce."""
    return {
        "source": component["source"],
        "include": component.get("include", []),
        "exclude": component.get("exclude", []),
    }


def sha256_of(repo, revision, path):
    quoted = urllib.parse.quote(path)
    digest = hashlib.sha256()
    with request(f"{endpoint()}/{repo}/resolve/{revision}/{quoted}") as response:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def lock(extension):
    manifest = json.loads((extension / "extension.json").read_text())
    trees = {}
    result = {"schema": 1, "models": {}}
    for model in manifest.get("models", []):
        components = {}
        for component in model["components"]:
            source = component["source"]
            repo, revision = source["hf"], source["revision"]
            if not REVISION.match(revision):
                sys.exit(f"{model['id']}/{component['role']}: revision must be a 40-hex commit")
            if (repo, revision) not in trees:
                trees[(repo, revision)] = tree(repo, revision)
            files = trees[(repo, revision)]
            paths = selected(files, component.get("include", []), component.get("exclude", []))
            if not paths:
                sys.exit(f"{model['id']}/{component['role']}: no file of {repo}@{revision} matches")
            entries = []
            for path in paths:
                entry = files[path]
                lfs = entry.get("lfs")
                digest = lfs["oid"] if lfs else sha256_of(repo, revision, path)
                entries.append({"path": path, "size": entry["size"], "sha256": digest})
            components[component["role"]] = {**spec(component), "files": entries}
        result["models"][model["id"]] = components
    path = extension / LOCK_NAME
    path.write_text(json.dumps(result, indent=2) + "\n")
    total = sum(f["size"] for c in result["models"].values() for e in c.values() for f in e["files"])
    print(f"wrote {path}: {len(result['models'])} models, {total / 1e9:.2f} GB")


def check(extension):
    """Problems with the extension's lock, without network access."""
    manifest = json.loads((extension / "extension.json").read_text())
    models = manifest.get("models", [])
    path = extension / LOCK_NAME
    if not models:
        return [f"{path}: no models in extension.json"] if path.exists() else []
    if not path.exists():
        return [f"{path}: missing; run tools/models/lock.py {extension}"]
    locked = json.loads(path.read_text()).get("models", {})
    problems = []
    if sorted(locked) != sorted(m["id"] for m in models):
        problems.append(f"{path}: models differ from extension.json")
    for model in models:
        entries = locked.get(model["id"], {})
        if sorted(entries) != sorted(c["role"] for c in model["components"]):
            problems.append(f"{path}: {model['id']} components differ from extension.json")
        for component in model["components"]:
            entry = entries.get(component["role"])
            if entry is None:
                continue
            name = f"{model['id']}/{component['role']}"
            if {k: entry.get(k) for k in spec(component)} != spec(component):
                problems.append(f"{path}: {name} is stale; run tools/models/lock.py {extension}")
                continue
            paths = [f["path"] for f in entry.get("files", [])]
            if not paths or selected(paths, entry["include"], entry["exclude"]) != sorted(paths):
                problems.append(f"{path}: {name} lists files its globs do not select")
            for item in entry.get("files", []):
                if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))):
                    problems.append(f"{path}: {name} {item.get('path')} has no SHA-256")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check locks offline")
    parser.add_argument("extensions", nargs="+", type=Path)
    args = parser.parse_args()
    if not args.check:
        for extension in args.extensions:
            lock(extension)
        return 0
    problems = [p for e in args.extensions for p in check(e)]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
