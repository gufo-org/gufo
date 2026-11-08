#!/usr/bin/env python3
"""Build a fixture extension as a module through the SDK and manage it with
`gufo extension`, against the actual executable and a private store."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

binary = sys.argv[1]
fixture = Path(__file__).resolve().parent / "module_fixture"


def gufo(*args, check=True):
    result = subprocess.run([binary, *args], text=True, capture_output=True,
                            env=env)
    if check and result.returncode != 0:
        raise AssertionError(f"gufo {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def tree_digest(root):
    """The digest gufo and the registry record for a source tree."""
    files = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = [d for d in subdirectories
                             if not (Path(directory) == root and d == ".git")]
        files += [(Path(directory) / n).relative_to(root) for n in names]
    hasher = hashlib.sha256()
    for relative in sorted(files, key=lambda p: p.parts):
        hasher.update(relative.as_posix().encode() + b"\0")
        hasher.update((root / relative).read_bytes())
    return "sha256:" + hasher.hexdigest()


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "-c", "core.autocrlf=false", *args], cwd=cwd, check=True,
                   capture_output=True)


def commit(message, cwd):
    git("commit", "--quiet", "-am", message, cwd=cwd)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def listing():
    return json.loads(gufo("extension", "list", "--json").stdout)["extensions"]


def installed_extensions():
    return [e for e in listing() if e["installed"]]


def extensions():
    return json.loads(gufo("diagnose", "--section", "extensions", "--json").stdout)


with tempfile.TemporaryDirectory() as temp:
    env = dict(os.environ, GUFO_EXTENSIONS_DIR=str(Path(temp) / "store"),
               GUFO_EXTENSION_REGISTRY=str(Path(temp) / "index.json"))
    (Path(temp) / "index.json").write_text('{"schema": 1, "extensions": {}}')
    assert all(e["builtin"] and not e["installed"] for e in listing())
    assert installed_extensions() == []

    refused = gufo("extension", "install", str(fixture), check=False)
    assert refused.returncode != 0 and "--yes" in refused.stderr, refused.stderr
    assert "has not been validated" in refused.stderr

    gufo("extension", "install", str(fixture), "--yes")
    [installed] = installed_extensions()
    assert installed["name"] == "module_fixture" and installed["status"] == "ready", installed
    assert installed["sourceKind"] == "path" and len(installed["digest"]) == 64

    report = extensions()
    [module] = [e for e in report["builtExtensions"] if e["name"] == "module_fixture"]
    assert module["source"].startswith("module:"), module
    [provider] = [p for p in report["providers"] if p["id"] == "module-fixture"]
    assert provider["available"] and provider["extension"] == "module_fixture", provider

    # The same extension again conflicts only with itself, which it replaces.
    copy = Path(temp) / "copy"
    shutil.copytree(fixture, copy)
    manifest = json.loads((copy / "extension.json").read_text())
    manifest["name"] = "module_fixture_copy"
    (copy / "extension.json").write_text(json.dumps(manifest))
    conflict = gufo("extension", "install", str(copy), "--yes", check=False)
    assert conflict.returncode != 0 and "provider module-fixture" in conflict.stderr, conflict.stderr

    # A module built for another gufo build is skipped until updated.
    modules = Path(temp) / "store" / "module_fixture" / "modules"
    [built] = list(modules.iterdir())
    built.rename(Path(temp) / built.name)
    assert installed_extensions()[0]["status"] == "needs update"
    stale = gufo("diagnose", "--section", "extensions", "--json")
    assert "gufo extension update module_fixture" in stale.stderr, stale.stderr
    assert all(e["name"] != "module_fixture" for e in json.loads(stale.stdout)["builtExtensions"])
    gufo("extension", "update", "module_fixture")
    assert "is up to date" in gufo("extension", "update").stdout
    assert installed_extensions()[0]["status"] == "ready"

    gufo("extension", "remove", "module_fixture")
    assert installed_extensions() == []
    assert not (Path(temp) / "store" / "module_fixture").exists()

    # Reviewed releases from a registry index: install by name with no
    # unvalidated-code prompt, refuse a tree that differs from its digest,
    # and move to a newer release on update.
    repository = Path(temp) / "repository"
    shutil.copytree(fixture, repository)
    git("init", "--quiet", cwd=repository)
    git("add", ".", cwd=repository)
    first = commit("0.1.0", cwd=repository)
    first_digest = tree_digest(repository)
    manifest = json.loads((repository / "extension.json").read_text())
    manifest["version"] = "0.2.0"
    (repository / "extension.json").write_text(json.dumps(manifest))
    second = commit("0.2.0", cwd=repository)
    second_digest = tree_digest(repository)
    host = report["extensionApiVersion"]
    platform = "windows" if os.name == "nt" else "linux"

    def release(version, revision, digest, api=">= 0.2.0"):
        return {"version": version, "revision": revision, "digest": digest,
                "api_version": api, "platforms": [platform]}

    def publish(*releases):
        (Path(temp) / "index.json").write_text(json.dumps({"schema": 1, "extensions": {
            "module_fixture": {"source": repository.as_uri(), "releases": list(releases)}}}))

    publish(release("0.1.0", first, "sha256:" + "0" * 64))
    tampered = gufo("extension", "install", "module_fixture", check=False)
    assert tampered.returncode != 0 and "does not match its reviewed release" in tampered.stderr, tampered.stderr
    assert installed_extensions() == []

    major, minor, _ = host.split(".")
    publish(release("0.2.0", second, second_digest, f">= {major}.{int(minor) + 1}.0"),
            release("0.1.0", first, first_digest))
    [available] = [e for e in listing() if e["name"] == "module_fixture"]
    assert available["newest"] == "0.1.0" and not available["installed"], available
    mistyped = gufo("extension", "install", "list", check=False)
    assert mistyped.returncode != 0 and "did you mean 'gufo extension list'" in mistyped.stderr
    gufo("extension", "install", "module_fixture")
    [installed] = installed_extensions()
    assert installed["sourceKind"] == "registry" and installed["revision"] == first, installed
    assert installed["version"] == "0.1.0" and installed["status"] == "ready", installed

    publish(release("0.2.0", second, second_digest), release("0.1.0", first, first_digest))
    [offered] = [e for e in listing() if e["name"] == "module_fixture"]
    assert offered["installed"] and offered["version"] == "0.1.0" and offered["newest"] == "0.2.0"
    assert "[installed, ready; 0.2.0 available]" in gufo("extension", "list").stdout
    assert "to 0.2.0" in gufo("extension", "update").stdout
    [installed] = installed_extensions()
    assert installed["version"] == "0.2.0" and installed["revision"] == second, installed
    gufo("extension", "remove", "module_fixture")
print("PASS: extension module install, load, conflict, update, remove and registry releases")
