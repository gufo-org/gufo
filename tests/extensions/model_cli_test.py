#!/usr/bin/env python3
"""Manage extension-provided model weights with `gufo model`, against the
actual executable, a private store and a local server that answers like
Hugging Face: /resolve redirects to a CDN host that serves byte ranges."""
import hashlib
import http.server
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading

binary = sys.argv[1]
REVISION = "0123456789abcdef0123456789abcdef01234567"
TOKEN = "hf_fixture_token"

# Two ranges of the downloader's 64 MiB chunks; the first breaks once.
FILES = {
    "weights/big.bin": random.Random(1).randbytes(70 << 20),
    "weights/config.json": b'{"fixture": true}\n',
    "extra.json": b'{"extra": true}\n',
    "gated.bin": b"gated\n",
}
BY_SHA = {hashlib.sha256(data).hexdigest(): data for data in FILES.values()}
requests = []  # (path, authorization)
broken = set()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        requests.append((self.path, self.headers.get("Authorization")))
        prefix = f"/owner/repo/resolve/{REVISION}/"
        if self.path.startswith(prefix):
            name = self.path[len(prefix):]
            if name not in FILES:
                return self.reply(404, b"")
            if name == "gated.bin" and self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self.reply(401, b"")
            sha = hashlib.sha256(FILES[name]).hexdigest()
            # Another host name: the token must not follow the redirect.
            port = self.server.server_address[1]
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{port}/cdn/{sha}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if self.path.startswith("/cdn/"):
            data = BY_SHA.get(self.path[5:])
            if data is None:
                return self.reply(404, b"")
            begin, end = 0, len(data) - 1
            if (value := self.headers.get("Range")):
                first, last = value.removeprefix("bytes=").split("-")
                begin, end = int(first), min(int(last), len(data) - 1)
            body = data[begin:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {begin}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if len(data) > 1 << 20 and begin == 0 and end > 1 << 20 and "big" not in broken:
                broken.add("big")
                self.wfile.write(body[:1 << 20])
                self.close_connection = True
                return None
            self.wfile.write(body)
            return None
        return self.reply(404, b"")

    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def component(role, option, install, include, entry=None):
    item = {"role": role, "option": option, "install": install,
            "source": {"hf": "owner/repo", "revision": REVISION}, "include": include}
    if entry:
        item["entry"] = entry
    return item


def lock_entry(item):
    paths = sorted(p for p in FILES if any(
        p == g or (g.endswith("*") and p.startswith(g[:-1])) for g in item["include"]))
    return {"source": item["source"], "include": item["include"], "exclude": [],
            "files": [{"path": p, "size": len(FILES[p]),
                       "sha256": hashlib.sha256(FILES[p]).hexdigest()} for p in paths]}


def extension(store, name, models, lock):
    source = store / name / "source"
    source.mkdir(parents=True)
    (source / "extension.json").write_text(json.dumps({"schema": 1, "name": name, "version": "1.0.0",
                                                       "models": models}))
    (source / "models.lock.json").write_text(json.dumps({"schema": 1, "models": lock}))
    return {"name": name, "version": "1.0.0", "digest": "",
            "source": {"kind": "path", "location": str(source), "revision": ""}}


def gufo(*args, check=True):
    result = subprocess.run([binary, *args], text=True, capture_output=True, env=env)
    if check and result.returncode != 0:
        raise AssertionError(f"gufo {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    weights = component("weights", "--model", "required", ["weights/*"], "weights/big.bin")
    extra = component("extra", "--extra-model", "optional", ["extra.json"])
    gated = component("weights", "--model", "required", ["gated.bin"])
    stale = component("weights", "--model", "required", ["extra.json"])
    extensions = root / "extensions"
    entries = [
        extension(extensions, "fixture_models",
                  [{"id": "fixture-model", "provider": "fixture", "license": "mit",
                    "description": "Fixture model", "components": [weights, extra]},
                   {"id": "gated-model", "provider": "fixture", "components": [gated]}],
                  {"fixture-model": {"weights": lock_entry(weights), "extra": lock_entry(extra)},
                   "gated-model": {"weights": lock_entry(gated)}}),
        extension(extensions, "fixture_stale",
                  [{"id": "stale-model", "provider": "fixture", "components": [stale]}],
                  {"stale-model": {"weights": {**lock_entry(stale), "include": ["other"]}}}),
    ]
    (extensions / "installed.json").write_text(json.dumps({"schema": 1, "extensions": entries}))
    store = root / "models"
    env = {**os.environ, "GUFO_EXTENSIONS_DIR": str(extensions), "GUFO_MODELS_DIR": str(store),
           "HF_ENDPOINT": f"http://127.0.0.1:{server.server_address[1]}",
           "HF_HOME": str(root / "hf"), "HF_TOKEN": ""}
    env.pop("HF_HUB_CACHE", None)
    # Plain listings never read the registry.
    registry = root / "index.json"
    registry.write_text("invalid registry")
    env["GUFO_EXTENSION_REGISTRY"] = str(registry)

    listing = json.loads(gufo("model", "list", "--json").stdout)
    rows = {m["id"]: m for m in listing["models"]}
    assert rows["fixture-model"]["extension"] == "fixture_models", rows
    assert rows["fixture-model"]["status"] == "available", rows
    assert rows["fixture-model"]["extensionStatus"] == "installed", rows
    assert rows["fixture-model"]["downloadable"] is True, rows
    assert "stale-model" not in rows, rows
    assert any("fixture_stale" in p and "stale" in p for p in listing["problems"]), listing
    assert not any("extension registry" in p for p in listing["problems"]), listing

    # --registry reads models from each reviewed commit, without installing
    # or building extension code.
    api = json.loads(gufo("diagnose", "--section", "extensions", "--json").stdout)["extensionApiVersion"]
    remote = {"id": "registry-model", "provider": "remote", "description": "Registry weights",
              "components": [gated]}
    repository = root / "registry_fixture"
    repository.mkdir()
    (repository / "extension.json").write_text(json.dumps({
        "schema": 1, "name": "registry_fixture", "version": "1.0.0",
        "models": [remote, {**remote, "id": "fixture-model"}],
        "providers": [{"id": "remote", "model_ids": ["Legacy-Recognition", "legacy-recognition"]}]}))
    (repository / "models.lock.json").write_text(json.dumps({"schema": 1, "models": {
        name: {"weights": lock_entry(gated)} for name in ("registry-model", "fixture-model")}}))
    git = ["git", "-C", str(repository), "-c", "user.name=fixture", "-c", "user.email=fixture@invalid"]
    subprocess.run([*git, "init", "--quiet"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", "fixture"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], check=True, text=True,
                            capture_output=True).stdout.strip()
    release = {"version": "1.0.0", "revision": commit, "digest": "sha256:" + "a" * 64,
               "api_version": ">= " + api, "platforms": ["linux", "windows"]}
    unreachable = "file://" + str(root / "missing")
    registry.write_text(json.dumps({"schema": 1, "extensions": {
        "registry_fixture": {"source": repository.as_uri(), "description": "Registry fixture",
                             "releases": [{**release, "version": "2.0.0", "api_version": ">= 99.0.0",
                                           "revision": "f" * 40}, release]},
        # Neither an installed extension nor an incompatible one is fetched.
        "fixture_models": {"source": unreachable, "releases": [{**release, "revision": "e" * 40}]},
        "unsupported_fixture": {"source": unreachable,
                                "releases": [{**release, "api_version": ">= 99.0.0"}]},
    }}))
    listing = json.loads(gufo("model", "list", "--json", "--registry").stdout)
    rows = {row["id"]: row for row in listing["models"]}
    assert rows["registry-model"]["extensionStatus"] == "not installed", rows
    assert rows["registry-model"]["status"] == "extension not installed", rows
    assert rows["registry-model"]["downloadable"] is True, rows
    assert rows["registry-model"]["extensionVersion"] == "1.0.0", rows
    assert rows["fixture-model"]["extension"] == "fixture_models", rows
    assert any("already provided by fixture_models" in p for p in listing["problems"]), listing
    legacy = [row for row in listing["models"] if row["id"].lower() == "legacy-recognition"]
    assert len(legacy) == 1 and legacy[0]["downloadable"] is False, legacy
    assert legacy[0]["description"] == "Registry fixture", legacy
    assert not any(row["extension"] == "unsupported_fixture" for row in listing["models"]), rows
    assert not any("cannot read" in p for p in listing["problems"]), listing
    table = gufo("model", "list", "--registry").stdout
    assert "extension not installed" in table and "gufo extension install" in table, table
    assert "gufo extension install" not in gufo("model", "list").stdout
    assert not store.exists() and not (extensions / "registry_fixture").exists()
    # Reviewed commits are immutable: the second listing reads the cache.
    assert (extensions / ".registry-metadata" / f"{commit}.json").is_file()
    # Git marks objects read-only, which Windows refuses to delete.
    repository.rename(root / "moved_registry_fixture")
    cached = json.loads(gufo("model", "list", "--json", "--registry").stdout)
    assert any(row["id"] == "registry-model" for row in cached["models"]), cached
    registry.write_text("invalid registry")
    offline = json.loads(gufo("model", "list", "--json", "--registry").stdout)
    assert any("extension registry" in problem for problem in offline["problems"]), offline
    assert any(row["id"] == "fixture-model" for row in offline["models"]), offline

    missing = gufo("prompt", "--model", "fixture-model", check=False)
    assert missing.returncode == 1 and "gufo model install fixture-model" in missing.stderr, missing

    # Without a token the gated file is refused, with a hint.
    denied = gufo("model", "install", "gated-model", check=False)
    assert denied.returncode == 1 and "HF_TOKEN" in denied.stderr, denied.stderr

    env["HF_TOKEN"] = TOKEN
    requests.clear()
    gufo("model", "install", "fixture-model", "--connections", "4")
    directory = store / "fixture-model"
    for path in ("weights/big.bin", "weights/config.json"):
        assert (directory / path).read_bytes() == FILES[path], path
    assert not (directory / "extra.json").exists()
    assert "big" in broken, "the interrupted range was not exercised"
    # The token reaches the endpoint only, never the CDN host.
    assert all(auth == f"Bearer {TOKEN}" for path, auth in requests if "/resolve/" in path), requests
    assert all(auth is None for path, auth in requests if path.startswith("/cdn/")), requests
    partials = {p.name.split(".")[0] for p in (store / "downloads").iterdir()}
    assert partials <= {hashlib.sha256(FILES["gated.bin"]).hexdigest()}, partials

    gufo("model", "install", "fixture-model", "--component", "extra")
    assert (directory / "extra.json").read_bytes() == FILES["extra.json"]
    row = {m["id"]: m for m in json.loads(gufo("model", "list", "--json").stdout)["models"]}["fixture-model"]
    assert row["status"] == "installed" and row["path"] == str(directory / "weights/big.bin"), row
    text = gufo("model", "list").stdout
    assert "fixture-model" in text and "fixture_models" in text and "installed (weights, extra)" in text, text

    unknown = gufo("model", "install", "fixture-model", "--component", "nope", check=False)
    assert unknown.returncode == 1 and "has no component nope" in unknown.stderr, unknown.stderr

    assert "fixture-model: ok" in gufo("model", "verify").stdout
    gufo("model", "install", "gated-model")
    gufo("model", "remove", "fixture-model")
    assert not directory.exists()
    remaining = [p.name for p in (store / "blobs").iterdir()]
    assert remaining == [hashlib.sha256(FILES["gated.bin"]).hexdigest()], remaining
    gufo("model", "remove", "gated-model")
    assert not list((store / "blobs").iterdir())
server.shutdown()
print("model CLI test passed")
