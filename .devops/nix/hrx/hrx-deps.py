#!/usr/bin/env python3
"""Lists, fetches and rewrites the pinned third-party downloads of HRX.

HRX resolves its dependencies at CMake configure time from two pinned
sources: MODULE.cmake.lock (archives and single files, with SHA256 hashes) and
Loom's AMDGPU ISA XML archive descriptor. The Nix sandbox has no network, so
`fetch` downloads every pinned URL into a fixed-output store path and `rewrite`
points the source tree at those files with file:// URLs. HRX's own hash checks
and dependency patches still run on the local copies.

Usage:
  hrx-deps.py fetch <hrx-source> <out-dir>
  hrx-deps.py rewrite <hrx-source> <deps-dir>
"""

import hashlib
import os
import re
import subprocess
import sys

LOCK = "MODULE.cmake.lock"
ARCHIVE_DESCRIPTORS = ["loom/build_tools/amdgpu/isa_xml/archive.bzl"]


def pinned_downloads(source):
    """Returns [(sha256, [urls])] for every pinned download."""
    downloads = []
    with open(os.path.join(source, LOCK), encoding="utf-8") as lock:
        text = lock.read()
    hashes = dict(re.findall(r'set\(IREE_DEP_(\w+)_SHA256 "([0-9a-f]{64})"\)', text))
    for name, block in re.findall(r"set\(IREE_DEP_(\w+)_URLS\n(.*?)\)", text, re.S):
        urls = re.findall(r'"([^"]+)"', block)
        if urls and name in hashes:
            downloads.append((hashes[name], urls))
    for descriptor in ARCHIVE_DESCRIPTORS:
        with open(os.path.join(source, descriptor), encoding="utf-8") as f:
            text = f.read()
        sha256 = re.search(r'"sha256": "([0-9a-f]{64})"', text).group(1)
        urls = re.findall(r'"(https://[^"]+)"', text)
        downloads.append((sha256, urls))
    return downloads


def store_name(sha256, url):
    # Keep the URL's file name so CMake recognizes the archive type.
    base = url.split("?")[0].rstrip("/").split("/")[-1] or "file"
    return f"{sha256}-{base}"


RAW_GITHUB = re.compile(
    r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([0-9a-f]{40})/(.+)")


def mirrors(url):
    """The same file at the same commit from mirrors of raw GitHub files.

    raw.githubusercontent.com rate limits shared CI addresses; every download
    is still checked against its pinned SHA-256."""
    match = RAW_GITHUB.fullmatch(url)
    if not match:
        return []
    owner, repo, commit, path = match.groups()
    found = [f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{commit}/{path}"]
    if (owner, repo) == ("torvalds", "linux"):
        found.append("https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/"
                     f"linux.git/plain/{path}?id={commit}")
    return found


def fetch(source, out):
    os.makedirs(out, exist_ok=True)
    for sha256, urls in pinned_downloads(source):
        target = os.path.join(out, store_name(sha256, urls[0]))
        candidates = urls + [mirror for url in urls for mirror in mirrors(url)]
        for url in candidates:
            partial = target + ".partial"
            result = subprocess.run(
                # Transient failures, such as rate limiting (HTTP 429), back
                # off exponentially, or as Retry-After says, before a mirror.
                ["curl", "-fsSL", "--retry", "5", "--retry-max-time", "120",
                 "-o", partial, url])
            if result.returncode != 0:
                continue
            with open(partial, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if digest == sha256:
                os.rename(partial, target)
                break
            os.remove(partial)
        else:
            sys.exit(f"no URL matched {sha256}: {candidates}")


def rewrite(source, deps):
    replacements = {}
    for sha256, urls in pinned_downloads(source):
        local = "file://" + os.path.join(deps, store_name(sha256, urls[0]))
        if not os.path.exists(local[len("file://"):]):
            sys.exit(f"missing prefetched download for {urls[0]}")
        for url in urls:
            replacements[url] = local
    files = [LOCK] + ARCHIVE_DESCRIPTORS
    for relative in files:
        path = os.path.join(source, relative)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for url, local in replacements.items():
            text = text.replace(f'"{url}"', f'"{local}"')
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    command, source, directory = sys.argv[1:4]
    {"fetch": fetch, "rewrite": rewrite}[command](source, directory)
