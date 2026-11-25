#!/usr/bin/env python3
"""Prune the repository's Actions caches for .github/workflows/caches.yml.

Keeps main's warm caches inside the 10 GB Actions cache budget:

  closed      delete a closed pull request's entries
  caller      delete the caller ref's superseded per-run entries, then
              evict other refs' entries until the total fits the budget
  sweep       both for every ref, plus closed PRs and entries of refs
              other than main that no run used for STALE_SECONDS

An entry created or restored in the last RECENT_SECONDS is never pruned: a
concurrent run may be downloading it. Budget eviction ranks entries by size
times idle time, so a few large idle entries go before thousands of tiny
Magic Nix Cache entries, and one run deletes at most MAX_DELETES entries;
the nightly sweep continues where a run stopped.

GitHub rate-limits cache deletes. A 429, or a 403 that reports a rate limit,
is retried after Retry-After, or X-RateLimit-Reset once the primary quota is
spent, else after 60 s doubling, for at most RETRY_SECONDS in total. Past that the prune stops with a
warning and succeeds. Any other failed delete, such as a 403 for a missing
actions: write permission, fails the job; a 404 means it is already gone.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

PR_REF = re.compile(r"refs/pull/([0-9]+)/")
PER_RUN = re.compile(r"-([0-9]{8,})-([0-9]+)$")


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: str
    error: str = ""


class RateLimited(Exception):
    """The retry budget ran out while GitHub kept rate-limiting."""


def parse(stdout: str, stderr: str, returncode: int) -> Response:
    """Split `gh api --include` output into status, headers and body."""
    head, _, body = stdout.replace("\r\n", "\n").partition("\n\n")
    lines = head.split("\n")
    match = re.match(r"HTTP/\S+\s+([0-9]{3})", lines[0]) if lines else None
    status = int(match.group(1)) if match else (0 if returncode else 200)
    if not match:
        found = re.search(r"HTTP ([0-9]{3})", stderr)
        if found:
            status = int(found.group(1))
    headers = {}
    for line in lines[1:] if match else []:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    return Response(status, headers, body if match else stdout, stderr.strip())


def gh_runner(args: list[str]) -> Response:
    result = subprocess.run(["gh", "api", "--include", *args], capture_output=True, text=True)
    return parse(result.stdout, result.stderr, result.returncode)


def rate_limited(response: Response) -> bool:
    if response.status == 429:
        return True
    if response.status != 403:
        return False
    text = (response.body + " " + response.error).lower()
    return ("rate limit" in text or response.headers.get("x-ratelimit-remaining") == "0"
            or "retry-after" in response.headers)


@dataclass
class Api:
    repo: str
    run: Callable[[list[str]], Response] = gh_runner
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time
    retry_seconds: float = 240.0
    pause_seconds: float = 1.0
    waited: float = 0.0

    def wait_for(self, response: Response, attempt: int) -> float:
        if response.headers.get("retry-after", "").isdigit():
            return float(response.headers["retry-after"])
        # The reset time belongs to the primary hourly quota; it only applies
        # once that quota is spent. A secondary limit with quota left waits at
        # least a minute, as GitHub asks, doubling up to the retry budget.
        reset = response.headers.get("x-ratelimit-reset", "")
        if response.headers.get("x-ratelimit-remaining") == "0" and reset.isdigit():
            return max(1.0, float(reset) - self.clock())
        return 60.0 * 2**attempt

    def call(self, args: list[str]) -> Response:
        attempt = 0
        while True:
            response = self.run(args)
            if not rate_limited(response):
                return response
            delay = self.wait_for(response, attempt)
            if self.waited + delay > self.retry_seconds:
                raise RateLimited(f"rate limited for {self.waited:.0f} s ({response.status})")
            print(f"rate limited ({response.status}); retrying in {delay:.0f} s", flush=True)
            self.sleep(delay)
            self.waited += delay
            attempt += 1

    def caches(self, ref: str | None = None) -> list[dict]:
        entries, page = [], 1
        while True:
            query = f"repos/{self.repo}/actions/caches?per_page=100&page={page}"
            if ref:
                query += f"&ref={ref}"
            response = self.call(["-X", "GET", query])
            if response.status != 200:
                raise RuntimeError(f"listing caches failed: HTTP {response.status} {response.error}")
            batch = json.loads(response.body).get("actions_caches", [])
            entries += batch
            if len(batch) < 100:
                return entries
            page += 1

    def pr_state(self, pr: str) -> str:
        response = self.call(["-X", "GET", f"repos/{self.repo}/pulls/{pr}"])
        if response.status != 200:
            return "unknown"
        return json.loads(response.body).get("state", "unknown")

    def delete(self, entry_id: int) -> Response:
        response = self.call(["-X", "DELETE", f"repos/{self.repo}/actions/caches/{entry_id}"])
        # GitHub asks for a pause between mutating requests.
        self.sleep(self.pause_seconds)
        return response


@dataclass
class Pruner:
    api: Api
    protected: str = "refs/heads/main"
    budget: int = 7 * 2**30
    stale: float = 2 * 86400
    recent: float = 900
    max_deletes: int = 100
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deleted: set = field(default_factory=set)
    errors: list = field(default_factory=list)

    def idle(self, entry: dict) -> float:
        """Seconds since the entry was created or last restored."""
        stamps = [entry.get("last_accessed_at"), entry.get("created_at")]
        newest = max(datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps if s)
        return (self.now - newest).total_seconds()

    def remove(self, entries: list[dict], reason: str) -> bool:
        """Delete entries; False once the per-run cap stops the prune."""
        for entry in entries:
            if entry["id"] in self.deleted:
                continue
            if len(self.deleted) >= self.max_deletes:
                print(f"::warning::stopped after {self.max_deletes} deletes; the nightly sweep continues")
                return False
            response = self.api.delete(entry["id"])
            if response.status in (200, 204, 404):
                self.deleted.add(entry["id"])
                if response.status != 404:
                    print(f"deleted ({reason}) {entry['ref']} {entry['key']} "
                          f"{entry['size_in_bytes'] / 2**20:.0f} MiB")
            else:
                self.errors.append(f"{entry['ref']} {entry['key']}: HTTP {response.status} {response.error}")
        return True

    def superseded(self, entries: list[dict]) -> list[dict]:
        """Every entry of a per-run family except the newest run's, per ref."""
        best, rows = {}, []
        for entry in entries:
            match = PER_RUN.search(entry["key"])
            if not match:
                continue
            family = (entry["ref"], entry["key"][: match.start()])
            run = (int(match.group(1)), int(match.group(2)))
            rows.append((family, run, entry))
            best[family] = max(best.get(family, run), run)
        chosen = [e for family, run, e in rows if run < best[family] and self.idle(e) > self.recent]
        return sorted(chosen, key=lambda e: -e["size_in_bytes"])

    def over_budget(self, entries: list[dict]) -> list[dict]:
        """Other refs' entries, large and idle first, until the total fits."""
        live = [e for e in entries if e["id"] not in self.deleted]
        total = sum(e["size_in_bytes"] for e in live)
        print(f"cache total {total / 2**30:.2f} GiB, budget {self.budget / 2**30:.2f} GiB")
        victims = sorted((e for e in live if e["ref"] != self.protected and self.idle(e) > self.recent),
                         key=lambda e: -e["size_in_bytes"] * self.idle(e))
        chosen = []
        for entry in victims:
            if total <= self.budget:
                break
            chosen.append(entry)
            total -= entry["size_in_bytes"]
        return chosen

    def closed(self, pr: str) -> None:
        for ref in (f"refs/pull/{pr}/merge", f"refs/pull/{pr}/head"):
            entries = sorted(self.api.caches(ref), key=lambda e: -e["size_in_bytes"])
            if not self.remove(entries, "closed PR"):
                return

    def caller(self, ref: str) -> None:
        if self.remove(self.superseded(self.api.caches(ref)), "superseded"):
            self.remove(self.over_budget(self.api.caches()), "over budget")

    def sweep(self) -> None:
        entries = self.api.caches()
        prs = sorted({m.group(1) for e in entries if (m := PR_REF.match(e["ref"]))}, key=int)
        closed = {pr for pr in prs if self.api.pr_state(pr) == "closed"}
        by_size = lambda items: sorted(items, key=lambda e: -e["size_in_bytes"])
        steps = [
            (by_size(e for e in entries
                     if (m := PR_REF.match(e["ref"])) and m.group(1) in closed), "closed PR"),
            (self.superseded(entries), "superseded"),
            (by_size(e for e in entries if e["ref"] != self.protected and self.idle(e) > self.stale), "unused"),
        ]
        for chosen, reason in steps:
            if not self.remove(chosen, reason):
                return
        self.remove(self.over_budget(entries), "over budget")


def main(env: dict[str, str] = os.environ, api: Api | None = None,
         now: datetime | None = None) -> int:
    api = api or Api(env["GH_REPO"], retry_seconds=float(env.get("RETRY_SECONDS", "240")))
    pruner = Pruner(
        api,
        protected=env.get("PROTECTED_REF", "refs/heads/main"),
        budget=int(env.get("BUDGET_BYTES", str(7 * 2**30))),
        stale=float(env.get("STALE_SECONDS", "172800")),
        recent=float(env.get("RECENT_SECONDS", "900")),
        max_deletes=int(env.get("MAX_DELETES", "100")),
        now=now or datetime.now(timezone.utc),
    )
    mode = env["MODE"]
    try:
        if mode == "pull_request_target":
            pruner.closed(env["CLOSED_PR"])
        elif mode in ("schedule", "workflow_dispatch"):
            pruner.sweep()
        elif mode == "caller":
            pruner.caller(env["CALLER_REF"])
        else:
            print(f"::error::unknown prune mode {mode}")
            return 1
    except RateLimited as stop:
        print(f"::warning::{stop}; stopped after {len(pruner.deleted)} deletes, the nightly sweep continues")
    for line in pruner.errors:
        print(f"::error::deleting a cache failed: {line}")
    return 1 if pruner.errors else 0


if __name__ == "__main__":
    sys.exit(main())
