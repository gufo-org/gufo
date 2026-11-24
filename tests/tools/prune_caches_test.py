#!/usr/bin/env python3
"""CPU-only regressions for the Actions cache prune, with a stubbed gh."""

import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import re
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prune_caches", ROOT / "tools/ci/prune_caches.py")
PRUNE = importlib.util.module_from_spec(SPEC)
sys.modules["prune_caches"] = PRUNE
SPEC.loader.exec_module(PRUNE)

NOW = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
MIB = 2**20


def stamp(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


def entry(id_, ref, key, mib, hours_ago=5):
    return dict(id=id_, ref=ref, key=key, size_in_bytes=int(mib * MIB),
                created_at=stamp(hours_ago), last_accessed_at=stamp(hours_ago))


class StubGh:
    """Serves a cache list and answers deletes from a scripted queue."""

    def __init__(self, entries, deletes=()):
        self.entries = entries
        self.deletes = list(deletes)
        self.deleted = []
        self.calls = 0

    def __call__(self, args):
        self.calls += 1
        method, path = args[1], args[2]
        if method == "GET":
            ref = path.partition("&ref=")[2] or None
            page = int(re.search(r"[?&]page=([0-9]+)", path).group(1))
            rows = [e for e in self.entries if ref is None or e["ref"] == ref]
            body = json.dumps({"actions_caches": rows[(page - 1) * 100: page * 100]})
            return PRUNE.Response(200, {}, body)
        response = self.deletes.pop(0) if self.deletes else PRUNE.Response(204, {}, "")
        if response.status == 204:
            self.deleted.append(int(path.rsplit("/", 1)[1]))
        return response


def api(stub, retry_seconds=60):
    slept = []
    clock = lambda: NOW.timestamp()
    return PRUNE.Api("o/r", run=stub, sleep=slept.append, clock=clock,
                     retry_seconds=retry_seconds), slept


def pruner(api_, **kwargs):
    return PRUNE.Pruner(api_, now=NOW, **kwargs)


def run_main(api_, **env):
    env = {"MODE": "caller", "CALLER_REF": "refs/pull/7/merge", "GH_REPO": "o/r",
           "BUDGET_BYTES": str(10 * MIB), **env}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = PRUNE.main(env, api_, now=NOW)
    return code, out.getvalue()


TOO_MANY = PRUNE.Response(429, {"retry-after": "7"}, "", "gh: HTTP 429")
SECONDARY = PRUNE.Response(
    403, {}, '{"message":"You have exceeded a secondary rate limit."}', "gh: HTTP 403")
FORBIDDEN = PRUNE.Response(
    403, {}, '{"message":"Resource not accessible by integration"}', "gh: HTTP 403")


class ParseTest(unittest.TestCase):
    def test_include_output(self):
        r = PRUNE.parse("HTTP/2.0 429 Too Many Requests\r\nRetry-After: 30\r\n\r\n{}", "gh: HTTP 429", 1)
        self.assertEqual((r.status, r.headers["retry-after"], r.body), (429, "30", "{}"))
        self.assertTrue(PRUNE.rate_limited(r))

    def test_secondary_limit_is_a_rate_limit_and_permissions_are_not(self):
        self.assertTrue(PRUNE.rate_limited(SECONDARY))
        self.assertTrue(PRUNE.rate_limited(PRUNE.Response(403, {"x-ratelimit-remaining": "0"}, "")))
        self.assertFalse(PRUNE.rate_limited(FORBIDDEN))


class RetryTest(unittest.TestCase):
    def test_429_is_retried_after_retry_after(self):
        stub = StubGh([], [TOO_MANY, SECONDARY])
        a, slept = api(stub, retry_seconds=240)
        self.assertEqual(a.delete(1).status, 204)
        self.assertEqual(slept[:2], [7.0, 120.0])  # Retry-After, then backoff.

    def test_reset_header_applies_once_the_quota_is_spent(self):
        reset = str(int(NOW.timestamp()) + 42)
        spent = PRUNE.Response(429, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset}, "")
        a, slept = api(StubGh([], [spent]))
        a.delete(1)
        self.assertEqual(slept[0], 42.0)

    def test_secondary_limit_with_quota_left_backs_off_a_minute(self):
        # Every response carries the primary reset, up to an hour away.
        reset = str(int(NOW.timestamp()) + 2400)
        left = PRUNE.Response(429, {"x-ratelimit-remaining": "4800", "x-ratelimit-reset": reset}, "")
        stub = StubGh([], [left, left])
        a, slept = api(stub, retry_seconds=240)
        self.assertEqual(a.delete(1).status, 204)
        self.assertEqual(slept[:2], [60.0, 120.0])
        a, slept = api(StubGh([], [left] * 5), retry_seconds=240)
        with self.assertRaises(PRUNE.RateLimited):
            a.delete(1)
        self.assertEqual(slept, [60.0, 120.0])  # 240 more would pass the budget.

    def test_retry_budget_stops_the_prune_without_failing(self):
        entries = [entry(i, "refs/pull/7/merge", f"k{i}", 5) for i in range(1, 4)]
        stub = StubGh(entries, [TOO_MANY] * 50)
        a, slept = api(stub, retry_seconds=20)
        code, out = run_main(a)
        self.assertEqual(code, 0, out)
        self.assertIn("::warning::rate limited", out)
        self.assertLessEqual(sum(s for s in slept if s > 1), 20)
        self.assertEqual(stub.deleted, [])

    def test_permission_error_fails_the_job(self):
        entries = [entry(1, "refs/pull/7/merge", "k1", 20)]
        a, _ = api(StubGh(entries, [FORBIDDEN]))
        code, out = run_main(a)
        self.assertEqual(code, 1)
        self.assertIn("::error::deleting a cache failed", out)

    def test_missing_entry_is_not_an_error(self):
        entries = [entry(1, "refs/pull/7/merge", "k1", 20)]
        a, _ = api(StubGh(entries, [PRUNE.Response(404, {}, "")]))
        self.assertEqual(run_main(a)[0], 0)


class SelectionTest(unittest.TestCase):
    def test_budget_evicts_large_idle_entries_before_tiny_ones(self):
        nix = [entry(100 + i, "refs/pull/8/merge", f"{i:032d}.narinfo-x", 0.01, 30) for i in range(500)]
        big = entry(1, "refs/pull/8/merge", "gufo-ccache-v1-a-12345678-1", 300, 10)
        main = entry(2, "refs/heads/main", "gufo-therock-v2-x", 900, 40)
        p = pruner(api(StubGh([]))[0], budget=1000 * MIB)
        chosen = p.over_budget(nix + [big, main])
        self.assertEqual([e["id"] for e in chosen], [1])

    def test_recent_and_main_entries_are_spared(self):
        recent = entry(1, "refs/pull/8/merge", "a", 500, hours_ago=0.1)
        main = entry(2, "refs/heads/main", "b", 500)
        self.assertEqual(pruner(api(StubGh([]))[0], budget=0).over_budget([recent, main]), [])

    def test_superseded_keeps_the_newest_run(self):
        rows = [entry(1, "refs/pull/8/merge", "gufo-rust-v1-x-45000000001-1", 10),
                entry(2, "refs/pull/8/merge", "gufo-rust-v1-x-45000000002-1", 10),
                entry(3, "refs/heads/main", "gufo-rust-v1-x-45000000000-1", 10)]
        self.assertEqual([e["id"] for e in pruner(api(StubGh([]))[0]).superseded(rows)], [1])

    def test_delete_cap_stops_with_a_warning(self):
        entries = [entry(i, "refs/pull/9/merge", f"k{i}", 1) for i in range(1, 30)]
        stub = StubGh(entries)
        a, _ = api(stub)
        code, out = run_main(a, MODE="pull_request_target", CLOSED_PR="9", MAX_DELETES="10")
        self.assertEqual((code, len(stub.deleted)), (0, 10))
        self.assertIn("stopped after 10 deletes", out)


if __name__ == "__main__":
    sys.exit(unittest.main())
