#!/usr/bin/env python3
"""matched_kl.py against a fake gufo endpoint: history, scoring and KL."""

import json
import math
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality"))

import matched_kl  # noqa: E402

VOCAB = 12


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def logits(history, shift):
    mix = len(history)
    for token in history:
        mix = (mix * 31 + token) % 1000003
    return [((mix + i * 7) % 11) * 0.5 + (shift if i == mix % VOCAB else 0.0)
            for i in range(VOCAB)]


def log_softmax(values):
    peak = max(values)
    total = peak + math.log(sum(math.exp(v - peak) for v in values))
    return [v - total for v in values]


def make_server(shift):
    """A fake endpoint whose distribution is shifted by `shift` logits."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if "prompt_logprobs" in body:
                tokens = body["prompt"]
                start = body.get("prompt_logprobs_start", 1)
                k = body["prompt_logprobs"]
                entries = [None] * start
                for position in range(start, len(tokens)):
                    lp = log_softmax(logits(tokens[:position], shift))
                    order = sorted(range(VOCAB), key=lambda t: (-lp[t], t))
                    entries.append({
                        "token": tokens[position],
                        "logprob": lp[tokens[position]],
                        "rank": order.index(tokens[position]) + 1,
                        "top_logprobs": [{"token": t, "logprob": lp[t]}
                                         for t in order[:k]],
                    })
                choice = {"text": "", "prompt_logprobs": entries}
            else:
                check(body["return_token_ids"] is True, "asks for token IDs")
                check(body["temperature"] == 0, "greedy history")
                prompt_ids = [ord(c) % VOCAB for c in body["prompt"][-6:]]
                history = list(prompt_ids)
                for _ in range(body["max_tokens"]):
                    lp = logits(history, shift)
                    history.append(max(range(VOCAB), key=lambda t: lp[t]))
                choice = {"text": "x", "prompt_token_ids": prompt_ids,
                          "token_ids": history[len(prompt_ids):]}
            data = json.dumps({"model": f"fake-{shift}",
                               "choices": [choice]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def test_compare_position():
    same = {"token": 1, "logprob": math.log(0.6), "rank": 1, "nonfinite": 0,
            "top": [[1, math.log(0.6)], [2, math.log(0.3)]]}
    row = matched_kl.compare_position(same, same)
    check(abs(row["kl"]) < 1e-12 and row["top1"] and row["overlap"] == 1.0,
          "identical distributions have zero KL")
    other = {"token": 1, "logprob": math.log(0.3), "rank": 2, "nonfinite": 0,
             "top": [[3, math.log(0.5)], [1, math.log(0.3)]]}
    row = matched_kl.compare_position(same, other)
    # Teacher bins {1, 2, rest}; candidate's 2 is imputed min(0.3, 0.2).
    expected = (0.6 * math.log(0.6 / 0.3) + 0.3 * math.log(0.3 / 0.2) +
                0.1 * math.log(0.1 / 0.5))
    check(abs(row["kl"] - expected) < 1e-12, f"coarse KL {row['kl']}")
    check(not row["top1"] and row["overlap"] == 0.5 and row["imputed"],
          "disagreement statistics")
    check(abs(row["nll_candidate"] + math.log(0.3)) < 1e-12,
          "NLL of the teacher token under the candidate")


def test_end_to_end():
    teacher, teacher_requests = make_server(0.0)
    candidate, candidate_requests = make_server(0.7)
    identical, _ = make_server(0.0)
    try:
        url = lambda s: f"http://127.0.0.1:{s.server_address[1]}"
        with tempfile.TemporaryDirectory() as directory:
            matched_kl.main(["run", "--teacher", url(teacher), "--candidate",
                             url(candidate), "--out-dir", directory,
                             "--limit", "5", "--max-tokens", "12",
                             "--top-k", "4"])
            report = json.loads(Path(directory, "report.json").read_text())
            overall = report["overall"]
            check(report["sequences"] == 5, "five histories scored")
            check(overall["positions"] == 5 * 12, "generated positions only")
            check(overall["kl_mean"] > 0.0, "a shifted candidate diverges")
            check(overall["teacher_rank1"] == 1.0,
                  "the teacher's greedy history is its own top-1")
            check(overall["kl_p99"] >= overall["kl_p90"] >=
                  overall["kl_median"], "ordered percentiles")
            check(set(report["categories"]) == {"code"},
                  "the first five built-in prompts are code")
            history = matched_kl.read_jsonl(str(Path(directory,
                                                     "history.jsonl")))
            scored = {tuple(r["prompt"]): r["prompt_logprobs_start"]
                      for r in candidate_requests if "prompt_logprobs" in r}
            check(scored == {tuple(h["tokens"]): h["start"] for h in history},
                  "the candidate scores the teacher's exact token IDs, "
                  "from the end of the prompt")
            check(not any("return_token_ids" in r for r in candidate_requests),
                  "the candidate never generates")
            check(len([r for r in teacher_requests
                       if "return_token_ids" in r]) == 5,
                  "the teacher generates each history once")

            # A resumed run reuses every completed result.
            before = len(teacher_requests)
            matched_kl.main(["run", "--teacher", url(teacher), "--candidate",
                             url(candidate), "--out-dir", directory,
                             "--limit", "5", "--max-tokens", "12",
                             "--top-k", "4"])
            check(len(teacher_requests) == before, "resume repeats nothing")

            same = str(Path(directory, "same.jsonl"))
            matched_kl.main(["score", "--endpoint", url(identical),
                             "--history", str(Path(directory,
                                                   "history.jsonl")),
                             "--out", same, "--top-k", "4"])
            control = matched_kl.compare(str(Path(directory,
                                                  "teacher.jsonl")), same)
            check(control["overall"]["kl_max"] < 1e-12 and
                  control["overall"]["top1_agreement"] == 1.0,
                  "an identical candidate has zero KL")
    finally:
        for server in (teacher, candidate, identical):
            server.shutdown()


if __name__ == "__main__":
    test_compare_position()
    test_end_to_end()
    print("matched_kl_test: ok")
