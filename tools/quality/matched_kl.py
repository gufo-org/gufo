#!/usr/bin/env python3
"""Matched-token quality comparison between two gufo text endpoints.

Follows docs/BENCHMARKS.md "Quality method (matched-token, per position)":
the teacher generates one fixed greedy history per prompt, returned as token
IDs; teacher and candidate then score exactly those IDs with
`/v1/completions` `prompt_logprobs`, and the comparison reads both next-token
distributions per position.

Each endpoint reports its top-K tokens plus the actual next token. KL is
computed over the coarse distribution {teacher top-K tokens} + {rest}; a
teacher top-K token outside the candidate's top-K takes the smaller of the
candidate's K-th probability and an equal share of its unlisted mass (counted
as `imputed`). Coarse-graining never increases KL, so without imputed
positions this is a lower bound of the full-vocabulary KL; imputation can push
it either way. Teacher-token NLL, top-1 agreement and top-K overlap are exact.

    matched_kl.py history --teacher URL --out history.jsonl
    matched_kl.py score --endpoint URL --history history.jsonl --out t.jsonl
    matched_kl.py compare t.jsonl c.jsonl [--json report.json]
    matched_kl.py run --teacher URL --candidate URL --out-dir DIR

Standard library only.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

# Qwen chat framing for raw completions. The tokenizer parses these control
# tokens in completion text, so the history is the chat-shaped prompt the
# chat template would render, with an empty reasoning block (no thinking).
FRAMINGS = {
    "chat": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
    "chat-think": "<|im_start|>user\n{prompt}<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n",
    "raw": "{prompt}",
}

# 64 generic prompts, eight per category.
PROMPTS = {
    "code": [
        "Write a Python function that merges two sorted lists into one sorted "
        "list without using sort(). Explain its complexity.",
        "Explain the difference between a process and a thread, with a short "
        "C example that starts two threads.",
        "Write a SQL query that returns the three most recent orders for each "
        "customer from a table orders(id, customer_id, created_at).",
        "Implement a least-recently-used cache class in TypeScript with get "
        "and put methods that run in constant time.",
        "Review this code and point out the bug: def avg(xs): return "
        "sum(xs) / len(xs) if xs else sum(xs) / len(xs)",
        "Write a Bash script that finds the ten largest files under a "
        "directory and prints their sizes in human-readable form.",
        "Explain what a race condition is and show how a mutex fixes one in "
        "Go.",
        "Write a Rust function that parses a line like 'key=value' into a "
        "tuple and returns an error for malformed input.",
    ],
    "math": [
        "Solve for x: 3x + 7 = 2x - 5. Show each step.",
        "What is the sum of the integers from 1 to 1000? Explain the method.",
        "A rectangle has a perimeter of 36 cm and its length is twice its "
        "width. Find its area.",
        "Prove that the square root of 2 is irrational.",
        "Compute the derivative of f(x) = x^3 * sin(x) and explain the rule "
        "you used.",
        "What is the probability of getting exactly two heads in four fair "
        "coin flips?",
        "Explain how to find the greatest common divisor of 252 and 105 with "
        "the Euclidean algorithm.",
        "A train travels 180 km in 2.5 hours. What is its average speed in "
        "meters per second?",
    ],
    "reasoning": [
        "If all bloops are razzies and all razzies are lazzies, are all "
        "bloops definitely lazzies? Explain.",
        "Three boxes are labeled apples, oranges and mixed, and every label "
        "is wrong. You may take one fruit from one box. How do you relabel "
        "them correctly?",
        "A farmer must cross a river with a wolf, a goat and a cabbage. The "
        "boat holds the farmer and one item. How does he do it?",
        "You have two ropes that each take an hour to burn unevenly. How can "
        "you measure 45 minutes?",
        "Which is heavier, a kilogram of feathers or a kilogram of iron? "
        "Explain why people often get this wrong.",
        "In a race you overtake the person in second place. What place are "
        "you in now?",
        "Plan a weekly schedule for learning a new language with 30 minutes "
        "per day, and justify the order of activities.",
        "Compare the trade-offs between renting and buying a home for "
        "someone who may move in three years.",
    ],
    "writing": [
        "Write a short story of about 200 words about a lighthouse keeper "
        "who finds a message in a bottle.",
        "Write a haiku sequence of three poems about autumn in a city.",
        "Draft a polite email asking a colleague to review a document by "
        "Friday.",
        "Write a persuasive paragraph arguing that public libraries are "
        "still important.",
        "Describe a busy market at dawn using vivid sensory details.",
        "Write a product description for a reusable water bottle that keeps "
        "drinks cold for 24 hours.",
        "Write the opening paragraph of a mystery novel set on a train.",
        "Rewrite this sentence to be clearer and more concise: 'Due to the "
        "fact that the meeting was delayed, we were unable to finish all of "
        "the items that were on the agenda.'",
    ],
    "knowledge": [
        "Explain how photosynthesis converts light into chemical energy.",
        "What caused the fall of the Western Roman Empire? Give several "
        "factors.",
        "Explain how a bill becomes a law in a parliamentary democracy.",
        "Describe the water cycle and the role of evaporation and "
        "condensation.",
        "What is the difference between weather and climate?",
        "Explain how the internet routes a packet from one computer to "
        "another.",
        "Summarize the main ideas of the theory of plate tectonics.",
        "Explain why the sky appears blue during the day and red at sunset.",
    ],
    "instructions": [
        "Give step-by-step instructions for changing a flat bicycle tire.",
        "Explain how to make a simple loaf of bread at home.",
        "List the steps to set up a new Git repository and push it to a "
        "remote.",
        "How do I plant and care for tomato seedlings on a balcony?",
        "Describe how to back up a laptop to an external drive and verify "
        "the backup.",
        "Explain how to prepare for a job interview in one week.",
        "Give a checklist for packing for a three-day hiking trip.",
        "Explain how to brew pour-over coffee and what affects the taste.",
    ],
    "multilingual": [
        "Traduce al inglés: 'El tren sale a las ocho y llega a Madrid "
        "antes del mediodía.' Luego explica la gramática.",
        "Écris un court paragraphe en français sur les avantages du vélo en "
        "ville.",
        "Erkläre auf Deutsch, wie ein Regenbogen entsteht.",
        "用中文简要介绍一下长城的历史。",
        "日本語で、四季それぞれの特徴を短く説明してください。",
        "Spiega in italiano la differenza tra 'essere' e 'stare'.",
        "Translate to Portuguese: 'Please close the window when you leave "
        "the room.'",
        "한국어로 건강한 수면 습관 다섯 가지를 알려주세요.",
    ],
    "conversation": [
        "I just moved to a new city and don't know anyone. How can I make "
        "friends?",
        "What are some good questions to ask when meeting someone new?",
        "Can you recommend a few classic novels for someone who enjoys "
        "adventure stories?",
        "I keep procrastinating on a big project. Any practical tips?",
        "What should I consider when adopting a cat for the first time?",
        "Help me plan a birthday party for a ten-year-old who loves space.",
        "How can I keep a houseplant alive if I travel often?",
        "What is a good way to start learning to draw as an adult?",
    ],
}


def post(url, path, body, timeout):
    request = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    if os.environ.get("GUFO_API_KEY"):
        request.add_header("Authorization",
                           "Bearer " + os.environ["GUFO_API_KEY"])
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{path} returned {error.code}: {detail}") from None


def history_digest(record):
    text = json.dumps([record["tokens"], record["start"]])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_prompts(path):
    if path is None:
        return [(category, index, prompt) for category, prompts in
                PROMPTS.items() for index, prompt in enumerate(prompts)]
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    # {"category": ["prompt", ...]} or [{"category":..., "prompt":...}]
    if isinstance(data, dict):
        return [(category, index, prompt) for category, prompts in
                data.items() for index, prompt in enumerate(prompts)]
    return [(item.get("category", "default"), index, item["prompt"])
            for index, item in enumerate(data)]


def read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_history(args):
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    done = {record["id"] for record in read_jsonl(args.out)}
    framing = FRAMINGS[args.framing]
    with open(args.out, "a", encoding="utf-8") as out:
        for category, index, prompt in prompts:
            ident = f"{category}-{index}"
            if ident in done:
                continue
            started = time.monotonic()
            response = post(args.teacher, "/v1/completions", {
                "prompt": framing.format(prompt=prompt),
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "return_token_ids": True,
            }, args.timeout)
            choice = response["choices"][0]
            prompt_ids = choice["prompt_token_ids"]
            generated = choice["token_ids"]
            if not generated:
                print(f"{ident}: teacher generated nothing; skipped",
                      file=sys.stderr)
                continue
            record = {
                "id": ident,
                "category": category,
                "teacher_model": response.get("model"),
                "framing": args.framing,
                "max_tokens": args.max_tokens,
                "tokens": prompt_ids + generated,
                "start": len(prompt_ids),
            }
            record["digest"] = history_digest(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"{ident}: {len(prompt_ids)} prompt + {len(generated)} "
                  f"generated tokens in {time.monotonic() - started:.1f}s",
                  file=sys.stderr)


def score_one(endpoint, record, top_k, include_prompt, timeout):
    start = 1 if include_prompt else record["start"]
    started = time.monotonic()
    response = post(endpoint, "/v1/completions", {
        "prompt": record["tokens"],
        "prompt_logprobs": top_k,
        "prompt_logprobs_start": start,
        "max_tokens": 0,
    }, timeout)
    entries = response["choices"][0]["prompt_logprobs"]
    positions = []
    for position, entry in enumerate(entries[start:], start):
        if entry["token"] != record["tokens"][position]:
            raise RuntimeError(f"{record['id']}: position {position} scored "
                               f"token {entry['token']}, expected "
                               f"{record['tokens'][position]}")
        positions.append({
            "token": entry["token"],
            # The server reports a non-finite target logit as null.
            "logprob": (-math.inf if entry["logprob"] is None
                        else entry["logprob"]),
            "rank": entry["rank"],
            "nonfinite": entry.get("nonfinite", 0),
            "top": [[item["token"], item["logprob"]]
                    for item in entry["top_logprobs"]],
        })
    return {
        "id": record["id"],
        "category": record["category"],
        "digest": record["digest"],
        "model": response.get("model"),
        "start": start,
        "top_k": top_k,
        "seconds": round(time.monotonic() - started, 3),
        "positions": positions,
    }


def score(endpoint, history, out_path, top_k, include_prompt, concurrency,
          timeout):
    records = read_jsonl(history)
    previous = read_jsonl(out_path)
    # A resume keeps earlier results, so they must use the same top-K.
    other = sorted({record["top_k"] for record in previous} - {top_k})
    if other:
        raise RuntimeError(f"{out_path} was scored with top-K {other}, not "
                           f"{top_k}; use another output")
    done = {(record["id"], record["digest"]) for record in previous}
    todo = [record for record in records
            if (record["id"], record["digest"]) not in done]
    with open(out_path, "a", encoding="utf-8") as out, \
            concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(score_one, endpoint, record, top_k,
                               include_prompt, timeout) for record in todo]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f"{endpoint} {result['id']}: {len(result['positions'])} "
                  f"positions in {result['seconds']:.1f}s", file=sys.stderr)


def compare_position(teacher, candidate):
    """KL(teacher || candidate) over teacher top-K + rest, and exact stats."""
    teacher_top = {token: math.exp(logprob) for token, logprob in teacher["top"]}
    candidate_top = {token: math.exp(logprob)
                     for token, logprob in candidate["top"]}
    # The actual token's log-probability is exact on both sides.
    teacher_top.setdefault(teacher["token"], math.exp(teacher["logprob"]))
    candidate_top.setdefault(candidate["token"], math.exp(candidate["logprob"]))
    candidate_listed = sum(math.exp(logprob) for _, logprob in candidate["top"])
    floor = min((math.exp(logprob) for _, logprob in candidate["top"]),
                default=0.0)
    missing = [token for token in teacher_top if token not in candidate_top]
    unlisted = max(0.0, 1.0 - candidate_listed)
    share = unlisted / len(missing) if missing else 0.0
    eps = 1e-12
    kl = 0.0
    covered_candidate = 0.0
    for token, p in teacher_top.items():
        q = candidate_top.get(token, min(floor, share))
        q = max(q, eps)
        covered_candidate += q
        if p > 0.0:
            kl += p * (math.log(p) - math.log(q))
    rest_teacher = max(0.0, 1.0 - sum(teacher_top.values()))
    rest_candidate = max(eps, 1.0 - covered_candidate)
    if rest_teacher > 0.0:
        kl += rest_teacher * (math.log(rest_teacher) - math.log(rest_candidate))
    k = max(len(teacher["top"]), 1)
    overlap = len({t for t, _ in teacher["top"]} &
                  {t for t, _ in candidate["top"]}) / k
    teacher_argmax = teacher["top"][0][0] if teacher["top"] else None
    candidate_argmax = candidate["top"][0][0] if candidate["top"] else None
    return {
        "kl": max(kl, 0.0),
        "top1": teacher_argmax is not None and
                teacher_argmax == candidate_argmax,
        "nll_candidate": -candidate["logprob"],
        "nll_teacher": -teacher["logprob"],
        "overlap": overlap,
        "imputed": bool(missing),
        "teacher_rank1": teacher["rank"] == 1,
        "nonfinite": teacher["nonfinite"] + candidate["nonfinite"],
    }


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = fraction * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(rows):
    kls = [row["kl"] for row in rows]
    finite_nll = [row["nll_candidate"] for row in rows
                  if math.isfinite(row["nll_candidate"])]
    teacher_nll = [row["nll_teacher"] for row in rows
                   if math.isfinite(row["nll_teacher"])]
    return {
        "positions": len(rows),
        "kl_mean": statistics.fmean(kls) if kls else float("nan"),
        "kl_median": percentile(kls, 0.5),
        "kl_p90": percentile(kls, 0.9),
        "kl_p99": percentile(kls, 0.99),
        "kl_max": max(kls) if kls else float("nan"),
        "top1_agreement": statistics.fmean(row["top1"] for row in rows)
                          if rows else float("nan"),
        "topk_overlap": statistics.fmean(row["overlap"] for row in rows)
                        if rows else float("nan"),
        "nll_candidate": statistics.fmean(finite_nll)
                         if finite_nll else float("nan"),
        "nll_teacher": statistics.fmean(teacher_nll)
                       if teacher_nll else float("nan"),
        "ppl_candidate": math.exp(statistics.fmean(finite_nll))
                         if finite_nll else float("nan"),
        "ppl_teacher": math.exp(statistics.fmean(teacher_nll))
                       if teacher_nll else float("nan"),
        "infinite_nll": len(rows) - len(finite_nll),
        "imputed_fraction": statistics.fmean(row["imputed"] for row in rows)
                            if rows else float("nan"),
        # Greedy histories: how often the teacher's scored top-1 is the token
        # it generated. Below 1.0 when decode and scoring numerics differ.
        "teacher_rank1": statistics.fmean(row["teacher_rank1"] for row in rows)
                         if rows else float("nan"),
        "nonfinite_logits": sum(row["nonfinite"] for row in rows),
    }


def compare(teacher_path, candidate_path):
    teacher = {record["id"]: record for record in read_jsonl(teacher_path)}
    candidate = {record["id"]: record for record in read_jsonl(candidate_path)}
    rows_by_category = {}
    skipped = []
    for ident, t in teacher.items():
        c = candidate.get(ident)
        if c is None:
            skipped.append(ident)
            continue
        if (t["digest"] != c["digest"] or t["start"] != c["start"] or
                t["top_k"] != c["top_k"] or
                len(t["positions"]) != len(c["positions"])):
            raise RuntimeError(f"{ident}: teacher and candidate scored "
                               "different histories")
        for tp, cp in zip(t["positions"], c["positions"]):
            if tp["token"] != cp["token"]:
                raise RuntimeError(f"{ident}: scored tokens differ")
            rows_by_category.setdefault(t["category"], []).append(
                compare_position(tp, cp))
    all_rows = [row for rows in rows_by_category.values() for row in rows]
    report = {
        "teacher_model": next(iter(teacher.values()), {}).get("model"),
        "candidate_model": next(iter(candidate.values()), {}).get("model"),
        "sequences": len(teacher) - len(skipped),
        "missing_candidate_sequences": skipped,
        "top_k": next(iter(teacher.values()), {}).get("top_k"),
        "overall": summarize(all_rows),
        "categories": {category: summarize(rows) for category, rows in
                       sorted(rows_by_category.items())},
    }
    return report


def print_report(report):
    columns = [("positions", "{:d}"), ("kl_mean", "{:.5f}"),
               ("kl_median", "{:.5f}"), ("kl_p90", "{:.5f}"),
               ("kl_p99", "{:.5f}"), ("top1_agreement", "{:.4f}"),
               ("topk_overlap", "{:.4f}"), ("nll_teacher", "{:.4f}"),
               ("nll_candidate", "{:.4f}"), ("imputed_fraction", "{:.3f}"),
               ("teacher_rank1", "{:.4f}")]
    header = ["category"] + [name for name, _ in columns]
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    rows = list(report["categories"].items()) + [("ALL", report["overall"])]
    for name, summary in rows:
        cells = [name] + [fmt.format(summary[key]) for key, fmt in columns]
        lines.append(" | ".join(cells))
    print(f"teacher={report['teacher_model']} "
          f"candidate={report['candidate_model']} "
          f"sequences={report['sequences']} top_k={report['top_k']}")
    print("\n".join(lines))
    if report["missing_candidate_sequences"]:
        print("missing candidate sequences: " +
              ", ".join(report["missing_candidate_sequences"]))
    overall = report["overall"]
    if overall["nonfinite_logits"] or overall["infinite_nll"]:
        print(f"WARNING: {overall['nonfinite_logits']} non-finite logits, "
              f"{overall['infinite_nll']} positions with infinite NLL")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--timeout", type=float, default=1800.0)

    history = sub.add_parser("history", parents=[common])
    history.add_argument("--teacher", required=True)
    history.add_argument("--out", required=True)
    history.add_argument("--prompts", help="JSON file; default: built-in 64")
    history.add_argument("--max-tokens", type=int, default=400)
    history.add_argument("--framing", choices=sorted(FRAMINGS), default="chat")
    history.add_argument("--limit", type=int, default=0)

    scoring = sub.add_parser("score", parents=[common])
    scoring.add_argument("--endpoint", required=True)
    scoring.add_argument("--history", required=True)
    scoring.add_argument("--out", required=True)
    scoring.add_argument("--top-k", type=int, default=20)
    scoring.add_argument("--include-prompt", action="store_true",
                         help="also score the prompt positions")
    scoring.add_argument("--concurrency", type=int, default=1)

    comparing = sub.add_parser("compare")
    comparing.add_argument("teacher")
    comparing.add_argument("candidate")
    comparing.add_argument("--json")

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--teacher", required=True)
    run.add_argument("--candidate", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--prompts")
    run.add_argument("--max-tokens", type=int, default=400)
    run.add_argument("--framing", choices=sorted(FRAMINGS), default="chat")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--include-prompt", action="store_true")
    run.add_argument("--concurrency", type=int, default=1)

    args = parser.parse_args(argv)
    if getattr(args, "top_k", 0) > 20:
        parser.error("--top-k is at most 20")
    if args.command == "history":
        generate_history(args)
    elif args.command == "score":
        score(args.endpoint, args.history, args.out, args.top_k,
              args.include_prompt, args.concurrency, args.timeout)
    elif args.command == "compare":
        report = compare(args.teacher, args.candidate)
        print_report(report)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        args.out = os.path.join(args.out_dir, "history.jsonl")
        generate_history(args)
        outputs = {"teacher": os.path.join(args.out_dir, "teacher.jsonl"),
                   "candidate": os.path.join(args.out_dir, "candidate.jsonl")}
        # Different hosts: score both at once.
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(score, getattr(args, role), args.out, path,
                                args.top_k, args.include_prompt,
                                args.concurrency, args.timeout)
                    for role, path in outputs.items()]
            for job in jobs:
                job.result()
        report = compare(outputs["teacher"], outputs["candidate"])
        print_report(report)
        with open(os.path.join(args.out_dir, "report.json"), "w",
                  encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
