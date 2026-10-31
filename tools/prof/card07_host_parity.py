#!/usr/bin/env python3
"""Card 07: per-source compile-flag + object parity for every moved Qwen source.

Old tree = parent revision (Card 06) production build build/card06-release.
New tree = working copy production build build/card07-release.
"""
import json, re, shlex, subprocess, collections, os, sys

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', required=True)
parser.add_argument('--before', required=True)
parser.add_argument('--after', required=True)
args = parser.parse_args()
REPO, OLD, NEW = args.root, args.before, args.after


def load(path):
    out = {}
    for entry in json.load(open(path)):
        out[entry["file"].replace(REPO + "/", "")] = entry
    return out


def map_path(old):
    """Directory mapping applied by the Card 07 split."""
    r = old[len("src/models/qwen/"):]
    F = "src/families/qwen/"
    if r.startswith("hip/kernels/"):
        return F + "kernels/" + r[len("hip/kernels/"):]
    if r.startswith("hip/ops/"):
        return F + "kernels/ops/" + r[len("hip/ops/"):]
    if r == "hip/ops.hpp":
        return F + "kernels/ops.hpp"
    if r in ("tokenizer.hpp", "tokenizer.cpp", "chat_template.hpp", "chat_template.cpp"):
        return F + "text/" + r
    if r in ("continuation_hooks.hpp", "gemm_route.hpp", "hip/quant_ops.hpp",
             "hip/execution_policy.hpp", "hip/detail/attention_policy.hpp",
             "hip/detail/recurrent_state.hpp"):
        return F + "hip_common/" + os.path.basename(r)
    if r == "vision/rope.hpp":
        return F + "hip_common/rope.hpp"
    if r.startswith("vision/"):
        return F + "vision/" + r[len("vision/"):]
    return old  # stays with the Qwen 27B model library


DROP_VALUES = {"-o", "-c", "-MF", "-MT", "-MQ"}


def normalize(entry):
    flags, skip = [], False
    for arg in shlex.split(entry["command"]):
        if skip:
            skip = False
            continue
        if arg in DROP_VALUES:
            skip = True
            continue
        if (arg.endswith((".cpp", ".hip", ".cpp.o", ".hip.o"))
                or arg.startswith(REPO)):
            continue
        flags.append(arg)
    return [f for f in flags if not f.startswith(("-DGUFO_FFMPEG_EXECUTABLE=", "-DGUFO_FFPROBE_EXECUTABLE="))]


def symbols(obj):
    out = subprocess.run(["nm", "--defined-only", "--demangle", obj],
                         capture_output=True, text=True, check=True).stdout
    return {line.split(" ", 2)[-1] for line in out.splitlines() if line.strip()}


def disasm(obj):
    """symbol -> list of (mnemonic, operand text); leading addresses dropped."""
    proc = subprocess.run(["objdump", "-d", "--no-show-raw-insn", obj],
                          capture_output=True, text=True, check=True).stdout
    funcs, current, body = {}, None, []
    for line in proc.splitlines():
        if line and not line.startswith(" ") and line.endswith(">:") and "<" in line:
            if current:
                funcs[current] = body
            current = line[line.index("<") + 1:-2]
            body = []
        elif line.strip().endswith(":") and line.startswith(" "):
            continue
        elif current and line.strip():
            # keep mnemonic + operands, drop the leading address
            parts = line.strip().split(" ", 1)
            body.append((parts[0], parts[1].split("\t")[-1].strip() if len(parts) > 1 else ""))
    if current:
        funcs[current] = body
    return funcs


CUID = re.compile(r'__hip_cuid_[0-9a-fA-F]+')


def classify(symbol, before, after):
    """Return None when equal, a category for an expected move artifact, else a reason.

    Moving a translation unit changes only two things in host code: the
    compiler-generated per-TU `__hip_cuid_*` identifier and the offsets of
    `__FILE__`-derived string literals inside the local label pool. Any other
    difference is a real code change and must fail the gate.
    """
    if before == after:
        return None
    if len(before) != len(after):
        return "instruction count changed"
    if [m for m, _ in before] != [m for m, _ in after]:
        return "opcode or control flow changed"
    moved = [(o1, o2) for (_, o1), (_, o2) in zip(before, after) if o1 != o2]
    # The per-TU CU identifier also appears as a call target inside functions
    # (for example __clang_call_terminate), so mask it before comparing.
    if all(CUID.sub('__hip_cuid_*', o1) == CUID.sub('__hip_cuid_*', o2)
           for o1, o2 in moved):
        return None
    if all(".L" in o1 or ".L" in o2 for o1, o2 in moved):
        return None
    return "operand changed outside the CU id/local label pool: " + str(moved[0])[:120]


def main():
    old, new = load(OLD), load(NEW)
    sources = sorted(s for s in old if s.startswith("src/models/qwen/"))
    counts = collections.Counter()
    report = []
    for src_old in sources:
        src_new = map_path(src_old)
        if src_new == src_old and src_old not in new:
            counts["stale-path"] += 1
            report.append(("STALE", src_old, src_new, "", []))
            continue
        if src_new not in new:
            counts["missing"] += 1
            report.append(("MISSING-IN-NEW", src_old, src_new, "", []))
            continue
        fo, fn = normalize(old[src_old]), normalize(new[src_new])
        flag_status = "IDENTICAL" if fo == fn else "FLAG-DIFF"
        extra = [f for f in fn if f not in fo]
        gone = [f for f in fo if f not in fn]
        o_obj, n_obj = old[src_old]["output"], new[src_new]["output"]
        so, sn = symbols(o_obj), symbols(n_obj)
        sym_status = "SYMBOLS-IDENTICAL" if so == sn else "SYMBOL-DIFF"
        code, unexplained = "n/a", []
        if os.path.exists(o_obj) and os.path.exists(n_obj):
            do, dn = disasm(o_obj), disasm(n_obj)
            common = set(do) & set(dn)
            same = 0
            for name in sorted(common):
                reason = classify(name, do[name], dn[name])
                if reason is None:
                    same += 1
                else:
                    unexplained.append(f"{name}: {reason}")
            code = (f"{same}/{len(common)} functions disassembly-equal or "
                    f"move-artifact-only (old {len(do)}, new {len(dn)})")
            if unexplained:
                code += " UNEXPLAINED=" + str(len(unexplained))
        counts[(flag_status, sym_status)] += 1
        report.append((f"{flag_status}/{sym_status}", src_old, src_new,
                       f"{code} -{gone} +{extra}", unexplained))
    print("=== per-source parity (old card06-release -> new card07-release) ===")
    for key, value in sorted(counts.items(), key=lambda kv: str(kv[0])):
        print(f"  {key}: {value}")
    print(f"  total compiled sources compared: {len(sources)}")
    print("\n=== non-clean rows ===")
    shown = 0
    for row in report:
        if row[0].startswith("IDENTICAL/SYMBOLS-IDENTICAL") and not row[4]:
            continue
        shown += 1
        print(" ", row[0], "|", row[1], "->", row[2])
        if row[3]:
            print("      ", row[3])
        for detail in row[4]:
            print("        UNEXPLAINED", detail)
    if shown == 0:
        print("  (none)")
    failures = [row for row in report
                if not row[0].startswith("IDENTICAL") or row[4]]
    print(f"\nhost-parity: sources={len(sources)} unexplained-code-differences="
          f"{sum(len(row[4]) for row in report)} "
          f"flag-or-path-failures={len([r for r in failures if not r[4]])}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
