#!/usr/bin/env python3
"""GPU kernel profiler front end for rocprofv3 result databases.

Beyond a flat kernel table this reports the three things that actually drive
optimization decisions on gfx1151:

  * pipeline stage rollup -- kernels grouped into model stages, so a change is
    scored against the stage it targets rather than a kernel name;
  * occupancy of the timeline -- GPU busy time versus wall span, plus the
    largest idle gaps and which dispatch precedes them, which is how launch
    bound and host bound phases become visible;
  * A/B comparison -- two runs diffed per kernel and per stage, which is the
    form every experiment in benchmarks/qwen3.8-27b/README.md is recorded in.

Usage
-----
  tools/prof/prof.py run  [-o DIR] [--stages qwen] -- <command> [args...]
  tools/prof/prof.py show DB [--stages qwen] [--top N] [--gaps N] [--json]
  tools/prof/prof.py diff BEFORE_DB AFTER_DB [--stages qwen] [--top N]

`run` needs rocprofv3 on PATH, so invoke it inside `nix develop`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

# Kernel-name substring -> pipeline stage. First match wins, so order matters.
STAGE_MAPS: dict[str, list[tuple[str, str]]] = {
    "ds4": [
        ("GroupedQ2Down", "moe: down"),
        ("moe_gate_up", "moe: gate+up"),
        ("moe_down", "moe: down"),
        ("moe_sum", "moe: reduce"),
        ("moe_", "moe: routing/layout"),
        ("mul_mat_q", "moe/dense: mmq"),
        ("dspark", "support: markov/fusion"),
        ("indexer", "attention: indexer"),
        ("compress", "attention: compressor"),
        ("attention", "attention"),
        ("attn_", "attention"),
        ("flash_attn", "attention"),
        ("hc_", "hyperconnections"),
        ("matmul", "dense projections"),
        ("Cijk", "dense: hipblaslt"),
        ("gemm", "dense: gemm"),
        ("rms", "norm"),
        ("rope", "rope"),
        ("quantize", "quantize"),
        ("f16", "convert/dense"),
        ("argmax", "sample"),
        ("fillBuffer", "runtime: fill"),
        ("copyBuffer", "runtime: copy"),
    ],
    "qwen": [
        ("W8A8Dual", "gemm: ffn gate+up"),
        ("W8A8Blocked", "gemm: blocked w8a8"),
        ("W8A8Wmma", "gemm: w8a8 16-row"),
        ("Cijk", "gemm: hipblaslt bf16"),
        ("TiledBatchedQ8_0GEMM", "gemm: tiled q8"),
        ("QuantGEMV", "gemm: quant gemv"),
        ("GEMVKernel", "gemm: gemv"),
        ("hipblas", "gemm: hipblas"),
        ("DeltaNetPrep", "ssm: deltanet prologue"),
        ("DeltaNet", "ssm: deltanet recurrence"),
        ("SSMConv", "ssm: conv1d"),
        ("SSMPostNormGate", "ssm: post-norm gate"),
        ("SSM", "ssm: other"),
        ("Attention", "attention"),
        ("attn_fwd", "attention: aotriton prefix"),
        ("QKNormRoPE", "attention: qk-norm+rope+kv"),
        ("RoPE", "attention: rope"),
        ("PackTiledAttentionKv", "attention: kv pack"),
        ("FusedRMSNormQuantize", "norm+quantize (fused)"),
        ("SwiGLU", "ffn: swiglu"),
        ("Quantize", "quantize: activations"),
        ("RMSNorm", "norm"),
        ("ResidualAdd", "residual"),
        ("FloatToBfloat16", "convert"),
        ("Embedding", "embed"),
        ("UnpackQG", "unpack"),
        ("Argmax", "sample"),
        ("Dequantize", "dequant"),
        ("fillBuffer", "runtime: fill"),
        ("copyBuffer", "runtime: copy"),
    ],
}


def stage_of(name: str, stages: list[tuple[str, str]]) -> str:
    for needle, stage in stages:
        if needle in name:
            return stage
    return "other"


def short(name: str) -> str:
    # A __global__ function with internal linkage (anonymous namespace) can reach
    # the database with an empty display_name, which would otherwise silently
    # collapse several kernels into one nameless row. Surface that instead of
    # hiding it -- the fix belongs in the kernel, not here.
    if not name or not name.strip():
        return "<unnamed: internal-linkage kernel>"
    name = name.split("(")[0]
    for prefix in ("void ", "gufo::hip::", "gufo::"):
        name = name.replace(prefix, "")
    return name


@dataclass
class KernelStat:
    calls: int = 0
    ns: float = 0.0
    max_ns: float = 0.0
    shape: str = ""
    grids: set[str] = field(default_factory=set)


@dataclass
class Profile:
    kernels: dict[str, KernelStat]
    busy_ns: float
    wall_ns: float
    gaps: list[tuple[float, str, str]]  # (ns, before, after)
    invalid_dispatches: int = 0

    @property
    def total_ns(self) -> float:
        return sum(k.ns for k in self.kernels.values())

    @property
    def dispatches(self) -> int:
        return sum(k.calls for k in self.kernels.values())


def load(db_path: str) -> Profile:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT ks.display_name, kd.start, kd.end,
               kd.grid_size_x, kd.grid_size_y, kd.grid_size_z,
               kd.workgroup_size_x, kd.workgroup_size_y, kd.workgroup_size_z
        FROM rocpd_kernel_dispatch kd
        JOIN rocpd_info_kernel_symbol ks ON ks.id = kd.kernel_id
        ORDER BY kd.start
        """
    ).fetchall()
    conn.close()
    if not rows:
        raise SystemExit(f"{db_path}: no kernel dispatches recorded")

    kernels: dict[str, KernelStat] = defaultdict(KernelStat)
    intervals: list[tuple[float, float, str]] = []
    invalid_dispatches = 0
    for name, start, end, gx, gy, gz, wx, wy, wz in rows:
        if end <= start:
            invalid_dispatches += 1
            continue
        key = short(name)
        dur = float(end - start)
        st = kernels[key]
        st.calls += 1
        st.ns += dur
        wg = wx * wy * wz or 1
        blocks = (gx // wx if wx else gx, gy // wy if wy else gy, gz // wz if wz else gz)
        st.grids.add(f"{blocks[0]}x{blocks[1]}x{blocks[2]}/{wg}")
        if dur > st.max_ns:
            st.max_ns = dur
            st.shape = f"blocks={blocks[0]}x{blocks[1]}x{blocks[2]} wg={wg}"
        intervals.append((float(start), float(end), key))

    if not intervals:
        raise SystemExit(f"{db_path}: no valid kernel dispatches recorded")

    # Union of busy intervals plus the biggest idle gaps between them.
    intervals.sort()
    busy = 0.0
    gaps: list[tuple[float, str, str]] = []
    cur_start, cur_end, cur_name = intervals[0]
    prev_name = cur_name
    for start, end, name in intervals[1:]:
        if start > cur_end:
            busy += cur_end - cur_start
            gaps.append((start - cur_end, prev_name, name))
            cur_start, cur_end = start, end
            prev_name = name
        elif end > cur_end:
            cur_end = end
            prev_name = name
    busy += cur_end - cur_start
    wall = cur_end - intervals[0][0]
    gaps.sort(reverse=True)
    return Profile(dict(kernels), busy, wall, gaps, invalid_dispatches)


def rollup(prof: Profile, stages: list[tuple[str, str]]) -> dict[str, tuple[int, float]]:
    out: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for name, st in prof.kernels.items():
        entry = out[stage_of(name, stages)]
        entry[0] += st.calls
        entry[1] += st.ns
    return {k: (int(v[0]), v[1]) for k, v in out.items()}


def cmd_show(args: argparse.Namespace) -> int:
    prof = load(args.db)
    stages = STAGE_MAPS.get(args.stages, [])
    total = prof.total_ns

    if args.json:
        print(
            json.dumps(
                {
                    "total_ms": total / 1e6,
                    "busy_ms": prof.busy_ns / 1e6,
                    "wall_ms": prof.wall_ns / 1e6,
                    "dispatches": prof.dispatches,
                    "invalid_dispatches": prof.invalid_dispatches,
                    "kernels": {
                        n: {"calls": s.calls, "ms": s.ns / 1e6, "shape": s.shape}
                        for n, s in prof.kernels.items()
                    },
                    "stages": {k: {"calls": c, "ms": ns / 1e6} for k, (c, ns) in rollup(prof, stages).items()},
                },
                indent=2,
            )
        )
        return 0

    idle = prof.wall_ns - prof.busy_ns
    print(f"dispatches      : {prof.dispatches}")
    if prof.invalid_dispatches:
        print(f"invalid records : {prof.invalid_dispatches} (ignored)")
    print(f"kernel time sum : {total / 1e6:10.2f} ms")
    print(f"gpu busy (union): {prof.busy_ns / 1e6:10.2f} ms")
    print(f"wall span       : {prof.wall_ns / 1e6:10.2f} ms")
    print(
        f"idle in span    : {idle / 1e6:10.2f} ms  ({100.0 * idle / prof.wall_ns:.1f}% "
        f"of span -- launch or host bound if large)"
    )
    overlap = total - prof.busy_ns
    if overlap > 0.01 * total:
        print(f"concurrent exec : {overlap / 1e6:10.2f} ms of kernel time overlapped")

    if stages:
        print("\n-- pipeline stages --")
        print(f"{'stage':<28} {'calls':>7} {'total ms':>10} {'%':>7}")
        for stage, (calls, ns) in sorted(rollup(prof, stages).items(), key=lambda kv: -kv[1][1]):
            print(f"{stage:<28} {calls:>7} {ns / 1e6:>10.2f} {100.0 * ns / total:>6.1f}%")

    print(f"\n-- kernels (top {args.top}) --")
    print(f"{'kernel':<52} {'calls':>6} {'total ms':>10} {'mean us':>9} {'%':>7}  shape")
    for name, st in sorted(prof.kernels.items(), key=lambda kv: -kv[1].ns)[: args.top]:
        extra = f"{st.shape}"
        if len(st.grids) > 1:
            extra += f"  (+{len(st.grids) - 1} other shapes)"
        print(
            f"{name[:52]:<52} {st.calls:>6} {st.ns / 1e6:>10.2f} "
            f"{st.ns / st.calls / 1e3:>9.2f} {100.0 * st.ns / total:>6.1f}%  {extra}"
        )

    if args.gaps and prof.gaps:
        print(f"\n-- largest idle gaps (top {args.gaps}) --")
        print(f"{'gap us':>10}  after -> before")
        for ns, before, after in prof.gaps[: args.gaps]:
            print(f"{ns / 1e3:>10.1f}  {before[:34]} -> {after[:34]}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    a = load(args.before)
    b = load(args.after)
    stages = STAGE_MAPS.get(args.stages, [])
    if a.invalid_dispatches or b.invalid_dispatches:
        print(
            "invalid records ignored: "
            f"before={a.invalid_dispatches} after={b.invalid_dispatches}"
        )
    print(
        f"kernel time  {a.total_ns / 1e6:.2f} ms -> {b.total_ns / 1e6:.2f} ms "
        f"({100.0 * (b.total_ns - a.total_ns) / a.total_ns:+.1f}%)"
    )
    print(
        f"gpu busy     {a.busy_ns / 1e6:.2f} ms -> {b.busy_ns / 1e6:.2f} ms   "
        f"wall {a.wall_ns / 1e6:.2f} -> {b.wall_ns / 1e6:.2f} ms"
    )

    if stages:
        ra, rb = rollup(a, stages), rollup(b, stages)
        print("\n-- stage delta (ms, sorted by absolute change) --")
        print(f"{'stage':<28} {'before':>10} {'after':>10} {'delta':>10} {'%':>8}")
        keys = set(ra) | set(rb)
        deltas = []
        for k in keys:
            before = ra.get(k, (0, 0.0))[1] / 1e6
            after = rb.get(k, (0, 0.0))[1] / 1e6
            deltas.append((abs(after - before), k, before, after))
        for _, k, before, after in sorted(deltas, reverse=True):
            pct = (100.0 * (after - before) / before) if before else float("inf")
            print(f"{k:<28} {before:>10.2f} {after:>10.2f} {after - before:>+10.2f} {pct:>+7.1f}%")

    print(f"\n-- kernel delta (top {args.top} by absolute change) --")
    print(f"{'kernel':<52} {'before':>10} {'after':>10} {'delta':>10}")
    keys = set(a.kernels) | set(b.kernels)
    rows = []
    for k in keys:
        before = a.kernels[k].ns / 1e6 if k in a.kernels else 0.0
        after = b.kernels[k].ns / 1e6 if k in b.kernels else 0.0
        rows.append((abs(after - before), k, before, after))
    for _, k, before, after in sorted(rows, reverse=True)[: args.top]:
        print(f"{k[:52]:<52} {before:>10.2f} {after:>10.2f} {after - before:>+10.2f}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        raise SystemExit("run: expected a command after --")
    if shutil.which("rocprofv3") is None:
        raise SystemExit("rocprofv3 not on PATH; run inside `nix develop`")
    out_dir = args.out or tempfile.mkdtemp(prefix="strixprof-")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag
    proc = subprocess.run(
        ["rocprofv3", "--kernel-trace", "-d", out_dir, "-o", tag, "--", *args.command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode:
        sys.stdout.write(proc.stdout)
        raise SystemExit(proc.returncode)
    db = os.path.join(out_dir, f"{tag}_results.db")
    if not os.path.exists(db):
        sys.stdout.write(proc.stdout)
        raise SystemExit(f"rocprofv3 produced no database at {db}")
    if args.show_output:
        sys.stdout.write(proc.stdout)
    print(f"database: {db}\n")
    args.db = db
    return cmd_show(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--stages", default="qwen", help="stage map name, or '' to disable")
        p.add_argument("--top", type=int, default=25)

    p_show = sub.add_parser("show", help="analyze an existing rocprofv3 database")
    p_show.add_argument("db")
    p_show.add_argument("--gaps", type=int, default=8)
    p_show.add_argument("--json", action="store_true")
    common(p_show)
    p_show.set_defaults(func=cmd_show)

    p_run = sub.add_parser("run", help="profile a command and analyze the result")
    p_run.add_argument("-o", "--out", default=None, help="output directory")
    p_run.add_argument("--tag", default="prof")
    p_run.add_argument("--gaps", type=int, default=8)
    p_run.add_argument("--json", action="store_true")
    p_run.add_argument("--show-output", action="store_true")
    common(p_run)
    p_run.add_argument("command", nargs=argparse.REMAINDER)
    p_run.set_defaults(func=cmd_run)

    p_diff = sub.add_parser("diff", help="compare two profiles")
    p_diff.add_argument("before")
    p_diff.add_argument("after")
    common(p_diff)
    p_diff.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
