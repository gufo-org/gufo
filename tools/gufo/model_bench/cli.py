"""Command line: run tables for one target, or render BENCHMARKS.md."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from gufo.serving_bench import load_fingerprint, source_identity

from .config import TARGETS, BenchConfig, load_config, parse_file_args

FILE_ROLES = ("gguf", "draft", "mtp", "dspark", "mmproj")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-bench",
        description="Measure docs/models/<model>/BENCHMARKS.md tables for Gufo and its reference server.",
    )
    parser.add_argument("--model", required=True, help="docs/models/<model> directory name")
    parser.add_argument("--gufo", type=Path, default=Path("result/bin/gufo"), help="Gufo binary")
    for role in FILE_ROLES:
        parser.add_argument(f"--{role}", action="append", default=[], metavar="[VARIANT=]PATH")
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/model-bench"),
                        help="server logs (ignored directory)")
    parser.add_argument("--config", type=Path, default=None, help="bench.json override (experiments)")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="write/read artifacts here instead of docs/models/<model>/artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure tables for one target")
    run.add_argument("--target", choices=TARGETS, required=True)
    run.add_argument("--table", action="append", default=[], help="table id (repeatable, comma-separated)")
    run.add_argument("--todo", action="store_true", help="measure only rows that are TODO in BENCHMARKS.md")
    run.add_argument("--reference-binary", default=None, help="reference server executable (default from bench.json)")
    run.add_argument("--drop-caches", default=None,
                     help="shell command that evicts model file pages before each loading launch")
    run.add_argument("--repetitions", type=int, default=None,
                     help="override each table's repetitions (mean ± sd is reported above 1)")
    run.add_argument("--fresh", action="store_true",
                     help="discard rows of an existing artifact instead of merging into them")
    run.add_argument("--depths", default=None, help="comma-separated depths to measure in single-user tables")
    run.add_argument("--mode", action="append", default=[], help="restrict to a mode: ar or the speculative mode")
    run.add_argument("--context", type=int, default=None,
                     help="override the single-user tables' context capacity (e.g. to fit a reference server in RAM)")

    render = sub.add_parser("render", help="rewrite marked tables in BENCHMARKS.md from artifacts")
    render.add_argument("--table", action="append", default=[])
    render.add_argument("--check", action="store_true", help="exit 1 when the document would change")
    render.add_argument("--no-charts", action="store_true", help="skip SVG charts under artifacts/charts")

    sub.add_parser("tables", help="list table ids and their artifacts")
    return parser


def _table_ids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def cmd_tables(config: BenchConfig) -> int:
    from .artifacts import artifact_path

    for table in config.tables():
        sources = table.workload_tables() or [table]
        paths = [artifact_path(config, source, target) for source in sources for target in TARGETS]
        if table.kind == "multi":
            modes = table.spec.get("modes", ["ar"])
            paths = [artifact_path(config, source, "gufo", m) for source in sources for m in modes]
            paths += [artifact_path(config, source, "reference", None if m == "ar" else m)
                      for source in sources for m in modes]
        present = [p.name for p in paths if p.exists()]
        print(f"{table.id:28} {table.kind:14} {', '.join(present) or '-'}")
    return 0


def cmd_render(config: BenchConfig, args: argparse.Namespace) -> int:
    from .render import render_document

    document = config.benchmarks_path.read_text(encoding="utf-8")
    updated, rendered = render_document(config, document, _table_ids(args.table) or None)
    charts: list[str] = []
    if not args.no_charts and not args.check:
        from .charts import render_charts

        updated, charts = render_charts(config, updated, _table_ids(args.table) or None)
    if args.check:
        if updated != document:
            print(f"{config.benchmarks_path}: tables out of date")
            return 1
        print("up to date")
        return 0
    if updated != document:
        config.benchmarks_path.write_text(updated, encoding="utf-8")
    from .render import hand_cells, multi_summary

    for line in multi_summary(config):
        print(line)
    for table_id, rows in hand_cells(config, updated).items():
        print(f"{table_id}: hand-entered Gufo cells kept for rows {', '.join(rows)} (no artifact covers them)")
    print(f"rendered {len(rendered)} tables and {len(charts)} charts in {config.benchmarks_path}")
    return 0


def cmd_run(config: BenchConfig, args: argparse.Namespace) -> int:
    if config.category != "llm":
        raise SystemExit(f"category {config.category!r} is not implemented yet")
    from . import llm

    root = config.root
    gufo_binary = args.gufo if args.gufo.is_absolute() else (root / args.gufo)
    if not gufo_binary.exists():
        raise SystemExit(f"Gufo binary not found: {gufo_binary} (run `nix build`)")
    reference_binary = args.reference_binary or config.data["reference"]["server"]
    if args.target == "reference":
        binaries = [reference_binary, config.speculative.get("reference", {}).get("server")]
        for binary in filter(None, binaries):
            if shutil.which(binary) is None:
                raise SystemExit(f"{binary} not on PATH; enter `nix develop`")
    revision, dirty = source_identity(root)
    fingerprint = load_fingerprint(None, gufo_binary)
    document = config.benchmarks_path.read_text(encoding="utf-8") if config.benchmarks_path.exists() else ""
    session = llm.Session(
        config, args.target, gufo_binary=gufo_binary, reference_binary=reference_binary,
        source={"revision": revision, "dirty": dirty, "buildMode": "nix-release"},
        fingerprint=fingerprint, log_dir=(args.log_dir if args.log_dir.is_absolute() else root / args.log_dir),
        document=document, todo_only=args.todo, drop_caches=args.drop_caches,
        repetitions=args.repetitions, fresh=args.fresh,
        depths=[int(d) for d in args.depths.split(",")] if args.depths else None,
        modes=args.mode or None, context=args.context,
    )
    wanted = _table_ids(args.table)
    tables = [t for t in config.tables() if not wanted or t.id in wanted]
    unknown = wanted - {t.id for t in config.tables()}
    if unknown:
        raise SystemExit(f"unknown tables: {', '.join(sorted(unknown))}")
    for table in tables:
        print(f"== {table.id} ({args.target})")
        llm.run_table(session, table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(_root(), args.model, args.config)
    if args.artifacts_dir is not None:
        config.artifacts_override = args.artifacts_dir.resolve()
    parse_file_args(config, {role: getattr(args, role) for role in FILE_ROLES})
    if args.command == "tables":
        return cmd_tables(config)
    if args.command == "render":
        return cmd_render(config, args)
    return cmd_run(config, args)
