"""Render BENCHMARKS.md tables from artifacts, keeping hand-entered cells."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .artifacts import artifact_path, load_artifact
from .config import BenchConfig, TableSpec

TODO = "TODO"
NA = "n/a"  # reference cell that cannot be measured on this host (see artifacts/unavailable.json)
MARKER_RE = re.compile(
    r"<!-- bench:(?P<id>[\w-]+) -->\n(?P<body>.*?)<!-- /bench -->", re.DOTALL
)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


@dataclass
class Column:
    header: str
    owner: str  # gufo | reference | gain | exact | label
    align: str = "---:"
    unit: str | None = None

    @property
    def label(self) -> str:
        return f"{self.header} ({self.unit})" if self.unit else self.header


@dataclass
class Layout:
    columns: list[Column]
    rows: list[str]  # row labels in order


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.replace("**", "").strip()
    if text == TODO or text in ("", "-", "n/a"):
        return None
    match = NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return TODO
    return f"{value:.{digits}f}{suffix}"


def _fmt_stat(row: dict[str, Any] | None, key: str, digits: int = 2, suffix: str = "") -> str | None:
    if row is None or row.get(key) is None:
        return None
    text = _fmt(row[key], digits)
    sd = row.get(f"{key}_sd")
    if sd is not None:
        text += f" ± {sd:.{digits}f}"
    return text + suffix


def gain(gufo: float | None, reference: float | None, better: str, unavailable: bool = False) -> str:
    if unavailable:
        return NA
    if gufo is None or reference is None or gufo <= 0 or reference <= 0:
        return TODO
    ratio = gufo / reference if better == "higher" else reference / gufo
    return f"{(ratio - 1) * 100:+.1f}%"


def _depth_label(depth: int) -> str:
    return f"{depth:,}"


def _merged_tokens(size: int) -> int:
    return (size // 32) ** 2


def model_label(config: BenchConfig, table: TableSpec) -> str:
    model = config.data.get("model", {})
    parts = [model.get("label", model.get("id", config.model))]
    if table.variant:
        parts.append(config.variant_label(table.variant))
    if table.kind == "single":
        parts.append(config.speculative["label"] if table.speculative else "AR")
    elif table.kind == "multi":
        parts.extend("AR" if mode == "ar" else config.speculative["label"]
                     for mode in table.spec.get("modes", ["ar"]))
    elif table.kind == "memory":
        parts.append("AR")
    elif table.kind == "image-encoder":
        parts.append("vision")
    return " ".join(parts)


def layout_for(config: BenchConfig, table: TableSpec) -> Layout:
    layout = _layout_for(config, table)
    layout.columns[0].header = f"{model_label(config, table)}<br>{layout.columns[0].header}"
    if table.kind in ("single", "multi", "loading"):
        for column in layout.columns[1:]:
            if column.owner in ("gufo", "reference"):
                column.unit = "s" if table.kind == "loading" else "tok/s"
    return layout


def _layout_for(config: BenchConfig, table: TableSpec) -> Layout:
    ref = config.reference_name
    spec_label = config.speculative["label"]
    kind = table.kind
    if kind == "loading":
        return Layout(
            [Column("Target", "label", "---"), Column("Gufo ready", "gufo"),
             Column(f"{ref} ready", "reference"), Column("Gain", "gain")],
            [config.variant_label(v) for v in config.variants],
        )
    if kind == "single" and not table.speculative:
        return Layout(
            [Column("Depth (tokens)", "label"), Column("Gufo pp", "gufo"), Column(f"{ref} pp", "reference"),
             Column("Gain", "gain"), Column("Gufo tg", "gufo"), Column(f"{ref} tg", "reference"),
             Column("Gain", "gain")],
            [_depth_label(d) for d in table.spec["depths"]],
        )
    if kind == "single":
        workloads = table.workload_tables()
        if workloads:
            columns = [Column("Depth (tokens)", "label"), Column("Gufo pp", "gufo")]
            if config.reference_speculative:
                columns += [Column(f"{ref} pp", "reference"), Column("Gain pp", "gain")]
            for workload in workloads:
                label = workload.spec["label"]
                reference = ref if config.reference_speculative else f"{ref} AR"
                columns += [Column(f"Gufo tg {label}", "gufo"),
                            Column(f"{reference} tg {label}", "reference"),
                            Column(f"Gain {label}", "gain")]
            return Layout(columns, [_depth_label(d) for d in table.spec["depths"]])
        if config.reference_speculative:
            columns = [Column("Depth (tokens)", "label"),
                       Column("Gufo pp", "gufo"), Column(f"{ref} pp", "reference"), Column("Gain", "gain"),
                       Column("Gufo tg", "gufo"), Column(f"{ref} tg", "reference"), Column("Gain", "gain")]
        else:
            columns = [Column("Depth (tokens)", "label"), Column("Gufo pp", "gufo"), Column("Gufo tg", "gufo"),
                       Column(f"{ref} AR tg", "reference"),
                       Column(f"Gain vs {ref} AR", "gain")]
        return Layout(columns, [_depth_label(d) for d in table.spec["depths"]])
    if kind == "multi":
        workloads = table.workload_tables()
        if workloads:
            columns = [Column("Users", "label")]
            for workload in workloads:
                label = workload.spec["label"]
                columns += [Column(f"Gufo {label}", "gufo"), Column(f"{ref} {label}", "reference"),
                            Column("Gain", "gain")]
            return Layout(columns, [str(c) for c in table.spec["concurrency"]])
        modes = table.spec.get("modes", ["ar"])
        if len(modes) == 1:
            label = "AR" if modes[0] == "ar" else spec_label
            return Layout(
                [Column("Users", "label"), Column(f"Gufo {label}", "gufo"),
                 Column(f"{ref} {label}", "reference"), Column("Gain", "gain")],
                [str(c) for c in table.spec["concurrency"]],
            )
        if config.reference_speculative:
            speculative = [Column(f"Gufo {spec_label}", "gufo"), Column(f"{ref} {spec_label}", "reference"),
                           Column("Gain", "gain")]
        else:
            speculative = [Column(f"Gufo {spec_label}", "gufo"), Column(f"Gain vs {ref} AR", "gain")]
        return Layout(
            [Column("Users", "label"), Column("Gufo AR", "gufo"), Column(f"{ref} AR", "reference"),
             Column("Gain", "gain"), *speculative, Column("Exact", "exact")],
            [str(c) for c in table.spec["concurrency"]],
        )
    if kind == "memory":
        return Layout(
            [Column("Workload", "label", "---"), Column("Gufo GiB", "gufo"),
             Column(f"{ref} GiB", "reference"), Column("Gain", "gain")],
            [w["id"] for w in table.spec["workloads"]],
        )
    if kind == "image-encoder":
        return Layout(
            [Column("RGB image", "label", "---"), Column("Merged tokens", "label"),
             Column("Gufo ms", "gufo"), Column(f"{ref} ms", "reference"), Column("Gain", "gain")],
            [f"{s}×{s}" for s in table.spec["sizes"]],
        )
    raise SystemExit(f"no renderer for table {table.id}")


def parse_table(body: str) -> dict[str, dict[str, str]]:
    """Existing cells keyed by row label and metric header, without display units."""
    lines = [line for line in body.strip().splitlines() if line.startswith("|")]
    if len(lines) < 2:
        return {}
    headers = [re.sub(r" \((?:tok/s|s)\)$", "", h.strip())
               for h in lines[0].strip().strip("|").split("|")][1:]
    rows: dict[str, dict[str, str]] = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0].replace("**", "")] = dict(zip(headers, cells[1:]))
    return rows


def existing_tables(document: str) -> dict[str, dict[str, dict[str, str]]]:
    return {m.group("id"): parse_table(m.group("body")) for m in MARKER_RE.finditer(document)}


def _row_values(config: BenchConfig, table: TableSpec) -> dict[str, dict[str, str | None]]:
    """Fresh cell text per row label from artifacts; None means no artifact value."""
    values: dict[str, dict[str, str | None]] = {}
    kind = table.kind
    workloads = table.workload_tables()
    if workloads:
        for workload in workloads:
            unavailable = unavailable_rows(config, workload) or set()
            for label, row in _row_values(config, workload).items():
                if "*" in unavailable or label in unavailable:
                    row["ref_unavailable"] = True
                values.setdefault(label, {}).update(
                    {f"{key}_{workload.spec['label']}": value for key, value in row.items()}
                )
        return values
    if kind == "multi":
        modes = table.spec.get("modes", ["ar"])
        if len(modes) == 1:
            mode = modes[0]
            gufo = load_artifact(artifact_path(config, table, "gufo", mode))
            reference = load_artifact(artifact_path(config, table, "reference", None if mode == "ar" else mode))
            for users in table.spec["concurrency"]:
                g, r = _serving_rate(gufo, users), _serving_rate(reference, users)
                values[str(users)] = {
                    "gufo": None if g is None else _fmt(g),
                    "reference": None if r is None else _fmt(r),
                    "ref_unavailable": _serving_unavailable(reference, users),
                }
            return values
        mode = config.speculative["mode"]
        gufo_ar = load_artifact(artifact_path(config, table, "gufo", "ar"))
        gufo_spec = load_artifact(artifact_path(config, table, "gufo", mode))
        reference = load_artifact(artifact_path(config, table, "reference"))
        ref_spec = load_artifact(artifact_path(config, table, "reference", mode))
        def rate(report: dict[str, Any] | None, users: int) -> str | None:
            value = _serving_rate(report, users)
            return None if value is None else _fmt(value)

        for users in table.spec["concurrency"]:
            values[str(users)] = {
                "gufo_ar": rate(gufo_ar, users),
                "reference": rate(reference, users),
                "gufo_spec": rate(gufo_spec, users),
                "ref_spec": rate(ref_spec, users),
                "exact": _serving_exact(reference, users) if reference else None,
                "ref_unavailable": _serving_unavailable(reference, users),
                "ref_spec_unavailable": _serving_unavailable(ref_spec, users),
            }
        return values

    gufo = load_artifact(artifact_path(config, table, "gufo"))
    if kind == "single" and table.speculative and not config.reference_speculative:
        ar_table = config.table("single-ar" + (f"-{table.variant}" if table.variant else ""))
        reference = load_artifact(artifact_path(config, ar_table, "reference"))
    else:
        reference = load_artifact(artifact_path(config, table, "reference"))
    g_rows = (gufo or {}).get("rows", {})
    r_rows = (reference or {}).get("rows", {})

    if kind == "loading":
        for variant in config.variants:
            values[config.variant_label(variant)] = {
                "gufo": _fmt_stat(g_rows.get(variant), "ready_s", 2),
                "reference": _fmt_stat(r_rows.get(variant), "ready_s", 2),
            }
    elif kind == "single":
        for depth in table.spec["depths"]:
            g, r = g_rows.get(str(depth)), r_rows.get(str(depth))
            values[_depth_label(depth)] = {
                "gufo_pp": _fmt_stat(g, "pp"), "gufo_tg": _fmt_stat(g, "tg"),
                "ref_pp": _fmt_stat(r, "pp"), "ref_tg": _fmt_stat(r, "tg"),
                "ref_unavailable": bool(r and r.get("unavailable")),
            }
    elif kind == "memory":
        for workload in table.spec["workloads"]:
            key = workload["id"]
            values[key] = {"gufo": _fmt_stat(g_rows.get(key), "gib"),
                           "reference": _fmt_stat(r_rows.get(key), "gib")}
    elif kind == "image-encoder":
        for size in table.spec["sizes"]:
            key = str(size)
            values[f"{size}×{size}"] = {"gufo": _fmt_stat(g_rows.get(key), "ms", 1),
                                        "reference": _fmt_stat(r_rows.get(key), "ms", 1)}
    return values


def _serving_rate(report: dict[str, Any] | None, users: int) -> float | None:
    """Mean of the summed individual decode rates in each measured cohort."""
    if report is None:
        return None
    result = report.get("results", {}).get(f"c{users}")
    if result is None or "unavailable" in result:
        return None
    rounds = result.get("rounds", [])
    samples = result.get("samples", [])
    if not rounds or not samples:
        return None
    if sum(round_.get("sampleCount", 0) for round_ in rounds) != len(samples):
        return None
    rates = [sample.get("decode_tokens_per_second") for sample in samples]
    if any(rate is None or not math.isfinite(rate) or rate < 0 for rate in rates):
        return None
    return math.fsum(rates) / len(rounds)


def _serving_unavailable(report: dict[str, Any] | None, users: int) -> bool:
    if report is None:
        return False
    result = report.get("results", {}).get(f"c{users}")
    return bool(result and result.get("unavailable"))


def _serving_exact(report: dict[str, Any] | None, users: int) -> str | None:
    if report is None:
        return None
    result = report.get("results", {}).get(f"c{users}")
    if result and "unavailable" in result:
        return None
    exactness = (result or {}).get("completionExactness")
    if not isinstance(exactness, dict):
        return None
    return f"{exactness['exactRequests']}/{exactness['comparedRequests']}"


def _pick(fresh: str | None, existing: str | None) -> str:
    """Gufo cells keep hand-entered text when no artifact covers them."""
    if fresh is not None:
        return fresh
    return existing if existing else TODO


def _ref(fresh: str | None, unavailable: bool = False) -> str:
    """Reference cells come from artifacts only, so layout changes never carry stale values."""
    if fresh is not None:
        return fresh
    return NA if unavailable else TODO


def unavailable_rows(config: BenchConfig, table: TableSpec) -> set[str] | None:
    """Row labels whose reference cells are declared unmeasurable; None for none, {'*'} for all."""
    path = config.artifacts_dir / "unavailable.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data.get(table.id) or data.get(table.base)
    if not entry:
        return None
    return set(entry)


def render_table(config: BenchConfig, table: TableSpec, existing: dict[str, dict[str, str]] | None) -> str:
    layout = layout_for(config, table)
    values = _row_values(config, table)
    better = table.spec.get("better", "higher")
    kind = table.kind
    lines = ["| " + " | ".join(c.label for c in layout.columns) + " |",
             "| " + " | ".join(c.align for c in layout.columns) + " |"]
    unavailable = unavailable_rows(config, table) or set()
    spec_unavailable = unavailable_rows(config, TableSpec(table.id, table.base + "-speculative", table.variant, table.spec)) or set()
    for label in layout.rows:
        old = (existing or {}).get(label, {})
        cell = old.get
        fresh = values.get(label, {})
        na = "*" in unavailable or label in unavailable or bool(fresh.get("ref_unavailable"))
        na_spec = na or "*" in spec_unavailable or label in spec_unavailable or bool(fresh.get("ref_spec_unavailable"))
        if kind == "loading" or kind == "memory":
            g = _pick(fresh.get("gufo"), cell("Gufo ready") or cell("Gufo GiB"))
            r = _ref(fresh.get("reference"), na)
            cells = [g, r, gain(_number(g), _number(r), better, r == NA)]
        elif kind == "single" and not table.speculative:
            gp = _pick(fresh.get("gufo_pp"), cell("Gufo pp")); rp = _ref(fresh.get("ref_pp"), na)
            gt = _pick(fresh.get("gufo_tg"), cell("Gufo tg")); rt = _ref(fresh.get("ref_tg"), na)
            cells = [gp, rp, gain(_number(gp), _number(rp), better, rp == NA),
                     gt, rt, gain(_number(gt), _number(rt), better, rt == NA)]
        elif kind == "single":
            workloads = table.workload_tables()
            if workloads:
                def best_pp(engine: str) -> str | None:
                    candidates = [fresh.get(f"{engine}_pp_{w.spec['label']}") for w in workloads]
                    return max((v for v in candidates if _number(v) is not None),
                               key=_number, default=None)

                gp = _pick(best_pp("gufo"), cell("Gufo pp"))
                cells = [gp]
                if config.reference_speculative:
                    rp = _ref(best_pp("ref"), na or all(
                        fresh.get(f"ref_unavailable_{w.spec['label']}") for w in workloads))
                    cells += [rp, gain(_number(gp), _number(rp), better, rp == NA)]
                for workload in workloads:
                    name = workload.spec["label"]
                    gt = _pick(fresh.get(f"gufo_tg_{name}"), cell(f"Gufo tg {name}"))
                    rt = _ref(fresh.get(f"ref_tg_{name}"),
                              na or bool(fresh.get(f"ref_unavailable_{name}")))
                    cells += [gt, rt, gain(_number(gt), _number(rt), better, rt == NA)]
                lines.append("| " + " | ".join([label, *cells]) + " |")
                continue
            gp = _pick(fresh.get("gufo_pp"), cell("Gufo pp")); gt = _pick(fresh.get("gufo_tg"), cell("Gufo tg"))
            rp = _ref(fresh.get("ref_pp"), na); rt = _ref(fresh.get("ref_tg"), na)
            if config.reference_speculative:
                cells = [gp, rp, gain(_number(gp), _number(rp), better, rp == NA),
                         gt, rt, gain(_number(gt), _number(rt), better, rt == NA)]
            else:
                cells = [gp, gt, rt, gain(_number(gt), _number(rt), better, rt == NA)]
        elif kind == "multi" and table.workload_tables():
            cells = []
            for workload in table.workload_tables():
                name = workload.spec["label"]
                g = _pick(fresh.get(f"gufo_{name}"), cell(f"Gufo {name}"))
                r = _ref(fresh.get(f"reference_{name}"), bool(fresh.get(f"ref_unavailable_{name}")))
                cells += [g, r, gain(_number(g), _number(r), better, r == NA)]
        elif kind == "multi" and len(table.spec.get("modes", ["ar"])) == 1:
            g = _pick(fresh.get("gufo"), cell(layout.columns[1].header))
            r = _ref(fresh.get("reference"), na)
            cells = [g, r, gain(_number(g), _number(r), better, r == NA)]
        elif kind == "multi":
            ga = _pick(fresh.get("gufo_ar"), cell("Gufo AR")); ra = _ref(fresh.get("reference"), na)
            gs = _pick(fresh.get("gufo_spec"), cell(f"Gufo {config.speculative['label']}")); ex = _ref(fresh.get("exact"), na)
            cells = [ga, ra, gain(_number(ga), _number(ra), better, ra == NA), gs]
            if config.reference_speculative:
                rs = _ref(fresh.get("ref_spec"), na_spec)
                cells += [rs, gain(_number(gs), _number(rs), better, rs == NA), ex]
            else:
                cells += [gain(_number(gs), _number(ra), better, ra == NA), ex]
        elif kind == "image-encoder":
            size = int(label.split("×")[0])
            g = _pick(fresh.get("gufo"), cell("Gufo ms")); r = _ref(fresh.get("reference"), na)
            cells = [str(_merged_tokens(size)), g, r, gain(_number(g), _number(r), better, r == NA)]
        else:
            raise SystemExit(f"no renderer for table {table.id}")
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines) + "\n"


def render_document(config: BenchConfig, document: str, only: set[str] | None = None) -> tuple[str, list[str]]:
    existing = existing_tables(document)
    known = {t.id: t for t in config.tables()}
    rendered: list[str] = []

    def replace(match: re.Match[str]) -> str:
        table_id = match.group("id")
        if table_id not in known or (only and table_id not in only):
            return match.group(0)
        body = render_table(config, known[table_id], existing.get(table_id))
        rendered.append(table_id)
        return f"<!-- bench:{table_id} -->\n{body}<!-- /bench -->"

    return MARKER_RE.sub(replace, document), rendered


def _serving_output_tokens(result: dict[str, Any]) -> int | None:
    samples = result.get("samples")
    if isinstance(samples, list):
        return sum(int(s.get("completion_tokens", 0)) for s in samples)
    return None


def multi_summary(config: BenchConfig) -> list[str]:
    """One line per concurrency artifact: top-level request latency, acceptance and cache hits."""
    lines: list[str] = []
    for table in (workload for parent in config.tables()
                  for workload in (parent.workload_tables() or [parent])):
        if table.kind != "multi":
            continue
        pairs = [(target, None if target == "reference" and mode == "ar" else mode)
                 for target in ("gufo", "reference") for mode in table.spec.get("modes", ["ar"])]
        for target, suffix in pairs:
            report = load_artifact(artifact_path(config, table, target, suffix))
            if report is None:
                continue
            measured = [int(k[1:]) for k, v in report.get("results", {}).items() if "unavailable" not in v]
            if not measured:
                continue
            top = max(measured)
            result = report["results"][f"c{top}"]
            latency = result.get("latency", {}).get("request_ms", {})
            spec = result.get("speculative") or {}
            parts = [f"{table.id} {target}-{suffix or 'ar'} C{top}:",
                     f"request median {latency.get('median', 0) / 1000:.2f} s / p95 {latency.get('p95', 0) / 1000:.2f} s"]
            accepted = spec.get("acceptedTokens")
            output = _serving_output_tokens(result)
            if accepted and output and output > accepted:
                parts.append(f"accepted/step {accepted / (output - accepted):.2f}")
            parts.append(f"cache hits {spec.get('cacheHits', 0)}/{result.get('requestCount')}")
            lines.append(" ".join(parts))
    return lines


def hand_cells(config: BenchConfig, document: str) -> dict[str, list[str]]:
    """Rows per table whose Gufo cells hold text no artifact provides (hand-entered, possibly stale)."""
    tables = existing_tables(document)
    result: dict[str, list[str]] = {}
    for table in config.tables():
        if table.id not in tables:
            continue
        layout = layout_for(config, table)
        owned = [c.header for c in layout.columns[1:] if c.owner == "gufo"]
        fresh = _row_values(config, table)
        rows = []
        for label, cells in tables[table.id].items():
            provided = any(v is not None for k, v in fresh.get(label, {}).items() if k.startswith("gufo"))
            has_text = any(cells.get(h, TODO) != TODO for h in owned)
            if has_text and not provided:
                rows.append(label)
        if rows:
            result[table.id] = rows
    return result


def todo_rows(config: BenchConfig, document: str, table: TableSpec, target: str,
              workload: str | None = None) -> set[str] | None:
    """Row labels whose target-owned cells are TODO; None when the table is absent."""
    tables = existing_tables(document)
    if table.id not in tables:
        return None
    layout = layout_for(config, table)
    owned = [c.header for c in layout.columns[1:]
             if (c.owner == target or (target == "reference" and c.owner == "exact"))
             and (workload is None or c.header.endswith(f" {workload}")
                  or (table.kind == "single" and c.header.endswith(" pp")))]
    todo: set[str] = set()
    for label, cells in tables[table.id].items():
        if any(cells.get(h, TODO) == TODO for h in owned):
            todo.add(label)
    return todo
