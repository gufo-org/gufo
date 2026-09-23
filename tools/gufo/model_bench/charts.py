"""SVG charts rendered from the tables in BENCHMARKS.md (an addition to the tables)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .config import BenchConfig, TableSpec
from .render import MARKER_RE, _number, layout_for, parse_table

# Categorical slots from the validated default palette: Gufo, reference, Gufo speculative.
COLORS = {"gufo": "#2a78d6", "reference": "#eb6834", "spec": "#1baf7a", "ref_spec": "#eda100"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
CHART_DIR = "artifacts/charts"
IMAGE_RE = r"\n\n!\[[^\]]*\]\(" + re.escape(CHART_DIR) + r"/{id}\.svg\)"


def _plt() -> Any:
    try:
        import matplotlib
    except ImportError as exception:  # pragma: no cover - environment dependent
        raise SystemExit("charts need matplotlib; enter `nix develop`") from exception
    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "text.color": TEXT,
        "axes.labelcolor": TEXT_SECONDARY, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.hashsalt": "gufo-model-bench", "svg.fonttype": "none",
    })
    return plt


def _series(rows: dict[str, dict[str, str]], labels: list[str], header: str) -> list[float]:
    values = []
    for label in labels:
        value = _number(rows.get(label, {}).get(header))
        values.append(math.nan if value is None else value)
    return values


def _depth_ticks(labels: list[str]) -> list[str]:
    """Compact tick labels: 0, 4K, 8K, ... for depth tables."""
    out = []
    for label in labels:
        value = int(label.replace(",", ""))
        out.append(f"{value // 1024}K" if value >= 1024 and value % 1024 == 0 else str(value))
    return out


def _has_data(*series: list[float]) -> bool:
    return any(not math.isnan(v) for s in series for v in s)


def _finish_axes(ax: Any, labels: list[str], series: list[tuple[str, list[float], str]], ylabel: str,
                 ticks: list[str] | None = None) -> None:
    ax.set_xticks(range(len(labels)), ticks or labels)
    ax.set_ylabel(ylabel)
    top = max((v for _, values, _ in series for v in values if not math.isnan(v)), default=1.0)
    ax.set_ylim(0, top * 1.2)
    ax.grid(axis="x", visible=False)


def _lines(ax: Any, labels: list[str], series: list[tuple[str, list[float], str]], ylabel: str,
           ticks: list[str] | None = None) -> None:
    for name, values, color in series:
        points = [(i, v) for i, v in enumerate(values) if not math.isnan(v)]
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=name)
    _finish_axes(ax, labels, series, ylabel, ticks)


def _bars(ax: Any, labels: list[str], series: list[tuple[str, list[float], str]], ylabel: str) -> None:
    width = 0.8 / len(series)
    for offset, (name, values, color) in enumerate(series):
        x = [i - 0.4 + width * (offset + 0.5) for i in range(len(labels))]
        ax.bar(x, [0 if math.isnan(v) else v for v in values], width=width - 0.04, color=color,
               label=name, linewidth=0)
    _finish_axes(ax, labels, series, ylabel)


def chart_for(config: BenchConfig, table: TableSpec, rows: dict[str, dict[str, str]], path: Path) -> bool:
    """Write one SVG for a rendered table; return False when there is nothing to draw."""
    layout = layout_for(config, table)
    labels = layout.rows
    ref = config.reference_name
    spec_label = config.speculative["label"]
    kind = table.kind
    title = table.spec.get("title", table.id)
    if table.variant:
        title += f" ({config.variant_label(table.variant)})"
    plt = _plt()

    if kind == "single" and not table.speculative:
        gp, rp, gt, rt = (_series(rows, labels, h) for h in ("Gufo pp", f"{ref} pp", "Gufo tg", f"{ref} tg"))
        if not _has_data(gp, gt):
            return False
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
        ticks = _depth_ticks(labels)
        _lines(a1, labels, [("Gufo", gp, COLORS["gufo"]), (ref, rp, COLORS["reference"])], "prefill tok/s", ticks)
        _lines(a2, labels, [("Gufo", gt, COLORS["gufo"]), (ref, rt, COLORS["reference"])], "generation tok/s", ticks)
        a1.set_xlabel("context depth (tokens)")
        a2.set_xlabel("context depth (tokens)")
        a1.legend(loc="lower left")
    elif kind == "single":
        gp, gt, acc = (_series(rows, labels, h) for h in ("Gufo pp", "Gufo tg", "Gufo accepted/step"))
        if not _has_data(gt):
            return False
        ticks = _depth_ticks(labels)
        if config.reference_speculative:
            rp, rt, racc = (_series(rows, labels, h) for h in (f"{ref} pp", f"{ref} tg", f"{ref} accepted/step"))
            fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(11, 3.2))
            _lines(a0, labels, [(f"Gufo {spec_label}", gp, COLORS["spec"]), (f"{ref} {spec_label}", rp, COLORS["ref_spec"])],
                   "prefill tok/s", ticks)
            a0.set_xlabel("context depth (tokens)")
            a0.legend(loc="lower left")
            ref_name = f"{ref} {spec_label}"
            acceptance = [("Gufo accepted/step", acc, COLORS["spec"]), (f"{ref} accepted/step", racc, COLORS["ref_spec"])]
        else:
            rt = _series(rows, labels, f"{ref} AR tg")
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
            ref_name = f"{ref} AR"
            acceptance = [(f"Gufo {spec_label} accepted/step", acc, COLORS["spec"])]
        _lines(a1, labels, [(f"Gufo {spec_label}", gt, COLORS["spec"]), (ref_name, rt, COLORS["ref_spec"])],
               "generation tok/s", ticks)
        _lines(a2, labels, acceptance, "accepted draft tokens per step", ticks)
        a2.legend(loc="lower right")
        a1.set_xlabel("context depth (tokens)")
        a2.set_xlabel("context depth (tokens)")
        a1.legend(loc="lower left")
    elif kind == "multi":
        ga, ra, gs = (_series(rows, labels, h) for h in ("Gufo AR", f"{ref} AR", f"Gufo {spec_label}"))
        if not _has_data(ga, gs):
            return False
        series = [("Gufo AR", ga, COLORS["gufo"]), (f"{ref} AR", ra, COLORS["reference"]),
                  (f"Gufo {spec_label}", gs, COLORS["spec"])]
        if config.reference_speculative:
            series.append((f"{ref} {spec_label}", _series(rows, labels, f"{ref} {spec_label}"), COLORS["ref_spec"]))
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        _bars(ax, labels, series, "sum of request decode tok/s")
        ax.set_xlabel("concurrent users")
        ax.legend(loc="upper left")
    elif kind in ("loading", "memory", "image-encoder"):
        unit_header = {"loading": "ready", "memory": "GiB", "image-encoder": "ms"}[kind]
        g, r = _series(rows, labels, f"Gufo {unit_header}"), _series(rows, labels, f"{ref} {unit_header}")
        if not _has_data(g):
            return False
        unit = {"loading": "seconds to ready", "memory": "GiB", "image-encoder": "encode ms"}[kind]
        fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(labels) + 2), 3.0))
        _bars(ax, labels, [("Gufo", g, COLORS["gufo"]), (ref, r, COLORS["reference"])], unit)
        ax.legend(loc="upper right")
    else:
        return False

    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    return True


def render_charts(config: BenchConfig, document: str, only: set[str] | None = None) -> tuple[str, list[str]]:
    """Draw charts for rendered tables and place an image line after each table."""
    known = {t.id: t for t in config.tables()}
    written: list[str] = []
    chart_dir = config.model_dir / CHART_DIR

    def replace(match: re.Match[str]) -> str:
        table_id = match.group("id")
        block = match.group(0)
        if table_id not in known or (only and table_id not in only):
            return block
        path = chart_dir / f"{table_id}.svg"
        if chart_for(config, known[table_id], parse_table(match.group("body")), path):
            written.append(table_id)
            return block + f"\n\n![{known[table_id].spec.get('title', table_id)}]({CHART_DIR}/{table_id}.svg)"
        return block

    # Replace only selected charts; untouched table links must survive a partial render.
    for table_id in known:
        if not only or table_id in only:
            document = re.sub(IMAGE_RE.format(id=re.escape(table_id)), "", document)
    return MARKER_RE.sub(replace, document), written
