#!/usr/bin/env python3
"""Build the ESD-variant comparison table for the report.

For each ESD variant (basic / +preservation / +preservation+anchor / normalized)
this scans its (ngs x lr) grid, picks the run with the best erasure-vs-quality
balance, and emits a booktabs LaTeX table to ``report/esd_results_table.tex``.

Run selection: among a grid's runs, take the one with the lowest concept
fire-detection-rate subject to a concept-CLIP floor (so a quality/desaturation
collapse cannot masquerade as erasure); ties go to the higher concept CLIP.

Rows are labelled by method name only; source experiments are named in the
caption, not in a column. Metric coverage is uneven across variants (colorfulness
only on exp015, DOVER only on the earlier athena grids); missing metrics render
as blank cells.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TEX = REPO_ROOT / "report" / "esd_results_table.tex"

# Run-selection guard: minimum concept CLIP for a run to count as a real erasure
# rather than a quality collapse (base-model concept CLIP ~= 0.30; collapsed
# runs sit at 0.15-0.22).
CLIP_FLOOR = 0.27

# Decimal places per metric.
CDR_DP = 2
CLIP_DP = 2
COLORFUL_DP = 1
DOVER_DP = 2

BLANK = "--"  # rendered for metrics never recorded for a run

# Groups within each eval whose scores we surface.
CONCEPT = "concept"
UNRELATED = "unrelated"


@dataclass(frozen=True)
class Variant:
    method: str  # display name (the row label)
    grid_dir: Path  # the grid_* folder holding run_*/outputs


@dataclass(frozen=True)
class MetricCell:
    """A formatted (concept, unrelated) value pair for one metric."""

    concept: str
    unrelated: str


# One row per ESD variant; the representative run is chosen by select_run().
VARIANTS: list[Variant] = [
    Variant("ESD (basic)",
            REPO_ROOT / "experiments/exp009_esd_fire_grid/grid_20260425_115620"),
    Variant("ESD + preservation",
            REPO_ROOT / "experiments/exp011_esd_preservation_grid/grid_20260504_121415"),
    Variant("ESD + preservation + anchor",
            REPO_ROOT / "experiments/exp013_esd_preservation_anchor_grid/grid_20260504_122442"),
    Variant("ESD (normalized)",
            REPO_ROOT / "experiments/exp015_esd_normalized_grid/grid_20260609_210937"),
]


def latest_scores(output_dir: Path) -> dict[str, dict[str, float]]:
    """Return the final-step ``{group: {metric: value}}`` scores for a run.

    Prefers ``summary.json`` (last eval block); falls back to the highest-step
    ``eval_step_*/metrics.json`` for older runs without a summary.
    """
    summary = output_dir / "summary.json"
    if summary.exists():
        evals = json.loads(summary.read_text()).get("eval", [])
        if evals:
            return evals[-1].get("scores", {})

    step_dirs = sorted(
        output_dir.glob("eval_step_*"),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    for step_dir in reversed(step_dirs):
        metrics = step_dir / "metrics.json"
        if metrics.exists():
            return json.loads(metrics.read_text())

    raise FileNotFoundError(f"No eval scores found under {output_dir}")


def select_run(grid_dir: Path) -> Path:
    """Pick the run output dir with the best erasure-vs-quality balance.

    Lowest concept fire-detection-rate among runs whose concept CLIP clears
    CLIP_FLOOR; ties broken by higher concept CLIP. Falls back to the highest
    concept CLIP if no run clears the floor (whole grid collapsed).
    """
    candidates: list[tuple[float, float, Path]] = []  # (cdr, -clip, dir)
    fallback: list[tuple[float, Path]] = []  # (clip, dir)
    for run_dir in sorted(grid_dir.glob("run_*")):
        output_dir = run_dir / "outputs"
        try:
            concept = latest_scores(output_dir).get(CONCEPT, {})
        except FileNotFoundError:
            continue  # crashed run with no eval
        cdr = concept.get("fire_detection_rate")
        clip = concept.get("clip_score_mean")
        if cdr is None or clip is None:
            continue
        fallback.append((clip, output_dir))
        if clip >= CLIP_FLOOR:
            candidates.append((cdr, -clip, output_dir))

    if candidates:
        return min(candidates)[2]
    if fallback:
        return max(fallback)[0:2][1]
    raise FileNotFoundError(f"No usable runs under {grid_dir}")


def _format(value: float | None, decimals: int) -> str:
    return BLANK if value is None else f"{value:.{decimals}f}"


def _get(group_scores: dict[str, float], key: str) -> float | None:
    return group_scores.get(key)


def _get_dover(group_scores: dict[str, float]) -> float | None:
    """DOVER is disabled on helios (mean pinned to 0.0); treat that as missing."""
    value = group_scores.get("dover_aesthetic_mean")
    return value if value else None


def build_cells(scores: dict[str, dict[str, float]]) -> list[MetricCell]:
    concept = scores.get(CONCEPT, {})
    unrelated = scores.get(UNRELATED, {})
    return [
        MetricCell(
            _format(_get(concept, "fire_detection_rate"), CDR_DP),
            _format(_get(unrelated, "fire_detection_rate"), CDR_DP),
        ),
        MetricCell(
            _format(_get(concept, "clip_score_mean"), CLIP_DP),
            _format(_get(unrelated, "clip_score_mean"), CLIP_DP),
        ),
        MetricCell(
            _format(_get(concept, "colorfulness_mean"), COLORFUL_DP),
            _format(_get(unrelated, "colorfulness_mean"), COLORFUL_DP),
        ),
        MetricCell(
            _format(_get_dover(concept), DOVER_DP),
            _format(_get_dover(unrelated), DOVER_DP),
        ),
    ]


HEADER = r"""\begin{table}[t]
\centering
\caption{ESD variants for fire-concept unlearning. C = fire-concept prompts,
U = unrelated (preservation) prompts. CDR is the concept-detection rate. Each
row is the best erasure-vs-quality run of that variant's (ngs $\times$ lr) grid:
basic = exp009, with preservation = exp011, with preservation + anchor = exp013
(grid truncated at 600 steps), normalized = exp015. Blank cells mark metrics not
recorded for that run (colorfulness added on the later run; DOVER disabled on
helios).}
\label{tab:esd-variants}
\begin{tabular}{l cc cc cc cc}
\toprule
& \multicolumn{2}{c}{CDR $\downarrow$} & \multicolumn{2}{c}{CLIP $\uparrow$} & \multicolumn{2}{c}{Colorful.} & \multicolumn{2}{c}{DOVER-aes $\uparrow$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & C & U & C & U & C & U & C & U \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
\end{table}"""


def render_row(method: str, cells: list[MetricCell]) -> str:
    values = " & ".join(f"{c.concept} & {c.unrelated}" for c in cells)
    return f"{method} & {values} \\\\"


def build_table() -> str:
    lines = [HEADER]
    for variant in VARIANTS:
        cells = build_cells(latest_scores(select_run(variant.grid_dir)))
        lines.append(render_row(variant.method, cells))
    lines.append(FOOTER)
    return "\n".join(lines) + "\n"


def main() -> None:
    table = build_table()
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(table)
    print(f"Wrote {OUTPUT_TEX.relative_to(REPO_ROOT)}")
    print(table)


if __name__ == "__main__":
    main()
