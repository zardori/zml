#!/usr/bin/env python3
"""Build the per-method quantitative comparison table for the report.

Reads the final-step evaluation scores of a curated set of runs and emits a
booktabs LaTeX table to ``report/results_table.tex``. Metric coverage is uneven
across runs (DOVER only on early athena ESD runs, fire-area only on recent
frame-replace runs); missing metrics render as blank cells.

Rows are labelled by method name only -- never by experiment number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TEX = REPO_ROOT / "report" / "results_table.tex"

# Decimal places per metric.
CDR_DP = 2
AREA_DP = 3
CLIP_DP = 2
COLORFUL_DP = 1
DOVER_DP = 2

BLANK = "--"  # rendered for metrics never recorded for a run

# Groups within each eval whose scores we surface.
CONCEPT = "concept"
UNRELATED = "unrelated"


@dataclass(frozen=True)
class TableRow:
    method: str  # method name, possibly shared across consecutive rows
    descriptor: str  # short qualitative variant label
    output_dir: Path  # run output directory holding summary.json / eval_step_*


@dataclass(frozen=True)
class MetricCell:
    """A formatted (concept, unrelated) value pair for one metric."""

    concept: str
    unrelated: str


# Curated selection: two contrasting runs per method (erasure- vs preservation-leaning).
# Source experiments are intentionally not surfaced in the rendered table.
ROWS: list[TableRow] = [
    TableRow("ESD", "basic",
             REPO_ROOT / "experiments/exp009_esd_fire_grid/grid_20260425_115620/run_007/outputs"),
    TableRow("ESD", "with preservation",
             REPO_ROOT / "experiments/exp006_esd_100_prompts_gs_1_lr_00005/outputs_20260419_205935"),
    TableRow("unhype", "basic",
             REPO_ROOT / "experiments/exp029_unhype_promptvar_stepfix/outputs_20260608_010505"),
    TableRow("unhype", "stabilized",
             REPO_ROOT / "experiments/exp031_unhype_cosine_stabilized/outputs_20260608_193200"),
    TableRow("frame-replace", "velocity reconstruction",
             REPO_ROOT / "experiments/exp043_frame_replace_retention/outputs_20260624_161343"),
    TableRow("frame-replace", "denoising-redirection",
             REPO_ROOT / "experiments/exp046_frame_replace_redirect/outputs_20260626_010355"),
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
            _format(_get(concept, "fire_area_score_mean"), AREA_DP),
            _format(_get(unrelated, "fire_area_score_mean"), AREA_DP),
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
\caption{Fire-concept unlearning: per-method comparison. C = fire-concept
prompts, U = unrelated (preservation) prompts. CDR is the concept-detection
rate. Blank cells mark metrics not recorded for that run (DOVER only on the
early runs; colorfulness and fire-area added later).}
\label{tab:unlearn-results}
\begin{tabular}{l cc cc cc cc cc}
\toprule
& \multicolumn{2}{c}{CDR $\downarrow$} & \multicolumn{2}{c}{Fire-area $\downarrow$} & \multicolumn{2}{c}{CLIP $\uparrow$} & \multicolumn{2}{c}{Colorful.} & \multicolumn{2}{c}{DOVER-aes $\uparrow$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Method & C & U & C & U & C & U & C & U & C & U \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
\end{table}"""


def render_row(row: TableRow, cells: list[MetricCell]) -> str:
    label = f"{row.method} ({row.descriptor})"
    values = " & ".join(f"{c.concept} & {c.unrelated}" for c in cells)
    return f"{label} & {values} \\\\"


def build_table() -> str:
    lines = [HEADER]
    for i, row in enumerate(ROWS):
        cells = build_cells(latest_scores(row.output_dir))
        lines.append(render_row(row, cells))
        # rule between method blocks (not after the last row)
        is_block_end = i + 1 == len(ROWS) or ROWS[i + 1].method != row.method
        if is_block_end and i + 1 != len(ROWS):
            lines.append(r"\midrule")
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
