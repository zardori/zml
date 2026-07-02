#!/usr/bin/env python3
"""Build the unhype-variant comparison table for the report.

Surfaces the two real unhype runs -- basic (exp029) and stabilized (exp031) --
and emits a booktabs LaTeX table to ``report/unhype_results_table.tex``. The
debug control runs (exp028 distill-control, exp030 static-apply-control) are
intentionally excluded.

Colorfulness is the load-bearing metric here: the unhype "basic" run drives fire
CDR to zero but collapses into a desaturated, washed-out state (very high
colorfulness with collapsed CLIP), which the stabilized variant tames. DOVER is
disabled on helios for both runs, so that column is dropped.

Rows are labelled by method name only; source experiments are named in the
caption, not in a column.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TEX = REPO_ROOT / "report" / "unhype_results_table.tex"

# Decimal places per metric.
CDR_DP = 2
CLIP_DP = 2
COLORFUL_DP = 1

BLANK = "--"  # rendered for metrics never recorded for a run

# Groups within each eval whose scores we surface.
CONCEPT = "concept"
UNRELATED = "unrelated"


@dataclass(frozen=True)
class TableRow:
    method: str  # display name (the row label)
    output_dir: Path  # run output directory holding summary.json / eval_step_*


@dataclass(frozen=True)
class MetricCell:
    """A formatted (concept, unrelated) value pair for one metric."""

    concept: str
    unrelated: str


# The two real unhype runs (debug controls exp028/exp030 are excluded).
ROWS: list[TableRow] = [
    TableRow("unhype (basic)",
             REPO_ROOT / "experiments/exp029_unhype_promptvar_stepfix/outputs_20260608_010505"),
    TableRow("unhype (stabilized)",
             REPO_ROOT / "experiments/exp031_unhype_cosine_stabilized/outputs_20260608_193200"),
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
    ]


HEADER = r"""\begin{table}[t]
\centering
\caption{unhype variants for fire-concept unlearning. C = fire-concept prompts,
U = unrelated (preservation) prompts. CDR is the concept-detection rate;
colorfulness flags the desaturation collapse (high values are washed-out runs).
basic = exp029, stabilized = exp031; the debug control runs (distill-control,
static-apply-control) are excluded, and DOVER is omitted (disabled on helios).}
\label{tab:unhype-variants}
\begin{tabular}{l cc cc cc}
\toprule
& \multicolumn{2}{c}{CDR $\downarrow$} & \multicolumn{2}{c}{CLIP $\uparrow$} & \multicolumn{2}{c}{Colorful.} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Method & C & U & C & U & C & U \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
\end{table}"""


def render_row(method: str, cells: list[MetricCell]) -> str:
    values = " & ".join(f"{c.concept} & {c.unrelated}" for c in cells)
    return f"{method} & {values} \\\\"


def build_table() -> str:
    lines = [HEADER]
    for row in ROWS:
        cells = build_cells(latest_scores(row.output_dir))
        lines.append(render_row(row.method, cells))
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
