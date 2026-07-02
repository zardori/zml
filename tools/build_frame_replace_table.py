#!/usr/bin/env python3
"""Build the frame-replace ablation table for the report.

Surfaces the frame-replace ablation chain -- offline (exp038), online (exp039),
+retention (exp043), +masked loss (exp044), +x0-space loss (exp045), and
denoising-redirection (exp046) -- and emits a booktabs LaTeX table to
``report/frame_replace_results_table.tex``.

Fire-area (``fire_area_score_mean``) is the load-bearing metric for this family
and was only added in the retention era, so it is blank for the offline/online
baselines. DOVER is disabled on helios for every run, so that column is dropped.

Rows are labelled by method name only; source experiments are named in the
caption, not in a column.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TEX = REPO_ROOT / "report" / "frame_replace_results_table.tex"

# Decimal places per metric.
CDR_DP = 2
AREA_DP = 3
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
    step: int | None = None  # specific eval step; None -> final (latest) eval


@dataclass(frozen=True)
class MetricCell:
    """A formatted (concept, unrelated) value pair for one metric."""

    concept: str
    unrelated: str


# The frame-replace ablation chain, in narrative order.
ROWS: list[TableRow] = [
    TableRow("frame-replace (offline)",
             REPO_ROOT / "experiments/exp038_frame_replace_fire_longer_stronger/outputs_20260620_234835"),
    TableRow("frame-replace (online)",
             REPO_ROOT / "experiments/exp039_frame_replace_online/outputs_20260622_011035"),
    TableRow("frame-replace + retention",
             REPO_ROOT / "experiments/exp043_frame_replace_retention/outputs_20260624_161343"),
    TableRow("frame-replace + masked loss",
             REPO_ROOT / "experiments/exp044_frame_replace_masked/outputs_20260625_010847"),
    TableRow("frame-replace + x0-space loss",
             REPO_ROOT / "experiments/exp045_frame_replace_x0loss/outputs_20260625_164320"),
    TableRow("frame-replace (denoising-redirection, step 500)",
             REPO_ROOT / "experiments/exp046_frame_replace_redirect/outputs_20260626_010355",
             step=500),
    TableRow("frame-replace (denoising-redirection, step 1000)",
             REPO_ROOT / "experiments/exp046_frame_replace_redirect/outputs_20260626_010355"),
]


def scores_for(output_dir: Path, step: int | None) -> dict[str, dict[str, float]]:
    """Return the ``{group: {metric: value}}`` scores for a run.

    ``step=None`` selects the final eval; otherwise the eval at that exact step.
    Prefers ``summary.json``; falls back to ``eval_step_*/metrics.json`` for
    older runs without a summary.
    """
    summary = output_dir / "summary.json"
    if summary.exists():
        evals = json.loads(summary.read_text()).get("eval", [])
        if evals:
            if step is None:
                return evals[-1].get("scores", {})
            for entry in evals:
                if entry.get("step") == step:
                    return entry.get("scores", {})
            raise ValueError(f"No eval at step {step} in {summary}")

    if step is not None:
        metrics = output_dir / f"eval_step_{step}" / "metrics.json"
        if metrics.exists():
            return json.loads(metrics.read_text())
        raise FileNotFoundError(f"No eval at step {step} under {output_dir}")

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
    ]


HEADER = r"""\begin{table}[h!]
\centering
\caption{frame-replace ablation chain for fire-concept unlearning. C =
fire-concept prompts, U = unrelated (preservation) prompts. CDR is the
concept-detection rate; fire-area is the continuous fire-coverage score.
Fire-area was added in the retention era, so it is blank for the offline/online
baselines; DOVER is omitted (disabled on helios). For denoising-redirection both
the step-500 checkpoint (where fire is erased) and the final step-1000
checkpoint are shown.}
\label{tab:frame-replace-results}
\begin{tabular}{l cc cc cc cc}
\toprule
& \multicolumn{2}{c}{CDR $\downarrow$} & \multicolumn{2}{c}{Fire-area $\downarrow$} & \multicolumn{2}{c}{CLIP $\uparrow$} & \multicolumn{2}{c}{Colorful.} \\
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
    for row in ROWS:
        cells = build_cells(scores_for(row.output_dir, row.step))
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
