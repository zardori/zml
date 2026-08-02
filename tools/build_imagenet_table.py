#!/usr/bin/env python3
"""Collect ImageNet object-erasure results into the T2VUnlearning-style comparison table.

Walks ``experiments/*/**/esr_psr.json`` (written by ``zml/eval/imagenet_eval.py``) and prints one row
per run, with the published numbers as fixed reference lines above them.

A base-model run (``erased_class: null``) carries ESR/PSR for every class in turn, so it prints as a
single mean±std row — the same shape as the papers' ``Original``. A per-class run prints its own four
numbers; several per-class runs of the same method are also aggregated into a mean±std row so ours is
read the same way as theirs.

Run:  uv run python tools/build_imagenet_table.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = ("ESR-1", "ESR-5", "PSR-1", "PSR-5")

# T2VUnlearning (arXiv 2505.17550) Table 4, CogVideoX-2B, mean±std over the ten erased classes.
# Reference only — a different base model and frame count, see docs/imagenet_objects.md.
PUBLISHED: dict[str, dict[str, tuple[float, float]]] = {
    "Original (paper, 2B)": {"ESR-1": (21.62, 20.13), "ESR-5": (5.09, 8.23),
                             "PSR-1": (78.38, 2.24), "PSR-5": (94.91, 0.92)},
    "NegPrompt (paper, 2B)": {"ESR-1": (48.59, 17.29), "ESR-5": (19.79, 11.52),
                              "PSR-1": (65.37, 3.90), "PSR-5": (88.62, 2.50)},
    "SAFREE (paper, 2B)": {"ESR-1": (61.65, 15.75), "ESR-5": (36.41, 17.65),
                           "PSR-1": (53.46, 3.23), "PSR-5": (79.17, 1.87)},
    "T2VUnlearning (paper, 2B)": {"ESR-1": (92.38, 6.44), "ESR-5": (77.09, 18.74),
                                  "PSR-1": (54.03, 6.17), "PSR-5": (82.14, 5.38)},
}


@dataclass(frozen=True)
class Row:
    label: str
    scores: dict[str, tuple[float, float | None]]  # metric -> (value, std or None)


def _row_from_report(path: Path, report: dict, experiments_dir: Path) -> Row:
    experiment = path.relative_to(experiments_dir).parts[0]  # experiments/<exp>/outputs_*/esr_psr.json
    erased = report.get("erased_class")
    if erased is None:
        # Base-model run: ESR/PSR for each class in turn, already summarized.
        label = f"{experiment} (all classes)"
        return Row(label, {m: (report["mean"][m], report["std"][m]) for m in METRICS})
    label = f"{experiment} [{erased}]"
    return Row(label, {m: (report[m], None) for m in METRICS})


def _aggregate(rows: list[Row], label: str) -> Row | None:
    """Mean±std across per-class rows, so our numbers are read like the published ones."""
    per_class = [r for r in rows if r.scores["ESR-1"][1] is None]
    if len(per_class) < 2:
        return None
    return Row(
        label,
        {m: (float(np.mean([r.scores[m][0] for r in per_class])),
             float(np.std([r.scores[m][0] for r in per_class]))) for m in METRICS},
    )


def _format(value: float, std: float | None) -> str:
    return f"{value:.2f}" if std is None else f"{value:.2f}±{std:.2f}"


def _print_table(rows: list[Row]) -> None:
    label_width = max([len(r.label) for r in rows] + [24])
    header = f"{'Method / run'.ljust(label_width)}  " + "  ".join(m.rjust(13) for m in METRICS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = "  ".join(_format(*row.scores[m]).rjust(13) for m in METRICS)
        print(f"{row.label.ljust(label_width)}  {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments_dir", default=str(REPO_ROOT / "experiments"))
    parser.add_argument("--skip_published", action="store_true", help="Show only our runs")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    reports = sorted(experiments_dir.glob("*/**/esr_psr.json"))
    if not reports:
        print(f"No esr_psr.json found under {experiments_dir}. Pull results first.")
        return

    ours = [_row_from_report(p, json.loads(p.read_text()), experiments_dir) for p in reports]
    aggregate = _aggregate(ours, "OUR RUNS (mean over classes)")

    rows: list[Row] = []
    if not args.skip_published:
        rows += [Row(name, {m: scores[m] for m in METRICS}) for name, scores in PUBLISHED.items()]
    rows += ours
    if aggregate is not None:
        rows.append(aggregate)

    _print_table(rows)
    print(
        "\nPaper rows are CogVideoX-2B / 17-frame; ours are CogVideoX-5b / 49-frame. "
        "See docs/imagenet_objects.md for the deviations before comparing directly."
    )


if __name__ == "__main__":
    main()
