"""Append-only, AI-readable metrics sink that runs alongside wandb/mlflow.

Motivation: wandb charts are great for a human eye but awkward to feed back to an
agent — you end up hand-copying numbers. This writes two plain files into the run's
output dir (both rsync-able by ``pull_results.sh``):

  * ``metrics.jsonl`` — append-only; one JSON object per flushed train window and per
    eval. Full (downsampled) history, robust to crashes, trivially machine-parsed.
  * ``summary.json``  — overwritten on every update; config echo, per-metric train
    trends, per-checkpoint eval scores, and derived *health flags* that answer
    "did training actually start?" at a glance.

Train metrics are logged every step but buffered and flushed as window aggregates
(mean/last/min/max) every ``flush_interval`` steps, so a 2000-step run becomes ~40
readable rows instead of 2000.
"""

import json
import math
import os
from collections import defaultdict
from typing import Any


def _sig(x: float, n: int = 4) -> float:
    """Round to ``n`` significant figures, preserving tiny magnitudes (e.g. 3.7e-18)
    that fixed-decimal rounding would flatten to 0."""
    if not isinstance(x, (int, float)) or x == 0 or not math.isfinite(x):
        return x
    digits = n - 1 - math.floor(math.log10(abs(x)))
    return round(x, digits)


# Below this, a squared-error loss is treated as numerically degenerate (no signal).
_DEGENERATE_LOSS = 1e-12


class MetricsRecorder:
    """Collects train/eval metrics and renders the two sink files."""

    def __init__(self, output_dir: str, run_name: str, config: dict[str, Any], flush_interval: int) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, "metrics.jsonl")
        self.summary_path = os.path.join(output_dir, "summary.json")
        self.run_name = run_name
        self.config = config
        self.flush_interval = max(1, flush_interval)

        # Truncate any stale sink from a previous run in the same dir.
        open(self.jsonl_path, "w").close()

        self._buffer: dict[str, list[float]] = defaultdict(list)
        self._train_rows: list[dict[str, Any]] = []  # flushed window aggregates
        self._eval_rows: list[dict[str, Any]] = []
        self._last_step = 0

    def log_train(self, step: int, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._buffer[k].append(float(v))
        self._last_step = step
        if (step + 1) % self.flush_interval == 0:
            self._flush_train(step)

    def log_eval(self, step: int, payload: dict[str, Any]) -> None:
        row = {"type": "eval", "step": step, **_round(payload)}
        self._eval_rows.append(row)
        self._append_jsonl(row)
        self._write_summary()

    def close(self) -> None:
        """Flush any partial train window and write the final summary."""
        if self._buffer:
            self._flush_train(self._last_step)
        else:
            self._write_summary()

    def _flush_train(self, step: int) -> None:
        agg: dict[str, dict[str, float]] = {}
        for k, vals in self._buffer.items():
            if not vals:
                continue
            agg[k] = {
                "mean": _sig(sum(vals) / len(vals)),
                "last": _sig(vals[-1]),
                "min": _sig(min(vals)),
                "max": _sig(max(vals)),
            }
        row = {"type": "train", "step": step, "count": len(next(iter(self._buffer.values()))), "metrics": agg}
        self._train_rows.append(row)
        self._append_jsonl(row)
        self._buffer.clear()
        self._write_summary()

    def _append_jsonl(self, obj: dict[str, Any]) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def _train_trends(self) -> dict[str, dict[str, float]]:
        """Per-metric trend across all flushed windows: first/last/min/max + recent mean."""
        trends: dict[str, dict[str, float]] = {}
        keys = {k for row in self._train_rows for k in row["metrics"]}
        for k in keys:
            rows = [r["metrics"][k] for r in self._train_rows if k in r["metrics"]]
            trends[k] = {
                "first": rows[0]["mean"],
                "recent": rows[-1]["mean"],
                "last": rows[-1]["last"],
                "min": _sig(min(r["min"] for r in rows)),
                "max": _sig(max(r["max"] for r in rows)),
            }
        return trends

    def _health(self, trends: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Derived flags + plain-language hints aimed at an agent reading the run."""
        flags: dict[str, Any] = {}
        notes: list[str] = []

        # Generic loss check (any method that logs train/loss): caught divergence / stall.
        loss = trends.get("train/loss", {})
        if loss:
            loss_max = loss.get("max")
            if loss_max is not None and not math.isfinite(loss_max):
                flags["loss_diverged"] = True
                notes.append("train/loss became non-finite: training diverged.")
            else:
                first, recent = loss.get("first"), loss.get("recent")
                if first is not None and recent is not None:
                    flags["loss_first_to_recent"] = [first, recent]
                    if recent >= first:
                        notes.append("train/loss not decreasing vs first window: erasure may be stalled.")

        # ESD-normalized progress: fraction of the path from the base concept prediction to
        # the ESD target that the student has covered (see unlearn_model_normalized.py).
        progress = trends.get("train/erase_progress", {}).get("recent")
        if progress is not None:
            flags["erase_progress_recent"] = _sig(progress)
            if progress < 0.05:
                notes.append("erase_progress ~0: student still matches the base concept prediction; LoRA may be a no-op.")
            elif progress >= 0.9:
                notes.append("erase_progress ~1: student has converged onto the ESD target.")

        pred = trends.get("train/predicted_step_norm", {}).get("recent")
        tgt = trends.get("train/target_step_norm", {}).get("recent")
        if pred is not None and tgt:
            ratio = pred / tgt if tgt else 0.0
            flags["predicted_vs_target_ratio_recent"] = _sig(ratio)
            if ratio < 0.1:
                notes.append("predicted_step << target_step: hypernet is not tracking the trajectory yet.")

        rem = trends.get("train/loss_remove", {})
        if rem:
            degenerate = (rem.get("max", 0.0) or 0.0) < _DEGENERATE_LOSS
            flags["loss_remove_degenerate"] = degenerate
            if degenerate:
                notes.append("loss_remove pinned ~0: removal signal degenerate (zero-trajectory wins).")

        if self._eval_rows:
            concept = [r.get("theta_S_norm_concept_mean") for r in self._eval_rows if "theta_S_norm_concept_mean" in r]
            if concept:
                flags["theta_S_concept_first_to_last"] = [concept[0], concept[-1]]
                if (concept[-1] or 0.0) < 0.1:
                    notes.append("theta_S ~0 on eval prompts: adapter is effectively a no-op.")
            last = self._eval_rows[-1]
            std = last.get("theta_S_norm_concept_std")
            mean = last.get("theta_S_norm_concept_mean")
            if std is not None and mean:
                if std / mean < 0.02:
                    notes.append("theta_S barely varies across prompts: weak prompt conditioning.")

        flags["notes"] = notes
        return flags

    def _write_summary(self) -> None:
        trends = self._train_trends()
        summary = {
            "run": self.run_name,
            "updated_step": self._last_step,
            "config": self.config,
            "train": trends,
            "eval": self._eval_rows,
            "health": self._health(trends),
        }
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, self.summary_path)  # atomic; readers never see a half-written file


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return _sig(obj)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj
