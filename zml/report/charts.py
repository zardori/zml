"""Inline SVG sparklines for a checkpoint trajectory.

The shape of the data forces the form. A frame_replace run is judged on four or five metrics at once
— an erasure rate in [0, 1], colourfulness around 20-50, a motion score around 0-2 — and the whole
point is how they move *against each other* over training steps. Putting them on one pair of axes
would need two y-scales, which is the one thing a chart may never do, so each metric gets its own
small multiple with its own range and they share an x-axis by being drawn the same width.

Two series per chart at most: the concept prompt set (what should be erased) against a preservation
set (what must survive). That pairing is the project's central comparison, and two is also where a
sparkline stops being readable.
"""
from __future__ import annotations

from dataclasses import dataclass

from zml.report.metrics import SHARED_METRICS

VIEW_WIDTH = 240
VIEW_HEIGHT = 68
PAD_LEFT = 4
PAD_RIGHT = 34  # room for the direct end-label
PAD_TOP = 10
PAD_BOTTOM = 14

LINE_WIDTH = 2
END_MARKER_RADIUS = 4  # 8px diameter, the marker floor
Y_PADDING = 0.08  # keeps a flat series off the frame edge
LABEL_BASELINE_SHIFT = 3  # centres a 9px label on its marker
LABEL_MIN_GAP = 9  # one line-height, so two converging series stay legible

MAX_CHARTS = 5  # past five small multiples a card stops being scannable

# Series identity is fixed, never cycled: slot 1 is always what is being erased.
SERIES_SLOTS = ("series-1", "series-2")
PREFERRED_SERIES = ("concept", "unrelated", "related", "anchor")


@dataclass(frozen=True)
class Series:
    name: str
    slot: str
    points: list[tuple[float, float]]  # (step, value)


def _value_at(entry: dict, prompt_set: str, label: str) -> float | None:
    value = (entry.get("values") or {}).get(prompt_set, {}).get(label)
    return float(value) if isinstance(value, (int, float)) else None


def _series_for(trajectory: list[dict], prompt_sets: list[str], label: str) -> list[Series]:
    series = []
    for slot, prompt_set in zip(SERIES_SLOTS, prompt_sets):
        points = [
            (float(entry["step"]), value)
            for entry in trajectory
            if (value := _value_at(entry, prompt_set, label)) is not None
        ]
        if len(points) >= 2:
            series.append(Series(prompt_set, slot, points))
    return series


def _scale(points: list[tuple[float, float]], x_range: tuple[float, float],
           y_range: tuple[float, float]) -> list[tuple[float, float]]:
    (x_min, x_max), (y_min, y_max) = x_range, y_range
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    plot_width = VIEW_WIDTH - PAD_LEFT - PAD_RIGHT
    plot_height = VIEW_HEIGHT - PAD_TOP - PAD_BOTTOM
    return [
        (
            PAD_LEFT + (x - x_min) / x_span * plot_width,
            PAD_TOP + (1 - (y - y_min) / y_span) * plot_height,
        )
        for x, y in points
    ]


def _format_value(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    decimals = 3 if magnitude < 1 else (2 if magnitude < 10 else 1)
    return f"{value:.{decimals}f}"


def _tooltip(prompt_set: str, step: float, value: float) -> str:
    return f"{prompt_set} · step {step:g}: {_format_value(value)}"


def _separated_labels(ends: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Nudge end-labels apart when two series finish at nearly the same value.

    Two lines converging is the *interesting* case here — an erasure metric meeting its preservation
    baseline is the result — and it is exactly when the two numbers would print on top of each other.
    """
    placed: list[tuple[float, float, float]] = []
    for end_x, end_y, value in sorted(ends, key=lambda end: end[1]):
        label_y = end_y + LABEL_BASELINE_SHIFT
        if placed and label_y - placed[-1][1] < LABEL_MIN_GAP:
            label_y = placed[-1][1] + LABEL_MIN_GAP
        placed.append((end_x, label_y, value))
    return placed


def sparkline(label: str, trajectory: list[dict], prompt_sets: list[str]) -> str | None:
    """One metric's small multiple, or ``None`` if there is not enough of it to plot."""
    series = _series_for(trajectory, prompt_sets, label)
    if not series:
        return None

    all_points = [point for one in series for point in one.points]
    x_range = (min(x for x, _ in all_points), max(x for x, _ in all_points))
    low, high = min(y for _, y in all_points), max(y for _, y in all_points)
    pad = (high - low) * Y_PADDING or (abs(high) * Y_PADDING or 1.0)
    y_range = (low - pad, high + pad)

    parts = [
        f'<svg class="spark" viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" '
        f'role="img" aria-label="{label} over training steps" preserveAspectRatio="none">',
        # A single recessive baseline; a grid would out-weigh the data at this size.
        f'<line class="spark-base" x1="{PAD_LEFT}" x2="{VIEW_WIDTH - PAD_RIGHT}" '
        f'y1="{VIEW_HEIGHT - PAD_BOTTOM}" y2="{VIEW_HEIGHT - PAD_BOTTOM}" />',
    ]

    ends = []
    for one in series:
        scaled = _scale(one.points, x_range, y_range)
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in scaled)
        end_x, end_y = scaled[-1]
        ends.append((end_x, end_y, one.points[-1][1]))
        parts += [
            f'<polyline class="spark-line" style="stroke:var(--{one.slot})" '
            f'stroke-width="{LINE_WIDTH}" points="{path}" />',
            f'<circle class="spark-end" style="fill:var(--{one.slot})" '
            f'cx="{end_x:.1f}" cy="{end_y:.1f}" r="{END_MARKER_RADIUS}" />',
        ]
        parts += [
            f'<circle class="spark-hit" cx="{x:.1f}" cy="{y:.1f}" r="6" '
            f'data-tip="{_tooltip(one.name, step, value)}" />'
            for (x, y), (step, value) in zip(scaled, one.points)
        ]

    # Direct labels in ink, never in the series colour — the dot beside each carries identity.
    for end_x, label_y, value in _separated_labels(ends):
        parts.append(
            f'<text class="spark-label" x="{end_x + END_MARKER_RADIUS + 3:.1f}" '
            f'y="{label_y:.1f}">{_format_value(value)}</text>'
        )

    parts += [
        f'<text class="spark-axis" x="{PAD_LEFT}" y="{VIEW_HEIGHT - 3}">step {x_range[0]:g}</text>',
        f'<text class="spark-axis spark-axis-end" x="{VIEW_WIDTH - PAD_RIGHT}" '
        f'y="{VIEW_HEIGHT - 3}">{x_range[1]:g}</text>',
        "</svg>",
    ]
    return "".join(parts)


def choose_prompt_sets(trajectory: list[dict]) -> list[str]:
    """At most two prompt sets to plot, in the order a reader compares them."""
    available = {name for entry in trajectory for name in (entry.get("values") or {})}
    return [name for name in PREFERRED_SERIES if name in available][:2]


def chart_labels(trajectory: list[dict], prompt_sets: list[str]) -> list[str]:
    """Metric labels worth plotting: one erasure metric, then the quality metrics it trades against.

    A concept records two or three near-equivalent erasure metrics (nudity by frame, by video and by
    area all move together), so plotting them all spends the card's four slots saying one thing and
    leaves no room for the motion and colour columns — which is where this method's damage shows up.
    """
    seen: dict[str, None] = {}
    for entry in trajectory:
        for prompt_set in prompt_sets:
            for label in (entry.get("values") or {}).get(prompt_set, {}):
                seen.setdefault(label, None)

    shared = [label.label for label in SHARED_METRICS]
    erasure = [label for label in seen if label not in shared]
    quality = [label for label in shared if label in seen]
    return (erasure[:1] + quality)[:MAX_CHARTS]
