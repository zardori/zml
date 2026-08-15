"""Rendering ``data.json`` into the deck.

Plain string templating rather than a template engine: the repo declares no templating dependency,
and the alternative is one more thing to install on three machines to regenerate a document that is
thrown away weekly. Everything is emitted into a single self-contained ``index.html`` beside its
``media/`` directory, so the deck can be opened from disk with no server.

The page is walked through in a meeting and then read again afterwards, which sets two constraints
the CSS has to meet: it must survive both colour schemes (nobody's laptop is configured the same),
and it must paginate under Ctrl+P without cutting a card in half.
"""
from __future__ import annotations

from datetime import datetime
from html import escape

from experiments_index import THREAD_DOCS, format_elapsed  # via zml/report/__init__.py's path shim

from zml.report import charts

# Threads in the order the project works on them; anything unrecognised is appended.
THREAD_ORDER = ("nudity", "imagenet", "face_identity", "shared")
UNGROUPED = "other"

STATUS_TONE = {
    "done": "good",
    "active": "warning",
    "ready": "neutral",
    "superseded": "muted",
    "abandoned": "muted",
}
OUTCOME_TONE = {"completed": "good", "running": "warning", "timeout": "critical", "failed": "critical"}

MIN_TRAJECTORY_POINTS = 3  # two points is a line, not a trajectory


def _esc(value: object) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def _tag(name: str, content: str, cls: str = "", **attrs: str) -> str:
    parts = [f' class="{cls}"'] if cls else []
    parts += [f' {key.replace("_", "-")}="{_esc(value)}"' for key, value in attrs.items() if value]
    return f"<{name}{''.join(parts)}>{content}</{name}>"


def _chip(text: str, tone: str = "neutral") -> str:
    return _tag("span", _esc(text), f"chip chip-{tone}")


def _pretty_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%a %d %b, %H:%M")
    except ValueError:
        return iso


def _thread_label(thread: str | None) -> str:
    return (thread or UNGROUPED).replace("_", " ")


# --- results ------------------------------------------------------------------------------------

def _score_table(table: dict, caption: str) -> str:
    """A ``{prompt_set: {metric: formatted}}`` block as a table.

    The arrow in each header is load-bearing: erasure metrics fall and utility metrics rise, and
    without it a reader has no way to tell a good column from a bad one.
    """
    columns = table.get("columns") or []
    if not columns:
        return ""

    def direction(column: dict) -> str:
        if column["lower_is_better"] is None:
            return ""
        return _tag("span", "↓" if column["lower_is_better"] else "↑", "dir")

    header = "".join(
        _tag("th", f"{_esc(column['label'])} {direction(column)}") for column in columns
    )
    rows = "".join(
        _tag(
            "tr",
            _tag("th", _esc(name), "row-head")
            + "".join(
                _tag("td", _esc(values.get(column["label"]) or "—")) for column in columns
            ),
        )
        for name, values in (table.get("prompt_sets") or {}).items()
    )
    return _tag(
        "div",
        _tag("table", _tag("thead", _tag("tr", _tag("th", "") + header)) + _tag("tbody", rows))
        + _tag("p", _esc(caption), "table-note"),
        "table-wrap",
    )


def _stat(value: str, label: str, tone: str = "") -> str:
    return _tag(
        "div",
        _tag("div", _esc(value), "stat-value") + _tag("div", _esc(label), "stat-label"),
        f"stat {tone}".strip(),
    )


def _dataset_block(dataset: dict) -> str:
    """A precompute build's yield — the number that gates every downstream erasure run."""
    usable = dataset.get("usable")
    headline = f"{usable}/{dataset['built']}" if usable is not None else str(dataset["built"])
    stats = [
        _stat(headline, "usable / built" if usable is not None else "targets built"),
        _stat(f"{dataset['yield']:.0%}" if dataset.get("yield") is not None else "—", "yield"),
        _stat(str(dataset["skipped"]), "skipped at build"),
    ]
    return _tag("div", "".join(stats), "stats stats-inline")


def _run_chips(run: dict) -> str:
    chips = []
    if run.get("arm"):
        chips.append(_chip(run["arm"]))
    if run.get("job_type"):
        chips.append(_chip(run["job_type"]))
    if run.get("cluster"):
        chips.append(_chip(run["cluster"]))
    if run.get("elapsed_s"):
        chips.append(_chip(format_elapsed(run["elapsed_s"])))
    if run.get("outcome"):
        chips.append(_chip(run["outcome"], OUTCOME_TONE.get(run["outcome"], "neutral")))
    return "".join(chips)


def _charts_block(run: dict) -> str:
    trajectory = run.get("trajectory") or []
    if len(trajectory) < MIN_TRAJECTORY_POINTS:
        return ""

    prompt_sets = charts.choose_prompt_sets(trajectory)
    figures = []
    for label in charts.chart_labels(trajectory, prompt_sets):
        svg = charts.sparkline(label, trajectory, prompt_sets)
        if svg:
            figures.append(_tag("figure", _tag("figcaption", _esc(label)) + svg, "spark-figure"))
    if not figures:
        return ""

    legend = "".join(
        _tag(
            "span",
            f'<span class="swatch" style="background:var(--{slot})"></span>{_esc(name)}',
            "legend-item",
        )
        for slot, name in zip(charts.SERIES_SLOTS, prompt_sets)
    )
    return _tag(
        "div",
        _tag("div", legend, "legend") + _tag("div", "".join(figures), "sparks"),
        "charts",
    )


def _media_block(entries: list[dict]) -> str:
    blocks = []
    for entry in entries:
        frames = "".join(
            f'<img src="{_esc(src)}" alt="{_esc(entry.get("label"))} frame" loading="lazy">'
            for src in entry.get("frames") or []
        )
        clip = (
            f'<video src="{_esc(entry["clip"])}" controls preload="metadata" playsinline></video>'
            if entry.get("clip")
            else ""
        )
        caption = entry.get("caption") or entry.get("label") or ""
        blocks.append(
            _tag(
                "figure",
                _tag("div", frames, "strip") + clip + _tag("figcaption", _esc(caption)),
                "media",
            )
        )
    return _tag("div", "".join(blocks), "media-row") if blocks else ""


def _run_block(run: dict) -> str:
    parts = [_tag("div", _run_chips(run) + _tag("code", _esc(run["dir"]), "path"), "run-head")]

    if run.get("dataset"):
        parts.append(_dataset_block(run["dataset"]))
    if run.get("final_scores"):
        step = run.get("final_step")
        parts.append(_score_table(run["final_scores"], f"eval at step {step}"))
    if run.get("esr_psr"):
        parts.append(_score_table(run["esr_psr"], "ESR/PSR by erased class (%)"))
    if run.get("id_similarity"):
        parts.append(_score_table(run["id_similarity"], "ID similarity by erased identity"))
    parts.append(_charts_block(run))

    if run.get("health_notes"):
        notes = "".join(_tag("li", _esc(note)) for note in run["health_notes"])
        parts.append(_tag("ul", notes, "health"))
    return _tag("div", "".join(parts), "run")


def _card(experiment: dict) -> str:
    header = _tag(
        "div",
        _tag("h3", f'{_esc(experiment["id"])} · {_esc(experiment["name"].split("_", 1)[-1])}')
        + _tag(
            "div",
            _chip(experiment["status"], STATUS_TONE.get(experiment["status"], "neutral"))
            + _chip(experiment["concept"])
            + _chip(experiment["method"]),
            "chips",
        ),
        "card-head",
    )

    body = [header]
    if experiment.get("commentary"):
        body.append(_tag("p", _esc(experiment["commentary"]), "commentary"))
    if experiment.get("takeaway"):
        body.append(_tag("p", _esc(experiment["takeaway"]), "takeaway"))
    body += [_run_block(run) for run in experiment.get("runs") or []]
    body.append(_media_block(experiment.get("media") or []))

    classes = "card" + (" card-highlight" if experiment.get("highlight") else "")
    return _tag("article", "".join(body), classes, id=experiment["id"])


def _planned_list(experiments: list[dict]) -> str:
    if not experiments:
        return ""
    items = "".join(
        _tag(
            "li",
            _tag("strong", _esc(experiment["id"]))
            + f' {_esc(experiment["name"].split("_", 1)[-1])} '
            + _chip(experiment["status"], STATUS_TONE.get(experiment["status"], "neutral"))
            + _tag("span", _esc(experiment["takeaway"]), "planned-note"),
        )
        for experiment in experiments
    )
    return _tag("div", _tag("h4", "Staged and in flight") + _tag("ul", items, "planned"), "planned-box")


def _thread_section(thread: str | None, experiments: list[dict]) -> str:
    cards = [e for e in experiments if e.get("include") and not e.get("planned")]
    planned = [e for e in experiments if e.get("planned")]
    if not cards and not planned:
        return ""

    blurb, doc = THREAD_DOCS.get(thread or "", ("", None))
    heading = _tag("h2", _esc(_thread_label(thread)))
    subtitle = _tag("p", _esc(blurb) + (f" · <code>{_esc(doc)}</code>" if doc else ""), "thread-note")
    return _tag(
        "section",
        heading + subtitle + "".join(_card(e) for e in cards) + _planned_list(planned),
        "thread",
    )


def _gaps_section(gaps: list[dict]) -> str:
    """Results the week produced that are not on this machine.

    Rendered rather than dropped on purpose: ``pull_results.sh`` never downloads ``notes.md``, so an
    experiment's write-up arrives by git while its numbers arrive only if someone pulled. Silently
    omitting those turns "not here" into "did not happen".
    """
    if not gaps:
        return ""
    rows = "".join(
        _tag(
            "tr",
            _tag("th", _esc(gap["id"]), "row-head")
            + _tag("td", _esc(gap["name"].split("_", 1)[-1]))
            + _tag("td", _esc(_thread_label(gap.get("thread"))))
            + _tag("td", _esc(gap.get("method")))
            + _tag("td", _tag("code", _esc(gap["fix"]))),
        )
        for gap in gaps
    )
    headers = ("", "Experiment", "Thread", "Method", "Pull with")
    table = _tag(
        "table",
        _tag("thead", _tag("tr", "".join(_tag("th", h) for h in headers))) + _tag("tbody", rows),
    )
    return _tag(
        "section",
        _tag("h2", f"Not pulled ({len(gaps)})")
        + _tag("p", "Written up this week, but the results are not on this machine.", "thread-note")
        + _tag("div", table, "table-wrap"),
        "thread gaps",
    )


def _next_week(items: list[str]) -> str:
    if not items:
        return ""
    return _tag(
        "section",
        _tag("h2", "Next week") + _tag("ol", "".join(_tag("li", _esc(item)) for item in items), "next"),
        "thread",
    )


def _appendix(data: dict) -> str:
    commits = data.get("commits") or []
    if not commits:
        return ""
    rows = "".join(
        _tag(
            "tr",
            _tag("td", _tag("code", _esc(commit["sha"])))
            + _tag("td", _esc(commit["date"][:10]))
            + _tag("td", _esc(commit["author"]))
            + _tag("td", _esc(commit["subject"])),
        )
        for commit in commits
    )
    table = _tag("div", _tag("table", _tag("tbody", rows)), "table-wrap")
    return _tag(
        "details",
        _tag("summary", f"Commit log ({len(commits)})") + table,
        "appendix",
    )


def _header(data: dict) -> str:
    narrative = data.get("narrative") or {}
    experiments = data.get("experiments") or []
    cards = [e for e in experiments if e.get("include") and not e.get("planned")]
    runs = [run for e in experiments for run in e.get("runs") or []]
    failed = [run for run in runs if run.get("outcome") in ("timeout", "failed")]
    findings = [c for c in data.get("commits") or [] if c.get("finding")]

    start = _pretty_date(data["range"]["start"])[:-7]
    end = _pretty_date(data["range"]["end"])[:-7]

    stats = "".join([
        _stat(str(len(cards)), "experiments reported"),
        _stat(str(len(runs)), "runs on disk"),
        _stat(str(len(failed)), "failed or timed out", "stat-critical" if failed else ""),
        _stat(str(len(data.get("gaps") or [])), "results not pulled"),
        _stat(str(len(findings)), "findings committed"),
    ])

    parts = [
        _tag("p", "Weekly report", "eyebrow"),
        _tag("h1", _esc(data["week"])),
        _tag("p", f"{_esc(start)} — {_esc(end)}", "dates"),
    ]
    if narrative.get("headline"):
        parts.append(_tag("p", _esc(narrative["headline"]), "headline"))
    if narrative.get("summary"):
        parts.append(_tag("p", _esc(narrative["summary"]), "summary"))
    parts.append(_tag("div", stats, "stats stats-hero"))
    return _tag("header", "".join(parts), "deck-head")


def render(data: dict) -> str:
    experiments = data.get("experiments") or []
    by_thread: dict[str | None, list[dict]] = {}
    for experiment in experiments:
        by_thread.setdefault(experiment.get("thread"), []).append(experiment)

    order = [t for t in THREAD_ORDER if t in by_thread]
    order += [t for t in by_thread if t not in THREAD_ORDER]

    body = (
        _header(data)
        + "".join(_thread_section(thread, by_thread[thread]) for thread in order)
        + _gaps_section(data.get("gaps") or [])
        + _next_week((data.get("narrative") or {}).get("next_week") or [])
        + _appendix(data)
        + _tag("footer", f"Generated {_esc(data.get('generated_at'))} · "
                         "tools/weekly_report.py", "foot")
    )
    return PAGE.format(title=_esc(data["week"]), style=STYLE, body=body, script=SCRIPT)


STYLE = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --rule: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --good: #0ca30c;
  --warning: #fab219;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --rule: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --grid: #2c2c2a; --rule: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926;
}

* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 1.5rem 4rem;
  background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main, header, section { max-width: 62rem; margin: 0 auto; }
h1 { font-size: 2.6rem; margin: 0.2rem 0 0; letter-spacing: -0.02em; }
h2 { font-size: 1.35rem; margin: 3rem 0 0.25rem; letter-spacing: -0.01em; }
h3 { font-size: 1.05rem; margin: 0; }
h4 { font-size: 0.85rem; margin: 0 0 0.5rem; text-transform: uppercase;
     letter-spacing: 0.08em; color: var(--muted); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em; }

.deck-head { padding: 3rem 0 1rem; border-bottom: 1px solid var(--rule); }
.eyebrow { margin: 0; text-transform: uppercase; letter-spacing: 0.14em;
           font-size: 0.75rem; color: var(--muted); }
.dates { margin: 0.35rem 0 0; color: var(--ink-2); }
.headline { font-size: 1.35rem; line-height: 1.35; margin: 1.5rem 0 0; max-width: 44rem; }
.summary { margin: 0.75rem 0 0; color: var(--ink-2); max-width: 44rem; }

.stats { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.25rem 0; }
.stats-hero { margin: 2rem 0 1rem; }
.stat { flex: 1 1 8rem; background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.7rem 0.85rem; }
.stat-value { font-size: 1.6rem; line-height: 1.1; }
.stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 0.15rem; }
.stat-critical .stat-value { color: var(--critical); }
/* Inside a card the stats are a supporting row, not the page's headline. */
.stats-inline .stat { flex: 0 1 auto; padding: 0.4rem 0.7rem; }
.stats-inline .stat-value { font-size: 1.1rem; }

.thread-note { margin: 0 0 1.25rem; color: var(--muted); font-size: 0.85rem; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
        padding: 1.1rem 1.25rem; margin: 0 0 1rem; break-inside: avoid; }
.card-highlight { border-color: var(--series-1); box-shadow: 0 0 0 1px var(--series-1); }
.card-head { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: baseline;
             justify-content: space-between; }
.chips { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.chip { font-size: 0.72rem; padding: 0.1rem 0.45rem; border-radius: 99px;
        border: 1px solid var(--border); color: var(--ink-2); white-space: nowrap; }
.chip-good { color: var(--good); border-color: var(--good); }
.chip-warning { color: var(--ink); border-color: var(--warning); }
.chip-critical { color: var(--critical); border-color: var(--critical); }
.chip-muted { color: var(--muted); }

.commentary { margin: 0.9rem 0 0; font-size: 1.02rem; }
.takeaway { margin: 0.6rem 0 0; color: var(--ink-2); font-size: 0.9rem; }
.run { margin-top: 1.1rem; padding-top: 0.9rem; border-top: 1px solid var(--grid); }
.run-head { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
.path { color: var(--muted); margin-left: auto; }

.table-wrap { overflow-x: auto; margin: 0.8rem 0 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem;
        font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 0.3rem 0.55rem; border-bottom: 1px solid var(--grid);
         white-space: nowrap; }
thead th { color: var(--muted); font-weight: 500; font-size: 0.78rem; }
.row-head, tbody td:first-child, .appendix td { text-align: left; }
.row-head { font-weight: 600; }
.dir { color: var(--muted); font-weight: 400; }
.table-note { margin: 0.35rem 0 0; font-size: 0.75rem; color: var(--muted); }
.health { margin: 0.7rem 0 0; padding-left: 1.1rem; color: var(--ink-2); font-size: 0.85rem; }

.charts { margin-top: 1rem; }
.legend { display: flex; gap: 0.9rem; font-size: 0.75rem; color: var(--ink-2);
          margin-bottom: 0.4rem; }
.legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
.swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.sparks { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.9rem; }
.spark-figure { margin: 0; }
.spark-figure figcaption { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.15rem; }
.spark { width: 100%; height: 68px; overflow: visible; }
.spark-line { fill: none; stroke-linejoin: round; stroke-linecap: round; }
.spark-base { stroke: var(--rule); stroke-width: 1; }
.spark-label { font-size: 9px; fill: var(--ink-2); }
.spark-axis { font-size: 8px; fill: var(--muted); }
.spark-axis-end { text-anchor: end; }
.spark-hit { fill: transparent; cursor: crosshair; }

.media-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
             gap: 1rem; margin-top: 1rem; }
.media { margin: 0; }
.strip { display: flex; gap: 2px; }
.strip img { width: 100%; min-width: 0; border-radius: 3px; display: block; }
/* Capped so a clip supports the strip above it rather than swallowing the card. */
.media video { width: 100%; max-height: 220px; margin-top: 2px; border-radius: 3px;
               background: #000; object-fit: contain; }
.media figcaption { font-size: 0.75rem; color: var(--muted); margin-top: 0.3rem; }

.planned-box { background: var(--surface); border: 1px dashed var(--border);
               border-radius: 12px; padding: 1rem 1.25rem; }
.planned { margin: 0; padding-left: 1.1rem; font-size: 0.88rem; }
.planned li { margin-bottom: 0.5rem; }
/* Takeaways are paragraphs; clamped here because this list is an inventory, not a set of cards.
   The full text stays in the DOM for search and for anyone who wants it. */
.planned-note { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                overflow: hidden; color: var(--muted); font-size: 0.8rem; }
.next { font-size: 0.95rem; }
.gaps .row-head { font-weight: 600; }

.appendix { margin: 3rem auto 0; max-width: 62rem; }
.appendix summary { cursor: pointer; color: var(--ink-2); font-size: 0.9rem; }
.appendix table { margin-top: 0.6rem; }
.foot { max-width: 62rem; margin: 3rem auto 0; color: var(--muted); font-size: 0.75rem; }

#tip { position: fixed; z-index: 10; pointer-events: none; opacity: 0;
       background: var(--surface); color: var(--ink); border: 1px solid var(--border);
       border-radius: 6px; padding: 0.25rem 0.5rem; font-size: 0.75rem;
       font-variant-numeric: tabular-nums; transition: opacity 90ms; }

@media print {
  body { background: #fff; padding: 0; }
  .card, .planned-box, .thread { break-inside: avoid; }
  .appendix { display: none; }
  video { display: none; }
}
"""

SCRIPT = """
const tip = document.getElementById('tip');
document.addEventListener('mouseover', (event) => {
  const text = event.target.dataset && event.target.dataset.tip;
  if (!text) return;
  tip.textContent = text;
  tip.style.opacity = '1';
});
document.addEventListener('mousemove', (event) => {
  if (tip.style.opacity !== '1') return;
  tip.style.left = Math.min(event.clientX + 12, window.innerWidth - 220) + 'px';
  tip.style.top = (event.clientY + 16) + 'px';
});
document.addEventListener('mouseout', (event) => {
  if (event.target.dataset && event.target.dataset.tip) tip.style.opacity = '0';
});
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly report — {title}</title>
<style>{style}</style>
</head>
<body>
<main>{body}</main>
<div id="tip" role="status"></div>
<script>{script}</script>
</body>
</html>
"""
