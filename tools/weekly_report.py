#!/usr/bin/env python3
"""Build the weekly mentor report: an HTML deck of what the week's experiments did.

Two steps on purpose. ``collect`` gathers the mechanical half — which experiments ran, what they
scored, what was written into their notes, and which results are missing from this machine — into
``report/weekly/<week>/data.json``. A person (or Claude, via the ``weekly-report`` skill) then writes
the headline and the per-experiment commentary into that file, and ``render`` turns it into
``index.html``. Re-running ``collect`` never overwrites what was written; see
``zml/report/collect.py::merge_curation``.

Reads only local files and git, so it needs no cluster access — but it can only report results that
have already been pulled. Anything missing is listed in the deck's gaps section rather than dropped.

Run:
    uv run python tools/weekly_report.py collect            # the last 7 days
    uv run python tools/weekly_report.py collect --week 2026-W33
    uv run python tools/weekly_report.py render --week 2026-W33
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from zml.report import html
from zml.report.collect import (
    DATA_NAME,
    collect,
    load_data,
    merge_curation,
    save_data,
    seed_media,
    week_dir,
)
from zml.report.media import MEDIA_DIR_NAME, media_bytes
from zml.report.window import Window, resolve_window

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FRAMES = 4
DEFAULT_MAX_CLIPS = 6
MEDIA_WARN_MB = 50


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _resolve(args: argparse.Namespace) -> Window:
    return resolve_window(args.week, getattr(args, "since", None))


def run_collect(args: argparse.Namespace) -> None:
    window = _resolve(args)
    print(f"Collecting {window.label}: {window.start:%Y-%m-%d %H:%M} -> {window.end:%Y-%m-%d %H:%M}")

    data = merge_curation(collect(window), load_data(window.label))
    for warning in seed_media(data, args.frames, args.max_clips):
        print(f"  warning: {warning}")

    path = save_data(window.label, data)
    cards = [exp for exp in data["experiments"] if exp["include"] and not exp["planned"]]
    planned = [exp for exp in data["experiments"] if exp["planned"]]
    uncommented = [exp for exp in cards if not exp["commentary"].strip()]

    print(f"Wrote {_relative(path)}")
    print(f"  {len(cards)} result cards, {len(planned)} staged/in-flight, {len(data['gaps'])} gaps")

    size_mb = media_bytes(week_dir(window.label) / MEDIA_DIR_NAME) / 1e6
    if size_mb > MEDIA_WARN_MB:
        print(f"  warning: media is {size_mb:.0f} MB (over {MEDIA_WARN_MB} MB)")
    if not data["narrative"]["headline"] or uncommented:
        print(f"  next: write narrative.headline and commentary for {len(uncommented)} card(s) "
              f"in {DATA_NAME}, then run `render`")


def run_render(args: argparse.Namespace) -> None:
    window = _resolve(args)
    data = load_data(window.label)
    if data is None:
        raise SystemExit(f"No {DATA_NAME} for {window.label} — run `collect` first")

    path = week_dir(window.label) / "index.html"
    path.write_text(html.render(data))
    print(f"Wrote {_relative(path)}")
    if args.open:
        webbrowser.open(path.as_uri())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_window_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--week", help="ISO week to report on, e.g. 2026-W33")
        sub.add_argument("--since", help="trailing window instead of a week, e.g. 7d or 2w")

    collect_parser = subparsers.add_parser("collect", help="gather the week's facts into data.json")
    add_window_flags(collect_parser)
    collect_parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                                help="frames per strip (default: %(default)s)")
    collect_parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS,
                                help="playable clips across the whole deck (default: %(default)s)")
    collect_parser.set_defaults(func=run_collect)

    render_parser = subparsers.add_parser("render", help="turn data.json into index.html")
    add_window_flags(render_parser)
    render_parser.add_argument("--open", action="store_true", help="open the deck in a browser")
    render_parser.set_defaults(func=run_render)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
