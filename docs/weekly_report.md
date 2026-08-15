# Weekly report

How the weekly mentor deck is built: what `tools/weekly_report.py` collects, how it decides what "this
week" contains, and what a person has to add before it is a presentation. Reference for
`tools/weekly_report.py` and the `zml/report/` package. The curation half — what to include, how to
write the commentary — is the **`weekly-report` skill** (`.claude/skills/weekly-report/SKILL.md`);
this page is the mechanism behind it.

## 1. The problem

Results have to be presented every week, and the material is scattered across four places that no
single tool read together:

| Where | What only it has | Tracked? |
|---|---|---|
| `experiments/**/notes.md` | the authored interpretation — takeaway, results sections, open questions | yes |
| `outputs_*/summary.json`, `eval_step_*/metrics.json` | the numbers | no (gitignored) |
| `outputs_*/run_info.json` | what actually ran: cluster, wall time, timeout/failure | no |
| git log | the week's findings, as unprefixed `expNNN: <claim>` subjects | yes |

A busy week is 20–60 commits from three people across a dozen experiments, and none of the four
sources is complete on its own: a job can finish without its notes being updated, and notes are
routinely written about results that live on someone else's machine.

## 2. Mechanism

`collect` produces `report/weekly/<week>/data.json`; `render` turns it into `index.html` beside a
`media/` directory. `report/` is gitignored, which is deliberate — the deck is derived, and the
durable interpretation belongs in `notes.md`.

### 2.1 The window

`--week 2026-W33` pins an ISO week (Monday 00:00 local to the following Monday); `--since 7d` takes a
trailing span, labelled by the ISO week it ends in. A span other than 7 days gets the span appended to
the label (`2026-W33-last30d`) so a monthly sweep cannot overwrite the week it ends in.

Two comparison points come out of git, both resolved **by date rather than by `HEAD@{...}`** — the
reflog is per-checkout and would give each of the three of us a different answer for the same week:

- **base** — the last commit before the window opened.
- **head** — the last commit inside the window, or the *working tree* when the window has not closed
  yet. The working tree matters because the notes for the week being presented are usually still
  uncommitted on the morning of the meeting; using it for a *closed* week would write last month's
  report up with this month's findings.

### 2.2 What counts as "this week"

Two sources, unioned; neither subsumes the other.

**Runs on disk.** Every `outputs_{TS}/` and `grid_{TS}/run_NNN/outputs/` whose interval overlaps the
window. The time is taken from `run_info.json`'s `started_at`/`ended_at`, falling back to
`summary.json`'s `runtime`, then `runtime.json`, then the directory-name timestamp — which is the only
one present for *every* job type. **Filesystem mtimes are never used**: `pull_results.sh` rsyncs with
`-u` so mtimes come from the cluster, and the post-hoc scorers rewrite `metrics.json` days later,
which would date a July run to this week.

**Notes that changed.** Every `notes.md` whose content differs between base and head, **matched by
experiment id rather than by path**. This is load-bearing: `f048777` moved all 122 experiments into
`experiments/<thread>/`, and a path-keyed diff reads that as the entire project being new. For the
same reason, markdown link targets and `experiments/<thread>/expNNN_...` paths quoted in the prose are
normalised away before two revisions are compared — the regroup rewrote those in every file, and they
are not results. On a real week this cuts the changed-notes count from 55 to 47.

### 2.3 What each card carries

Registry metadata comes from `tools/experiments_index.py`'s `discover()` — the repo's only reader of
the `notes.md` frontmatter, reused so it cannot drift from `--check`. Per run: the `run_info.json`
fields, the eval trajectory, and whichever headline artifact the job type produces — `esr_psr.json`
for ImageNet, `id_similarity.json` for faces, and for a precompute build its **yield**
(`metadata.json` built, `skipped.json` skipped, `<outputs>_screened.json` and
`metadata_human_filtered*.json` kept), which is the result a dataset build reports.

Scores are read by `zml/results_io.py::latest_eval_scores`, which **merges** the two on-disk copies
rather than choosing one — see §4.

### 2.4 Curation, and why re-collecting is safe

`data.json` holds derived facts and a small set of authored fields: top-level `narrative`, and per
experiment `commentary`, `include`, `highlight`, `media`. `collect` re-applies the authored fields on
top of freshly gathered facts, matched by experiment id, so running it again after new results land
never discards what was written — the same guarantee, for the same reason, as
`zml/metrics_file.py::update_metrics_json`.

`media` is seeded only when empty, so a card the curator deliberately emptied stays empty.

## 3. Knobs

| Flag | Does | Too low | Too high |
|---|---|---|---|
| `--week` / `--since` | picks the window | a short span splits one result across two decks | a long span buries the week in a month of context |
| `--frames` (4) | frames per strip | 2 frames cannot show a mid-clip splice seam | strips shrink past legibility in the card's column |
| `--max-clips` (6) | playable clips in the whole deck | motion claims fall back to stills, which cannot show them | the deck gets heavy and stops scanning; warns past 50 MB |

## 4. Two traps this code exists to avoid

**DOVER `0.0` means "not measured".** DOVER's import fails on helios (aarch64) and the scorer records
a literal `0.0`. `zml/results_io.py` drops those, and the deck shows `—`. Reading them as scores would
report the project's only technical-quality metric as catastrophic on every helios run.

**The two copies of an eval are each incomplete.** `summary.json` keeps 4 significant figures but only
seven keys and is never touched by a post-hoc scorer; `eval_step_*/metrics.json` has the full metric
set and is where `score_dover.py` / `score_nudity_frame_rate.py` / `score_q16.py` merge their columns,
but `zml/unlearn/eval.py` rounds it to **2 decimal places** on write — which flattens `0.003924` to
`0.0`. Preferring either file silently loses a column, so `latest_eval_scores` merges: the summary
supplies precision for the keys it has, the step file supplies everything added since, and the step
file wins only where the summary's value is an unmeasured DOVER zero.

## 5. Status

| Date | What |
|---|---|
| 2026-08-16 | Built. Verified on 2026-W32 (24 cards, 58 runs) and 2026-W33 (25 cards, 27 gaps). `latest_eval_scores` extracted from `build_frame_replace_table.py` / `build_results_table.py`; both still emit byte-identical `.tex`. |

Known limits, in the order they will bite:

- **Registry metadata is always current, not historical.** A past week's deck shows each experiment's
  status as it is *today* (`superseded`, not the `active` it was then). Fine for a recent week;
  misleading for an old one.
- **Only what has been pulled can be reported.** The gaps section names the rest and prints the
  `./pull_results.sh --range N` that fixes it, but the deck cannot fetch results itself.
- **Grid arms each render a card section**, so a 9-arm grid is nine tables. Curate it down or
  highlight one arm.

## 6. Generalization

Nothing here is concept-specific except `zml/report/metrics.py`, which maps a concept to the metrics it
is judged on and the direction each moves. A new concept needs one entry in `CONCEPT_METRICS` — the
same one-place-per-concept rule as `zml/benchmarks/registry.py::build_detector`. Everything else keys
off the registry frontmatter and the artifact names, both of which every thread already produces.
