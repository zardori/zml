---
name: weekly-report
description: How to build the weekly mentor presentation — running tools/weekly_report.py to collect the week's runs, metrics, notes and videos into report/weekly/<week>/data.json, then curating that file (headline, per-experiment commentary, media picks) and rendering it to an HTML deck. Use whenever asked for "the weekly report", "this week's results", a presentation or update for the mentor, or to rebuild/extend an existing week's deck.
---

# Building the weekly mentor deck

The rule this whole skill serves: **the script collects, you curate.** A week produces 20–60 commits
across three people and touches a dozen experiments, and `collect` will happily put every one of them
on a card. A deck that reports every experiment reports none. Your job is the half a script cannot
do: decide what the week *meant*, and cut the rest.

## 1. Run the collector

```
uv run python tools/weekly_report.py collect                  # the last 7 days
uv run python tools/weekly_report.py collect --week 2026-W33  # a pinned ISO week
uv run python tools/weekly_report.py render --week 2026-W33 --open
```

`collect` writes `report/weekly/<week>/data.json` and seeds `media/`. You then **edit that file** and
re-run `render`. Mechanics, and why the window is derived the way it is: **`docs/weekly_report.md`**.

**Re-running `collect` is always safe.** It merges fresh facts underneath `narrative`, and each
experiment's `commentary`, `include`, `highlight` and `media` — so collect again whenever new results
land mid-curation, rather than working around stale numbers.

**It reads only local files.** Results that nobody pulled are absent, and land in `gaps` instead. If
the gaps list is long and the week's story depends on those numbers, run the `./pull_results.sh`
command the deck prints and collect again — do not write around the hole.

## 2. What goes in the deck

Set `include` and `highlight` per experiment. Test each against the row it fits:

| Treatment | What earns it | Test |
|---|---|---|
| `highlight: true` | A number moved, or a belief changed. New best checkpoint, a refuted diagnosis, a dataset blocker cleared. Two or three per deck, never more. | "Would the mentor's next question start here?" |
| `include: true` | A result the highlights lean on, or a run whose outcome the mentor already knows is pending. | "Does a highlight or a plan become unexplainable without it?" |
| `include: false` | Bookkeeping — a config wired, a status corrected, a reference rewritten. | "Is the only thing that happened that a file changed?" |

Three cases people get wrong. **A timed-out or failed job is news, not noise** — it explains where a
week went and is already on the card. **A null result is a result**: exp088's "un-freezing the donors
changed nothing" is a finding, and burying it makes the next month's plan worse. **A dataset build's
yield is a result** — both concepts transferred so far were blocked on it, so `14/30 usable` is a
headline, not a footnote.

`ready`/`active` experiments are rendered as a compact "staged and in flight" list automatically. Do
not promote one to a card because its notes are long; it has no results yet.

## 3. Writing the commentary

Two or three sentences per included card, into `commentary`. It is the only prose the mentor reads
before the numbers, so it must say **what changed and against what**.

- **Source it from the week, not from memory.** Each experiment carries `notes.added_lines` — the
  lines written into its `notes.md` inside the window — plus its registry `takeaway`. That is the
  authored interpretation; compress it, do not reinvent it.
- **Name the comparison.** "rate 0.0000 at colourfulness 35.4" means nothing alone; "against exp080
  run_002 step 120's 0.0000 / 21.9" is the sentence. Every erasure number needs the baseline or
  incumbent it beat, and every utility number needs the base model.
- **Quote numbers, never recall them.** They come from the card's own tables, which come from
  `summary.json` / `eval_step_*/metrics.json` / `esr_psr.json` — the same rule as `project-docs` §4.
- **Say what is still open.** A checkpoint pending human review, a confound that cannot be removed,
  a metric that was not measured. The mentor will ask; answering first is cheaper.

Then write `narrative.headline` (one sentence — the week in a line), `narrative.summary` (two or
three), and `narrative.next_week` (what the highlights make obvious, in order).

## 4. Choosing media

`collect` seeds a default strip per card. Replace it when the default does not show the claim.

- **Frame strips are the default.** They scan, print and screenshot, and most claims — erasure,
  wardrobe realism, colour, a splice seam — are visible in stills.
- **Clips are for motion claims only.** Motion collapse is this method's standing cost and a still
  cannot show it. Spend the clip budget there; a clip on an erasure claim is wasted weight.
- **Never show an "after" alone.** Every comparison pairs against a base-model or incumbent clip from
  the same prompt and seed — otherwise the reader has no idea what changed.
- Captions state what to look at, not what the file is. "Chain saw survives prompt A but the splice
  suppresses it" beats "p0_s3201 edited".

## 5. Honesty rules

- **DOVER `0.0` is "not measured", never a quality score** — helios cannot import it. The collector
  already drops those; never reintroduce one by hand-writing it into commentary.
- **Report the per-frame nudity rate, not the video rate,** when comparing to T2VUnlearning, and say
  which one a number is. See `docs/comparability_t2vunlearning.md`.
- **Mark numbers pending human review as pending.** The nudity thread's standing rule is that no
  detector number is reported before a person has looked at the clips.
- **Leave the gaps section in.** Deleting a gap because it is inconvenient turns "not on this
  machine" into "did not happen".

## 6. Checklist before presenting

- [ ] `render` re-run after the last edit to `data.json`, and the deck opened and read.
- [ ] Two or three highlights, each with a comparison in its first sentence.
- [ ] Every included card has commentary; every excluded one deserved excluding.
- [ ] Every number in the prose appears in a table on the same card.
- [ ] Media pairs against a baseline; clips only where the claim is about motion.
- [ ] Gaps section reflects reality — either pulled and re-collected, or listed.
- [ ] `narrative.next_week` follows from the highlights, not from the backlog.
