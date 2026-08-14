---
status: done
concept: nudity
method: split_prompt/precompute
thread: nudity
takeaway: >
  Answered: no ceiling in [0.85, 1.0], but no further improvement either — NudeNet gap declines
  slightly (0.560 @ 0.85 -> 0.540 @ 1.0). Direct frame-by-frame visual check (not just the
  detector) at the temporal seam found the clothed->nude transition is a hard cut at EVERY
  split_step_frac tested, 0.2 through 1.0 — including the fully degenerate 1.0 case with zero
  shared/heal-phase steps. Clip coherence (same person/pose/lighting/background across the cut)
  does not come from the heal-phase conditioning on prompt C; it's already fully present with
  zero heal steps. Picked **0.85** over 1.0: ties for best score, keeps a nonzero heal phase, and
  1.0 is the most out-of-distribution point in the sweep (zero jointly-conditioned steps) for no
  measured benefit.
---
# exp076 — split_step_frac sweep, high end (0.85-1.0)

## Why
exp074 swept `split_step_frac` in [0.2, 0.8] and, per human video review (see exp074's notes.md),
found 0.2/0.3 inconsistent (don't produce true nudity in all 5 cases) and 0.4-0.8 all consistently
good with a **slight upward tendency toward 0.8** — no seam artifact observed even at the top of
the range. That leaves the question genuinely open: does quality keep improving past 0.8, or does
`docs/split_prompt.md`'s predicted "too long a split phase -> visible hard seam" failure mode show
up somewhere in [0.85, 1.0]? `split_step_frac=1.0` is the fully degenerate case — the entire
schedule stays in the A/B split, the shared neutral prompt C never runs, so there is no seam-healing
at all. That's the natural upper bound to test against.

## Setup
Identical construction to exp074: `zml/precompute/split_prompt_precompute.py`, grid over
`split_step_frac: [0.85, 0.9, 0.95, 1.0]`, same 5 rows/seeds from `prompts/split_nudity_sweep.csv`,
`split_latent_frame: 7` fixed, `skip_plain_abc: true`, `save_latents: false`. `0.8` itself is not
re-run (already have it from exp074 `run_007`) — read this sweep's `0.85` onward against exp074's
`0.8` result for continuity.

## Evaluation plan
Same as exp074: once the grid finishes, run `scripts/benchmark.py` locally against a `nudity_report.py`
`grid_dir` config pointed at this experiment's `grid_{TIMESTAMP}` (no need for a separate cluster
job, it's CPU-only and fast) — but per exp074's own finding, **the NudeNet metric alone is not
trustworthy near a "does this look like real nudity" call**: the run_002/0.3 case scored confidently
"localized" by the metric while human review found it unreliable. Treat any automated read here as
a first pass only; the real verdict needs the same visual review exp074 got, especially watching
for the hard-seam artifact at 0.95/1.0 that the detector has no way to see at all.

## Results (grid_20260804_160538, all 4 runs, 5 clips each)

| split_step_frac | first_half_max | second_half_max | gap |
|---:|---:|---:|---:|
| 0.85 | 0.162 | 0.722 | **0.560** — best in this batch |
| 0.90 | 0.165 | 0.716 | 0.551 |
| 0.95 | 0.175 | 0.717 | 0.542 |
| 1.00 | 0.172 | 0.711 | 0.540 |

Slight monotonic decline from 0.85 to 1.0, not a rise — the "upward tendency" exp074's human
review flagged does not continue past 0.85. No dramatic collapse either; this is a soft, not hard,
signal.

## Visual analysis (mine, not delegated to the metric)

Pulled the videos and extracted frames straddling the temporal seam (~frame 22 clothed / ~frame 27
nude, out of 49) for two seeds, across split_step_frac = 0.2, 0.3, 0.5 (from exp074) and 0.85, 1.0
(from this run). Same finding at every single value, **including split_step_frac=1.0 — the fully
degenerate case where the shared neutral prompt C never runs at all**: the clothed->nude transition
is a hard cut, frame to frame, with identical person/pose/lighting/room on both sides of the cut.
No visible seam artifact, but also no visible smoothing effect from having more (or any) shared
heal-phase steps — 0.5 (50% heal phase) and 1.0 (0% heal phase) look the same at the seam.

**Conclusion:** clip coherence in this method does not come from the prompt-C healing phase.
It's much more likely a consequence of the whole latent sharing one initial noise sample and one
joint DPM-solver trajectory (a single scheduler step is applied to the full spliced latent every
timestep, by construction — see `split_prompt_precompute.generate_split_clip`), with the VAE decode
smoothing whatever's left at the pixel level. Can't cleanly separate "shared trajectory" from "VAE"
as the mechanism without inspecting raw latents pre-decode, which weren't saved for this sweep.

Also visually confirmed exp074's human-review finding of *inconsistency* (not clean failure) at
low split_step_frac: at 0.2, the same seed (3103) that renders clear nudity at 0.3/0.5/0.85/1.0
instead renders a jumpsuit/coverall — the 80%-of-schedule heal phase fully overwrites the concept
commitment at 0.2. At 0.3, that same seed *does* show real nudity, consistent with "hit or miss"
across the 5-clip batch rather than a uniform washout.

**Consequence for the value to pick:** since more heal-phase steps doesn't cost anything visible
in coherence but zero heal-phase steps is the most out-of-distribution point in the sweep for no
measured gain (metric ties or slightly favors 0.85 anyway), **picked 0.85** over the higher values
— not because 1.0 looked worse, but because there's no reason to take on that risk for nothing in
return. Reasoning discussed with the user directly; see exp078's notes.md for where this value
gets used next.

## Status
- [x] Submitted.
- [x] Aggregate report run (locally, mirroring exp074/exp075) — see Results above.
- [x] Visually reviewed — by me (frame extraction + inspection), not just the user this time; see
      Visual analysis above.
- [x] Final value picked: **0.85**. `exp078`'s config updated accordingly (submitted separately at
      0.8, before this landed — see exp078's notes.md for that timing note).
- [x] `split_prompt_precompute.py`'s and `frame_replace_split_precompute.py`'s hardcoded defaults
      updated from 0.5 to 0.85, so new experiments don't silently default back to the old
      exp059-era number.
