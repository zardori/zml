---
status: ready
concept: nudity
method: precompute
thread: nudity
takeaway: >
  Fully-clothed retention anchors, replacing exp079 — whose human-filtered set was 11/20
  exposed-skin wardrobe and therefore fought the erase term (that is why exp085 lost to exp086's
  fire anchors). 40 prompts, 8 categories x 5, shot grammar matched to the training targets.
  RESULT: 39/40 approved (only p20 / seed 603021 rejected), a 97.5% yield against exp078's 26% on
  split-prompt triples — plain single-prompt generation has no A/B/C seam to heal, so this set is
  cheap to extend if exp105 shows it working. Feeds exp105.
---
# exp104 — fully-clothed retention anchors

## Why exp079 has to be replaced rather than reused

exp085 (exp079 nudity anchors) erased *worse* than exp086 (exp041 fire anchors), which is backwards
from the premise exp079 was built on. The cause is compositional.

exp085 trained on `exp079_nudity_preservation_precompute/metadata_human_filtered.json` — 20 entries,
not the 30-prompt CSV. Scanning those 20 for wardrobe:

| | n |
|---|---|
| swimwear (bikini, trunks, one-piece, general) | 4 |
| athletic (leotard, sports bra) | 2 |
| sleepwear (pyjamas) | 1 |
| towels (bathing, spa) | 2 |
| close-ups of bare shoulders / midriff | 2 |
| **exposed skin, total** | **11 of 20** |

The retention loss was therefore pulling the model toward *keeping* exposed torsos while the erase
loss pushed away from the same features. exp041's fire anchors share nothing with the concept
region, so they never pull back — the "wrong" set won because it was the only one not competing.

Two things made this hard to see:

1. **Three of the eleven sit in categories labelled `closeup_clothed` / `multiperson_clothed`.** The
   label said clothed; the prompt said "bare shoulders and upper back", "midriff ... sports bra",
   "three friends in swimsuits". Category names cannot be trusted as a composition audit.
2. **The human filter made it worse, not better.** Against the source CSV: medical 4→1, parenting
   2→1, bathing 3→1, while swimwear kept 4 of 5. Skin-heavy prompts render more reliably, so
   selecting on visual quality drifted the set skin-ward by accident.

## The design rule this produces

**A training retention set must be semantically disjoint from the concept. Adjacent content is an
evaluation instrument, not a training anchor.** Swimwear and medical belong in a `related` column
where destroying them is collateral damage we measure and report — see
[`docs/frame_replace.md`](../../docs/frame_replace.md).

Disjoint alone is not sufficient, though, or exp041 would already be the answer. Retention only
helps where the erase term does damage, so `prompts/cogvideox_nudity_retention_clothed.csv` keeps
the **shot grammar** of the training targets — exp078's close-up and multi-person framings, indoor
and bedroom settings, one and two people, slow deliberate motion — with every subject fully covered.
That is the difference from exp041: same regions, different content.

40 prompts, 8 categories x 5 (formal_wear, casual, outerwear, uniform, closeup_clothed,
multiperson_clothed, intimacy_clothed, domestic_indoor), seeds 603001-603040. Verified: no seed
collision with 601xxx (exp079) or 602xxx (control_related), and a keyword scan for
swim/bikini/leotard/towel/pyjama/bathing/sports bra/singlet/midriff/strapless/bare-torso returns
zero hits.

## Review rule — this is the part that broke exp079

**Filter within each category and preserve the category balance.** Do not rank the whole set by
visual quality and keep the top N; that is exactly how exp079 became 55% skin. If a category loses
more than one or two entries, regenerate those prompts rather than let the composition shift.
Record the surviving per-category counts here before anything trains on it.

Write the filtered metadata to the **experiment root**, not under `outputs_*/` — that path is
gitignored, and a filtered metadata file living there is what aborted exp085's first submission.

## Status
- [x] Prompt set written and validated (40 rows, balanced, no collisions, no excluded wardrobe).
- [x] Submitted and completed (1 job).
- [x] Videos reviewed 2026-08-10 — **39/40 approved**, only `p20` (seed 603021) rejected.
- [ ] `metadata_human_filtered.json` written to the experiment root and committed.

### Per-category survival
| category | generated | kept |
|---|---|---|
| formal_wear | 5 | 5 |
| casual | 5 | 5 |
| outerwear | 5 | 5 |
| uniform | 5 | 5 |
| closeup_clothed | 5 | **4** |
| multiperson_clothed | 5 | 5 |
| intimacy_clothed | 5 | 5 |
| domestic_indoor | 5 | 5 |
| **total** | **40** | **39** |

Balance held — the one rejection is a single close-up, well clear of the collapse threshold in
`tools/filter_retention_metadata.py`. Contrast exp079, where filtering on per-clip quality took
medical 4→1 and bathing 3→1 while swimwear kept 4 of 5, and the set silently became 55% skin.

**The 97.5% yield is itself a finding.** exp078's split-prompt run kept 13 of 50 (26%). These are
plain single-prompt generations with no A/B/C seam to heal, so they render reliably — which means
the retention set is cheap to extend, unlike the training triples where each usable example costs
roughly three rejected ones. If exp105 shows clothed anchors working, scaling this set is a much
better use of GPU time than scaling the triples.

## Downstream
exp105 points `retention_metadata_file` / `retention_latents_dir` here. Nothing else consumes it
yet; if it works, exp088's successor should adopt it too (clean data + clothed retention).
