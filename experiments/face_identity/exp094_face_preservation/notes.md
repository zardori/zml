---
status: done
concept: face
method: preservation/precompute
thread: face_identity
takeaway: >
  Retention anchors for the face-identity axis: 5 identities x 3 prompts + 10 generic-person
  anchors (25 total, seeds 7601-7625). Submitted and complete (`outputs_20260811_185230`, 43m on
  helios) — exp095/exp096 now point at the real output dir. Human video review of the 25 anchors
  has not been done yet (see Status); not a blocker for exp095 to submit, since retention anchors
  are used wholesale rather than per-triple human-filtered.
---
# exp094 — face-identity preservation/retention precompute

## Why
Every erase run needs a retention branch to anchor non-target behavior (`docs/frame_replace.md` §4.3
covers why generically; `docs/imagenet_objects.md` §4's `retention_exclude` pattern is the direct
precedent this reuses unchanged). T2VUnlearning anchors face erasure on **one randomly chosen**
remaining identity — reasonable for their purposes, but the draw is unpublished (not reproducible)
and a single anchor is noisier than several. This builds a stronger, reproducible replacement: all
four non-erased identities, plus a category the paper has no equivalent of at all.

## Setup
`prompts/face_preservation.csv`, 25 rows, one CSV covering two tiers (deviating from the plan's
original two-file sketch — `preservation_precompute.py` takes exactly one `csv_path` and already
carries every extra column through into `metadata.json`, so one file with a `class_name` column is
simpler and functionally identical):

- **Identity tier** (15 rows, 3 per identity): plain everyday scenes naming each of the 5 identities
  — an airport terminal, a bookstore, a golf course, and so on — disjoint from both the 30 published
  eval prompts and the 30 split-training A-prompts (verified at generation time). `class_name` =
  identity name, so `retention_exclude` drops exactly the identity being erased, the same mechanism
  `exp069`'s `retention_exclude: "chain saw"` already uses — **zero code change** needed for this.
- **Generic tier** (10 rows): unnamed-person scenes — a woman walking a dog, a man playing guitar,
  and so on — `class_name: generic`, which never matches any `retention_exclude` value (always an
  identity name), so every erase run keeps these regardless of which identity it targets. These are
  the anchor for `face_present_rate` specifically: the real collateral risk of identity erasure is
  the model losing the ability to render faces at all, not just this-identity-vs-that-identity
  confusion, and none of the five-identity anchors would necessarily catch that on their own.

## What to watch
- Per `docs/face_identity.md` §3.1, the same NudeNet lesson applies here in spirit even though the
  detector differs: an anchor that fails to render a clear face gives ArcFace nothing to work with,
  so a human video-review pass (mirroring exp079's nudity-preservation review, which found 9 of 30
  rejects were blank renders invisible to the detector) matters before trusting these as training
  anchors.
- Category coverage after any rejects — 3 anchors per identity is already thin; losing one to a bad
  render leaves 2, which the note on exp079's `medical`/`bathing`/`parenting` categories suggests can
  meaningfully weaken that identity's specific retention signal.

## Downstream
Feeds exp095/exp096's `retention_metadata_file`/`retention_latents_dir`, each with
`retention_exclude` set to the identity that run erases.

## Status
- [x] Submitted — complete, `outputs_20260811_185230` (25/25 latents, exit_code 0, 43m on helios).
- [ ] Human video review of all 25 anchors (genuinely the right identity / a genuine unnamed face,
      not a blank or malformed render). Not yet done — exp095/exp096 use the anchors unfiltered in
      the meantime; revisit if `face_present_rate` on the retention set looks off during training.
- [ ] Category coverage recorded after any rejects.
