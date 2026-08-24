---
status: ready
concept: nudity
method: frame_replace
thread: nudity
takeaway: >
  ABLATION ARM: is hand-reviewing a dataset worth the hours? Identical to exp146 except it trains
  the 155 gen5 clips passing an automated A-side gate (nobody watched a video) instead of the 27 a
  person hand-picked from shard 1. Human labels proved essentially unpredictable from the metadata
  (best rule 68% vs a 54% keep-everything baseline), so this is not an attempt to imitate review —
  it is a test of whether review buys anything. If this ties or wins, we stop reviewing builds.
---
# exp147 — gen5 with the automated gate (the "is review worth it" arm)

## Why this exists
Hand review does not scale. Three concepts are being unlearned in parallel, every one needs
datasets, and shard 1 alone took a person 50 clips. So the question is worth one job: **does the
hand-picked set actually train better than a permissive automated gate?**

It is a fair question because the human labels turned out to carry information no metadata field
holds. Against the 50 labelled clips of shard 1, the best single-threshold rule over every field in
the metadata reaches **68% accuracy against a 54% keep-everything baseline** — i.e. barely better
than not filtering. Review is either seeing something real that the detector cannot express, or it
is noise. This arm tells us which.

## The gate
`original_max_confidence >= 0.3`, and nothing else. 155/200 clips.

| threshold | kept (shard 1) | recall of the human picks | precision |
|---|---|---|---|
| 0.3 | 36/50 | **85%** | 64% |
| 0.5 | 29/50 | 70% | 66% |

Two deliberate choices:

* **A-side only.** This asks whether the base model rendered the concept under prompt A at all. A
  clip where it did not has nothing to erase, so it teaches a pure wardrobe edit — the failure a
  uniform-navy donor set is most exposed to. It is objective, and it is the one criterion the human
  and the detector visibly agree on (`original_max_confidence` averages 0.555 on the kept clips vs
  0.347 on the rejected).
* **Nothing filters the edited side.** Target "cleanliness" *anti*-correlates with results across
  our three datasets:

  | dataset | mean frame conf of the target | outcome |
  |---|---|---|
  | CLEAN-75 | 0.001 | worst |
  | OLD-31 | 0.044 | **best checkpoint** |
  | GEN4-100 | 0.089 | middle (0.150) |

  The long-standing claim that old data had a "0.001 floor" was that number mis-attributed — 0.001
  is clean-75's. Residual concept in the target has never been shown to cap erasure here, and the
  clips where it survives partially are wanted for a separate reason (exp146's curriculum note).

## Read it against exp146
Same regime, same eta sweep, same eval sets, same seed. The two differ in filter *and* size (27 vs
155), which is the actual choice on the table rather than a clean single variable. If they separate,
the cheap tiebreaker is a third arm on the gate restricted to shard 1 (36 clips), which isolates the
filter from the size.

## Status
Not submitted; blocked on the same cluster-side merge as exp146, into `combined_dataset_gate/`.
