---
status: ready
concept: nudity
method: eval
thread: nudity
takeaway: >
  Base-model reference on Ring-A-Bell (79 prompts), the only base generation missing from the
  T2VUnlearning-comparable table. The Gen half already exists: prompts/cogvideox_nudity.csv IS
  their released Gen set, and exp063 generated all 100 base clips on it — recovered from disk with
  tools/score_eval_videos.py, no GPU. One job. Not yet submitted.
---
# exp100 — base model on Ring-A-Bell nudity prompts

## Why
[`docs/comparability_t2vunlearning.md`](../../docs/comparability_t2vunlearning.md) establishes that
we are far closer to T2VUnlearning's protocol than the repo believed, and pins down what is actually
missing. Their CogVideoX-5B nudity table has two columns:

| Method | Gen | Ring-A-Bell |
|---|---|---|
| Original | 61.80 | 42.50 |
| NegPrompt | 46.35 | 14.91 |
| SAFREE | 35.12 | 10.64 |
| T2VUnlearning | **16.47** | **2.74** |

The **Gen** base row costs nothing: `prompts/cogvideox_nudity.csv` is byte-for-byte their released
`nudity_cogvideox.csv` (same 100 prompts, same order, same 100 seeds), and **exp063 already generated
all 100 base-model clips on it** on 2026-08-02. That run died in its CPU scoring phase, so the videos
sat on disk with no `metrics.json` for a week. `tools/score_eval_videos.py` recovers it from the
clips alone — it previously only handled the `run_NNN/outputs/` grid layout, which is why a
single-run experiment like exp063 was unrecoverable; it now handles `outputs_{timestamp}/` too.

So the only base-model **generation** still missing is Ring-A-Bell. That is this experiment.

## Setup
`prompts/ring_a_bell_nudity.csv` (79) — Ring-A-Bell's nudity prompts as redistributed by
T2VUnlearning's repo, built by `tools/build_t2vunlearning_evalsets.py`. The Ring-A-Bell repo itself
releases inverted prompts for Violence only, which is why `docs/external_eval_sets.md` recorded these
as unavailable; that entry is now corrected.

Same `control_unrelated_prompts` as exp082/exp083/exp084, so the collateral column stays comparable
across every row of our own table. No `related` set here — the paired safe rewrites are generated in
exp102, where they can be read against a method that actually changed something.

## What to watch
- **Do not call this robustness.** Their own Original baseline scores *lower* on Ring-A-Bell (42.50)
  than on plain Gen prompts (61.80), so on CogVideoX these short stylized art prompts are an easier
  distribution, not an adversarial one. A good score here is not evidence of attack resistance.
- **Seeds are ours, not theirs.** Ring-A-Bell ships none, so ours are hash-derived and frozen by
  committing the CSV. The column is comparable in prompt content, not in noise. Say so.
- Report `nudity_frame_rate` (their metric) next to `nudity_detection_rate` (ours). Both are written
  now; see the doc for why they are not interchangeable.

## Status
- [x] Prompt set built and committed.
- [ ] Submitted (1 job, 94 clips).
- [ ] Scored on x86_64 — NudeNet differs by ~1 video in 100 between aarch64 and x86_64 (see doc),
      so every reported row must be scored on one machine.
