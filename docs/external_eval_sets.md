# External nudity eval sets: what they are, where they came from, what we may claim

Built by `tools/build_external_nudity_evalsets.py` (deterministic — rerunning reproduces the
committed CSVs exactly). This page exists so nobody has to reverse-engineer the provenance of an
eval number later, and so the paper's claims about these sets stay inside what the data supports.

## 1. Why we needed them

> **Correction (2026-08-09).** The premise below is **wrong on its central factual claim** and is
> kept here only because it explains why these two sets exist. `prompts/cogvideox_nudity.csv` is
> **not** a set we wrote: it is byte-for-byte T2VUnlearning's released
> `evaluation/data/nudity_cogvideox.csv` — the same 100 prompts, same order, same 100 seeds — i.e.
> the "Gen" set of their Table 1. Our entire historical nudity series is therefore *already* on a
> published paper's own eval prompts and seeds. The comparability argument in point 1 does not
> apply; point 2 (we authored the *training* prompts) still does. Verified in code by
> `tools/build_t2vunlearning_evalsets.py:verify_gen_set`; full protocol mapping in
> [`comparability_t2vunlearning.md`](comparability_t2vunlearning.md).

Every nudity number the project had before 2026-08-07 was measured on `prompts/cogvideox_nudity.csv`
— a set **we wrote**. It is described in several places as "i2p-derived", which is misleading:
comparing it against the real I2P release finds **zero shared prompts**. It is I2P-*styled*
hand-written text, not the benchmark.

*(The "zero shared prompts with I2P" part is correct — the set is not I2P-derived. The inference that
it was therefore written in-house is what was wrong.)*

That is a problem on two independent counts:

1. **Comparability.** Published T2V unlearning work (T2VUnlearning, VideoEraser) reports nudity
   rates on real benchmarks. A number on our own prompts cannot be put in the same table, which
   removes the whole point of the "comparable to published work" framing in
   [`comparison_targets.md`](comparison_targets.md). — **Superseded: it is their benchmark.**
2. **Credibility.** We authored both the training prompts (`split_nudity*.csv`) and the eval
   prompts. That invites the reasonable objection that the eval vocabulary resembles what we
   trained on, and that a detection rate of 0.0 is partly an artefact of that overlap. Scoring on
   sets nobody on the team wrote removes the objection instead of arguing about it. — **Still
   stands for the training prompts, which are ours.**

`cogvideox_nudity.csv` is **not** retired — it stays as the set every historical run is measured on,
so exp062/exp073/exp077/exp080 remain mutually comparable, and (given the correction above) it is
the *most* comparable set we have, not the least. The external sets are added alongside it, not in
place of it.

## 2. `prompts/i2p_nudity.csv` — 95 prompts

**Source.** I2P (Inappropriate Image Prompts), Schramowski et al., *Safe Latent Diffusion*, CVPR
2023 — the standard prompt set for concept-erasure work. Fetched from the Ring-A-Bell repository's
copy of the 4703-prompt release (`data/unsafe-prompts4703.csv`).

**Filter.** `nudity_percentage > 50`, giving 95 of 4703 rows. `nudity_percentage` is a column *of
the benchmark*: the measured fraction of reference SD-1.4 samples for that prompt that contained
nudity. Filtering on it selects prompts that reliably elicit the concept — which is what an erasure
metric needs — and it is the benchmark's measurement, not our judgement. Rows are ordered by I2P's
own `case_number`.

**Seeds.** I2P's own `evaluation_seed`, carried through unchanged. The project seed policy
(CLAUDE.md) is therefore satisfied by the benchmark's seeds rather than any we invented. Note these
are large (up to ~4.3e9); `torch.Generator.manual_seed` accepts them (verified).

**Caveat that must appear in the paper.** I2P prompts were written for a text-to-*image* model.
They are comma-separated art-style prompts that frequently name artists ("art by ..."), not video
captions. Published T2V unlearning work reuses them as-is and so do we, but they are
out-of-distribution for a T2V model's usual caption style. That is precisely why the second set
below is worth reporting next to it — agreement across two very different prompt distributions is
a much stronger claim than either alone.

## 3. `prompts/safesora_nudity.csv` — 100 prompts

**Source.** SafeSora (PKU-Alignment), a text-to-**video** human-preference safety dataset, licence
**CC-BY-NC-4.0** (non-commercial research — fine for an academic paper, worth citing correctly).
Prompts taken from the released `config-test.json.gz`, deduplicated by `prompt_id`.

**Filter — ours, not SafeSora's.** SafeSora's released config files label prompts only
`safety_critical` / `safety_neutral`, with **no per-harm-category annotation**. The nudity subset is
therefore selected by a keyword filter we wrote (`SAFESORA_NUDITY_KEYWORDS` in the build script)
over the safety-critical prompts: 247 matched, capped to 100. **Do not describe this as "SafeSora's
nudity category"** — SafeSora does not define one in the release. Describe it as "the
safety-critical SafeSora prompts matching our published keyword filter", and point at the script.

**Seeds.** SafeSora ships none, so they are assigned deterministically from a SHA-256 hash of the
prompt text (`_stable_seed`) and frozen by committing the CSV. Hash-derived rather than positional
so that adding or reordering rows can never silently change an existing prompt's seed — the exact
failure the seed policy warns about.

**Character of the set.** Short, blunt, explicit phrasings (mean ~88 characters) — a very different
distribution from both I2P's long art prompts (~139 chars) and our own long cinematic training
prompts. Good for generalization claims; also the set most likely to expose that our training
vocabulary was too narrow.

## 4. What we deliberately did NOT build

**Ring-A-Bell adversarial nudity prompts.** — **Superseded (2026-08-09): these are available and
are now built.** The reasoning below is correct about the *Ring-A-Bell* repository, which releases
inverted prompts for Violence only, but it missed that **T2VUnlearning ships the 79 nudity prompts
it used** in its own repo (`evaluation/data/nudity-ring-a-bell.csv`). Their Ring-A-Bell column is
reproducible by download, not by re-running the attack. Built as `prompts/ring_a_bell_nudity.csv` by
`tools/build_t2vunlearning_evalsets.py`, together with the paired safe rewrites shipped alongside
them (`prompts/ring_a_bell_nudity_safe.csv`). The original reasoning, for the record:

> But the repository releases inverted (adversarial) prompts for **Violence only**
> (`data/InvPrompt/Violence/`); there is no nudity equivalent to download. Producing them means
> running their genetic-algorithm attack against our text encoder using the released
> `Concept Vectors/Nudity_vector.npy` — an implementation task, not a fetch.

One caveat survives: these are Ring-A-Bell prompts *as redistributed by T2VUnlearning*, and we have
not run the attack ourselves. Cite them as such. Note also that they are not adversarially strong
against CogVideoX — T2VUnlearning's own Original baseline scores *lower* on them (42.50) than on the
plain Gen set (61.80), so this is a second distribution, not a robustness test.

**exp079's preservation set is now an EVAL set, not a training set (2026-08-10).** It was built as a
training retention anchor, and this doc has described it that way throughout. exp085 established
that it does not work in that role: its human-filtered 20 entries are 11/20 exposed-skin wardrobe, so
the retention term pulled toward keeping exposed torsos while the erase term pushed away from the
same features, and every arm erased worse than the same grid on exp041's fire anchors
(see [`frame_replace.md`](frame_replace.md) §4.x). Its replacement for *training* is
`prompts/cogvideox_nudity_retention_clothed.csv` (exp104).

The upside: once nothing trains on it, `prompts/cogvideox_nudity_preservation.csv` becomes a
**legitimate held-out preservation set** — 30 nudity-adjacent prompts with committed seeds, already
generated and reviewed. For the swimwear question specifically it is *better* than
`cogvideox_nudity_control_related.csv`, because near-miss content is exactly where a nudity eraser
does its collateral damage, and that is a number to report rather than avoid. Do **not** use it to
score exp062-exp085, which trained against it.

**A held-out nudity `related` set.** — **Done (2026-08-08/09).**
`prompts/cogvideox_nudity_control_related.csv` (36 held-out nudity-adjacent prompts, 9 categories,
seeds 602001-602036, zero overlap with exp079's training anchors), plus the *paired*
`prompts/ring_a_bell_nudity_safe.csv`, where each safe rewrite carries the same seed as its
adversarial partner so the two differ only in wording. The original note, still correct on why the
exp079 set does not fill this slot:

> `prompts/cogvideox_nudity_preservation.csv` (exp079) is a *training* retention anchor set — once a
> run trains against it, scoring on it is evaluating on the training set.

## 5. How they are used

| exp | what | sets |
|---|---|---|
| [exp082](../experiments/nudity/exp082_eval_base_external_nudity/notes.md) | base-model "Original" reference | both, one grid job each |
| [exp083](../experiments/nudity/exp083_negprompt_nudity/notes.md) | NegPrompt training-free baseline | both, one grid job each |
| *(pending)* | our best checkpoint, once exp080 picks an LR | both |

The three together fill a comparison table whose rows are all measured by us on identical
`(prompt, seed)` pairs, with only the intervention differing.

NegPrompt support in the shared eval path is the `negative_prompt` field on
`zml/eval/eval_model.py:Config`, read by `zml/unlearn/eval.py:evaluate` via `getattr` (so no
training call site changed, and it stays `None` there). It is applied to **every** prompt set
including `unrelated`, because NegPrompt is a deployed inference-time defence — the collateral
damage it causes is part of what is being measured, exactly as PSR captures it on the object side.
When set, it is recorded as `_negative_prompt` in the run's `metrics.json` so a NegPrompt output
directory is identifiable on its own.
