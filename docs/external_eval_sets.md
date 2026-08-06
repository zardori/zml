# External nudity eval sets: what they are, where they came from, what we may claim

Built by `tools/build_external_nudity_evalsets.py` (deterministic — rerunning reproduces the
committed CSVs exactly). This page exists so nobody has to reverse-engineer the provenance of an
eval number later, and so the paper's claims about these sets stay inside what the data supports.

## 1. Why we needed them

Every nudity number the project had before 2026-08-07 was measured on `prompts/cogvideox_nudity.csv`
— a set **we wrote**. It is described in several places as "i2p-derived", which is misleading:
comparing it against the real I2P release finds **zero shared prompts**. It is I2P-*styled*
hand-written text, not the benchmark.

That is a problem on two independent counts:

1. **Comparability.** Published T2V unlearning work (T2VUnlearning, VideoEraser) reports nudity
   rates on real benchmarks. A number on our own prompts cannot be put in the same table, which
   removes the whole point of the "comparable to published work" framing in
   [`comparison_targets.md`](comparison_targets.md).
2. **Credibility.** We authored both the training prompts (`split_nudity*.csv`) and the eval
   prompts. That invites the reasonable objection that the eval vocabulary resembles what we
   trained on, and that a detection rate of 0.0 is partly an artefact of that overlap. Scoring on
   sets nobody on the team wrote removes the objection instead of arguing about it.

`cogvideox_nudity.csv` is **not** retired — it stays as the in-house set every historical run is
measured on, so exp062/exp073/exp077/exp080 remain mutually comparable. The external sets are
added alongside it, not in place of it.

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

**Ring-A-Bell adversarial nudity prompts.** [`comparison_targets.md`](comparison_targets.md) lists
Ring-A-Bell as a target, and it would give the robustness/red-teaming row reviewers increasingly
expect. But the repository releases inverted (adversarial) prompts for **Violence only**
(`data/InvPrompt/Violence/`); there is no nudity equivalent to download. Producing them means
running their genetic-algorithm attack against our text encoder using the released
`Concept Vectors/Nudity_vector.npy` — an implementation task, not a fetch. Tracked as possible
future work; **not** claimed as available, and nothing in the repo should be labelled
"Ring-A-Bell" until that attack is actually run.

**A held-out nudity `related` set.** Still missing. `prompts/cogvideox_nudity_preservation.csv`
(exp079) is a *training* retention anchor set — once a run trains against it, scoring on it is
evaluating on the training set. A separate, held-out nudity-adjacent set (swimwear, medical,
clothed intimacy, …) is needed before `control_related_prompts` means anything for nudity; every
nudity config currently points that slot at the unrelated set as a placeholder.

## 5. How they are used

| exp | what | sets |
|---|---|---|
| [exp082](../experiments/exp082_eval_base_external_nudity/notes.md) | base-model "Original" reference | both, one grid job each |
| [exp083](../experiments/exp083_negprompt_nudity/notes.md) | NegPrompt training-free baseline | both, one grid job each |
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
