# Partial-Fire Prompt Search

An autonomous loop that discovers `(prompt, seed)` pairs which render **partial fire** — a clip
with a fire-free part and a part with clearly visible flames — for the `frame_replace_online`
unlearning method. That method edits a clip by replacing its fire frames with nearby fire-free
"donor" frames (`zml/unlearn/frame_replace_ops.py::edit_latent`), so it can only use clips that
contain enough fire-free frames. Hand-curating such pairs is slow because a pair's behaviour is
only known after generating and watching the video. This search automates the discovery.

## The loop

A single SLURM job (`job_type: search`) loads CogVideoX once and runs `num_rounds` of:

1. **Propose** — an OpenRouter model (`proposer_model`, any OpenAI-compatible id) is shown a compact
   digest of the best and worst pairs so far and asked for `candidates_per_round` new prompts that
   should ignite once, partway through the clip.
2. **Generate** — each proposed prompt is generated with `seeds_per_prompt` deterministic seeds
   (derived from `global_seed` + round + prompt index, so accepted pairs are reproducible).
3. **Score** — per clip:
   - `VideoFireDetector.frame_fire_confidences` → a 49-value per-frame fire profile;
   - `VideoClipScorer` → text-video alignment (quality);
   - `VideoColorfulnessScorer` → desaturation guard.
   These feed `zml/search/scorer.py::score`, which computes a **separation_score ∈ [0,1]** and an
   **accept** decision.
4. **Feed back** — the round's results update the digest for the next proposal.

Implementation: `zml/search/{scorer,proposer,partial_fire_search}.py`, entrypoint
`scripts/search.py`.

## Scoring

`separation_score` rewards a single clean ignition near the middle and punishes flicker. It blends
(weights are tunable constants at the top of `scorer.py`):

- **transition** — `1 / num_transitions`; a single off→on boundary scores 1.0, flicker decays it;
- **contiguity** — the two largest contiguous (no-fire, fire) blocks as a fraction of the clip;
- **balance** — peaks when fire occupies ~half the clip (border in the middle);
- **margin** — confident fire frames vs. confident no-fire frames.

**Acceptance** gates on *method-usability + quality*, not centeredness (per design): fire present,
≥`min_nofire_latent_frames` fire-free latent frames (counted via `build_latent_fire_mask`, matching
the trainer's `min_nofire_frames`), `clip_score ≥ clip_min`, `colorfulness ≥ colorfulness_min`.
Off-center pairs are still accepted — they just carry a lower `separation_score`, which is the
ranking signal.

## Outputs (in `outputs_{timestamp}/`)

- `results.jsonl` — one row per `(prompt, seed)`: full metrics **plus the raw per-frame
  confidences**, so thresholds/scoring can be re-evaluated offline without regenerating.
- `accepted_pairs.csv` — `prompt,seed,concept,concept_type,separation_score,onset_frame,
  fire_fraction,clip_score`. The first four columns match the train CSV, so it merges straight into
  `prompts/cogvideox_partial_fire.csv` (`_load_train_prompts` ignores the extra columns).
- `summary.json` — config echo, totals, per-round acceptance rate + mean separation (watch this to
  see whether the proposer is improving), and the top accepted pairs.
- `proposer_log.jsonl` — each round's request digest + raw model response.
- `videos/` — saved per the `save_videos` policy (`all` | `accepted` | `none`).

## Running it

Intended for **helios** (its compute nodes have outbound internet for the API call).

1. Put the API key on the cluster as `$HOME/.openrouter_env` (untracked), e.g.
   `OPENROUTER_API_KEY=sk-or-...`. `slurm/helios.sh` sources it before launching.
2. Submit (owners submit jobs manually):
   `./submit_job.py helios experiments/exp040_partial_fire_search/config.yaml`
3. Pull results: `./pull_results.sh --cluster helios`
4. Review `summary.json` / `accepted_pairs.csv`, then merge the best rows into
   `prompts/cogvideox_partial_fire.csv`.

For a quick end-to-end smoke test, override the config to a tiny budget (`num_rounds: 1`,
`candidates_per_round: 2`, `seeds_per_prompt: 1`, `num_inference_steps: 10`) and confirm all output
files are written.

## Key config fields (`SearchConfig`)

`proposer_model`, `proposer_temperature`, `candidates_per_round`, `num_rounds`, `seeds_per_prompt`,
`target_accepted` (early-stop once enough pairs are collected); generation (`num_inference_steps`,
`guidance_scale`, …); scoring (`frame_fire_threshold`, `min_nofire_latent_frames`, `clip_min`,
`colorfulness_min`); `save_videos`, `global_seed`.
