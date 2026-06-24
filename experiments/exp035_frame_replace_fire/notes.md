# exp035 — frame_replace (self-substitution) fire unlearning

## Idea
Fire prompts often produce clips where fire is present only in *some* frames (e.g. fire dies out
halfway). We take the model's own fire-free frames and substitute them — in latent space — over
the fire frames, then fine-tune a LoRA so the fire prompt maps toward that fireless version of
its own output. First feasibility check; not tuned, not expecting spectacular results.

## Pipeline
1. **Precompute** (`zml/precompute/frame_replace_precompute.py`): generate each clip, decode,
   run the per-frame fire detector, map the 49 pixel frames to 13 latent frames (1+4k), replace
   fire latent frames with the nearest fire-free donor frame, save `x0_edited` + `metadata.json`.
2. **Train** (`method: frame_replace`): supervised v-prediction SFT toward `x0_edited` (no
   teacher / CFG / negative guidance), PEFT LoRA on the attention projections.

## How to run
```
# 1. build the dataset as a precompute run (owners do this); it lands in that
#    experiment's outputs_{timestamp}/ (latents/ + metadata.json)
./submit_job.py athena experiments/exp034_frame_replace_precompute/config.yaml
# 2. point this experiment's metadata_file/latents_dir at that outputs_{timestamp} dir,
#    then submit training
./submit_job.py athena experiments/exp035_frame_replace_fire/config.yaml
```

## Things to watch
- **Coverage**: the method only applies to prompts that actually produce partial-fire clips.
  the precompute run's `skipped.json` tells us how many fire prompts were unusable (all-fire /
  no-fire / too few donor frames) — a key viability signal. Prompts engineered for a clean
  fire/no-fire split (see `prompts/part_fire_prompts.txt`, exp033) are the ideal source; consider
  building a seeded CSV of such prompts if `cogvideox_fire.csv` yields too few usable clips.
- **No retention loss in v1** (kept minimal). If preservation degrades, the natural next step is
  an SFT anchor toward the *original* (unedited) latents of preservation prompts.

## Notes / TODO
- The existing `unlearn_with_precomputed_latents.py` and `unlearn_model_normalized.py` call the
  transformer **without** `image_rotary_emb`, i.e. they train CogVideoX-5b without RoPE while eval
  generates *with* RoPE. This trainer passes RoPE explicitly. Worth flagging / fixing in the
  other trainers.

## Results
- (pending first run)
