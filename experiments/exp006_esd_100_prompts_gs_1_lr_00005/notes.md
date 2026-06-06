# exp006_esd_100_prompts_gs_1_lr_00005 — ESD fire, larger prompt set

## Setup

Basic ESD direct LoRA (no hypernet), reusing the lr/ngs that "worked" (in the FDR sense) in
exp005 run_005, but swapping in a **larger, more diverse prompt set** to test the intuition that
broader coverage gives better generalization of the erasure.

- prompts: `prompts/cogvideox_fire_100.csv` (100 fire prompts — vs exp005's smaller set)
- `negative_guidance_scale`: 1.0
- `learning_rate`: 5e-4
- `lora_rank`: 8, `lora_alpha`: 8.0
- `steps`: 1000, eval/checkpoint every 200

## Result (visual inspection)

Outputs were **not** synced to `summary.json`, so these are from watching the generated videos:

- concept fire-detection-rate ≈ **0.6** — no fire on **2 of 5** videos.
- Of the 5, only **1 collapsed to black-and-white**; the other **3 keep color but show visibly
  much less fire**.

## Verdict

**The most promising genuine erasure so far.** Unlike exp005 run_005 (whose FDR=0 was largely a
desaturation/quality collapse — see exp005 notes correction), here erasure is partial but comes
with **far less quality/color collapse**: most videos stay colorful while fire is clearly reduced.
The bigger, more diverse prompt set appears to be the key differentiator (better generalization).

## Caveats

- Numbers are visual, not from `summary.json` (outputs not pulled). Re-evaluate with the
  colorfulness guard added in exp027 to quantify the color preservation.
- The single B&W collapse shows the desaturation failure mode still lurks at the margin even at
  `ngs=1.0, lr=5e-4`.
