# ZML — Concept Unlearning for Text-to-Video Models

Research project for erasing target concepts (currently: **fire**) from [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b), a video diffusion transformer, without degrading general video quality. The method is based on **ESD** (Erased Stable Diffusion) adapted for video, using LoRA fine-tuning.

**Stack:** Python 3.12 · uv · SLURM on [PLGrid Athena](https://www.cyfronet.pl/en/computers/athena.html) and [Helios](https://www.cyfronet.pl/en/computers/helios.html) HPC clusters

---

## Repository Structure

```
zml/
├── zml/                         # shared library code
│   ├── unlearn/                 # ESD training methods
│   ├── precompute/              # precompute latent trajectories for faster training
│   ├── eval/                    # evaluation pipeline (generation + scoring)
│   └── benchmarks/              # one-off benchmarking scripts
├── experiments/                 # one folder per experiment
│   ├── exp001_esd_fire_lora8/
│   │   └── config.yaml          # all hyperparameters for this run
│   ├── exp005_esd_fire_grid/     # grid-search experiment
│   │   ├── config.yaml          # base config — list values = swept params
│   │   └── grid/
│   │       ├── run_001/
│   │       │   ├── config.yaml  # scalar config for this combination
│   │       │   ├── logs/        # SLURM stdout/stderr
│   │       │   └── outputs/     # checkpoints + per-step eval results
│   │       └── run_002/ ...
│   └── ...
├── scripts/                     # thin entrypoints (called by SLURM scripts)
│   ├── unlearn.py
│   └── eval.py
├── slurm/                       # SLURM templates
│   ├── unlearn.sh
│   └── eval.sh
├── prompts/                     # prompt datasets (CSV / TXT)
├── docs/                        # method write-ups & design notes (e.g. unhype.md)
├── legacy/                      # deprecated scripts, kept for reference
├── submit_job.py                # submit jobs to a cluster (single-run or grid search)
├── pull_results.sh              # rsync results from a cluster
├── watch_jobs.sh                # live SLURM queue monitor (both clusters)
├── interactive.sh               # open an interactive SLURM session
└── cluster.conf.example         # cluster config template
```

---

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure cluster access (fill in your credentials)
cp cluster.conf.example cluster.conf
```

`cluster.conf` fields:
- `SLURM_USERS` — comma-separated PLGrid usernames (for `watch_jobs.sh`)
- `ATHENA_HOST` / `HELIOS_HOST` — SSH hostnames of the clusters
- `ATHENA_REMOTE_DIR` / `HELIOS_REMOTE_DIR` — your remote working directory (used when submitting jobs)
- `ATHENA_REMOTE_DIRS` / `HELIOS_REMOTE_DIRS` — arrays of all team members' remote dirs (used when pulling results)

---

## Utility Scripts

### `submit_job.py` — submit jobs to a cluster

Commits must be pushed before submitting — the cluster runs `git pull` before each job.

```bash
# Single run on athena (default)
./submit_job.py slurm/unlearn.sh experiments/exp001_esd_fire_lora8/config.yaml

# Single run on helios
./submit_job.py --cluster helios slurm/helios_unlearn.sh experiments/exp001_esd_fire_lora8/config.yaml

# Grid search — any config field that is a list triggers Cartesian product expansion
./submit_job.py slurm/unlearn.sh experiments/exp005_esd_fire_grid/config.yaml
```

For a grid search, the script:
1. Expands list-valued fields into all combinations
2. Creates `experiments/EXP/grid/run_001/`, `run_002/`, … on the remote with scalar configs
3. Submits one `sbatch` job per combination

The script warns (but does not block) if there are uncommitted changes or unpushed commits.

---

### `pull_results.sh` — download results

Rsyncs `experiments/` and `mlruns/` from all team members' remote directories.

```bash
./pull_results.sh                            # full sync from athena (default)
./pull_results.sh --cluster helios           # full sync from helios
./pull_results.sh --logs-only                # only SLURM logs, skip outputs
./pull_results.sh --include-adapters         # also download .safetensors checkpoints (large)
```

---

### `watch_jobs.sh` — monitor SLURM queue

Polls `squeue` on both Athena and Helios for all team members and refreshes the terminal.

```bash
./watch_jobs.sh          # refresh every 30 s (default)
./watch_jobs.sh -i 60    # refresh every 60 s
```

---

## Python Entry Points (`scripts/`)

These are called by the SLURM templates. You can also run them locally if you have enough GPU memory.

### `scripts/unlearn.py`

Runs ESD concept-unlearning training. Wraps `zml.unlearn.unlearn_model.main()` with MLflow and W&B logging.

```
--config PATH       path to experiment config YAML  (required)
--output_dir PATH   where to write checkpoints and eval outputs  (default: .)
```

Experiment name is inferred from the config path:
- `experiments/exp005/grid/run_001/config.yaml` → `exp005`
- `experiments/exp001/config.yaml` → `exp001`

Logs to MLflow (`mlruns/`) and W&B (project `zml`, entity `zardori-zml`).

---

### `scripts/eval.py`

Generates videos with an optionally LoRA-adapted model and scores them. Wraps `zml.eval.eval_model.main()`.

```
--config PATH       path to experiment config YAML  (required)
--output_dir PATH   where to write generated videos and metrics.json  (default: .)
```

Metrics logged per prompt set:
- `fire_detection_rate` — fraction of videos where YOLOv8 detects fire
- `clip_score_mean` — CLIP text-image alignment (8 sampled frames per video)
- `dover_technical_mean` — DOVER technical video quality
- `dover_aesthetic_mean` — DOVER aesthetic video quality

---

## Experiment Config Format

All hyperparameters live in `config.yaml`. For a single run every field is a scalar:

```yaml
model_id: THUDM/CogVideoX-5b
prompts_path: prompts/cogvideox_fire.csv
control_concept_prompts: prompts/cogvideox_fire_control_fire.txt
control_related_prompts: prompts/cogvideox_fire_control_related.txt
control_unrelated_prompts: prompts/cogvideox_fire_control_unrelated.txt
lora_rank: 8
lora_alpha: 8.0
negative_guidance_scale: 2.0
steps: 1000
save_interval: 200
learning_rate: 0.0002
lora_dropout: 0.0
eval_num_prompts: 3
eval_inference_steps: 50
```

For a **grid search**, replace any scalar with a list — `submit_job.py` expands the Cartesian product:

```yaml
negative_guidance_scale: [0.5, 1.0, 2.0]
learning_rate: [0.0002, 0.0005]
# → 6 runs
```

---

## Research Workflow

1. **Add prompts** — put a CSV (with `prompt` and `seed` columns) or TXT file in `prompts/`
2. **Create experiment** — make `experiments/expXXX_NAME/config.yaml`
3. **Commit and push** — the cluster pulls from the repo; uncommitted changes won't be picked up
4. **Submit** — `./submit_job.py slurm/unlearn.sh experiments/expXXX_NAME/config.yaml`
5. **Monitor** — `./watch_jobs.sh`
6. **Download** — `./pull_results.sh`
7. **Analyze** — inspect `experiments/expXXX_NAME/outputs_*/` (or `grid/run_XXX/outputs/`) for videos and `metrics.json`
8. **Iterate** — adjust config, create a new experiment folder, repeat
