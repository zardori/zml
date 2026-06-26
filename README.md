# ZML — Concept Unlearning for Text-to-Video Models

Research project for erasing a target concept (currently: **fire**) from
[CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b), a video diffusion transformer, without
degrading the model's general video quality. We explore several unlearning methods rather than a
single one — classic **ESD** variants, the CLIP-guided **UnHype** hypernetwork, and a supervised
**frame-replace** approach — all implemented as LoRA fine-tunes and compared on a shared evaluation
suite.

**Stack:** Python 3.12 · uv · SLURM on [PLGrid Athena](https://www.cyfronet.pl/en/computers/athena.html)
(A100 40 GB) and [Helios](https://www.cyfronet.pl/en/computers/helios.html) (GH200 96 GB) HPC clusters.

---

## Repository Structure

```
zml/
├── zml/                         # shared library code
│   ├── unlearn/                 # unlearning methods (ESD, UnHype, frame-replace) + metrics logging
│   ├── precompute/              # precompute latents/targets to speed up training
│   ├── eval/                    # evaluation pipeline (generation + scoring)
│   ├── search/                  # autonomous (prompt, seed) search for partial-fire clips
│   └── benchmarks/              # one-off benchmarking scripts
├── experiments/                 # one folder per experiment (expNNN_name)
│   ├── exp001_esd_fire_lora8/
│   │   ├── config.yaml          # all hyperparameters for this run
│   │   ├── logs_{TIMESTAMP}/    # SLURM stdout/stderr
│   │   ├── outputs_{TIMESTAMP}/ # checkpoints, generated videos, metrics
│   │   └── notes.md             # what was tried, what happened
│   └── exp005_esd_fire_grid/    # grid-search experiment (alternative pattern)
│       ├── config.yaml          # base config — list values = swept params
│       └── grid_{TIMESTAMP}/
│           ├── run_001/         # config.yaml (scalar) + logs/ + outputs/
│           └── run_002/ ...
├── scripts/                     # thin entrypoints, dispatched by SLURM via JOB_TYPE
│   ├── unlearn.py
│   ├── eval.py
│   ├── precompute.py
│   └── search.py
├── slurm/                       # one generic SLURM script per cluster
│   ├── athena.sh                # dispatches on JOB_TYPE (unlearn|eval|precompute|search)
│   └── helios.sh
├── prompts/                     # prompt datasets (CSV / TXT), incl. vbench_prompts/
├── docs/                        # method write-ups & design notes
├── tools/                       # repo utilities (e.g. prompt-file merging)
├── legacy/                      # deprecated scripts, kept for reference
├── submit_job.py                # submit jobs to a cluster (single-run or grid search)
├── pull_results.sh              # rsync outputs + MLflow data from clusters
├── pull_weights.sh              # rsync checkpoint weights from a cluster
├── push_weights.sh              # push local weights to a cluster
├── watch_jobs.sh                # live SLURM queue monitor (both clusters)
├── interactive.sh               # open an interactive SLURM session
└── cluster.conf.example         # cluster config template
```

---

## Setup

You work in two places: **locally** you author experiments and submit/monitor jobs; on the
**cluster** the jobs actually run. Set up both.

### Local machine

```bash
# 1. Clone the repo
git clone <repo-url> zml && cd zml

# 2. Install dependencies (creates the .venv via uv)
uv sync

# 3. Configure cluster access (fill in your credentials)
cp cluster.conf.example cluster.conf
```

`cluster.conf` fields:
- `SLURM_USERS` — comma-separated PLGrid usernames (for `watch_jobs.sh`)
- `ATHENA_HOST` / `HELIOS_HOST` — SSH hostnames of the clusters
- `ATHENA_REMOTE_DIR` / `HELIOS_REMOTE_DIR` — your remote working directory (used when submitting jobs)
- `ATHENA_REMOTE_DIRS` / `HELIOS_REMOTE_DIRS` — arrays of all team members' remote dirs (used when pulling results)

> Local machines in this project have ≤ 8 GB VRAM, which cannot run CogVideoX. All training and
> generation happens on the cluster; only lightweight evaluation models (e.g. the YOLO fire
> detector) can be tested locally.

### Cluster (Athena / Helios)

Do this once per cluster you intend to use. Clone into your **group storage**, not `$HOME` — the
home quota is small and uv's dependency cache and the HF model cache are large.

```bash
# 1. Clone the repo into group storage (path must match cluster.conf's *_REMOTE_DIR)
cd "$PLG_GROUPS_STORAGE/plggtriplane/$USER"   # or your team's storage path
git clone <repo-url> zml && cd zml
```

```bash
# 2. Make sure uv is installed, and point its cache at group storage so deps don't fill $HOME.
#    Append to ~/.bashrc so every login shell (and SLURM job) sees it:
cat >> ~/.bashrc <<'EOF'

# --- zml ---
# Install uv if missing (skips if already on PATH)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Keep uv's package cache off the small home quota
export UV_CACHE_DIR="$PLG_GROUPS_STORAGE/plggtriplane/$USER/.uv_cache"

# Weights & Biases auth for training/eval jobs
export WANDB_API_KEY="<your-wandb-api-key>"
EOF

source ~/.bashrc
```

`uv sync` runs automatically inside each SLURM job via `uv run`, so you don't need to install
dependencies manually on the cluster — the first job just populates `UV_CACHE_DIR`. The HF model
cache is handled per-job by the SLURM scripts (`HF_HOME=hf_cache` inside the repo dir).

> Adjust `plggtriplane` and the storage path to your team's grant. `UV_CACHE_DIR` and the repo
> clone should live under the same group-storage tree.

---

## Job Types & Entry Points

Every job runs through one cluster SLURM script (`slurm/athena.sh` or `slurm/helios.sh`) that holds
only that cluster's account/partition/repo-dir and dispatches on the `JOB_TYPE` env var to the right
thin entrypoint in `scripts/`. `submit_job.py` supplies the job name, time, and log paths as `sbatch`
flags. The `job_type` config field selects the entrypoint:

| `job_type`   | Entrypoint            | Purpose                                                   |
|--------------|-----------------------|-----------------------------------------------------------|
| `unlearn`    | `scripts/unlearn.py`  | Run a concept-unlearning training method (default)        |
| `eval`       | `scripts/eval.py`     | Generate videos with a model and score them               |
| `precompute` | `scripts/precompute.py` | Precompute latents/targets reused during training       |
| `search`     | `scripts/search.py`   | Autonomously search for partial-fire `(prompt, seed)` pairs |

### Unlearning methods (`scripts/unlearn.py`)

The `method` config field selects which training routine runs:

`esd`, `esd_preservation`, `esd_preservation_anchor`, `esd_normalized`, `unhype`,
`frame_replace`, `frame_replace_online`, `smoke_test`.

Training logs to MLflow (`mlruns/`) and W&B (project `zml`, entity `zardori-zml`). The experiment
name is inferred from the config path (`experiments/exp005/grid_*/run_001/config.yaml` → `exp005`).

### Precompute methods (`scripts/precompute.py`)

The `method` field selects: `frame_replace` or `preservation`.

### Evaluation (`scripts/eval.py`)

Generates videos with an optionally LoRA-adapted model and scores them per prompt set:
- `fire_detection_rate` — fraction of videos where YOLO detects fire
- `fire_area_score_mean` — mean fire-area coverage across frames
- `clip_score_mean` — CLIP text–image alignment
- `colorfulness_mean` — Hasler–Süsstrunk colorfulness (catches quality collapse that CLIP misses)
- `dover_technical_mean` / `dover_aesthetic_mean` — DOVER video-quality scores

---

## Metrics Logging

Training runs log to wandb and MLflow, and in parallel write two plain files into the run's
`output_dir` (via `zml/unlearn/metrics_log.py`), meant to be read directly without the wandb UI:

- `metrics.jsonl` — append-only; one object per flushed train window and per eval. Downsampled
  full history, crash-robust, machine-parseable.
- `summary.json` — overwritten each update; the at-a-glance artifact. Holds the config echo,
  per-metric train trends, compact per-checkpoint eval scores, and a derived `health` block with
  flags and plain-language notes. **Read this first when analyzing a run.**

Train metrics are buffered and flushed every `metrics_log_interval` steps (default 50). Currently
wired into `unhype`, `frame_replace`, `frame_replace_online`, and `esd_normalized`.

---

## Experiment Config Format

All hyperparameters live in `config.yaml`. Two infra fields are always required/recognized:

- `slurm_time` — passed as the sbatch `--time` (e.g. `"0-4:00:00"`); **no default**, a missing value
  is rejected.
- `job_type` — `unlearn` (default) | `eval` | `precompute` | `search`.

A single run has scalar values throughout:

```yaml
slurm_time: "0-4:00:00"
job_type: unlearn
method: esd
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
eval_num_prompts: 3
eval_inference_steps: 50
```

For a **grid search**, replace any scalar with a list — `submit_job.py` expands the Cartesian
product into one job per combination:

```yaml
negative_guidance_scale: [0.5, 1.0, 2.0]
learning_rate: [0.0002, 0.0005]
# → 6 runs
```

---

## Seed Management

- **Training** uses a single global `seed` field (model init, batch ordering, dropout, …).
- **Evaluation** uses per-prompt seeds baked into the CSV prompt files. These are committed once and
  never changed, so every experiment is scored on identical `(prompt, seed)` pairs.
- Never use a global seed for evaluation — adding/removing/reordering prompts would silently change
  which seed each prompt gets.
- Exception: `frame_replace_online` generates its training targets online from the *attached* seed of
  each trusted `(prompt, seed)` pair in its train-prompts CSV; the global seed still governs everything
  else.

---

## Utility Scripts

### `submit_job.py` — submit jobs to a cluster

Commits must be pushed before submitting — the cluster runs `git pull` before each job.

```bash
# Single run (cluster is the first positional arg, then the config)
./submit_job.py athena experiments/exp001_esd_fire_lora8/config.yaml
./submit_job.py helios experiments/exp001_esd_fire_lora8/config.yaml

# Override the SLURM script
./submit_job.py helios experiments/exp001_esd_fire_lora8/config.yaml --slurm slurm/other.sh

# Grid search — any list-valued config field triggers Cartesian-product expansion
./submit_job.py athena experiments/exp005_esd_fire_grid/config.yaml
```

For a grid search the script expands list fields, creates `experiments/EXP/grid_{TIMESTAMP}/run_001/`,
`run_002/`, … on the remote with scalar configs, and submits one `sbatch` job per combination. It
warns (but does not block) on uncommitted changes or unpushed commits.

### `pull_results.sh` — download results

Rsyncs `experiments/` outputs and MLflow tracking data from all team members' remote directories.

```bash
./pull_results.sh                    # both clusters (default)
./pull_results.sh --cluster athena   # athena only
./pull_results.sh --cluster helios   # helios only
./pull_results.sh --logs-only        # skip outputs, only SLURM logs
./pull_results.sh --include-weights  # also download .safetensors/.pt checkpoints (excluded by default)
```

### `watch_jobs.sh` — monitor SLURM queue

```bash
./watch_jobs.sh          # combined squeue table for both clusters, refresh every 30 s
./watch_jobs.sh -i 60    # refresh every 60 s
```

### `interactive.sh` — open an interactive SLURM session on a cluster.

### `pull_weights.sh` / `push_weights.sh` — move checkpoint weights between local and cluster.

---

## Research Workflow

1. **Add prompts** — put a CSV (with `prompt` and `seed` columns) or TXT file in `prompts/`.
2. **Create experiment** — make `experiments/expXXX_NAME/config.yaml` (set `slurm_time`, `job_type`,
   `method`).
3. **Commit and push** — the cluster pulls from the repo; uncommitted changes won't be picked up.
4. **Submit** — `./submit_job.py athena experiments/expXXX_NAME/config.yaml`.
5. **Monitor** — `./watch_jobs.sh`.
6. **Download** — `./pull_results.sh`.
7. **Analyze** — read `outputs_*/summary.json` first, then inspect videos and `metrics.jsonl`.
8. **Iterate** — adjust the config or method, create a new experiment folder, repeat. Record what was
   tried in `notes.md`.

---

## Method Write-ups (`docs/`)

- [`unhype.md`](docs/unhype.md) — UnHype: CLIP-guided hypernetworks for dynamic LoRA unlearning.
- [`unhype_video_attempts.md`](docs/unhype_video_attempts.md) — porting UnHype to CogVideoX (exp016–exp031).
- [`frame_replace.md`](docs/frame_replace.md) — supervised v-prediction unlearning toward edited latents.
- [`partial_fire_search.md`](docs/partial_fire_search.md) — autonomous `(prompt, seed)` search for partial-fire clips.
</content>
</invoke>
