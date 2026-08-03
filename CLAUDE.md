# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Keep this file concise** — it is loaded into every session. It holds project-wide facts and
conventions only; detailed write-ups (methods, design rationale, status of a research thread) belong
in `docs/` and are linked from here. See the `project-docs` skill for placement rules and the
structure a `docs/` page should follow.

## Project Overview

The goal of this research project is to propose a method for effective concept unlearning from text to video models. The project uses CogVideoX-5b, a video diffusion transformer, as the primary model for experiments. The real challenge is to erase the target concept without harming the model's performance.

The main method is **frame_replace** (`docs/frame_replace.md`): supervised v-prediction fine-tuning toward a *concept-removed edit of the model's own output*, built by swapping the concept-containing latent frames for concept-free donor frames from the same clip. It was developed and validated on **fire**. We are now **transferring it to other concepts** — starting with **nudity** — so our numbers can be compared against published T2V unlearning papers.

The project uses python 3.12 and uv for python packages. Experiments are run on PLGrid HPC infrastructure athena cluster (A100 GPUs with 40GB VRAM) and helios cluster (GH200 chips with 96GB VRAM) via SLURM.

## Desired Repository Structure
```
zml/
├── zml/                         # shared "library" code
│   ├── unlearn/                 # scripts for unlearning
│   ├── precompute/              # scripts for precomputing latents used in unlearning
│   ├── benchmarks/              # concept detectors & reports (e.g. NudeNet wrapper)
│   ├── search/                  # prompt/hyperparameter search helpers
│   └── eval/                    # scripts and utils for evaluation
├── experiments/                 # one folder per experiment run; only LIVE work sits flat here
│   ├── INDEX.md                 # generated registry — read this first, not `ls`
│   ├── exp062_frame_replace_nudity_eta2/  # single-run experiment
│   │   ├── config.yaml          # hyperparameters, dataset info, etc.
│   │   ├── logs_{TIMESTAMP}/     # logs from the SLURM job (stdout, stderr)
│   │   ├── outputs_{TIMESTAMP}/  # generated videos, evaluation results, etc.
│   │   │   ├── metrics.jsonl    # metrics - one object per flushed train window and per eval
│   │   │   ├── summary.json     # metrics - overwritten each update
│   │   │   └── other outputs... # e.g. generated videos, eval results, etc.
│   │   └── notes.md             # registry frontmatter + what was tried, what happened
│   ├── exp0NN_some_grid/         # grid-search experiment (alternative pattern)
│   │   ├── config.yaml          # base config with list values for swept params
│   │   └── grid_{TIMESTAMP}/    # has one subfolder per hyperparameter combination
│   │       ├── run_001/
│   │       │   ├── config.yaml  # concrete config for this run (all values scalar)
│   │       │   ├── logs/        # SLURM stdout/stderr logs
│   │       │   └── outputs/     # checkpoints and per-step eval results
│   │       ├── run_002/
│   │       └── ...
│   └── archive/<thread>/         # retired threads (esd_fire, unhype, frame_replace_fire, ...)
│       └── exp0NN_.../           # same folder shape; nothing live may reference these
├── scripts/                     # thin generic entrypoints to the experiments (all should call zml/)
│   ├── unlearn.py               
│   ├── precompute.py            
│   └── eval.py                  
├── slurm/                       # one generic SLURM script per cluster
│   ├── athena.sh                # dispatches on JOB_TYPE (unlearn|eval|precompute)
│   ├── helios.sh                
│   ├── peer_roots.sh            # builds ZML_PEER_ROOTS: every member's repo root on this cluster
│   └── check_config_paths.sh    # pre-submit check that config inputs exist on the cluster
├── prompts/                     # prompts used in experiments
├── tools/                       # utility scripts
└── docs/                        # method write-ups & design notes
    ├── frame_replace.md         # main method: supervised SFT toward a concept-removed edit
    ├── split_prompt.md          # manufacturing partial-concept clips (A/B/C triples)
    ├── comparison_targets.md    # which concepts other T2V unlearning papers erase, and our order
    ├── imagenet_objects.md      # the ESR/PSR object-erasure protocol and our two-class pilot
    ├── unhype.md                # UnHype: CLIP-guided hypernetwork unlearning (the paper method)
    ├── unhype_video_attempts.md # porting UnHype to CogVideoX (exp016-exp031)
    ├── partial_fire_search.md   # autonomous search for partial-fire (prompt, seed) pairs
    └── experiment_registry.md   # notes.md frontmatter, INDEX.md, and the archive policy
```

### Compute Resources
Cluster access is rich, but not unlimited, so experiments should be designed to be research efficient. For example, we should avoid running grid search before the method used is proved to be effective. Short experiments are often sufficient to debug the method and refine the research direction.

### Working With the Clusters
**Minimize — ideally eliminate — steps that have to be done by hand on a cluster.** Every manual
`ssh` + edit + rerun is slow, has to be repeated in 6 remote repo roots (3 members × 2 clusters), and
is done inconsistently or forgotten. This is why `submit_job.py`, `pull_results.sh`,
`watch_jobs.sh` and `tools/migrate_experiments.sh` exist: each is run **locally** and does its
cluster work over `ssh` itself. When adding a workflow that touches a cluster, write it the same
way — a local entrypoint that `git pull`s on the cluster, acts on every target it owns, is
idempotent, and warns about uncommitted/unpushed local changes (the cluster only ever sees what was
pushed). Prefer extending one of the existing scripts over documenting a manual procedure.

### Desired Research Workflow

1. **Prepare Unlearning methods** (`zml/unlearn`): Add code for different unlearning methods there.
2. **Prepare Evaluation methods** (`zml/eval`): Prepare code for different evaluation methods there. Some functions from here should be used during unlearning for live evaluation.
3. **Prepare Precompute methods** (optional) (`zml/precompute`): If we can speed up unlearning, by precomputing some latents or other intermediate results, we add code here. This is also where frame_replace training targets are built (see "Partial-Concept Data Construction").
4. **Prepare thin generic entrypoints** (`scripts/`): These should be thin wrappers that parses arguments call the code in `zml/`.
5. **Prepare SLURM templates** (`slurm/`): There is one generic script per cluster (`slurm/athena.sh`, `slurm/helios.sh`). Each holds only that cluster's account/partition/repo-dir and dispatches on the `JOB_TYPE` env var to the right thin entrypoint. `submit_job.py` supplies the job name, time, and log paths as `sbatch` flags, so they are not baked into the scripts.
6. **Prepare experiments** (`experiments/`): For each experiment, create a new folder with a config file containing all hyperparameters, dataset info, etc. The experiment config should be in YAML format. Generate new prompt sets if needed. Also create `notes.md` with the registry frontmatter block (`status`/`concept`/`method`/`thread`/`takeaway`) — see "Experiment Registry" below.
7. **Run experiments** (`submit_job.py`): Submit jobs to a cluster. Pass the cluster name (`athena` or `helios`) as the first positional argument, then the config path. Optionally override the SLURM script with `--slurm`. The script SSHes into the cluster, runs `git pull`, verifies that every repo-relative data path in the config exists there — in your repo or in a peer's, the same search `zml/paths.py` does at runtime (`slurm/check_config_paths.sh`; a missing path aborts the submission, `--skip-path-check` overrides) — and calls `sbatch`. If the config has any list-valued fields a grid search is performed automatically — one job per combination. Cluster connection details are read from `cluster.conf` (copy from `cluster.conf.example`). Ensure all necessary content is committed before submitting. (Claude should not submit any jobs by itself — project owners do it manually.)
   Example: `./submit_job.py athena experiments/expXXX_NAME/config.yaml`
   Every config must set two infra fields: `slurm_time` (the sbatch `--time`, e.g. `slurm_time: "0-4:00:00"`; there is no default, so a missing value is rejected) and optionally `job_type` (`unlearn` (default) | `eval` | `precompute`), which selects the entrypoint via the `JOB_TYPE` env var.
8. **Collect results** (`pull_results.sh`): Download experiment outputs and MLflow tracking data from clusters via rsync. Defaults to pulling from both clusters. Use `--cluster athena` or `--cluster helios` to target one. Pass `--logs-only` to skip outputs, or `--include-weights` to include `.safetensors`/`.pt` checkpoints (excluded by default). Reads connection details from `cluster.conf`. Artifacts a member still keeps at a pre-archive path (they have not run `tools/migrate_experiments.sh`) are pulled into `experiments/archive/` instead of re-creating the flat folder locally.
9. **Evaluate, analyze, iterate**: Look on the results, optionally run additional evaluation scripts, analyze the results, and iterate on the unlearning method or hyperparameters.

### Experiment Registry

There are 70+ experiments and most are no longer relevant to a new one, so **start from
`experiments/INDEX.md`, not from `ls`**. It is generated by `tools/experiments_index.py` from a YAML
frontmatter block (`status`, `concept`, `method`, `thread`, `takeaway`) at the top of every
experiment's `notes.md` — a new experiment must have one. `status` is one of `ready` (configured,
not submitted yet) | `active` (in flight) | `done` (finished, still a live reference or dataset) |
`superseded` | `abandoned`; the first two are live and block archiving, the last two are what
`INDEX.md` flags as retirable. Retired threads live under `experiments/archive/<thread>/`;
`tools/archive_experiment.py` moves them there and enforces the one rule that keeps this honest:
**no live config may reference a path under `experiments/archive/`.** It also stacks the move onto
`tools/migrate_experiments.sh`, which every other member runs **once, locally** to bring their
checkout and both cluster repo roots to the new layout. Field reference and the archive procedure:
**`docs/experiment_registry.md`**.

### Utility Scripts
- `watch_jobs.sh`: Polls `squeue` on both athena and helios every 30 s and displays a combined job table. Reads `cluster.conf` for hostnames.
- `interactive.sh`: Opens an interactive SLURM session on the cluster.

### Metrics Logging

Runs log to wandb and mlflow (human-facing) and, in parallel, to two plain files written
into the run's `output_dir` by `zml/unlearn/metrics_log.py` (`MetricsRecorder`). These are
synced by `pull_results.sh` and meant to be read directly (by a person or an agent) to judge
a run without the wandb UI:

- `metrics.jsonl` — append-only; one object per flushed train window and per eval. Full
  (downsampled) history, crash-robust, machine-parseable.
- `summary.json` — overwritten each update; the at-a-glance artifact. Holds the config echo,
  per-metric train trends (`first/recent/last/min/max`), compact per-checkpoint eval scores,
  and a derived `health` block with flags + plain-language notes (e.g. "loss_remove pinned
  ~0", "predicted_step << target_step", "weak prompt conditioning").

Train metrics are buffered and flushed as window aggregates every `metrics_log_interval`
steps (config field, default 50) to keep the files small. When analyzing a run, prefer reading
`summary.json` first. Currently wired into `zml/unlearn/unhype.py`; other unlearning scripts
can adopt the recorder the same way.

### Partial-Concept Data Construction

frame_replace needs **partial-concept clips** (concept in some frames, concept-free donor frames in
the same clip). Fire is naturally partial; nudity and most other concepts are not — which is what
blocked the transfer. **split-prompt** (`zml/precompute/split_prompt_precompute.py` +
`frame_replace_split_precompute.py`) manufactures the partiality from an A/B/C prompt triple
(concept / safe / neutral): the early denoising steps condition one temporal region on A and the
other on B, the tail conditions everything on C to heal the seam. Full write-up, de-biasing knobs and
status: **`docs/split_prompt.md`**.

A new concept costs exactly two things: an A/B/C prompt CSV, and a per-frame detector registered in
`zml/benchmarks/registry.py` (`build_detector` is the one place a config's `concept` /
`concept_target` string is mapped to a detector — never branch on the concept anywhere else).

### Current Goals

1. **Nudity** — finish split-prompt → frame_replace (exp062 pilot: does erasure transfer, and is the
   positional shortcut gone?). Then scale the dataset and add a nudity `related`/preservation set.
2. **Second concept: ImageNet objects** — protocol implemented (per-frame ResNet-50, ESR/PSR via
   `mode: imagenet`), two-class pilot in exp064–exp072, chain saw and church. exp064 (base-model
   reference) is **done and passed the gate**; datasets exp066–exp068 are next. ESR/PSR is reported
   under two ranking conventions (1000-way and restricted to the ten classes) because the papers do
   not state theirs. Write-up: **`docs/imagenet_objects.md`**.
3. Keep improving the core method (retention/collateral, eta regime, localization).

Which concepts other T2V unlearning papers erase, with what detectors and prompt sets, and why we
pick them in this order: **`docs/comparison_targets.md`**.

### Seed Management Policy

- **Training**: use a single global `seed` field in `config.yaml`. It controls process-level randomness (model initialization, batch ordering, dropout, etc.).
- **Evaluation**: use per-prompt seeds baked into the CSV prompt files. Commit these seeds once and never change them, so every experiment is evaluated on identical `(prompt, seed)` pairs and results are comparable across runs.
- Never use a global seed for evaluation — adding, removing, or reordering prompts would silently change which seed each prompt gets.
- **Dataset construction** (`split_prompt` / `frame_replace_split` precompute): one seed per CSV row, shared by A, B, C and the combined clip; `split_jitter` derives from it, so rebuilds are reproducible. Commit these CSVs.
- **Exception — `frame_replace_online`**: this method generates its training targets online from the trusted `(prompt, seed)` pairs in its train-prompts CSV, using each pair's *attached* seed for generation (not the global seed). Those pairs are pre-checked to render partial fire, so a fixed seed is what makes them trustworthy. The global seed still governs everything else in the run (which pair is drawn, dropout, etc.).

### Additional Notes
- You should write clean and maintainable python code and use type hints.
- You should try to extract numeric constants to constants put at the top of the scripts, especially for values that need to be tuned
- You should avoid using too long functions or loops. If some logic is easily separable, extract it to a smaller function or class. However, be sane and don't force breaking code into functions or classes where it is not natural.
- It's usually better to pass and return dataclasses instead of dictionaries
- Inside unlearning scripts we should periodically run evaluation to check the progress.
- Our local computers don't have enough GPU memory (we have no more than 6 GB) to run the experiments, so we need to use the cluster.
- There are three people working on this project.
