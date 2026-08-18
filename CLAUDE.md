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
├── experiments/                 # grouped by THREAD, never flat — see "Experiment Layout" below
│   ├── INDEX.md                 # generated registry — read this first, not `ls`
│   ├── <thread>/                # nudity | imagenet | face_identity | shared
│   │   ├── exp062_frame_replace_nudity_eta2/  # single-run experiment
│   │   │   ├── config.yaml      # hyperparameters, dataset info, etc.
│   │   │   ├── logs_{TIMESTAMP}/     # logs from the SLURM job (stdout, stderr)
│   │   │   ├── outputs_{TIMESTAMP}/  # generated videos, evaluation results, etc.
│   │   │   │   ├── metrics.jsonl  # metrics - one object per flushed train window and per eval
│   │   │   │   ├── summary.json   # metrics - overwritten each update
│   │   │   │   ├── run_info.json  # cluster, node, elapsed, outcome — written even on timeout
│   │   │   │   └── other outputs... # e.g. generated videos, eval results, etc.
│   │   │   └── notes.md         # registry frontmatter + what was tried, what happened
│   │   └── exp0NN_some_grid/    # grid-search experiment (alternative pattern)
│   │       ├── config.yaml      # base config with list values for swept params
│   │       └── grid_{TIMESTAMP}/  # has one subfolder per hyperparameter combination
│   │           ├── run_001/
│   │           │   ├── config.yaml  # concrete config for this run (all values scalar)
│   │           │   ├── logs/    # SLURM stdout/stderr logs
│   │           │   └── outputs/ # checkpoints and per-step eval results
│   │           ├── run_002/
│   │           └── ...
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
│   └── check_config_paths.sh    # locates a config's inputs on the cluster (pre-submit check)
├── prompts/                     # prompt sets; per-concept subdirs, see "Prompt Layout" below
│   ├── imagenet_objects.csv     # the eval set for a concept stays at the top level
│   ├── imagenet_objects/        # everything derived from it, grouped
│   │   ├── chain_saw.csv        # per-class eval control set (tools/split_imagenet_prompts.py)
│   │   ├── others_chain_saw.csv # its preservation counterpart
│   │   └── split/               # A/B/C dataset-construction triples
│   │       ├── chain_saw.csv
│   │       └── church_closeup.csv
│   └── face_identities/         # same shape: control sets + split/
├── tools/                       # utility scripts
├── report/                      # generated, gitignored — LaTeX tables and the weekly decks
│   └── weekly/<ISO week>/       # data.json (collected + curated), media/, index.html
└── docs/                        # method write-ups & design notes
    ├── frame_replace.md         # main method: supervised SFT toward a concept-removed edit
    ├── split_prompt.md          # manufacturing partial-concept clips (A/B/C triples)
    ├── comparison_targets.md    # which concepts other T2V unlearning papers erase, and our order
    ├── imagenet_objects.md      # the ESR/PSR object-erasure protocol and our two-class pilot
    ├── face_identity.md         # celebrity ID-similarity protocol, 2-identity pilot (Obama, Merkel)
    ├── unhype.md                # UnHype: CLIP-guided hypernetwork unlearning (the paper method)
    ├── unhype_video_attempts.md # porting UnHype to CogVideoX (exp016-exp031)
    ├── partial_fire_search.md   # autonomous search for partial-fire (prompt, seed) pairs
    ├── weekly_report.md         # how the weekly mentor deck is collected and curated
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
7. **Run experiments** (`submit_job.py`): Submit jobs to a cluster. Pass the cluster name (`athena` or `helios`) as the first positional argument, then the config path. Optionally override the SLURM script with `--slurm`. The script SSHes into the cluster, runs `git pull`, verifies that every repo-relative data path in the config exists there — in your repo or in a peer's, the same search `zml/paths.py` does at runtime (`slurm/check_config_paths.sh`) — and calls `sbatch`. An input that is missing here but present on the **other** cluster is offered for copy and, once confirmed, transferred before the job is submitted (`zml/cluster_sync.py`; `--no-fetch-missing` disables it). A path nobody has anywhere aborts the submission, `--skip-path-check` overrides. If the config has any list-valued fields a grid search is performed automatically — one job per combination. Cluster connection details are read from `cluster.conf` (copy from `cluster.conf.example`). Ensure all necessary content is committed before submitting. `--yes` answers every confirmation, for non-interactive callers. (Claude must not run `submit_job.py` or `sbatch` itself. Jobs are submitted either by a project owner by hand, or by the autonomous research agent in `~/projects/research_agent` — and even there Claude only authors the config, while the agent's orchestrator validates it, enforces concurrency and daily caps, commits, pushes, and submits.)
   Example: `./submit_job.py athena experiments/expXXX_NAME/config.yaml`
   Every config must set two infra fields: `slurm_time` (the sbatch `--time`, e.g. `slurm_time: "0-4:00:00"`; there is no default, so a missing value is rejected) and optionally `job_type` (`unlearn` (default) | `eval` | `precompute`), which selects the entrypoint via the `JOB_TYPE` env var.
8. **Collect results** (`pull_results.sh`): Download experiment outputs and MLflow tracking data from clusters via rsync. Defaults to pulling from both clusters. Use `--cluster athena` or `--cluster helios` to target one. Narrow what is pulled with `--experiment PATH` (one dir), `--thread imagenet` (one thread, plus its `archive/` counterpart) or `--range 67-70` (experiment numbers, inclusive, single number allowed) — these can be combined and skip the MLflow sync. Pass `--logs-only` to skip outputs, or `--include-weights` to include `.safetensors`/`.pt` checkpoints (excluded by default). Reads connection details from `cluster.conf`. Artifacts a member still keeps at a pre-archive path (they have not run `tools/migrate_experiments.sh`) are pulled into `experiments/archive/` instead of re-creating the flat folder locally.
9. **Evaluate, analyze, iterate**: Look on the results, optionally run additional evaluation scripts, analyze the results, and iterate on the unlearning method or hyperparameters.

### Experiment Layout: grouped by thread

**Every experiment lives under its thread — `experiments/<thread>/expNNN_name/` while live, and
`experiments/archive/<thread>/expNNN_name/` once retired.** Threads are `nudity`, `imagenet`,
`face_identity` and `shared`; the thread comes from the `thread:` field in the experiment's
`notes.md`, so the folder and the registry can never disagree, and `tools/experiments_index.py
--check` fails if they do.

The reason is that three concepts are being unlearned in parallel, so experiment *numbers*
interleave and say nothing about what an experiment belongs to: the object thread is exp064–exp072,
then jumps to exp099 and exp117–exp119, with 30-odd nudity and face experiments in between. **Never
create an experiment directly in `experiments/`.** If one ends up there, `tools/regroup_experiments.py`
files it (dry run by default, `--apply` to move) — it rewrites every reference and stacks the move
onto `tools/migrate_experiments.sh` the same way archiving does.

Numbering stays globally sequential and one-number-one-experiment. Do **not** use suffixes like
`exp066_1` to tie a rebuild to what it rebuilds — that relationship belongs in the `status` and
`takeaway` frontmatter (`superseded` + "superseded by expNNN"), which is what `INDEX.md` renders.

### Experiment Registry

There are 100+ experiments and most are no longer relevant to a new one, so **start from
`experiments/INDEX.md`, not from `ls`**. It is generated by `tools/experiments_index.py` from a YAML
frontmatter block (`status`, `concept`, `method`, `thread`, `takeaway`) at the top of every
experiment's `notes.md` — a new experiment must have one, and the notes are what make a folder an
experiment (`config.yaml` is optional; some experiments are tool-driven and never submit a job).
`status` is one of `ready` (configured, not submitted yet) | `active` (in flight) | `done` (finished,
still a live reference or dataset) | `superseded` | `abandoned`; the first two are live and block
archiving, the last two are what `INDEX.md` flags as retirable. `tools/archive_experiment.py` retires
an experiment sideways into `experiments/archive/<thread>/` and enforces the one rule that keeps this
honest: **no live config may reference a path under `experiments/archive/`.** It also stacks the move
onto `tools/migrate_experiments.sh`, which every other member runs **once, locally** to bring their
checkout and both cluster repo roots to the new layout. Field reference and the archive procedure:
**`docs/experiment_registry.md`**.

### Prompt Layout: per-concept subdirectories

`prompts/` is shared by every concept and had grown to 60 flat files. The rule now:

- A concept's **eval set** stays at the top level (`prompts/imagenet_objects.csv`,
  `prompts/face_cogvideox.csv`) — it is the published, never-edited artifact.
- Everything **derived per target** goes in that concept's directory: control sets at
  `prompts/<concept>/<target>.csv`, and **A/B/C dataset-construction triples under
  `prompts/<concept>/split/<target>.csv`**. The `split/` directory is what distinguishes
  `prompts/imagenet_objects/chain_saw.csv` (20 eval prompts) from
  `prompts/imagenet_objects/split/chain_saw.csv` (30 A/B/C triples) — same target, different job.

Applied to `imagenet_objects/` and `face_identities/`, which have a per-target axis. The nudity and
fire sets are still flat: their split CSVs are generation-numbered (`split_nudity_gen4_part2.csv`)
rather than per-target, so there is no clean name to move them to. Fold them in when someone gives
them a per-target structure, not before.

### Utility Scripts
- `tools/sync_cluster_inputs.py`: Copies inputs that exist on one cluster into your repo on the
  other, at the same repo-relative path (`tools/sync_cluster_inputs.py helios --config <config>`, or
  bare paths). `submit_job.py` does this for the config it is submitting; use the script directly to
  stage data ahead of time, e.g. the sources of a `merge_dataset.sh` build. Transfers go
  cluster-to-cluster when the source login node can ssh to the target (agent forwarding), otherwise
  they are streamed through this machine.
- `watch_jobs.sh`: Polls `squeue` on both athena and helios every 30 s and displays a combined job table. Reads `cluster.conf` for hostnames.
- `interactive.sh`: Opens an interactive SLURM session on the cluster.
- `tools/weekly_report.py`: Builds the weekly mentor deck. `collect` gathers the week's runs,
  metrics, `notes.md` diffs and video frames into `report/weekly/<week>/data.json` (local files and
  git only — anything not pulled is listed as a gap, never dropped); a person then writes the
  headline and per-experiment commentary into that file and `render` emits `index.html`. Re-running
  `collect` merges fresh facts under the writing rather than overwriting it. Mechanism and knobs:
  **`docs/weekly_report.md`**; what to include and how to write it: the `weekly-report` skill.

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

**DOVER reads `0.0` on helios and that is "not measured", never a quality score.** `pyproject.toml`
gates `dover`/`decord` on x86_64, and helios' compute nodes are aarch64 (GH200), so the import
fails there. It is the only metric we have that measures technical quality (the softness human
review keeps catching and clip_score/colorfulness/motion keep missing), so don't read those zeros
as data. Scoring is post-hoc over saved `.mp4`s and needs no GPU job: pull the eval videos and run
`tools/score_dover.py <outputs_dir>` on any x86_64 machine (athena or locally); it rewrites only
the DOVER fields of each `eval_step_*/metrics.json`.

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

**Nothing inside precompute filters for quality, so always screen the build before training it.**
`tools/screen_split_dataset.py` scores each row on a within-clip differential (does the half
conditioned on prompt A read more concept than the half conditioned on B?) and separates the two
failures a new concept actually hits: the model never rendered the concept, or it rendered it in both
halves. Set `emit_whole_clip_target: true` on a first build so the A-side confidences can tell those
apart in one job. Both concepts transferred so far lost most of their first dataset to prompts the
base model does not render the concept under — write prompt A against the eval prompts, not from
scratch (`docs/split_prompt.md` §3.1–3.2).

### Current Goals

1. **Nudity** — finish split-prompt → frame_replace (exp062 pilot: does erasure transfer, and is the
   positional shortcut gone?). Then scale the dataset and add a nudity `related`/preservation set.
2. **Second concept: ImageNet objects** — **the pilot's erasure runs have landed, and they split.**
   exp069 erases chain saw semantically (top-1 0.506 → 0.00 from step 200, scene intact, on eval
   prompts with no object-free half); exp070 never erases church in the identical regime. So the
   method transfers, and how far is concept-dependent. Two follow-ups are the live work: **exp126**
   (`erase_esd_eta` sweep) attacks exp069's one defect — the concept clips freeze, motion 0.010 vs
   base 0.564, concept-conditional unlike nudity's global collapse — and **exp128** rebuilds church on
   repaired data. **exp071** reports the real 200-prompt ESR/PSR; exp072 is deliberately held. On the
   data side exp120 rejected `concept_guidance_scale` as a yield lever but confirmed the suppression
   mechanism, which **exp127** (`split_mode: trajectory`) now targets; exp121/exp122 confirmed
   re-seeding reproduces yield. ESR/PSR is reported under two ranking conventions (1000-way and
   restricted to the ten classes) because the papers do not state theirs — and exp065 shows it
   matters: NegPrompt reads ESR-1 70.9 one way and 17.2 the other. Write-up:
   **`docs/imagenet_objects.md`**.
3. **Third concept: face/celebrity identity** — ID-similarity protocol implemented (ArcFace + YuNet,
   `mode: face`), 2-identity pilot (Obama, Merkel) staged as exp090–exp098. Nothing submitted yet;
   exp090 (base-model reference, all 5 identities) is the hard gate everything else waits on.
   Write-up: **`docs/face_identity.md`**.
4. Keep improving the core method (retention/collateral, eta regime, localization).

Which concepts other T2V unlearning papers erase, with what detectors and prompt sets, and why we
pick them in this order: **`docs/comparison_targets.md`**.

**Comparability with T2VUnlearning (arXiv:2505.17550), our closest comparison: `docs/comparability_t2vunlearning.md`.**
Read it before reporting any nudity number. Two things it settles: `prompts/cogvideox_nudity.csv` *is*
their released "Gen" eval set (same 100 prompts and seeds — every historical run is already on their
prompts), and their "Nudity Rate" is **per-frame**, not per-video. Detectors now emit both
(`nudity_frame_rate` alongside `nudity_detection_rate`); they are not interchangeable, and
`tools/score_nudity_frame_rate.py` backfills the frame rate onto already-generated clips with no GPU.

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
