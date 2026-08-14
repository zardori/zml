# The experiment registry: notes frontmatter, `INDEX.md`, and the archive

Reference for `tools/experiments_index.py`, `tools/regroup_experiments.py` and
`tools/archive_experiment.py`, and for the `notes.md` frontmatter all three read. Covers what the
fields mean, where an experiment folder lives at each stage of its life, and the two-sided procedure
a move requires (repo *and* cluster).

Related: `.claude/skills/project-docs/SKILL.md` decides *what* goes in a `notes.md` at all; this
document only covers the machine-readable header on top of it.

## Motivation

The project passed 100 experiments. Almost all of them are irrelevant to the run being designed
today, but a flat `experiments/` gives no way to tell which — and the knowledge of "exp044 proved
the mask is inert, don't retry it" lived only in one person's head or buried in a 200-line
`notes.md`. Three people share this repo, so that is not a workable state.

There is a second, independent source of disorder: **three concepts are being unlearned in
parallel, so experiment numbers interleave.** The ImageNet object thread is exp064–exp072, then
jumps to exp099 and exp117–exp119, with 30-odd nudity and face experiments in between. A number
tells you *when* an experiment was created and nothing about what it belongs to, so consecutive
folders in a flat listing are usually unrelated.

Three mechanisms fix this, and they are deliberately separate:

- **The index answers "which of these matter?"** Every experiment declares its own status and a
  one-sentence takeaway; `INDEX.md` renders them into one scannable table, grouped by thread. This
  works regardless of where the folder sits.
- **The thread directory answers "where is this thread?"** Every live experiment sits at
  `experiments/<thread>/expNNN_name/`, so one thread is one directory.
- **The archive answers "which of these are over?"** Retired experiments move to
  `experiments/archive/<thread>/expNNN_name/` — the same grouping, one level down, so retiring is a
  move sideways rather than a re-shuffle.

The index is the load-bearing half. Grouping and archiving are housekeeping on top of it, which is
why the tools refuse to move anything whose metadata is missing.

### Numbering: sequential, never suffixed

Numbers stay globally sequential and one-number-one-experiment. A rebuild of exp066 is exp117, not
`exp066_1`. Suffixes were considered and rejected: they break the `exp_id` uniqueness that
`INDEX.md` and every `docs/` cross-reference depend on, they say nothing about the *next*
experiment in a thread, and they duplicate a relationship the frontmatter already records better —
`status: superseded` on the old one, "superseded by expNNN" in its `takeaway`.

## Frontmatter

Every experiment's `notes.md` starts with:

```yaml
---
status: superseded        # ready | active | done | superseded | abandoned
concept: fire             # fire | nudity | imagenet | none
method: frame_replace     # the pipeline; `foo/precompute` for a dataset build
thread: frame_replace_fire
takeaway: >
  eta=2 erases but motion collapses; superseded by exp057's interpolated target.
---
```

| field | meaning |
|---|---|
| `status` | `ready` — configured and committed, not submitted yet (often waiting on another run). `active` — in flight. `done` — finished, still a live reference or dataset. `superseded` — a later run replaced it. `abandoned` — configured but never run, or a dead end. The two live statuses (`ready`, `active`) block archiving; the two retired ones (`superseded`, `abandoned`) are what `INDEX.md` lists as ready to archive. |
| `concept` | Which concept the run is about, or `none` for infrastructure. Validated against a fixed list. |
| `method` | Mirrors `config.yaml`'s `method`, or the `job_type` when there is none (`eval`, `search`, `benchmark`). Dataset builds are written `frame_replace/precompute` so they are distinguishable from a training run of the same method in the index. |
| `thread` | Which research thread it belongs to; also the archive subfolder. Must be a key of `THREAD_DOCS` in `tools/experiments_index.py`. |
| `takeaway` | One or two sentences: what a person needs to know before designing the next run. Not a summary of the notes — the single thing that would change someone's mind. |

**On `takeaway` and honesty.** A run whose outcome was never written up says exactly that
("Outcome never written up"). Do not reconstruct a result from a guess; per
`.claude/skills/project-docs/SKILL.md` §4, numbers come from `summary.json` / `metrics.jsonl` or
they do not appear. A visible gap is more useful than a plausible fiction, because the next person
will otherwise cite it.

Threads and the write-up that summarises each live in `THREAD_DOCS`. A thread with no archived
members yet (`nudity`, `imagenet`, `shared`) is listed there anyway, so the taxonomy is one list.

## Generating the index

```bash
uv run tools/experiments_index.py            # rewrite experiments/INDEX.md
uv run tools/experiments_index.py --check    # validate only, non-zero exit on a problem
```

Discovery globs `experiments/**/config.yaml` and keeps directories named `expNNN_*`, so grid
`run_NNN/config.yaml` files are skipped and any nesting depth works.

`--check` fails on missing or unparseable frontmatter, an unknown `status`/`concept`/`thread`, a
`thread` that disagrees with the archive folder the experiment sits in, an archived experiment
still marked `active`, and:

> **The invariant: no live config may reference a path under `experiments/archive/`.**

That is the one rule that makes the archive meaningful. A live run reading archived data means the
archive is not dead — it is a hidden dependency, and the next person to prune it breaks a running
experiment. When a retired experiment's *data* is still needed, it stays live (this is why
`exp041_preservation_precompute` is not archived: it is fire-era but its retention anchors are
concept-agnostic, and the nudity and object runs still read them — it lives in the `shared` thread,
`experiments/shared/exp041_preservation_precompute/`).

## Filing a live experiment under its thread

```bash
uv run tools/regroup_experiments.py                  # dry run over every misfiled live experiment
uv run tools/regroup_experiments.py --apply
uv run tools/regroup_experiments.py expNNN --apply   # just these
```

New experiments should be created at `experiments/<thread>/expNNN_name/` directly; this tool exists
for the ones that are not, and for the one-time migration that introduced the layout.
`experiments_index.py --check` fails on any experiment sitting directly in `experiments/`, or whose
folder disagrees with its `thread:` field.

That check lives in `find_misfiled_experiments()`, deliberately *outside* `validate()` and therefore
outside `discover()`. Location is a property of where a folder is, not of whether its notes parse,
and `regroup_experiments.py` has to be able to run against a tree that is currently failing the
check — a check that blocked its own remedy would be useless.

The mechanics (reference rewriting, artifact carry-over, the migration ledger) are shared with
archiving and live in `tools/experiment_moves.py`; the two tools differ only in destination and in
the policy each enforces before moving.

## Archiving

```bash
uv run tools/archive_experiment.py exp0NN [exp0NN ...]          # dry run
uv run tools/archive_experiment.py exp0NN [exp0NN ...] --apply
```

The destination is `experiments/archive/<thread>/<expNNN_name>/`, with the thread taken from the
frontmatter so the folder and the `INDEX.md` group can never disagree. Since live experiments are
already grouped by thread, this is a move *sideways* — `experiments/imagenet/expNNN` to
`experiments/archive/imagenet/expNNN` — and only the live/retired axis changes. Per experiment the
tool:

1. **refuses** if `status` is `active`, if `thread` is missing, or if any config outside the moving
   set references the folder;
2. rewrites every reference to the old path across `*.yaml`, `*.md`, `*.py`, `*.sh` — **before**
   moving anything, because some referencing files (grid `run_NNN/config.yaml`, sibling notes) live
   inside a folder that is about to move;
3. `git mv`s the tracked files and `mv`s the untracked `outputs_*` / `logs_*` / `grid*` alongside;
4. stacks the move onto `tools/migrate_experiments.sh`, the companion every *other* member replays.

The reference rewrite matches the bare substring of the folder's *current* path, not an anchored
repo-relative path. That is deliberate: some eval configs carry an *absolute* cluster path
(`/net/.../zml/experiments/.../expNNN_.../`), and a repo-relative-only match would leave those behind
to fail silently at job time. Where one move's old path is a prefix of another's, the longest is
rewritten first, so the more specific replacement always wins.

**Why the `.gitignore` matters here.** Its patterns were `experiments/*/outputs_*/` — a single `*`
matches exactly one path component, so nesting an experiment one level deeper would have made every
checkpoint and video git-tracked. They are now `experiments/**/…`, which is what let live
experiments move under `experiments/<thread>/` without re-touching them. If you ever add a new
artifact pattern, use `**`.

## The other half: everyone else's trees

Archiving only fixes the tree it ran in. Each member also has a local checkout and their own repo
root on cluster scratch (`cluster.conf`: 3 members × 2 clusters = **6 remote trees**, only your own
writable). In every one of them `git pull` relocates the tracked `config.yaml`/`notes.md` while the
untracked `outputs_*`/`logs_*` stay behind at the old path. One command per member fixes all three
of their trees:

```bash
git pull && tools/migrate_experiments.sh          # local checkout + athena + helios
tools/migrate_experiments.sh --local              # or one target at a time
tools/migrate_experiments.sh --cluster helios
```

It always runs **locally** — nothing has to be typed on a cluster, per `CLAUDE.md` §Working With the
Clusters. Each cluster leg is one `ssh` that does `git pull` and then the migration, with the script
itself piped to `bash -s`, so the cluster checkout needs no copy of it and always executes the
version you have locally. Like `submit_job.py`, it warns (and asks) if you have uncommitted or
unpushed work first: the clusters pull from the remote, so an archive move that exists only in your
working tree would silently not happen there.

Per move it checks that the *tracked* half is already present (`git ls-files` on the destination)
before touching artifacts. After the remote `git pull` that is normally true; on a tree that is
behind — your local checkout, which is deliberately not auto-pulled — the artifacts are left alone
and the run exits non-zero, rather than moving them into a path that a later `git pull` will also
want to write.

The script carries **every** move ever emitted, not just the last round's, and re-checks all of them
on each run. That is what makes it both idempotent and sufficient: nine trees are never migrated in
one sitting, so a member who missed three archiving rounds still catches up with a single run of the
current script.

**What still depends on the remote trees being migrated.** `zml/paths.py` resolves a missing
repo-relative input by rebasing *the same relative string* onto each peer root. If the string is an
archive path and a peer has not migrated, the lookup misses. It degrades to `WARNING: config path …
not found` rather than crashing, but the job then fails on the missing file. Not destructive, but it
means: **do not submit a job that reads archived data until the migration has been run everywhere.**
Nothing in the live set does — only `exp041`, which does not move.

`pull_results.sh` no longer depends on it. It used to rsync an un-migrated peer's `experiments/`
verbatim, which re-created the flat folders locally and re-downloaded every archived artifact on
*every* pull — rsync compares paths, and the copy already sitting in `experiments/archive/` is
invisible to it (mtimes were never the problem; they survive the move intact). It now skips archived
folder names in the bulk transfer and pulls a peer's pre-archive copies straight into their archive
destination, where the size/mtime check makes them a no-op. The mapping is derived from the local
`experiments/archive/` tree, so it needs no bookkeeping of its own.

Per `CLAUDE.md`, cluster commands and job submission are done by the project owners.

## Status

- **2026-08-02 — the archive.** Moved 57 of 72 experiments into five retired threads: `esd_fire`
  (16), `unhype` (16), `frame_replace_fire` (20), `baselines` (2), `misc` (3). The remaining 15 —
  `exp041` plus the live nudity and ImageNet-object work — stayed flat.
- **2026-08-14 — thread directories for live work.** Moved all 62 live experiments into
  `experiments/<thread>/`: `nudity` (37), `imagenet` (13), `face_identity` (11), `shared` (1). Also
  changed experiment discovery from `config.yaml`-keyed to `notes.md`-keyed: `exp087` re-edited an
  existing dataset with a local tool and never submitted a job, so it had no config and had been
  invisible to `INDEX.md`, to validation and to the archive tool the whole time. The registry now
  holds 119 experiments, up from 118 for that reason alone.

## Human-review artifacts belong at the experiment root

`.gitignore` excludes `experiments/**/outputs_*/`, so anything written there travels to a cluster by
rsync (`pull_results.sh`) and **never by git**. A `metadata_human_filtered.json` produced by a local
review pass and saved into `outputs_{timestamp}/` therefore does not exist on any cluster, and a
config pointing at it fails `slurm/check_config_paths.sh` at submission time — which is how exp085
aborted on 2026-08-08.

Put review artifacts (`metadata_human_filtered.json`, `human_rejected.json`, approved-prompt CSVs)
at the **experiment folder root**, where they are tracked, and leave only cluster-generated data
(`latents/`, `videos/`, the unfiltered `metadata.json`) under `outputs_*`. A config then splits the
two naturally:

```yaml
retention_metadata_file: experiments/expNNN_.../metadata_human_filtered.json          # git
retention_latents_dir:   experiments/expNNN_.../outputs_{timestamp}/latents           # rsync
```

exp061 and exp079 both follow this split.
