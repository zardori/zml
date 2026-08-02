# The experiment registry: notes frontmatter, `INDEX.md`, and the archive

Reference for `tools/experiments_index.py` and `tools/archive_experiment.py`, and for the
`notes.md` frontmatter both read. Covers what the fields mean, when an experiment is retired into
`experiments/archive/`, and the two-sided procedure a move requires (repo *and* cluster).

Related: `.claude/skills/project-docs/SKILL.md` decides *what* goes in a `notes.md` at all; this
document only covers the machine-readable header on top of it.

## Motivation

The project passed 70 experiments. Almost all of them are irrelevant to the run being designed
today, but a flat `experiments/` gives no way to tell which — and the knowledge of "exp044 proved
the mask is inert, don't retry it" lived only in one person's head or buried in a 200-line
`notes.md`. Three people share this repo, so that is not a workable state.

Two mechanisms fix it, and they are deliberately separate:

- **The index answers "which of these matter?"** Every experiment declares its own status and a
  one-sentence takeaway; `INDEX.md` renders them into one scannable table. This works regardless of
  where the folder sits.
- **The archive answers "why is `ls` unreadable?"** Retired threads move under
  `experiments/archive/<thread>/`, so the flat listing shows only live work.

The index is the load-bearing half. Archiving is housekeeping on top of it, which is why the tool
refuses to move anything whose metadata is missing.

## Frontmatter

Every experiment's `notes.md` starts with:

```yaml
---
status: superseded        # active | done | superseded | abandoned
concept: fire             # fire | nudity | imagenet | none
method: frame_replace     # the pipeline; `foo/precompute` for a dataset build
thread: frame_replace_fire
takeaway: >
  eta=2 erases but motion collapses; superseded by exp057's interpolated target.
---
```

| field | meaning |
|---|---|
| `status` | `active` — in flight or about to be. `done` — finished, still a live reference or dataset. `superseded` — a later run replaced it. `abandoned` — configured but never run, or a dead end. Only `active` blocks archiving. |
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
experiment. When a retired experiment's *data* is still needed, it stays flat (this is why
`exp041_preservation_precompute` is still at the top level: it is fire-era but its retention
anchors are concept-agnostic, and the nudity and object runs still read them).

## Archiving

```bash
uv run tools/archive_experiment.py exp0NN [exp0NN ...]          # dry run
uv run tools/archive_experiment.py exp0NN [exp0NN ...] --apply
```

The destination is `experiments/archive/<thread>/<expNNN_name>/`, with the thread taken from the
frontmatter so the folder and the `INDEX.md` group can never disagree. Per experiment the tool:

1. **refuses** if `status` is `active`, if `thread` is missing, or if any config outside the moving
   set references the folder;
2. rewrites every reference to the old path across `*.yaml`, `*.md`, `*.py`, `*.sh` — **before**
   moving anything, because some referencing files (grid `run_NNN/config.yaml`, sibling notes) live
   inside a folder that is about to move;
3. `git mv`s the tracked files and `mv`s the untracked `outputs_*` / `logs_*` / `grid*` alongside;
4. writes `tools/migrate_experiments_remote.sh`, the cluster-side companion.

The reference rewrite matches the bare substring `experiments/expNNN_name`, not an anchored
repo-relative path. That is deliberate: some eval configs carry an *absolute* cluster path
(`/net/.../zml/experiments/expNNN_.../`), and a repo-relative-only match would leave those behind to
fail silently at job time.

**Why the `.gitignore` matters here.** Its patterns were `experiments/*/outputs_*/` — a single `*`
matches exactly one path component, so nesting an experiment one level deeper would have made every
checkpoint and video git-tracked. They are now `experiments/**/…`. If you ever add a new artifact
pattern, use `**`.

## The cluster half

Each member runs jobs from their own repo root on cluster scratch (`cluster.conf`:
3 roots × 2 clusters = **6 trees**), and only their own is writable. `git pull` relocates the tracked
`config.yaml`/`notes.md`; the untracked `outputs_*`/`logs_*` stay behind at the old path and must be
moved separately:

```bash
cd $ROOT && git pull && bash tools/migrate_experiments_remote.sh
```

The script is idempotent — already-migrated experiments are skipped — because six roots will not be
done in one sitting.

**Two things depend on all six being migrated.**

- `pull_results.sh` rsyncs every member's `experiments/` into one local tree. A member still on the
  old layout re-creates the old flat folders locally on the next pull.
- `zml/paths.py` resolves a missing repo-relative input by rebasing *the same relative string* onto
  each peer root. If the string is an archive path and a peer has not migrated, the lookup misses.
  It degrades to `WARNING: config path … not found` rather than crashing, but the job then fails on
  the missing file.

Neither is destructive; both mean the same thing: **do not submit a job that reads archived data
until the migration has been run everywhere.** Nothing in the live set does — only `exp041`, which
does not move.

Per `CLAUDE.md`, cluster commands and job submission are done by the project owners.

## Status

The initial migration (2026-08-02) moved 57 of 72 experiments into five threads — `esd_fire` (16),
`unhype` (16), `frame_replace_fire` (20), `baselines` (2), `misc` (3) — leaving 15 flat: `exp041`
plus the live nudity (`exp059`–`exp063`) and ImageNet-object (`exp064`–`exp072`) work.
