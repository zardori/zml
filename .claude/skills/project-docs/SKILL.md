---
name: project-docs
description: How to record project knowledge in this repo — where a piece of information belongs (CLAUDE.md vs docs/ vs experiments/*/notes.md vs nowhere), how to keep CLAUDE.md short by exporting detail to docs/, and the structure a docs/ write-up should follow. Use whenever writing to or updating CLAUDE.md, adding a docs/ write-up, documenting a new method or experiment result, or when asked to "document" / "write down" / "update the docs" for something.
---

# Project documentation placement

The rule this whole skill serves: **`CLAUDE.md` is an index, not a knowledge base.** It is loaded into
every session's context, so everything in it is paid for in every session, whether relevant or not.
Detail goes to `docs/`, which is read on demand.

## 1. Where does this information belong?

| Where | What goes there | Test |
|---|---|---|
| `CLAUDE.md` | Project-wide facts and conventions that shape *how work is done* in almost any session: repo layout, workflow, how to submit jobs, seed policy, coding standards, current goals. | "Would a session about an unrelated part of the project still need this?" |
| `docs/*.md` | Method write-ups, design rationale, algorithm mechanics, knob semantics, status of a research thread, comparisons with external work. | "Is this the *explanation* of something CLAUDE.md only names?" |
| `experiments/<exp>/notes.md` | Anything true of one run: goal, setup, what to look at, outcome, next step. | "Does this stop being true when the next experiment runs?" |
| Nowhere | Facts derivable from the code, config files, or git history. Restating them creates a second source of truth that goes stale. | "Could I get this by reading the file it describes?" |

When something is genuinely both (a convention *and* a mechanism), split it: the convention in
`CLAUDE.md`, the mechanism in `docs/`, linked.

## 2. Keeping CLAUDE.md short

**Budget:** a `CLAUDE.md` section is 1–2 short paragraphs or a handful of bullets. If a section grows
past roughly 15 lines, it is a `docs/` page that hasn't been extracted yet.

**Export pattern.** When a section outgrows the budget, move the body to `docs/<topic>.md` and leave
behind: what the thing is, why it exists, and a bolded path to the full write-up. Example of the
shape to aim for:

> frame_replace needs **partial-concept clips** (concept in some frames, concept-free donor frames in
> the same clip). Fire is naturally partial; nudity and most other concepts are not — which is what
> blocked the transfer. **split-prompt** (`zml/precompute/split_prompt_precompute.py`) manufactures
> the partiality from an A/B/C prompt triple. Full write-up, knobs and status: **`docs/split_prompt.md`**.

The reader must be able to tell from the stub alone whether they need to open the doc. A bare
"see `docs/x.md`" fails that test.

**Prefer updating over appending.** New information usually means an existing line is now wrong. The
common failure is leaving a stale claim ("previously we tried X, now we focus on Y") and adding the
new state next to it. Read the surrounding section before writing and rewrite what the change
invalidates — including the repo-structure diagram and `Current Goals`.

**Also update:** the structure diagram when a directory or notable doc appears, and any cross-links
in sibling `docs/` pages.

## 3. Structure of a `docs/` write-up

The shape `frame_replace.md` and `split_prompt.md` converged on — follow it unless the topic clearly
wants something else:

1. **Header** — one-paragraph statement of what the document covers and which source files it is the
   reference for (`zml/...`), plus links to related docs.
2. **Motivation / the problem** — what fails without this, concretely. State the *why*, not only the
   *what*; this is the part that cannot be recovered from the code.
3. **Mechanism** — how it works, in the order a reader would reconstruct it. Include the
   non-obvious design decisions and the reason behind them (e.g. "only one scheduler step is taken,
   because two would desynchronize the solver's multistep state").
4. **Knobs** — each tunable, what it does, and its failure mode *at both extremes*. A knob listed
   without its failure modes is not documented.
5. **Status** — what has been tried and what happened, as a small table keyed by experiment id, with
   the decision each experiment feeds. Keep per-run detail in the experiment's `notes.md` and only
   summarize here.
6. **Generalization / next steps** — what it would cost to apply this elsewhere.

Write for a collaborator who knows the project but not this thread. Three people work on this repo,
so nothing may rely on what one person happens to remember.

## 4. Accuracy rules

- **Verify before writing.** Paths, script names, config fields and experiment ids must be checked
  against the repo, not recalled. Numbers (keep rates, step counts, scores) come from the run's
  `metadata.json` / `summary.json` / `skipped.json`, not from an estimate.
- **Cite where a number came from** so it can be re-derived later.
- **Flag gaps rather than papering over them.** If a documented dataset depends on a file that is not
  committed, or a described eval set does not exist yet, say so in the doc.
- Do not document aspirational behavior as if implemented; mark it as planned.

## 5. Checklist before finishing

- [ ] Every added `CLAUDE.md` section is within budget, or has an extracted `docs/` page.
- [ ] Each stub says what the thing is and why, then points to the doc.
- [ ] Stale claims contradicted by the change were rewritten, not left in place.
- [ ] Structure diagram and cross-links updated if directories or docs were added.
- [ ] Every path, identifier and number was verified against the repo.
