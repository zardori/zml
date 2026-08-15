"""The reporting window, and what git says happened inside it.

Two halves of "last week" that have to be derived differently. Runs are found on disk by their
timestamps (``zml/report/artifacts.py``); the *interpretation* of those runs is only ever written into
``experiments/**/notes.md``, which is tracked, so the week's authored narrative has to come out of git.

The one trap: on 2026-08-15 commit ``f048777`` moved every experiment folder into
``experiments/<thread>/``. Any diff keyed on paths reads that as 122 deletions and 122 creations, and
a window spanning it reports the entire project as new. So every comparison here is keyed on the
**experiment id** parsed out of the path, and the two sides are matched by id before their contents
are compared — renames become invisible, which is exactly what they are.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from zml.report import REPO_ROOT

EXP_ID_RE = re.compile(r"exp(\d{2,4})")
# The thread directory is optional: before f048777 experiments sat directly in experiments/, and the
# "before" side of a window spanning that commit is full of such paths.
NOTES_PATH_RE = re.compile(r"^experiments/(?:.*/)?(exp\d{3}_[^/]+)/notes\.md$")
# Regrouping rewrote every cross-reference inside the notes — both markdown link targets and the
# `./submit_job.py athena experiments/<thread>/expNNN_.../config.yaml` lines quoted in the prose.
# Neither is a result, so both are normalised away before two revisions are compared.
LINK_TARGET_RE = re.compile(r"\]\([^)]*\)")
EXPERIMENT_PATH_RE = re.compile(r"experiments/(?:[\w.\-]+/)*(exp\d{3}_)")
SINCE_RE = re.compile(r"^(\d+)([dw])$", re.IGNORECASE)

DAYS_PER = {"d": 1, "w": 7}
WEEK_DAYS = 7

# Findings in this repo are committed as unprefixed, experiment-keyed subjects ("exp110 is the new
# best checkpoint; ..."), while plumbing uses conventional prefixes. The distinction is worth keeping
# in the deck: the first kind is a result, the second is context.
CHORE_PREFIX_RE = re.compile(r"^(feat|fix|docs?|chore|refactor|test|style|perf)(\(.+?\))?!?:", re.I)


def normalize_exp_id(raw: str) -> str:
    """``exp75`` / ``exp075`` / ``exp0075`` -> ``exp075``. Commit subjects use all three spellings."""
    return f"exp{int(raw):03d}"


def exp_ids_in(text: str) -> list[str]:
    """Every experiment id mentioned in a string, de-duplicated, in order of appearance."""
    seen: dict[str, None] = {}
    for raw in EXP_ID_RE.findall(text):
        seen.setdefault(normalize_exp_id(raw), None)
    return list(seen)


@dataclass(frozen=True)
class Window:
    """The half-open reporting interval ``[start, end)`` and the directory label it writes under."""

    label: str
    start: datetime
    end: datetime

    def contains(self, moment: datetime | None) -> bool:
        return moment is not None and self.start <= moment < self.end

    def overlaps(self, start: datetime | None, end: datetime | None) -> bool:
        """True if a run that began at ``start`` and finished at ``end`` touched this window.

        A 3-day precompute job started before the window and finishing inside it is this week's news,
        so containment of either endpoint is not enough.
        """
        if start is None:
            return self.contains(end)
        return start < self.end and (end is None or end >= self.start)


def _local_midnight(day: datetime) -> datetime:
    return day.replace(hour=0, minute=0, second=0, microsecond=0)


def _now() -> datetime:
    return datetime.now().astimezone()


def iso_week_window(week: str) -> Window:
    """``2026-W33`` -> that ISO week, Monday 00:00 local through the following Monday."""
    year, _, number = week.upper().partition("-W")
    if not number.isdigit():
        raise ValueError(f"Not an ISO week: {week!r} (expected e.g. 2026-W33)")
    monday = datetime.fromisocalendar(int(year), int(number), 1).astimezone()
    return Window(week.upper(), _local_midnight(monday), _local_midnight(monday) + timedelta(days=WEEK_DAYS))


def trailing_window(since: str) -> Window:
    """``7d`` / ``2w`` -> that span ending now.

    Labelled by the ISO week it ends in, so the usual Friday ``--since 7d`` lands in the same folder
    the pinned ``--week`` form would use. A longer span gets the span appended, because a 30-day sweep
    and the week it ends in are different documents and must not overwrite one another.
    """
    match = SINCE_RE.match(since)
    if not match:
        raise ValueError(f"Not a duration: {since!r} (expected e.g. 7d or 2w)")

    days = int(match.group(1)) * DAYS_PER[match.group(2).lower()]
    end = _now()
    year, number, _ = end.isocalendar()
    label = f"{year}-W{number:02d}"
    return Window(label if days == WEEK_DAYS else f"{label}-last{days}d", end - timedelta(days=days), end)


def resolve_window(week: str | None, since: str | None) -> Window:
    if week and since:
        raise ValueError("Pass --week or --since, not both")
    return iso_week_window(week) if week else trailing_window(since or f"{WEEK_DAYS}d")


# --- git ---------------------------------------------------------------------------------------

def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


@dataclass(frozen=True)
class Commit:
    sha: str
    author: str
    date: str  # ISO-8601 with offset, as git reports it
    subject: str
    exp_ids: list[str]

    @property
    def is_finding(self) -> bool:
        """An unprefixed, experiment-keyed subject — this repo's convention for reporting a result."""
        return bool(self.exp_ids) and not CHORE_PREFIX_RE.match(self.subject)


@dataclass(frozen=True)
class NotesChange:
    """What was written into one experiment's notes during the window."""

    exp_id: str
    rel_path: str  # path as of now
    added_lines: list[str]
    is_new: bool  # the notes did not exist at the window's base commit

    @property
    def added_prose(self) -> str:
        return "\n".join(self.added_lines).strip()


def _commit_before(moment: datetime) -> str | None:
    """Resolved by date rather than by ``HEAD@{...}``: the reflog is per-checkout and would give each
    of the three people working here a different answer for the same week."""
    sha = _git("rev-list", "-1", f"--before={moment.isoformat()}", "HEAD").strip()
    return sha or None


def base_commit(window: Window) -> str | None:
    """The last commit before the window opened — the "before" side of every comparison here."""
    return _commit_before(window.start)


def head_ref(window: Window) -> str | None:
    """The "after" side: the last commit inside the window, or ``None`` to mean the working tree.

    A window that has not closed yet is read from the working tree on purpose — notes for the week
    being presented are routinely still uncommitted on the morning of the meeting. A window that
    *has* closed must not be, or last month's report would be written up with this month's findings.
    """
    if window.end >= _now():
        return None
    return _commit_before(window.end)


def commits_in(window: Window) -> list[Commit]:
    """Commits authored inside the window, newest first, merges excluded."""
    fields = "%H%x1f%an%x1f%aI%x1f%s"
    raw = _git(
        "log", "--no-merges", f"--since={window.start.isoformat()}",
        f"--until={window.end.isoformat()}", f"--format={fields}",
    )
    commits = []
    for line in raw.splitlines():
        sha, _, rest = line.partition("\x1f")
        author, _, rest = rest.partition("\x1f")
        date, _, subject = rest.partition("\x1f")
        if sha:
            commits.append(Commit(sha[:7], author, date, subject, exp_ids_in(subject)))
    return commits


def _exp_id_of(notes_path: str) -> str:
    """The experiment id a notes path belongs to — the key everything here is matched on."""
    folder = NOTES_PATH_RE.match(notes_path).group(1)
    return normalize_exp_id(EXP_ID_RE.match(folder).group(1))


def _notes_paths_at(ref: str) -> dict[str, str]:
    """``{exp_id: path}`` for every experiment notes file present at ``ref``."""
    return {
        _exp_id_of(path): path
        for path in _git("ls-tree", "-r", "--name-only", ref, "--", "experiments").splitlines()
        if NOTES_PATH_RE.match(path)
    }


def _notes_at(ref: str | None) -> dict[str, tuple[str, str]]:
    """``{exp_id: (rel_path, text)}`` at ``ref``, or from the working tree when ``ref`` is None."""
    if ref is None:
        notes = {}
        for path in sorted((REPO_ROOT / "experiments").glob("**/notes.md")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if NOTES_PATH_RE.match(rel):
                notes[_exp_id_of(rel)] = (rel, path.read_text())
        return notes
    return {exp_id: (path, _git("show", f"{ref}:{path}")) for exp_id, path in _notes_paths_at(ref).items()}


def _normalise_paths(line: str) -> str:
    return EXPERIMENT_PATH_RE.sub(r"experiments/\1", LINK_TARGET_RE.sub("](#)", line))


def _added_lines(old: str, new: str) -> list[str]:
    """Lines present in ``new`` but not ``old``, compared with link targets normalised away.

    Comparing normalised lines but returning the *original* ones is what keeps a mass reference
    rewrite (``../exp109_x/notes.md`` -> ``../../nudity/exp109_x/notes.md``, applied to 122 files by
    ``f048777``) out of the deck without also mangling the prose that is reported.
    """
    new_lines = new.splitlines()
    matcher = SequenceMatcher(
        a=[_normalise_paths(line) for line in old.splitlines()],
        b=[_normalise_paths(line) for line in new_lines],
        autojunk=False,
    )
    return [
        line
        for tag, _, _, start, stop in matcher.get_opcodes()
        if tag in ("insert", "replace")
        for line in new_lines[start:stop]
    ]


def notes_changes(window: Window) -> dict[str, NotesChange]:
    """Per experiment, the lines added to its ``notes.md`` since the window's base commit.

    Matched by experiment id rather than by path, so the thread-regrouping move is not mistaken for
    122 new experiments. Only added lines are kept: a deletion is almost always a correction being
    rewritten, and the rewritten text is already in the additions.
    """
    base = base_commit(window)
    before = _notes_at(base) if base else {}
    changes = {}

    for exp_id, (rel_path, text) in _notes_at(head_ref(window)).items():
        old = before[exp_id][1] if exp_id in before else ""
        added = _added_lines(old, text)
        if not added:
            continue
        changes[exp_id] = NotesChange(exp_id, rel_path, added, is_new=exp_id not in before)
    return changes
