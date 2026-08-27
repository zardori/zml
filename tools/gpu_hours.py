#!/usr/bin/env -S uv run
"""Report GPU-hour consumption on athena and helios: the grants, and what your jobs spent.

Compute is the binding constraint on this project, but the two numbers that matter live in
different places and only on the clusters: `hpc-grants` knows what the *group's* allocation has
burned (and when it runs out), while Slurm accounting knows what *your* jobs cost and which
experiment they belonged to. This runs locally and does its cluster work over ssh (CLAUDE.md,
"Working With the Clusters"), so one command answers both, for both clusters, side by side.

It also cross-checks the allocations against `sacctmgr show assoc`: an exhausted or expired grant
loses its Slurm association, so the account still baked into `slurm/<cluster>.sh` can silently stop
accepting submissions. Those allocations are flagged NO ACCESS.

GPU-hours are elapsed wall time x allocated GPUs (`gres/gpu` in `AllocTRES`), summed per job, which
is how the allocation is billed. Jobs are grouped by experiment via the job name — `submit_job.py`
names every job after its experiment directory (grid runs get a `_run_NNN` suffix, stripped here).

Usage:
    tools/gpu_hours.py                             # both clusters, current grant period
    tools/gpu_hours.py --cluster helios --days 30
    tools/gpu_hours.py --since 2026-06-01 --top 20
    tools/gpu_hours.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections.abc import Callable
from datetime import date, timedelta

from zml.cluster_sync import KNOWN_CLUSTERS, ClusterSyncError, load_cluster_conf

# sacct is asked for one wide window and filtered here, so the report window can be derived from the
# grants (only known after the same ssh call) without a second round trip. PLGrid's sacct refuses
# ranges beyond roughly half a year ("Too wide of a date range in query"), which caps this.
FETCH_WINDOW_DAYS = 180
# Used only when hpc-grants tells us nothing about when the current allocation started.
FALLBACK_WINDOW_DAYS = 90
DEFAULT_TOP = 10
SACCT_FIELDS = "JobID,JobName,Account,State,Elapsed,Start,AllocTRES"
# Hours spent by jobs that produced no result. Worth seeing next to the total.
WASTED_STATES = frozenset({"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"})

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# `gres/gpu=2` and the typed `gres/gpu:a100=2` can both appear; the plain one is the total.
GPU_PLAIN_RE = re.compile(r"gres/gpu=(\d+)")
GPU_TYPED_RE = re.compile(r"gres/gpu:[^=,]+=(\d+)")
ELAPSED_RE = re.compile(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)")
RUN_SUFFIX_RE = re.compile(r"_run_\d+$")

SECTION_GRANTS = "###GRANTS"
SECTION_ASSOC = "###ASSOC"
SECTION_JOBS = "###JOBS"


@dataclass(frozen=True)
class Allocation:
    """One compute allocation of a grant, as `hpc-grants` reports it (consumption is group-wide)."""

    grant: str
    name: str
    resource: str
    status: str
    start: str
    end: str
    limit_hours: float
    consumed_hours: float

    @property
    def remaining_hours(self) -> float:
        return self.limit_hours - self.consumed_hours


@dataclass(frozen=True)
class Job:
    job_id: str
    name: str
    account: str
    state: str
    elapsed_hours: float
    start: date | None
    gpus: int

    @property
    def gpu_hours(self) -> float:
        return self.elapsed_hours * self.gpus

    @property
    def experiment(self) -> str:
        return RUN_SUFFIX_RE.sub("", self.name)

    @property
    def wasted(self) -> bool:
        return self.state.split()[0] in WASTED_STATES


@dataclass
class ClusterReport:
    cluster: str
    allocations: list[Allocation] = field(default_factory=list)
    associations: set[str] = field(default_factory=set)
    jobs: list[Job] = field(default_factory=list)
    error: str | None = None
    jobs_error: str | None = None

    @property
    def window_start(self) -> date:
        """Report from the current grant period, so the totals line up with what hpc-grants shows."""
        starts = [_parse_date(a.start) for a in self.allocations]
        known = [s for s in starts if s is not None]
        return min(known) if known else date.today() - timedelta(days=FALLBACK_WINDOW_DAYS)


def _parse_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _parse_hours(text: str) -> float | None:
    """`50 097.88 h` -> 50097.88; anything not denominated in hours (e.g. `5 000 GB`) -> None."""
    cleaned = text.replace(" ", " ").strip()
    if not cleaned.endswith("h"):
        return None
    try:
        return float(cleaned[:-1].replace(" ", ""))  # spaces group the thousands
    except ValueError:
        return None


def parse_elapsed_hours(text: str) -> float:
    match = ELAPSED_RE.fullmatch(text.strip())
    if not match:
        return 0.0
    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return days * 24 + hours + minutes / 60 + seconds / 3600


def parse_gpu_count(alloc_tres: str) -> int:
    plain = GPU_PLAIN_RE.search(alloc_tres)
    if plain:
        return int(plain.group(1))
    typed = [int(m) for m in GPU_TYPED_RE.findall(alloc_tres)]
    return sum(typed)


def parse_grants(text: str) -> list[Allocation]:
    """Pull the compute allocations out of `hpc-grants` (storage ones carry GB, not hours)."""
    allocations: list[Allocation] = []
    grant = ""
    pending: dict[str, str] = {}

    def flush() -> None:
        limit = _parse_hours(pending.get("hours", ""))
        consumed = _parse_hours(pending.get("consumed", ""))
        if pending and limit is not None:
            allocations.append(
                Allocation(
                    grant=grant,
                    name=pending.get("name", "?"),
                    resource=pending.get("resource", "?"),
                    status=pending.get("status", "?"),
                    start=pending.get("start", "?"),
                    end=pending.get("end", "?"),
                    limit_hours=limit,
                    consumed_hours=consumed if consumed is not None else 0.0,
                )
            )
        pending.clear()

    for raw in text.splitlines():
        line = ANSI_RE.sub("", raw).strip()
        if line.startswith("Grant:"):
            flush()
            grant = line.split(":", 1)[1].strip()
        elif line.startswith("Allocation:"):
            flush()
            body = line.split(":", 1)[1]
            name, _, resource = body.partition(", resource:")
            pending.update(name=name.strip(), resource=resource.strip())
        elif line.startswith("Group:"):
            flush()  # the member list follows; nothing after it belongs to an allocation
        elif not pending:
            continue
        elif line.startswith("status:"):
            pending["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("start:"):
            start, _, end = line.split(":", 1)[1].partition(", end:")
            pending.update(start=start.strip(), end=end.strip())
        elif line.startswith("hours:"):
            pending["hours"] = line.split(":", 1)[1].strip()
        elif line.startswith("consumed resources:"):
            pending["consumed"] = line.split(":", 1)[1].strip()
    flush()
    return allocations


def parse_jobs(text: str) -> list[Job]:
    """Every GPU job sacct returned; CPU-only jobs cost the allocation nothing and are dropped."""
    jobs: list[Job] = []
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) != 7:
            continue
        job_id, name, account, state, elapsed, start, alloc_tres = fields
        gpus = parse_gpu_count(alloc_tres)
        if gpus == 0:
            continue
        jobs.append(
            Job(
                job_id=job_id,
                name=name,
                account=account,
                state=state,
                elapsed_hours=parse_elapsed_hours(elapsed),
                start=_parse_date(start.split("T")[0]),
                gpus=gpus,
            )
        )
    return jobs


def jobs_since(jobs: list[Job], window_start: date) -> list[Job]:
    """A job that never started (Start is `Unknown`) has no elapsed time to account for either."""
    return [j for j in jobs if j.start is not None and j.start >= window_start]


def remote_command(user: str | None, fetch_since: date) -> str:
    """The markers are quoted: an unquoted `###JOBS` is a comment to the remote shell, not a word."""
    who = user or "$USER"
    return "\n".join(
        [
            f"echo '{SECTION_GRANTS}'",
            "hpc-grants 2>/dev/null || true",
            f"echo '{SECTION_ASSOC}'",
            f"sacctmgr -n -P show assoc user={who} format=Account 2>/dev/null || true",
            f"echo '{SECTION_JOBS}'",
            f"sacct -X -u {who} -S {fetch_since.isoformat()} -P -n --format={SACCT_FIELDS} 2>&1 || true",
        ]
    )


def split_sections(output: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in output.splitlines():
        if line.strip() in (SECTION_GRANTS, SECTION_ASSOC, SECTION_JOBS):
            current = sections.setdefault(line.strip(), [])
        elif current is not None:
            current.append(line)
    return {key: "\n".join(lines) for key, lines in sections.items()}


def collect(cluster: str, user: str | None, fetch_since: date) -> ClusterReport:
    report = ClusterReport(cluster=cluster)
    try:
        host = load_cluster_conf(cluster)["HOST"]
    except ClusterSyncError as exc:
        report.error = str(exc)
        return report

    result = subprocess.run(
        ["ssh", host, remote_command(user, fetch_since)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        report.error = result.stderr.strip() or f"ssh {host} failed (exit {result.returncode})"
        return report

    sections = split_sections(result.stdout)
    report.allocations = parse_grants(sections.get(SECTION_GRANTS, ""))
    report.associations = {a.strip() for a in sections.get(SECTION_ASSOC, "").splitlines() if a.strip()}
    jobs_output = sections.get(SECTION_JOBS, "")
    report.jobs_error = next((l.strip() for l in jobs_output.splitlines() if l.startswith("sacct: error")), None)
    report.jobs = parse_jobs(jobs_output)
    return report


def _totals(jobs: list[Job]) -> tuple[int, float, float]:
    return len(jobs), sum(j.gpu_hours for j in jobs), sum(j.gpu_hours for j in jobs if j.wasted)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _group(jobs: list[Job], key: Callable[[Job], str]) -> list[tuple[str, int, float]]:
    grouped: dict[str, list[Job]] = {}
    for job in jobs:
        grouped.setdefault(key(job), []).append(job)
    rows = [(name, len(group), sum(j.gpu_hours for j in group)) for name, group in grouped.items()]
    return sorted(rows, key=lambda row: row[2], reverse=True)


def render(report: ClusterReport, window_start: date, top: int, clamped_from: date | None = None) -> None:
    print(f"=== {report.cluster} ===")
    if report.error:
        print(f"  unavailable: {report.error}\n")
        return

    if report.allocations:
        print("  Grants (group-wide consumption):")
        for alloc in report.allocations:
            access = "" if alloc.name in report.associations else "  [NO ACCESS]"
            print(
                f"    {alloc.name:<26} {alloc.status:<10}"
                f" {alloc.consumed_hours:>10,.1f} / {alloc.limit_hours:>9,.0f} h"
                f"  ({alloc.remaining_hours:>9,.1f} h left)  {alloc.start} -> {alloc.end}{access}"
            )
    else:
        print("  Grants: hpc-grants reported no compute allocation.")

    if report.jobs_error:
        print(f"\n  Your jobs: {report.jobs_error}\n")
        return

    jobs = report.jobs
    count, gpu_hours, wasted = _totals(jobs)
    running = sum(1 for j in jobs if j.state.startswith("RUNNING"))
    print(
        f"\n  Your jobs since {window_start}: {_plural(count, 'job')}, {gpu_hours:,.1f} GPU-h"
        f" ({wasted:,.1f} GPU-h failed/timeout/cancelled"
        + (f", {running} still running" if running else "")
        + ")"
    )
    if clamped_from:
        print(f"    (grant period began {clamped_from}; sacct only serves the last {FETCH_WINDOW_DAYS} days)")
    if not jobs:
        print()
        return

    print("    by account:")
    for name, n, hours in _group(jobs, lambda j: j.account):
        flag = "" if name in report.associations else "  [no longer submittable]"
        print(f"      {name:<26} {_plural(n, 'job'):>9} {hours:>10,.1f} GPU-h{flag}")

    by_experiment = _group(jobs, lambda j: j.experiment)
    shown = by_experiment if top <= 0 else by_experiment[:top]
    print(f"    by experiment ({len(shown)} of {len(by_experiment)}, most expensive first):")
    for name, n, hours in shown:
        print(f"      {name:<46} {_plural(n, 'job'):>9} {hours:>10,.1f} GPU-h")
    print()


def to_json(report: ClusterReport, window_start: date) -> dict:
    count, gpu_hours, wasted = _totals(report.jobs)
    return {
        "cluster": report.cluster,
        "error": report.error or report.jobs_error,
        "window_start": window_start.isoformat(),
        "allocations": [
            {
                "grant": a.grant,
                "name": a.name,
                "resource": a.resource,
                "status": a.status,
                "start": a.start,
                "end": a.end,
                "limit_hours": a.limit_hours,
                "consumed_hours": a.consumed_hours,
                "remaining_hours": a.remaining_hours,
                "submittable": a.name in report.associations,
            }
            for a in report.allocations
        ],
        "jobs": count,
        "gpu_hours": round(gpu_hours, 2),
        "wasted_gpu_hours": round(wasted, 2),
        "by_account": [
            {"account": n, "jobs": c, "gpu_hours": round(h, 2)}
            for n, c, h in _group(report.jobs, lambda j: j.account)
        ],
        "by_experiment": [
            {"experiment": n, "jobs": c, "gpu_hours": round(h, 2)}
            for n, c, h in _group(report.jobs, lambda j: j.experiment)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-hour usage per cluster: grant allocations and your own jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cluster",
        choices=KNOWN_CLUSTERS,
        action="append",
        help="Cluster to query (repeatable; default: all)",
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--since", help="Count jobs from this date (YYYY-MM-DD); default: grant start")
    window.add_argument("--days", type=int, help="Count jobs from N days ago")
    parser.add_argument("--user", help="Cluster login to report on (default: yours on that cluster)")
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Experiments to list, 0 for all (default: {DEFAULT_TOP})",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the text report")
    return parser.parse_args()


def resolve_window(args: argparse.Namespace) -> date | None:
    """The explicit window, or None to let each cluster derive it from its own grant."""
    if args.since:
        window = _parse_date(args.since)
        if window is None:
            print(f"Error: --since must be YYYY-MM-DD, got '{args.since}'", file=sys.stderr)
            sys.exit(1)
        return window
    if args.days:
        return date.today() - timedelta(days=args.days)
    return None


def main() -> None:
    args = parse_args()
    clusters = args.cluster or list(KNOWN_CLUSTERS)
    window = resolve_window(args)
    earliest = date.today() - timedelta(days=FETCH_WINDOW_DAYS)
    fetch_since = max(window, earliest) if window else earliest

    with ThreadPoolExecutor(max_workers=len(clusters)) as pool:
        reports = list(pool.map(lambda c: collect(c, args.user, fetch_since), clusters))

    payload = []
    for report in reports:
        requested = window or report.window_start  # fetched wide: the window is applied per cluster
        start = max(requested, fetch_since)
        report.jobs = jobs_since(report.jobs, start)
        if args.json:
            payload.append(to_json(report, start))
        else:
            render(report, start, args.top, clamped_from=requested if start > requested else None)
    if args.json:
        print(json.dumps(payload, indent=2))

    if all(r.error for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
