"""Finding an experiment's inputs on the other cluster, and copying them where they are needed.

Athena and helios each hold a separate copy of the repo tree, and an output lives only on the
cluster that produced it: exp068's preservation latents were precomputed on athena, so exp069
submitted to helios fails its path check even though the data exists one cluster over. Rebuilding it
there costs hours of GPU time for bytes that already exist, and copying it by hand means remembering
which of the three members' repo roots holds it.

This module closes that gap at submit time. It runs locally and does all its cluster work over ssh
(see CLAUDE.md, "Working With the Clusters"); the lookup on either side is the same
`slurm/check_config_paths.sh --locate` the pre-submit check already uses, so a peer's copy counts as
found exactly as it does at runtime (`zml/paths.py`). Inputs always land in the *running user's*
repo on the target, never in a peer's.

Two transfer routes, tried in that order:

1. **Direct** — `ssh -A <source> rsync ... <target>`: the bytes travel cluster-to-cluster over
   PLGrid's network and never touch the local link. Needs agent forwarding (`ForwardAgent yes` in
   `~/.ssh/config`) so the source login node can authenticate to the target as the user.
2. **Relay** — `ssh <source> tar -cf - | ssh <target> tar -xf -`: streams through this machine
   without staging anything on local disk. Slower, and only used when the direct hop is refused.

Either way the payload lands in a staging directory beside its destination and is moved into place
only once it is complete, so an interrupted copy can never be mistaken for the data itself.
"""

from __future__ import annotations

import posixpath
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from zml.paths import DATA_PREFIXES

REPO_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_CONF = REPO_ROOT / "cluster.conf"
PATH_CHECK_SCRIPT = "slurm/check_config_paths.sh"
KNOWN_CLUSTERS: tuple[str, ...] = ("athena", "helios")

# Options for the source -> target hop. BatchMode turns a missing forwarded agent into an immediate
# failure instead of a password prompt on a login node nobody is watching; accept-new lets the first
# hop record the target's host key without a confirmation nobody can give.
HOP_SSH_OPTS: tuple[str, ...] = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
)

_SIZE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


class ClusterSyncError(RuntimeError):
    """A cluster-side step failed; the caller should report it, not raise a traceback."""


@dataclass(frozen=True)
class Cluster:
    """One cluster's connection details, as seen from here and from the other cluster."""

    name: str
    host: str  # what *we* ssh to; may be a ~/.ssh/config alias
    remote_dir: str
    ssh_user: str
    ssh_hostname: str

    @property
    def address(self) -> str:
        """`user@host` as it must be written from another machine — ssh aliases are local-only."""
        return f"{self.ssh_user}@{self.ssh_hostname}"


@dataclass(frozen=True)
class RemoteFile:
    """A config input located on some cluster: where it is, and how much it costs to move."""

    rel_path: str
    abs_path: str
    size_bytes: int


@dataclass
class LocateResult:
    found: dict[str, RemoteFile] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    missing_config: list[str] = field(default_factory=list)


def load_cluster_conf(cluster: str) -> dict[str, str]:
    """Read HOST/REMOTE_DIR for one cluster out of `cluster.conf` (sourced, so it stays shell)."""
    if not CLUSTER_CONF.exists():
        raise ClusterSyncError(f"{CLUSTER_CONF} not found. Copy cluster.conf.example to cluster.conf.")
    script = f"""
source {shlex.quote(str(CLUSTER_CONF))}
case {shlex.quote(cluster)} in
    athena) echo "HOST=$ATHENA_HOST" && echo "REMOTE_DIR=$ATHENA_REMOTE_DIR" ;;
    helios) echo "HOST=$HELIOS_HOST" && echo "REMOTE_DIR=$HELIOS_REMOTE_DIR" ;;
    *) echo "Error: unknown cluster '{cluster}'" >&2; exit 1 ;;
esac
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise ClusterSyncError(result.stderr.strip() or f"unknown cluster '{cluster}'")
    conf: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, val = line.partition("=")
        conf[key.strip()] = val.strip()
    return conf


def _resolve_ssh_identity(host: str) -> tuple[str, str]:
    """The login and real hostname behind an ssh alias, via `ssh -G`.

    `cluster.conf` may name a `~/.ssh/config` alias ("helios"), which means nothing on the *other*
    cluster; the direct hop needs the address that machine would have to dial itself.
    """
    result = subprocess.run(["ssh", "-G", host], capture_output=True, text=True)
    if result.returncode != 0:
        raise ClusterSyncError(f"`ssh -G {host}` failed: {result.stderr.strip()}")
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        fields.setdefault(key, value)  # ssh -G prints the effective value first
    return fields.get("user", ""), fields.get("hostname", host)


def load_cluster(name: str) -> Cluster:
    conf = load_cluster_conf(name)
    user, hostname = _resolve_ssh_identity(conf["HOST"])
    return Cluster(
        name=name,
        host=conf["HOST"],
        remote_dir=conf["REMOTE_DIR"],
        ssh_user=user,
        ssh_hostname=hostname,
    )


def other_clusters(name: str) -> list[str]:
    return [cluster for cluster in KNOWN_CLUSTERS if cluster != name]


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in _SIZE_UNITS:
        if size < 1024 or unit == _SIZE_UNITS[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} {_SIZE_UNITS[-1]}"


def config_data_paths(config: dict[str, Any]) -> list[str]:
    """Repo-relative data paths the config references, in config order and de-duplicated.

    Uses the same `DATA_PREFIXES` rule as the runtime resolution in `zml/paths.py`, so what is
    verified (and fetched) is exactly what the entrypoints will later try to open, and nothing else
    — a HF model id like `THUDM/CogVideoX-5b` contains a slash but is not a path. List values are
    flattened so a grid that sweeps over a path field has all its variants covered.
    """
    paths: list[str] = []
    for value in config.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item.startswith(DATA_PREFIXES) and item not in paths:
                paths.append(item)
    return paths


def _ssh(host: str, command: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", host, command], capture_output=capture, text=True)


def git_pull(cluster: Cluster, check: bool = True) -> None:
    """Bring the cluster repo up to date. The pull may itself be what supplies a missing input."""
    result = _ssh(cluster.host, f"cd {shlex.quote(cluster.remote_dir)} && git pull", capture=not check)
    if result.returncode != 0 and check:
        raise ClusterSyncError(f"`git pull` failed on {cluster.name}")
    if result.returncode != 0:
        print(f"  (warning: `git pull` failed on {cluster.name}; using the checkout as it is)")


def locate_paths(cluster: Cluster, paths: Sequence[str], config: str | None = None) -> LocateResult:
    """Ask a cluster where each repo-relative path is, searching this user's repo and the peers'."""
    result = LocateResult()
    if not paths and not config:
        return result

    args = ["--locate"]
    if config:
        args += ["--config", config]
    args += [cluster.name, *paths]
    command = " ".join(shlex.quote(arg) for arg in [PATH_CHECK_SCRIPT, *args])
    proc = _ssh(cluster.host, f"cd {shlex.quote(cluster.remote_dir)} && {command}", capture=True)
    if proc.returncode != 0:
        raise ClusterSyncError(
            f"path lookup failed on {cluster.name}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)

    for line in proc.stdout.splitlines():
        kind, *fields = line.split("\t")
        if kind == "FOUND" and len(fields) == 3:
            rel, abs_path, size = fields
            result.found[rel] = RemoteFile(rel, abs_path, int(size))
        elif kind == "MISSING" and fields:
            result.missing.append(fields[0])
        elif kind == "MISSING_CONFIG" and fields:
            result.missing_config.append(fields[0])
    return result


def _confirm_transfer(
    source: Cluster, target: Cluster, items: Sequence[RemoteFile], assume_yes: bool
) -> bool:
    total = sum(item.size_bytes for item in items)
    print(f"  found on {source.name}:")
    for item in items:
        print(f"    {item.rel_path}  ({format_size(item.size_bytes)})  <- {item.abs_path}")
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        # Nobody is there to answer: reading stdin would either hang or blow up with an EOFError
        # whose traceback says nothing about the missing --yes. Fail with the actionable reason.
        raise ClusterSyncError(
            f"cross-cluster copy needed ({len(items)} path(s), {format_size(total)}, "
            f"{source.name} -> {target.name}) but stdin is not a terminal; "
            "re-run with --yes to approve transfers non-interactively, "
            "or --no-fetch-missing to skip them"
        )
    prompt = f"Copy {len(items)} path(s), {format_size(total)}, {source.name} -> {target.name}? [Y/n] "
    return input(prompt).strip().lower() in ("", "y", "yes")


def _direct_hop_available(source: Cluster, target: Cluster) -> bool:
    probe = f"ssh {' '.join(HOP_SSH_OPTS)} {shlex.quote(target.address)} true"
    proc = subprocess.run(["ssh", "-A", source.host, probe], capture_output=True, text=True)
    if proc.returncode == 0:
        return True
    reason = proc.stderr.strip().splitlines()
    print(f"  no direct {source.name} -> {target.name} ssh ({reason[-1] if reason else 'refused'});"
          " relaying through this machine")
    return False


def _copy_direct(source: Cluster, target: Cluster, item: RemoteFile, staging: str) -> None:
    """rsync straight from source to target, driven by the source login node over forwarded auth."""
    hop = "ssh " + " ".join(HOP_SSH_OPTS)
    remote_cmd = " ".join([
        "rsync", "-a", "--info=progress2", "-e", shlex.quote(hop),
        shlex.quote(item.abs_path),
        shlex.quote(f"{target.address}:{staging}/"),
    ])
    if subprocess.run(["ssh", "-A", source.host, remote_cmd]).returncode != 0:
        raise ClusterSyncError(f"rsync {source.name} -> {target.name} failed for {item.rel_path}")


def _copy_relayed(source: Cluster, target: Cluster, item: RemoteFile, staging: str) -> None:
    """Stream the payload through this machine, tar to tar, so nothing is staged on local disk."""
    src_parent, name = posixpath.split(item.abs_path.rstrip("/"))
    pack = f"tar -C {shlex.quote(src_parent)} -cf - -- {shlex.quote(name)}"
    unpack = f"tar -C {shlex.quote(staging)} -xf -"

    with subprocess.Popen(["ssh", source.host, pack], stdout=subprocess.PIPE) as producer:
        if producer.stdout is None:
            raise ClusterSyncError("could not open the tar stream from the source cluster")
        consumer = subprocess.run(["ssh", target.host, unpack], stdin=producer.stdout)
        producer.stdout.close()
    if producer.returncode != 0 or consumer.returncode != 0:
        raise ClusterSyncError(f"relayed copy of {item.rel_path} failed")


def _staging_dir(target: Cluster, item: RemoteFile) -> str:
    """Where an in-flight copy lands, beside its destination.

    Nothing appears at the real path until the whole payload is there: an interrupted transfer of a
    20 GB `latents/` would otherwise leave a directory that the next path check happily reports as
    present, and the job trains on a truncated dataset. Staging beside the destination keeps the
    rename on one filesystem, and a re-run resumes into the same directory instead of starting over.
    """
    dest = f"{target.remote_dir}/{item.rel_path}"
    return f"{posixpath.dirname(dest)}/.zml_incoming"


def copy_to_cluster(source: Cluster, target: Cluster, items: Sequence[RemoteFile]) -> None:
    """Copy located inputs into the running user's repo on `target`, at their repo-relative paths."""
    if not items:
        return
    staging_dirs = sorted({_staging_dir(target, item) for item in items})
    mkdir = " && ".join(f"mkdir -p {shlex.quote(staging)}" for staging in staging_dirs)
    if _ssh(target.host, mkdir, capture=True).returncode != 0:
        raise ClusterSyncError(f"could not create destination directories on {target.name}")

    direct = _direct_hop_available(source, target)
    print(f"  copying {len(items)} path(s) {source.name} -> {target.name}...")
    for item in items:
        staging = _staging_dir(target, item)
        print(f"    {item.rel_path} ({format_size(item.size_bytes)})")
        if direct:
            _copy_direct(source, target, item, staging)
        else:
            _copy_relayed(source, target, item, staging)
        _publish(target, item, staging)

    # Only empty now, and only if every payload was published; a leftover means something is unfinished.
    _ssh(target.host, " ; ".join(f"rmdir {shlex.quote(s)} 2>/dev/null" for s in staging_dirs), capture=True)


def _publish(target: Cluster, item: RemoteFile, staging: str) -> None:
    """Move a fully transferred payload from the staging dir to the path the config names."""
    dest = f"{target.remote_dir}/{item.rel_path}"
    name = posixpath.basename(item.abs_path.rstrip("/"))
    # -T so an unexpected directory at the destination is an error, never a nested copy inside it.
    move = f"mv -T -- {shlex.quote(f'{staging}/{name}')} {shlex.quote(dest)}"
    result = _ssh(target.host, move, capture=True)
    if result.returncode != 0:
        raise ClusterSyncError(
            f"copied {item.rel_path} but could not put it in place on {target.name}: "
            f"{result.stderr.strip()}"
        )


def fetch_missing_inputs(
    target: Cluster,
    missing: Sequence[str],
    sources: Sequence[str] | None = None,
    assume_yes: bool = False,
) -> list[str]:
    """Fetch paths missing on `target` from another cluster; return what is still missing.

    The caller keeps ownership of the abort decision: a path nobody has anywhere is a real error,
    but which of them is fatal depends on why it was being checked.
    """
    pending = list(missing)
    for name in sources or other_clusters(target.name):
        if not pending:
            break
        source = load_cluster(name)
        print(f"Searching {source.name} for {len(pending)} path(s) missing on {target.name}...")
        git_pull(source, check=False)  # only needed so --locate exists in an old checkout
        located = locate_paths(source, pending)
        items = [located.found[path] for path in pending if path in located.found]
        if not items:
            print(f"  nothing found on {source.name}.")
            continue
        if not _confirm_transfer(source, target, items, assume_yes):
            print("  skipped.")
            break
        copy_to_cluster(source, target, items)
        pending = [path for path in pending if path not in located.found]
    return pending
