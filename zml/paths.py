"""Resolution of config input paths across the project members' cluster repos.

Every member runs jobs from their own repo copy on cluster scratch, so a config field like
``metadata_file: experiments/exp061_.../outputs_.../metadata.json`` only resolves for whoever
actually ran exp061 — anyone else gets a missing file, even though the dirs are group-readable.

The fix is a search path. A repo-relative input that is missing in the running user's repo is
looked up under the other members' repo roots, taken from the ``ZML_PEER_ROOTS`` env var
(colon-separated, exported by ``slurm/*.sh``). The local repo is always tried first, so nothing
changes when the data is present locally, and outputs still go only to the running user's repo.
"""

import os
from pathlib import Path
from typing import Any

PEER_ROOTS_ENV = "ZML_PEER_ROOTS"

# Only repo-relative references to shared data are candidates for redirection. Restricting to
# these prefixes keeps the sweep over config values predictable: an ordinary string field can
# never be silently rewritten into a path.
DATA_PREFIXES = ("experiments/", "prompts/", "outputs/", "datasets/")


def peer_roots() -> list[Path]:
    """Repo roots of the other project members, in search order (empty off-cluster)."""
    raw = os.environ.get(PEER_ROOTS_ENV, "")
    local_root = Path.cwd().resolve()
    roots = []
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        root = Path(entry)
        if root.resolve() != local_root and root not in roots:
            roots.append(root)
    return roots


def resolve_input_path(path: str) -> str:
    """Return `path` unchanged if it exists locally, else the first peer copy that exists.

    Falls back to the original path when nothing is found, so the caller still fails with the
    path the user wrote rather than with a rewritten one.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    for root in peer_roots():
        candidate = root / path
        if candidate.exists():
            return str(candidate)
    return path


def resolve_config_paths(params: dict[str, Any]) -> dict[str, Any]:
    """Redirect repo-relative data paths in a loaded config to a peer's copy when missing locally.

    Applied by the thin entrypoints right after the YAML is read, so every method benefits without
    knowing which of its fields are paths.
    """
    resolved = dict(params)
    for key, value in params.items():
        if not isinstance(value, str) or not value.startswith(DATA_PREFIXES) or os.path.exists(value):
            continue
        found = resolve_input_path(value)
        if found != value:
            print(f"Config path '{key}': not in this repo, using peer copy {found}")
        else:
            searched = [str(root) for root in peer_roots()] or ["<none: ZML_PEER_ROOTS unset>"]
            print(f"WARNING: config path '{key}' = {value} not found locally nor in {searched}")
        resolved[key] = found
    return resolved
