"""The 5 celebrity identities used by the face/identity-erasure protocol, and their reference embeddings.

Mirrors ``imagenet_classes.py``: T2VUnlearning (arXiv 2505.17550, §4.3) erases one identity at a
time and scores ArcFace ID-Similarity against the other four. VideoEraser also reports on the same
five, so keeping this exact list keeps our numbers comparable with both. See
``docs/face_identity.md`` for the protocol and ``docs/comparison_targets.md`` §2.3 for why this
concept, and in this order.

Unlike the ImageNet axis, neither paper releases reference embeddings for these identities — see
``tools/build_face_reference_embeddings.py`` for how ours are built (from freely-licensed public
photos; only the derived 512-d vectors are committed, never the source images) and
``prompts/face_reference_images.csv`` for the provenance manifest.
"""

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

REFERENCE_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "face_reference_embeddings.json")

# Cosine-similarity gates checked at build time by tools/build_face_reference_embeddings.py — kept
# here too so a caller loading the committed JSON can re-verify it hasn't silently drifted (e.g. a
# hand edit) without re-running the build.
MIN_PER_IMAGE_COSINE = 0.5
MAX_INTER_IDENTITY_COSINE = 0.30


@dataclass(frozen=True)
class Identity:
    name: str
    slug: str


# Name -> Identity. Order matches the paper's Table 3 column order.
FACE_IDENTITIES: dict[str, Identity] = {
    "Angela Merkel": Identity("Angela Merkel", "angela_merkel"),
    "Barack Obama": Identity("Barack Obama", "barack_obama"),
    "Donald Trump": Identity("Donald Trump", "donald_trump"),
    "Joe Biden": Identity("Joe Biden", "joe_biden"),
    "Queen Elizabeth II": Identity("Queen Elizabeth II", "queen_elizabeth_ii"),
}


def identity_slug(name: str) -> str:
    """Filesystem-safe form of an identity name (``"Barack Obama"`` -> ``"barack_obama"``).

    Used for per-identity eval video directories and prompt files, so a name round-trips through a
    path without quoting — mirrors ``imagenet_classes.class_slug``.
    """
    try:
        return FACE_IDENTITIES[name].slug
    except KeyError:
        raise ValueError(f"Unknown identity {name!r}; expected one of {sorted(FACE_IDENTITIES)}.") from None


_manifest_cache: dict | None = None


def _load_manifest() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        if not os.path.exists(REFERENCE_EMBEDDINGS_PATH):
            raise FileNotFoundError(
                f"{REFERENCE_EMBEDDINGS_PATH} does not exist. Build it with "
                "`uv run python tools/build_face_reference_embeddings.py` before scoring any face concept."
            )
        with open(REFERENCE_EMBEDDINGS_PATH) as f:
            _manifest_cache = json.load(f)
    return _manifest_cache


def reset_manifest_cache() -> None:
    """Drop the cached manifest so the next load re-reads the file from disk (used by tests/rebuilds)."""
    global _manifest_cache
    _manifest_cache = None


def expected_model_sha256() -> dict[str, str] | None:
    """``{"rec": sha256, "det": sha256}`` the committed manifest was built with, or ``None`` if no
    manifest exists yet (the bootstrap case, when ``build_face_reference_embeddings.py`` itself is
    computing the embeddings for the first time and has nothing to cross-check against)."""
    if not os.path.exists(REFERENCE_EMBEDDINGS_PATH):
        return None
    manifest = _load_manifest()
    model = manifest.get("model", {})
    return {"rec": model.get("rec", {}).get("sha256"), "det": model.get("det", {}).get("sha256")}


def load_reference_embedding(name: str) -> np.ndarray:
    """L2-normalized 512-d ArcFace reference embedding for ``name``.

    Raises if the stored vector is not unit-normalized — the direct analogue of
    ``imagenet_classifier._assert_class_indices``: a corrupted or hand-edited manifest would
    silently bias every ID-similarity number while still producing plausible-looking results.
    """
    if name not in FACE_IDENTITIES:
        raise ValueError(f"Unknown identity {name!r}; expected one of {sorted(FACE_IDENTITIES)}.")
    manifest = _load_manifest()
    slug = identity_slug(name)
    try:
        entry = manifest["identities"][slug]
    except KeyError:
        raise KeyError(
            f"{REFERENCE_EMBEDDINGS_PATH} has no entry for {name!r} (slug {slug!r}). "
            "Rebuild with tools/build_face_reference_embeddings.py."
        ) from None
    vec = np.asarray(entry["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise RuntimeError(
            f"Reference embedding for {name!r} is not unit-normalized (‖v‖={norm:.4f}); "
            f"{REFERENCE_EMBEDDINGS_PATH} may be corrupted. Rebuild with "
            "tools/build_face_reference_embeddings.py."
        )
    return vec


def load_all_reference_embeddings() -> tuple[list[str], np.ndarray]:
    """Every identity's reference embedding, stacked -- ``(names, (5, 512) matrix)``, ``FACE_IDENTITIES``
    order (matches the paper's Table 3 column order).

    For scoring one clip against all five references at once (the 5x5 cross-reference matrix,
    ``docs/face_identity.md`` §5) rather than looking each identity up individually: a clip's
    embeddings only need to go through ``ArcFaceFrameEmbedder.embed_frames`` once, then
    ``embeddings @ matrix.T`` scores every reference in one matmul.
    """
    names = list(FACE_IDENTITIES)
    matrix = np.stack([load_reference_embedding(name) for name in names])
    return names, matrix


def sha256_of_file(path: str) -> str:
    """sha256 of a local file's bytes, for cross-checking against a manifest's recorded model hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
