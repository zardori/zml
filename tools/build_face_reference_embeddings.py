"""Build the ArcFace reference embedding for each of the 5 face-erasure identities.

T2VUnlearning (arXiv 2505.17550, §4.3) scores ArcFace ID-Similarity against a "ground-truth
identity" but never says what that reference is, and releases no embeddings — see
``docs/face_identity.md`` §4.2. This script builds ours: for each identity, download the freely-
licensed photos listed in ``prompts/face_reference_images.csv``, detect + align + embed the largest
face in each, and average into one L2-normalized 512-d reference vector.

**Only the derived embeddings are committed** (``zml/benchmarks/data/face_reference_embeddings.json``,
~40 KB, diffable). The source photos are never written to the repo — a 512-d ArcFace template is a
biometric derivative of a real person, and "freely licensed, never redistributed as pixels, full
provenance recorded" is the line this project draws. Every source photo's URL, Commons page,
pinned sha256, license and author are recorded in the manifest CSV and echoed into the output JSON.

Two gates abort the build rather than silently producing a bad reference set:

- **Per-image cosine to the identity mean >= MIN_PER_IMAGE_COSINE.** Catches a mis-cropped or
  wrong-person photo — this is not hypothetical: an earlier draft of the manifest included a Commons
  file labelled "Angela Merkel" that turned out, on inspection of the aligned crop, to be a different
  person entirely (a crowd photo where the "largest face" heuristic below picked someone in the
  foreground, not Merkel). Visually inspect the aligned crop of anything that fails this gate before
  swapping it out.
- **Every inter-identity cosine < MAX_INTER_IDENTITY_COSINE.** If two of the five references are not
  well separated, the whole metric is meaningless before a single video is generated.

Both matrices (per-image cosines and the 5x5 inter-identity matrix) are written into the output JSON
so a reader can audit the reference set without rebuilding it.

Run:
    uv run python tools/build_face_reference_embeddings.py
"""

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

from zml.benchmarks.arcface_embedder import ArcFaceFrameEmbedder
from zml.benchmarks.face_identities import FACE_IDENTITIES, identity_slug

MANIFEST_CSV = "prompts/face_reference_images.csv"
OUTPUT_PATH = "zml/benchmarks/data/face_reference_embeddings.json"

MIN_PER_IMAGE_COSINE = 0.5
MAX_INTER_IDENTITY_COSINE = 0.30

# Curated reference stills are already known-good, single-subject photos (verified by the earlier
# "largest face" mis-pick described above), and some are wide official shots where the face is a
# modest fraction of the frame. Unlike the 480x720 CogVideoX video frames ArcFaceFrameEmbedder's
# default min_face_px=48 is tuned for, there's no video-quality concern here — a real face at any
# resolution the source photo offers is worth embedding, and "largest face in the image" already
# discards spurious tiny false-positive detections.
REFERENCE_BUILD_MIN_FACE_PX = 8


def _fetch(url: str, expected_sha256: str, cache_dir: str | None) -> bytes:
    """Download ``url``, optionally through a local by-sha256 cache.

    ``cache_dir`` is a courtesy for repeat rebuilds and local development (Wikimedia rate-limits
    bursts of requests) — it is keyed by the manifest's *pinned* sha256, not the URL, so a stale or
    tampered cache entry can never silently substitute the wrong bytes: a cache hit is only ever
    served if its filename already equals the hash we require. Off by default; every cluster/CI run
    fetches over the network.
    """
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"{expected_sha256}.bin")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()

    request = urllib.request.Request(url, headers={"User-Agent": "zml-research-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{expected_sha256}.bin"), "wb") as f:
            f.write(raw)
    return raw


def _decode_image(raw: bytes) -> np.ndarray:
    array = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Downloaded bytes did not decode as an image (got HTML/an error page?).")
    return img


def _load_manifest(path: str) -> dict[str, list[dict]]:
    by_identity: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_identity[row["identity"]].append(row)

    missing = set(FACE_IDENTITIES) - set(by_identity)
    extra = set(by_identity) - set(FACE_IDENTITIES)
    if missing or extra:
        raise ValueError(
            f"{path} identities don't match zml.benchmarks.face_identities.FACE_IDENTITIES: "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}."
        )
    return by_identity


def build(manifest_path: str, output_path: str, cache_dir: str | None = None) -> None:
    manifest = _load_manifest(manifest_path)
    # verify_reference_sha256=False: this script is what *produces* the reference embeddings, so
    # there is nothing yet to cross-check the loaded checkpoint against (the bootstrap case).
    embedder = ArcFaceFrameEmbedder(verify_reference_sha256=False, min_face_px=REFERENCE_BUILD_MIN_FACE_PX)

    identity_means: dict[str, np.ndarray] = {}
    identity_records: dict[str, dict] = {}

    for name in FACE_IDENTITIES:
        rows = manifest[name]
        embeddings = []
        sources = []
        for row in rows:
            time.sleep(1.0)  # a courtesy delay; commons.wikimedia.org rate-limits bursts of fetches
            raw = _fetch(row["url"], row["sha256"], cache_dir)
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if actual_sha256 != row["sha256"]:
                raise RuntimeError(
                    f"{row['url']}: downloaded sha256 {actual_sha256} does not match the manifest's "
                    f"pinned {row['sha256']}. The Commons file may have changed since the manifest "
                    "was built, or the download was corrupted — do not use it without re-verifying "
                    "(view the file, update the pin deliberately if the change is benign)."
                )
            img = _decode_image(raw)
            faces = embedder.embed_frames([img])[0]
            if len(faces) == 0:
                raise RuntimeError(
                    f"No face detected in {row['url']} (identity {name!r}). Inspect the image and "
                    "either lower det_threshold or replace this manifest row."
                )
            areas = faces.boxes[:, 2] * faces.boxes[:, 3]
            largest = int(np.argmax(areas))
            embeddings.append(faces.embeddings[largest])
            sources.append(
                {
                    "url": row["url"],
                    "commons_page": row["commons_page"],
                    "sha256": actual_sha256,
                    "license": row["license"],
                    "author": row["author"],
                }
            )

        stacked = np.stack(embeddings)
        mean = stacked.mean(axis=0)
        mean = mean / np.linalg.norm(mean)
        per_image_cos = (stacked @ mean).tolist()

        worst = min(per_image_cos)
        if worst < MIN_PER_IMAGE_COSINE:
            worst_idx = per_image_cos.index(worst)
            raise RuntimeError(
                f"{name}: {sources[worst_idx]['url']} has cosine {worst:.3f} to the identity mean, "
                f"below the {MIN_PER_IMAGE_COSINE} gate. This is very likely a mis-cropped or "
                "wrong-person photo (see this script's docstring for a real prior instance) — "
                "visually inspect the aligned crop before trusting or swapping this row."
            )

        identity_means[name] = mean
        identity_records[name] = {"sources": sources, "per_image_cos": per_image_cos}
        print(f"{name}: n={len(rows)}, per-image cos to mean = {[round(c, 3) for c in per_image_cos]}")

    names = sorted(identity_means)
    inter_identity_cos = {
        a: {b: float(identity_means[a] @ identity_means[b]) for b in names} for a in names
    }
    worst_pair, worst_cos = None, 0.0
    for a in names:
        for b in names:
            if a != b and abs(inter_identity_cos[a][b]) > worst_cos:
                worst_pair, worst_cos = (a, b), abs(inter_identity_cos[a][b])
    if worst_cos >= MAX_INTER_IDENTITY_COSINE:
        raise RuntimeError(
            f"Inter-identity cosine between {worst_pair[0]!r} and {worst_pair[1]!r} is "
            f"{worst_cos:.3f}, at/above the {MAX_INTER_IDENTITY_COSINE} gate — these two identities "
            "are not well separated by this reference set, which would make ID-similarity "
            "meaningless before a single video is generated. Inspect both identities' source photos."
        )
    print(f"Inter-identity cosine matrix (max |off-diagonal| = {worst_cos:.3f}):")
    header = "              " + " ".join(f"{n[:10]:>10}" for n in names)
    print(header)
    for a in names:
        print(f"{a[:14]:>14} " + " ".join(f"{inter_identity_cos[a][b]:10.3f}" for b in names))

    output = {
        "model": {
            "rec": {
                "repo_id": "immich-app/buffalo_l",
                "filename": "recognition/model.onnx",
                "sha256": embedder.rec_sha256,
            },
            "det": {
                "name": "face_detection_yunet_2023mar.onnx",
                "sha256": embedder.det_sha256,
            },
        },
        "gates": {
            "min_per_image_cosine": MIN_PER_IMAGE_COSINE,
            "max_inter_identity_cosine": MAX_INTER_IDENTITY_COSINE,
        },
        "identities": {
            identity_slug(name): {
                "name": name,
                "embedding": identity_means[name].tolist(),
                **identity_records[name],
            }
            for name in names
        },
        "inter_identity_cos": inter_identity_cos,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST_CSV)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--cache_dir", default=None,
        help="optional local by-sha256 cache to avoid re-fetching on repeat rebuilds; off by default",
    )
    args = parser.parse_args()
    build(args.manifest, args.output, cache_dir=args.cache_dir)
