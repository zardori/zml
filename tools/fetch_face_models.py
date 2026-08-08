"""Fetch the two ONNX models the face/identity axis needs: a face detector and ArcFace.

Neither ships as a Python package, so both are downloaded once and cached rather than committed
(``w600k_r50.onnx`` alone is ~170 MB). ``zml/benchmarks/arcface_embedder.py`` loads them from the
same locations this script writes to, and cross-checks their sha256 against
``zml/benchmarks/data/face_reference_embeddings.json``'s recorded provenance at every run — a
silently swapped checkpoint would corrupt every ID-similarity number while still producing
plausible-looking results (the same failure mode ``imagenet_classifier._assert_class_indices``
guards against).

Models
------
**Detection + 5-point alignment landmarks** — YuNet (``face_detection_yunet_2023mar.onnx``), from
the OpenCV Zoo. Not on Hugging Face, so fetched by direct URL and cached under ``$HF_HOME`` (the
same env var ``slurm/*.sh`` already sets to ``hf_cache/`` for every other model this project uses)
so it lives next to everything ``hf_hub_download`` fetches without a second cache directory to
manage.

**Recognition** — ArcFace ``w600k_r50`` (the ``buffalo_l`` pack's recognition model), via
``hf_hub_download`` from ``immich-app/buffalo_l`` — a maintained mirror of the standard insightface
checkpoint (Immich, a photo-management app, depends on it for its own face recognition). Respects
``$HF_HOME`` automatically, like every other ``hf_hub_download`` call in this project.

**License note for the paper**: the insightface checkpoint's license
(https://github.com/deepinsight/insightface/tree/master/python-package#license) permits
non-commercial research use, which is what this project is. This is a model-licensing question,
distinct from the source-photo licensing tracked in ``prompts/face_reference_images.csv``.

Compute nodes may have no outbound network, so — like exp080's required merge step — run this once
on each cluster's **login** node before submitting a face job; ``slurm/check_config_paths.sh`` can't
see a HF cache directory, so a missing model fails inside the job instead of at submission time.

Run:
    uv run python tools/fetch_face_models.py
"""

import argparse
import hashlib
import os
import urllib.request

from huggingface_hub import hf_hub_download

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_LOCAL_NAME = "face_detection_yunet_2023mar.onnx"

ARCFACE_REPO_ID = "immich-app/buffalo_l"
ARCFACE_FILENAME = "recognition/model.onnx"
ARCFACE_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"


def _hf_home() -> str:
    return os.environ.get("HF_HOME", "hf_cache")


def yunet_local_path() -> str:
    """Where ``arcface_embedder.py`` expects the YuNet weights, alongside the HF hub cache."""
    return os.path.join(_hf_home(), "zml_face_models", YUNET_LOCAL_NAME)


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_sha256(path: str, expected: str, what: str) -> None:
    actual = _sha256_of_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{what} at {path} has sha256 {actual}, expected {expected}. The upstream file may have "
            "changed, or the download was corrupted — do not use it for scoring until this is resolved."
        )


def fetch_yunet(force: bool = False) -> str:
    out_path = yunet_local_path()
    if os.path.exists(out_path) and not force:
        _assert_sha256(out_path, YUNET_SHA256, "YuNet (cached)")
        print(f"YuNet already present and verified: {out_path}")
        return out_path

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with urllib.request.urlopen(YUNET_URL, timeout=120) as response:
        data = response.read()
    with open(tmp_path, "wb") as f:
        f.write(data)
    _assert_sha256(tmp_path, YUNET_SHA256, "YuNet (freshly downloaded)")
    os.replace(tmp_path, out_path)
    print(f"YuNet downloaded and verified: {out_path}")
    return out_path


def fetch_arcface() -> str:
    path = hf_hub_download(repo_id=ARCFACE_REPO_ID, filename=ARCFACE_FILENAME)
    _assert_sha256(path, ARCFACE_SHA256, "ArcFace w600k_r50")
    print(f"ArcFace w600k_r50 present and verified: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download YuNet even if cached")
    args = parser.parse_args()
    fetch_yunet(force=args.force)
    fetch_arcface()
