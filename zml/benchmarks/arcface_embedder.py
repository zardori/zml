"""Face detection + alignment + ArcFace embedding, shared by every identity in the face concept.

Mirrors ``imagenet_classifier.ImageNetFrameClassifier``: one instance is built once and handed to
every ``VideoFaceDetector`` (one per identity), so the ~170 MB recognition model is loaded once
regardless of how many identities a run scores.

Two ONNX models, both CPU (see ``docs/face_identity.md`` §8 for why aarch64/helios is fine here
even though DOVER is not):

- **Detection + 5-point landmarks** — YuNet (``cv2.FaceDetectorYN``), already available via
  ``opencv-python`` — no new dependency. Its five landmarks (right eye, left eye, nose, right mouth
  corner, left mouth corner) are exactly ArcFace's alignment template, in the same order.
- **Recognition** — ArcFace ``w600k_r50``, run directly through ``onnxruntime`` (not the
  ``insightface`` package, which would pull in its own detector/alignment stack we don't need).

Both weight files are fetched by ``tools/fetch_face_models.py``; this module locates them the same
way and refuses to run scoring against a checkpoint that doesn't match the reference embeddings'
recorded provenance (``zml/benchmarks/face_identities.expected_model_sha256``).
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime

from zml.benchmarks.face_identities import expected_model_sha256, sha256_of_file
from zml.benchmarks.ort_runtime import bound_opencv_threads, cpu_session_options

# Standard ArcFace 112x112 alignment template (right eye, left eye, nose, right mouth corner, left
# mouth corner) — insightface's ``arcface_dst``. Every embedding this module produces is aligned
# onto this template, which is what makes cosines from different photos/frames comparable at all;
# an unaligned crop moves cosines by roughly 0.1.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
ARCFACE_INPUT_SIZE = 112

# Below this, at 480x720 CogVideoX frames, ArcFace's embedding is unreliable — an extreme wide shot
# gives a face too small to embed meaningfully. Faces below this floor are treated as "not detected"
# (they count toward ``face_present_rate`` failing, not toward the identity-similarity pool).
DEFAULT_MIN_FACE_PX = 48
# YuNet's own score threshold; separate from the identity-similarity threshold applied downstream.
DEFAULT_DET_THRESHOLD = 0.6


@dataclass
class FrameFaces:
    """Every face embedding for one frame, aligned to nothing in particular — a caller compares
    against whichever identity reference it needs via ``ArcFaceFrameEmbedder.cosine``."""

    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 512), dtype=np.float32))
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))  # x, y, w, h
    det_scores: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))

    def __len__(self) -> int:
        return self.embeddings.shape[0]


def _yunet_local_path() -> str:
    # Matches tools/fetch_face_models.py::yunet_local_path without importing it (that module has an
    # argparse __main__ block this import should not trigger side effects from).
    hf_home = os.environ.get("HF_HOME", "hf_cache")
    return os.path.join(hf_home, "zml_face_models", "face_detection_yunet_2023mar.onnx")


def _arcface_local_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="immich-app/buffalo_l", filename="recognition/model.onnx")


class ArcFaceFrameEmbedder:
    """Wraps YuNet + ArcFace; one instance can embed faces from any number of frames/identities."""

    def __init__(
        self,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        det_threshold: float = DEFAULT_DET_THRESHOLD,
        min_face_px: int = DEFAULT_MIN_FACE_PX,
        verify_reference_sha256: bool = True,
    ):
        bound_opencv_threads()
        det_model_path = det_model_path or _yunet_local_path()
        rec_model_path = rec_model_path or _arcface_local_path()
        if not os.path.exists(det_model_path):
            raise FileNotFoundError(
                f"YuNet weights not found at {det_model_path!r}. Run "
                "`uv run python tools/fetch_face_models.py` first (once per cluster, on the login node)."
            )

        self.det_threshold = det_threshold
        self.min_face_px = min_face_px
        self.det_sha256 = sha256_of_file(det_model_path)
        self.rec_sha256 = sha256_of_file(rec_model_path)

        if verify_reference_sha256:
            self._verify_reference_sha256()

        # (0, 0) placeholder; real size is set per-image in _detect via setInputSize, since frames
        # in this project vary between the 480x720 CogVideoX render size and reference-photo sizes.
        self._detector = cv2.FaceDetectorYN.create(det_model_path, "", (0, 0), score_threshold=det_threshold)
        self._session = onnxruntime.InferenceSession(
            rec_model_path,
            # log_severity_level=3: this checkpoint declares a static batch=1 output shape, so
            # onnxruntime warns on every >1-face batch even though it computes the right thing —
            # see cpu_session_options' docstring.
            sess_options=cpu_session_options(log_severity_level=3),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        print(
            f"ArcFaceFrameEmbedder ready (det sha256={self.det_sha256[:12]}…, "
            f"rec sha256={self.rec_sha256[:12]}…, min_face_px={min_face_px})"
        )

    def _verify_reference_sha256(self) -> None:
        """Refuse to score against a checkpoint that doesn't match the committed reference embeddings.

        A silently swapped ArcFace/YuNet weight file would corrupt every ID-similarity number while
        still producing plausible-looking results — the same failure mode
        ``imagenet_classifier._assert_class_indices`` guards against for ResNet-50. ``None`` from
        ``expected_model_sha256`` means no manifest exists yet (the bootstrap case, when
        ``build_face_reference_embeddings.py`` is computing it for the first time), which is not an
        error.
        """
        expected = expected_model_sha256()
        if expected is None:
            return
        mismatches = []
        if expected.get("det") and expected["det"] != self.det_sha256:
            mismatches.append(f"det: manifest={expected['det']}, loaded={self.det_sha256}")
        if expected.get("rec") and expected["rec"] != self.rec_sha256:
            mismatches.append(f"rec: manifest={expected['rec']}, loaded={self.rec_sha256}")
        if mismatches:
            raise RuntimeError(
                "ArcFace/YuNet checkpoint sha256 mismatch against "
                "zml/benchmarks/data/face_reference_embeddings.json: " + "; ".join(mismatches) + ". "
                "Results would not be comparable to the committed reference embeddings — either "
                "restore the original checkpoint or rebuild the reference embeddings with "
                "tools/build_face_reference_embeddings.py using the current one."
            )

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Raw YuNet rows for one BGR frame, filtered to ``min_face_px``. Shape (n, 15): box(4) +
        5 landmarks(10) + score(1)."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return np.empty((0, 15), dtype=np.float32)
        keep = np.minimum(faces[:, 2], faces[:, 3]) >= self.min_face_px
        return faces[keep]

    def _align(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        landmarks = face_row[4:14].reshape(5, 2).astype(np.float32)
        transform, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE_112, method=cv2.LMEDS)
        return cv2.warpAffine(frame, transform, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE), borderValue=0)

    def _embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """L2-normalized 512-d embeddings for a batch of already-aligned 112x112 BGR crops."""
        if not aligned_faces:
            return np.empty((0, 512), dtype=np.float32)
        batch = np.stack(aligned_faces).astype(np.float32)  # (n, 112, 112, 3) BGR
        batch = batch[:, :, :, ::-1]  # BGR -> RGB
        batch = (batch - 127.5) / 127.5
        batch = batch.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        embeddings = self._session.run(None, {self._input_name: batch})[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)

    def embed_frames(self, frames: list[np.ndarray]) -> list[FrameFaces]:
        """Detect + align + embed every face (>= ``min_face_px``) in every frame.

        Frames must be BGR uint8, like the other detectors in this package. Detection runs per
        frame (YuNet has no native batching); the ArcFace forward pass is batched once across every
        face found in every frame, since that is the expensive half.
        """
        per_frame_rows: list[np.ndarray] = [self._detect(frame) for frame in frames]
        aligned: list[np.ndarray] = []
        for frame, rows in zip(frames, per_frame_rows):
            aligned.extend(self._align(frame, row) for row in rows)
        embeddings = self._embed_batch(aligned)

        results: list[FrameFaces] = []
        offset = 0
        for rows in per_frame_rows:
            n = len(rows)
            results.append(
                FrameFaces(
                    embeddings=embeddings[offset : offset + n],
                    boxes=rows[:, :4] if n else np.empty((0, 4), dtype=np.float32),
                    det_scores=rows[:, 14] if n else np.empty((0,), dtype=np.float32),
                )
            )
            offset += n
        return results

    @staticmethod
    def cosine(embeddings: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Cosine similarity of each row in ``embeddings`` against one L2-normalized ``reference``.

        Both operands are assumed already unit-normalized (every embedding this class produces is),
        so this is a plain dot product.
        """
        if embeddings.shape[0] == 0:
            return np.empty(0, dtype=np.float32)
        return embeddings @ reference
