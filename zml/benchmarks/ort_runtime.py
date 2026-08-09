"""Thread-capping helpers shared by every onnxruntime-backed detector.

Extracted from ``check_for_nudity.py`` so a second ONNX consumer (``arcface_embedder.py``) doesn't
duplicate the fix or, worse, forget it. The problem this works around: an ``InferenceSession`` built
with no explicit ``SessionOptions`` lets onnxruntime default ``intra_op_num_threads`` to the machine's
core count. On helios' 288-core GH200 that spawns ~288 threads to run a small model, and thread
dispatch dominates the actual inference — see ``_bounded_ort_sessions``'s docstring for the measured
numbers (NudeNet: 288x1 103s vs 1x16 11.9s over 30 clips). OpenCV's DNN backend (used by
``cv2.FaceDetectorYN``) has the identical pathology, so ``bound_opencv_threads`` covers it too.
"""

import os
from contextlib import contextmanager
from typing import Iterator

import cv2
import onnxruntime

# Intra-op threading is pure dispatch overhead for the small models this project scores with
# (NudeNet's 320x320 nano model, YuNet, ArcFace at 112x112) — see the measurement above. The
# parallelism that pays is one process per video/image, not a wide thread pool per session.
ORT_INTRA_OP_THREADS = 1
# ``sched_getaffinity`` rather than ``cpu_count`` so a SLURM cgroup's actual allocation is respected
# — ignoring that mask is what produces the pthread_setaffinity_np storm this module works around.
DEFAULT_NUM_WORKERS = min(
    32, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
)


@contextmanager
def bounded_ort_sessions(num_threads: int = ORT_INTRA_OP_THREADS) -> Iterator[None]:
    """Force every ``InferenceSession`` built inside the block onto a bounded thread pool.

    For a detector that constructs its own session with no ``SessionOptions`` hook (NudeNet). A
    caller that builds its own session directly should instead pass
    ``sess_options=cpu_session_options()`` and skip this context manager entirely.
    """
    original = onnxruntime.InferenceSession

    def bounded(*args, **kwargs):
        options = kwargs.pop("sess_options", None) or onnxruntime.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        return original(*args, sess_options=options, **kwargs)

    onnxruntime.InferenceSession = bounded
    try:
        yield
    finally:
        onnxruntime.InferenceSession = original


def cpu_session_options(
    num_threads: int = ORT_INTRA_OP_THREADS, log_severity_level: int | None = None
) -> onnxruntime.SessionOptions:
    """``SessionOptions`` for a caller that constructs its own ``InferenceSession`` directly.

    ``log_severity_level`` (0=verbose .. 4=fatal) is left at onnxruntime's default (2=warning)
    unless set — pass 3 to silence a benign per-call warning some ONNX exports trigger when the
    declared output shape uses a literal batch dim instead of a symbolic one (harmless: the actual
    batch still runs and returns the right shape, onnxruntime just logs about its own static
    shape-inference metadata being stale).
    """
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    if log_severity_level is not None:
        options.log_severity_level = log_severity_level
    return options


def bound_opencv_threads(num_threads: int = ORT_INTRA_OP_THREADS) -> None:
    """Cap OpenCV's own thread pool (DNN backend included) for the same reason as ``cpu_session_options``.

    Process-global, not scoped — matches ``cv2``'s own API (``cv2.setNumThreads`` has no context-manager
    form). Call once per process before constructing a ``cv2.FaceDetectorYN`` or similar.
    """
    cv2.setNumThreads(num_threads)
