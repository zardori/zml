"""ESR/PSR evaluation for ImageNet object erasure (``docs/imagenet_objects.md``).

The protocol comes from ESD via T2VUnlearning (arXiv 2505.17550 §4.2): erase one of ten ImageNet
classes, generate videos for *all* ten, classify every frame, and report

    ESR-k = 1 - top-k accuracy of the erased class      (erasure success)
    PSR-k = mean top-k accuracy of the other nine       (preservation success)

both as percentages. This does not fit ``zml/unlearn/eval.py::evaluate``, which scores one detector
across ``concept``/``related``/``unrelated`` sets: here there are ten sets, each scored against *its
own* target class. Hence a separate eval mode (``mode: imagenet``).

With ``erased_class`` unset the run scores the unmodified base model and reports ESR/PSR for every
class in turn as the hypothetical erased one, plus mean and std across the ten — which is exactly how
the papers' ``Original`` row and its ± are produced, from a single generation pass.

Generation is resumable: an existing non-empty video file is not regenerated, so a job that hits its
SLURM wall clock can be resubmitted and pick up where it stopped.
"""

import json
import os
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import torch
import wandb
from diffusers.utils import export_to_video

from zml.benchmarks.check_for_object import VideoObjectDetector
from zml.benchmarks.imagenet_classes import IMAGENETTE_CLASSES, class_slug
from zml.benchmarks.imagenet_classifier import ImageNetFrameClassifier
from zml.eval.clip_score import VideoClipScorer
from zml.eval.colorfulness import VideoColorfulnessScorer
from zml.eval.eval_model import build_eval_pipeline
from zml.eval.motion import VideoMotionScorer

# Standalone eval runs at a synthetic step so outputs share the eval_step_<n>/ layout.
EVAL_STEP = 0
VIDEO_FPS = 8
REQUIRED_COLUMNS = ("prompt", "seed", "class_name")
# k values of the protocol's ESR-k / PSR-k. Fixed rather than configurable: they define the metric
# every published table reports, and a list-valued config field would trip submit_job.py's grid search.
TOP_K = 5
# `negative_prompt: auto` expands to the erased class name — the NegPrompt baseline.
AUTO_NEGATIVE_PROMPT = "auto"


@dataclass
class Config:
    model_id: str
    output_dir: str
    eval_inference_steps: int
    prompts_csv: str  # columns: prompt, seed, class_name (+ optional class_idx, ignored)
    # Which class this model was trained to erase. None -> base model; ESR/PSR is then reported for
    # every class in turn, which fills the whole `Original` comparison row from one run.
    erased_class: str | None = None
    lora_checkpoint_dir: str | None = None
    # Inference-time negative prompt. "auto" resolves to `erased_class` (the NegPrompt baseline);
    # any other string is passed through verbatim.
    negative_prompt: str | None = None
    eval_num_prompts_per_class: int | None = None  # None -> every row of each class
    num_frames: int = 49
    guidance_scale: float = 6.0
    disable_mlflow: bool = False

    def __post_init__(self) -> None:
        if self.erased_class is not None and self.erased_class not in IMAGENETTE_CLASSES:
            raise ValueError(
                f"erased_class {self.erased_class!r} is not one of {sorted(IMAGENETTE_CLASSES)}."
            )
        if self.negative_prompt == AUTO_NEGATIVE_PROMPT and self.erased_class is None:
            raise ValueError("negative_prompt: auto needs erased_class to name what to negate.")

    def resolved_negative_prompt(self) -> str | None:
        if self.negative_prompt == AUTO_NEGATIVE_PROMPT:
            return self.erased_class
        return self.negative_prompt


@dataclass
class ClassScores:
    """Frame-pooled classification of every clip generated for one class."""

    top1: float
    top5: float
    num_videos: int


def load_class_prompts(csv_path: str, limit: int | None = None) -> dict[str, list[tuple[str, int]]]:
    """Group a prompt CSV into ``{class_name: [(prompt, seed), ...]}``, in file order.

    Seeds come from the file per the repo seed policy, so every run scores identical (prompt, seed)
    pairs and numbers stay comparable across experiments.
    """
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s) {sorted(missing)}.")

    unknown = set(df["class_name"]) - set(IMAGENETTE_CLASSES)
    if unknown:
        raise ValueError(f"{csv_path} has class_name values outside the protocol: {sorted(unknown)}.")

    grouped: dict[str, list[tuple[str, int]]] = {}
    for class_name, rows in df.groupby("class_name", sort=False):
        pairs = [(str(p), int(s)) for p, s in zip(rows["prompt"], rows["seed"])]
        grouped[str(class_name)] = pairs[:limit] if limit else pairs
    return grouped


def _generate_class_videos(pipe, config: Config, class_name: str, prompts: list[tuple[str, int]],
                           eval_root: str) -> str:
    video_dir = os.path.join(eval_root, class_slug(class_name))
    os.makedirs(video_dir, exist_ok=True)
    negative_prompt = config.resolved_negative_prompt()

    with torch.no_grad():
        for i, (prompt, seed) in enumerate(prompts):
            video_path = os.path.join(video_dir, f"video_{i}.mp4")
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                continue  # resume a killed job without paying for it again
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=config.num_frames,
                num_inference_steps=config.eval_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device=pipe.device).manual_seed(seed),
            )
            export_to_video(result.frames[0], video_path, fps=VIDEO_FPS)
            print(f"Saved eval video: {video_path}")
    return video_dir


def compute_esr_psr(per_class: dict[str, ClassScores], erased_class: str) -> dict[str, float]:
    """ESR/PSR percentages for one choice of erased class."""
    others = [s for name, s in per_class.items() if name != erased_class]
    if not others:
        raise ValueError("PSR needs at least one preserved class.")
    return {
        "ESR-1": 100.0 * (1.0 - per_class[erased_class].top1),
        "ESR-5": 100.0 * (1.0 - per_class[erased_class].top5),
        "PSR-1": 100.0 * float(np.mean([s.top1 for s in others])),
        "PSR-5": 100.0 * float(np.mean([s.top5 for s in others])),
    }


def _leave_one_out_report(per_class: dict[str, ClassScores]) -> dict:
    """ESR/PSR with each class in turn as the erased one, plus mean/std across the ten.

    This is what the published ``Original`` row reports: the ± spread comes from varying which class
    counts as erased, not from repeated sampling.
    """
    rows = {name: compute_esr_psr(per_class, name) for name in per_class}
    metrics = ["ESR-1", "ESR-5", "PSR-1", "PSR-5"]
    return {
        "per_erased_class": rows,
        "mean": {m: float(np.mean([r[m] for r in rows.values()])) for m in metrics},
        "std": {m: float(np.std([r[m] for r in rows.values()])) for m in metrics},
    }


def _quality_scores(video_dir: str, prompts: list[str]) -> dict[str, float]:
    """Generic quality signals per class, so a collapse in ESR can be read against video quality."""
    clip = VideoClipScorer(video_dir=video_dir, prompts=prompts).process_videos()
    color = VideoColorfulnessScorer(video_dir=video_dir).process_videos()
    motion = VideoMotionScorer(video_dir=video_dir).process_videos()
    return {
        "clip_score_mean": float(np.mean(clip)) if clip else 0.0,
        "colorfulness_mean": float(np.mean(color)) if color else 0.0,
        "motion_score_mean": float(np.mean(motion)) if motion else 0.0,
    }


def main(config: Config) -> dict:
    class_prompts = load_class_prompts(config.prompts_csv, config.eval_num_prompts_per_class)
    eval_root = os.path.join(config.output_dir, f"eval_step_{EVAL_STEP}")
    os.makedirs(eval_root, exist_ok=True)

    pipe = build_eval_pipeline(config.model_id, config.lora_checkpoint_dir)
    pipe.transformer.eval()

    video_dirs = {
        name: _generate_class_videos(pipe, config, name, prompts, eval_root)
        for name, prompts in class_prompts.items()
    }

    # One classifier shared across all ten target classes; loading ResNet-50 per class is wasteful.
    classifier = ImageNetFrameClassifier()
    per_class: dict[str, ClassScores] = {}
    quality: dict[str, dict[str, float]] = {}
    for name, video_dir in video_dirs.items():
        scores = VideoObjectDetector(
            video_dir=video_dir, target_class=name, top_k=TOP_K, classifier=classifier
        ).process_videos()
        per_class[name] = ClassScores(
            top1=scores["object_top1_accuracy"],
            top5=scores["object_top5_accuracy"],
            num_videos=int(scores["total_videos"]),
        )
        quality[name] = _quality_scores(video_dir, [p for p, _ in class_prompts[name]])
        print(f"{name}: top1={per_class[name].top1:.4f} top5={per_class[name].top5:.4f}")

    report: dict = {
        "erased_class": config.erased_class,
        "lora_checkpoint_dir": config.lora_checkpoint_dir,
        "negative_prompt": config.resolved_negative_prompt(),
        "per_class": {
            name: {"top1": round(s.top1, 4), "top5": round(s.top5, 4), "num_videos": s.num_videos}
            for name, s in per_class.items()
        },
    }
    if config.erased_class is not None:
        report.update(compute_esr_psr(per_class, config.erased_class))
        headline = {k: report[k] for k in ("ESR-1", "ESR-5", "PSR-1", "PSR-5")}
    else:
        report.update(_leave_one_out_report(per_class))
        headline = report["mean"]

    report = {**report, "quality": quality}
    with open(os.path.join(config.output_dir, "esr_psr.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"ESR/PSR: {json.dumps(headline, indent=2)}")

    if not config.disable_mlflow:
        for key, value in headline.items():
            mlflow.log_metric(f"eval/{key}", round(value, 2), step=EVAL_STEP)
    wandb.log({f"eval/{k}": round(v, 2) for k, v in headline.items()}, step=EVAL_STEP)

    return report
