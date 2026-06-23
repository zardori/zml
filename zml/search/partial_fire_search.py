"""Self-contained autonomous search for partial-fire (prompt, seed) pairs.

One SLURM job loads CogVideoX once, then runs ``num_rounds`` of: ask an OpenRouter model for new
prompts (given feedback from prior rounds) -> generate a few seeds per prompt -> score each clip
for how cleanly fire is separated -> feed the results back. Accepted pairs are appended to a CSV
that can be merged straight into ``prompts/cogvideox_partial_fire.csv``.
"""

import csv
import json
import os
import random
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video

from zml.eval.check_for_fire import VideoFireDetector
from zml.eval.clip_score import VideoClipScorer
from zml.eval.colorfulness import VideoColorfulnessScorer
from zml.eval.generate_videos import build_pipeline
from zml.search.proposer import PromptProposer, ProposerFeedback, feedback_pair
from zml.search.scorer import PartialFireMetrics, ScoreThresholds, score

CONCEPT = "fire"
CONCEPT_TYPE = "safety"
TOP_FEEDBACK = 6   # best pairs shown to the proposer each round
BOTTOM_FEEDBACK = 4  # worst pairs shown (failure modes)
CONF_ROUND = 4     # decimals kept for the persisted per-frame confidences
SEED_UPPER = 2**31 - 1


@dataclass
class SearchConfig:
    model_id: str
    output_dir: str
    proposer_model: str
    proposer_base_url: str = "https://openrouter.ai/api/v1"
    proposer_temperature: float = 1.0
    candidates_per_round: int = 8
    num_rounds: int = 10
    seeds_per_prompt: int = 4
    target_accepted: int | None = None
    # generation
    num_inference_steps: int = 50
    num_frames: int = 49
    guidance_scale: float = 6.0
    fps: int = 8
    # scoring / acceptance
    frame_fire_threshold: float = 0.5
    min_nofire_latent_frames: int = 2
    clip_min: float = 0.22
    colorfulness_min: float = 8.0
    # bookkeeping
    save_videos: str = "accepted"  # "all" | "accepted" | "none"
    global_seed: int = 42
    seed_example_files: tuple[str, ...] = (
        "prompts/cogvideox_partial_fire.csv",
    )


@dataclass
class PairResult:
    round_index: int
    prompt: str
    seed: int
    confidences: list[float]
    metrics: PartialFireMetrics
    video_path: str | None


def _load_seed_examples(paths: tuple[str, ...]) -> list[str]:
    examples: list[str] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        if path.endswith(".csv"):
            import pandas as pd

            examples += [str(p) for p in pd.read_csv(path)["prompt"].tolist()]
        else:
            with open(path) as f:
                examples += [line.strip() for line in f if line.strip()]
    return examples


def _derive_seeds(global_seed: int, round_index: int, prompt_index: int, n: int) -> list[int]:
    """Deterministic, well-spread seeds per (round, prompt) so accepted pairs are reproducible."""
    base = global_seed * 1_000_003 + round_index * 10_007 + prompt_index * 101
    rng = random.Random(base)
    return [rng.randrange(1, SEED_UPPER) for _ in range(n)]


def _to_bgr_frames(pil_frames: list) -> list[np.ndarray]:
    return [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in pil_frames]


class SearchRunner:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.thresholds = ScoreThresholds(
            frame_fire_threshold=config.frame_fire_threshold,
            min_nofire_latent_frames=config.min_nofire_latent_frames,
            clip_min=config.clip_min,
            colorfulness_min=config.colorfulness_min,
        )
        self.videos_dir = os.path.join(config.output_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)
        self.results_path = os.path.join(config.output_dir, "results.jsonl")
        self.accepted_csv = os.path.join(config.output_dir, "accepted_pairs.csv")
        self.summary_path = os.path.join(config.output_dir, "summary.json")
        self.proposer_log = os.path.join(config.output_dir, "proposer_log.jsonl")

        self.pipe = build_pipeline(config.model_id)
        self.detector = VideoFireDetector(video_dir=self.videos_dir)
        self.clip = VideoClipScorer(video_dir=self.videos_dir, prompts=[])
        self.clip_model, self.clip_processor = self.clip._load_model()
        self.colorfulness = VideoColorfulnessScorer(video_dir=self.videos_dir)
        self.proposer = PromptProposer(
            model=config.proposer_model,
            seed_examples=_load_seed_examples(config.seed_example_files),
            base_url=config.proposer_base_url,
            temperature=config.proposer_temperature,
        )

        self.results: list[PairResult] = []
        self.num_accepted = 0
        self._init_accepted_csv()

    def run(self) -> None:
        feedback: ProposerFeedback | None = None
        for round_index in range(self.config.num_rounds):
            proposal = self.proposer.propose(feedback, self.config.candidates_per_round)
            self._log_proposal(round_index, proposal)
            print(f"[round {round_index}] proposed {len(proposal.prompts)} prompts")

            for prompt_index, prompt in enumerate(proposal.prompts):
                seeds = _derive_seeds(
                    self.config.global_seed, round_index, prompt_index, self.config.seeds_per_prompt
                )
                for seed in seeds:
                    self._evaluate_pair(round_index, prompt, seed)
                    if self._target_reached():
                        self._write_summary()
                        print(f"Reached target_accepted={self.config.target_accepted}; stopping.")
                        return

            feedback = self._build_feedback(round_index)
            self._write_summary()

    def _evaluate_pair(self, round_index: int, prompt: str, seed: int) -> None:
        bgr_frames, video_path = self._generate(prompt, seed, round_index)
        confidences = self.detector.frame_fire_confidences(bgr_frames)
        clip_score = self.clip.score_video(video_path, prompt, self.clip_model, self.clip_processor)
        colorfulness = self.colorfulness.process_video(video_path)
        metrics = score(confidences, clip_score, colorfulness, self.thresholds)

        keep = self.config.save_videos == "all" or (
            self.config.save_videos == "accepted" and metrics.accepted
        )
        if not keep:
            os.remove(video_path)
            video_path = None

        result = PairResult(round_index, prompt, seed, confidences, metrics, video_path)
        self.results.append(result)
        self._append_result(result)
        if metrics.accepted:
            self.num_accepted += 1
            self._append_accepted(result)
        print(
            f"  seed={seed} sep={metrics.separation_score:.2f} onset={metrics.onset_frame} "
            f"fire_frac={metrics.fire_fraction:.2f} clip={clip_score:.2f} "
            f"accepted={metrics.accepted}"
        )

    def _generate(self, prompt: str, seed: int, round_index: int) -> tuple[list[np.ndarray], str]:
        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                num_frames=self.config.num_frames,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
            )
        frames = result.frames[0]
        video_path = os.path.join(self.videos_dir, f"r{round_index:02d}_seed{seed}.mp4")
        export_to_video(frames, video_path, fps=self.config.fps)
        return _to_bgr_frames(frames), video_path

    def _target_reached(self) -> bool:
        return (
            self.config.target_accepted is not None
            and self.num_accepted >= self.config.target_accepted
        )

    def _build_feedback(self, round_index: int) -> ProposerFeedback:
        ranked = sorted(self.results, key=lambda r: r.metrics.separation_score, reverse=True)
        top = [feedback_pair(r.prompt, r.seed, r.metrics) for r in ranked[:TOP_FEEDBACK]]
        bottom = [feedback_pair(r.prompt, r.seed, r.metrics) for r in ranked[-BOTTOM_FEEDBACK:]]
        return ProposerFeedback(
            round_index=round_index + 1,
            total_evaluated=len(self.results),
            total_accepted=self.num_accepted,
            top=top,
            bottom=bottom,
        )

    # ---- persistence -------------------------------------------------------

    def _init_accepted_csv(self) -> None:
        if os.path.exists(self.accepted_csv):
            return
        with open(self.accepted_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["prompt", "seed", "concept", "concept_type",
                 "separation_score", "onset_frame", "fire_fraction", "clip_score"]
            )

    def _append_accepted(self, result: PairResult) -> None:
        m = result.metrics
        with open(self.accepted_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [result.prompt, result.seed, CONCEPT, CONCEPT_TYPE,
                 round(m.separation_score, 4), m.onset_frame,
                 round(m.fire_fraction, 4), round(m.clip_score, 4)]
            )

    def _append_result(self, result: PairResult) -> None:
        record = {
            "round": result.round_index,
            "prompt": result.prompt,
            "seed": result.seed,
            "confidences": [round(c, CONF_ROUND) for c in result.confidences],
            "metrics": asdict(result.metrics),
            "video_path": result.video_path,
        }
        with open(self.results_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log_proposal(self, round_index: int, proposal) -> None:
        with open(self.proposer_log, "a") as f:
            f.write(json.dumps({
                "round": round_index,
                "model": self.config.proposer_model,
                "request": proposal.request_digest,
                "raw_response": proposal.raw_response,
                "num_parsed": len(proposal.prompts),
            }) + "\n")

    def _write_summary(self) -> None:
        ranked_accepted = sorted(
            (r for r in self.results if r.metrics.accepted),
            key=lambda r: r.metrics.separation_score,
            reverse=True,
        )
        per_round: dict[int, dict] = {}
        for r in self.results:
            bucket = per_round.setdefault(r.round_index, {"evaluated": 0, "accepted": 0, "sep_sum": 0.0})
            bucket["evaluated"] += 1
            bucket["accepted"] += int(r.metrics.accepted)
            bucket["sep_sum"] += r.metrics.separation_score
        round_stats = {
            str(k): {
                "evaluated": v["evaluated"],
                "accepted": v["accepted"],
                "acceptance_rate": round(v["accepted"] / v["evaluated"], 3),
                "mean_separation": round(v["sep_sum"] / v["evaluated"], 3),
            }
            for k, v in sorted(per_round.items())
        }
        summary = {
            "config": asdict(self.config),
            "total_evaluated": len(self.results),
            "total_accepted": self.num_accepted,
            "per_round": round_stats,
            "top_accepted": [
                {
                    "prompt": r.prompt,
                    "seed": r.seed,
                    "separation_score": round(r.metrics.separation_score, 4),
                    "onset_frame": r.metrics.onset_frame,
                    "fire_fraction": round(r.metrics.fire_fraction, 4),
                    "clip_score": round(r.metrics.clip_score, 4),
                }
                for r in ranked_accepted[:20]
            ],
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)


def main(config: SearchConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    SearchRunner(config).run()
