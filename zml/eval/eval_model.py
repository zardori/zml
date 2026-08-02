from dataclasses import dataclass

import pandas as pd
import torch
from diffusers import CogVideoXPipeline
from peft import PeftModel

from zml.unlearn.eval import EvalPrompt, evaluate

# Standalone evaluation runs at a synthetic "step" so its outputs share the
# eval_step_<n>/ layout produced during training.
EVAL_STEP = 0


@dataclass
class Config:
    model_id: str
    output_dir: str
    eval_inference_steps: int
    eval_num_prompts: int | None = None  # None means use all prompts
    lora_checkpoint_dir: str | None = None
    # Named control subsets — any combination is valid, each appears under its own key.
    control_concept_prompts: str | None = None
    control_related_prompts: str | None = None
    control_unrelated_prompts: str | None = None
    concept: str = "fire"  # detector to score with at eval: see zml/benchmarks/registry.py
    concept_target: str | None = None  # target within a concept family, e.g. "chain saw" for "object"
    disable_mlflow: bool = False

    def __post_init__(self) -> None:
        if not any([
            self.control_concept_prompts,
            self.control_related_prompts,
            self.control_unrelated_prompts,
        ]):
            raise ValueError("At least one control prompt CSV must be provided.")


def _load_eval_prompts(path: str | None) -> list[EvalPrompt]:
    """Load a prompt CSV into EvalPrompts, using the per-prompt seed baked into the file
    (seed policy). Missing path -> empty set."""
    if path is None:
        return []
    df = pd.read_csv(path)
    return [EvalPrompt(prompt, seed) for prompt, seed in zip(df["prompt"], df["seed"])]


def build_eval_pipeline(model_id: str, lora_checkpoint_dir: str | None = None) -> CogVideoXPipeline:
    """Inference pipeline for evaluation: bf16, VAE slicing+tiling, optional LoRA adapter.

    Shared by every eval mode so a checkpoint is always loaded the same way (an unmatched dtype or a
    missing adapter would silently change what is being scored).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if lora_checkpoint_dir is not None:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_checkpoint_dir)
        print(f"Loaded LoRA checkpoint from {lora_checkpoint_dir}")
    return pipe


def main(config: Config) -> dict:
    concept = _load_eval_prompts(config.control_concept_prompts)
    related = _load_eval_prompts(config.control_related_prompts)
    unrelated = _load_eval_prompts(config.control_unrelated_prompts)

    pipe = build_eval_pipeline(config.model_id, config.lora_checkpoint_dir)

    return evaluate(
        pipe,
        pipe.transformer,
        config,
        EVAL_STEP,
        concept,
        related,
        unrelated,
        log_mlflow=not config.disable_mlflow,
        include_related=config.control_related_prompts is not None,
    )
