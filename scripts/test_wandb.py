"""Fake experiment to verify wandb logging works (no GPU/model required)."""

import math
import random
import time

import wandb

EXPERIMENT_NAME = "test_wandb_fake_run"
STEPS = 50
EVAL_INTERVAL = 10


def fake_train_loss(step: int) -> float:
    """Simulates a decaying loss with noise."""
    return math.exp(-step / 20) + random.gauss(0, 0.02)


def fake_eval_metrics(step: int) -> dict[str, dict[str, float]]:
    """Simulates eval metrics improving over training."""
    progress = step / STEPS
    return {
        "concept": {
            "fire_detection_rate": max(0.0, 0.9 - progress * 0.7 + random.gauss(0, 0.05)),
            "clip_score_mean": 0.3 + random.gauss(0, 0.01),
        },
        "related": {
            "fire_detection_rate": max(0.0, 0.5 - progress * 0.2 + random.gauss(0, 0.05)),
            "clip_score_mean": 0.28 + random.gauss(0, 0.01),
        },
        "unrelated": {
            "fire_detection_rate": 0.05 + random.gauss(0, 0.02),
            "clip_score_mean": 0.32 + random.gauss(0, 0.01),
        },
    }


def main() -> None:
    wandb.init(
        project="zml",
        name=EXPERIMENT_NAME,
        config={
            "model_id": "fake/model",
            "lora_rank": 8,
            "lora_alpha": 8.0,
            "negative_guidance_scale": 2.0,
            "steps": STEPS,
            "learning_rate": 0.0002,
            "note": "fake run for wandb smoke test",
        },
    )

    print(f"wandb run URL: {wandb.run.url}")

    for step in range(STEPS):
        loss = fake_train_loss(step)
        wandb.log({"train/loss": loss}, step=step)
        print(f"Step {step:3d} | loss={loss:.4f}")

        if (step + 1) % EVAL_INTERVAL == 0:
            metrics = fake_eval_metrics(step + 1)
            wandb.log(
                {
                    f"eval/{set_name}_{metric}": value
                    for set_name, scores in metrics.items()
                    for metric, value in scores.items()
                },
                step=step,
            )
            print(f"  [eval] {metrics}")

        time.sleep(0.05)  # slight delay to make the run visible in the UI

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
