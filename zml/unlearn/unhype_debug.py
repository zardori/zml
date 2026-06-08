"""Debug-only control modes for UnHype, dispatched from ``unhype.main`` by ``target_mode``.

These do not train the real online gradient-matching objective; they exist to isolate the
adapter apply/eval path from the (historically broken) online learning signal:

- ``distill`` — regress the hypernet's endpoint output H(c, S) directly onto a known-good
  erasing adapter θ*. A static, variance-free target that tests whether the hypernet can
  even reproduce a working adapter, independent of the noisy online finite-difference target.
- ``static_apply`` — inject θ* directly through the decode/apply path (no hypernet training)
  and eval once. A confound-free test of the apply/eval path: if fire survives here despite
  θ* having erased it as a plain PEFT adapter, the wiring (decode layout, alpha/rank scaling,
  or eval conditioning) is at fault rather than the learning signal.

Both share the heavy setup via ``UnhypeContext`` and reuse the online retain-loss/optimizer
tail (``apply_optimizer_step``) and eval block (``run_eval_checkpoint``)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from zml.unlearn.eval import evaluate
from zml.unlearn.unhype import (
    Config,
    UnhypeContext,
    apply_optimizer_step,
    run_eval_checkpoint,
    _tensor_norm,
)
from zml.unlearn.unhype_modules import clear_hypernet_output, load_peft_adapter_as_flat


def load_theta_star(ctx: UnhypeContext, config: Config) -> torch.Tensor:
    """Load the known-good erasing adapter θ* and flatten it to the hypernet's output layout.
    Shared by both debug modes."""
    if not config.distill_adapter_dir:
        raise ValueError(f"target_mode={config.target_mode!r} requires distill_adapter_dir")
    theta_star = load_peft_adapter_as_flat(
        config.distill_adapter_dir, ctx.lora_shapes, config.lora_rank
    ).to(ctx.device)
    if theta_star.numel() != ctx.hypernet.total_output:
        raise ValueError(
            f"theta_star dim {theta_star.numel()} != hypernet flat output {ctx.hypernet.total_output} "
            "(check lora_rank/lora_alpha and target_modules match the adapter)"
        )
    print(f"Target θ* loaded: {theta_star.numel():,} params, norm {float(theta_star.norm()):.4f}")
    return theta_star


def run_static_apply(ctx: UnhypeContext, config: Config) -> None:
    """Inject θ* into every adapter through the same decode/apply path the hypernet uses
    (constant across prompts), then eval once and stop. If fire drops here, the apply/eval
    path is sound and every past failure is upstream (the online learning signal); if fire
    survives despite θ* having erased it as a direct PEFT adapter, there is a wiring bug
    (decode layout, alpha/rank scaling, or eval conditioning)."""
    theta_star = load_theta_star(ctx, config)
    print(f"Static-apply eval: injecting θ* (norm {float(theta_star.norm()):.4f}) into all adapters.")

    def prepare_for_prompt(_prompt: str) -> None:
        ctx.apply_flat(theta_star)

    eval_metrics = evaluate(
        ctx.pipe, ctx.transformer, config, ctx.S,
        ctx.control_concept, ctx.control_related, ctx.control_unrelated,
        prepare_for_prompt=prepare_for_prompt,
    )
    clear_hypernet_output(ctx.hyper_modules)
    ctx.recorder.log_eval(ctx.S, {
        "theta_star_norm": float(theta_star.norm()),
        "scores": {
            set_name: {
                "fire_detection_rate": s["fire_detection_rate"],
                "clip_score_mean": s["clip_score_mean"],
                "colorfulness_mean": s["colorfulness_mean"],
                "dover_technical_mean": s["dover_technical_mean"],
                "dover_aesthetic_mean": s["dover_aesthetic_mean"],
            }
            for set_name, s in eval_metrics.items()
        },
    })
    ctx.recorder.close()
    print("Static-apply eval complete.")


def run_distill(ctx: UnhypeContext, config: Config) -> None:
    """Regress the hypernet endpoint H(c, S) onto θ* with a plain MSE loss. No diffusion
    forward in the loop, so the ~21 GB diffusion stack is offloaded to CPU (the rank-8
    hypernet's weight+grad otherwise overflows the GPU at backward) and brought back only
    for the periodic eval."""
    theta_star = load_theta_star(ctx, config)
    ctx.move_diffusion_stack("cpu")

    print("Starting UnHype training...")
    pbar = tqdm(range(config.steps))
    for step in pbar:
        s = ctx.S
        target_prompt = random.choice(ctx.target_mapping)[0]
        c_target_clip = ctx.encode_clip(target_prompt)
        theta_s = ctx.hypernet_predict(c_target_clip, s)
        loss_remove = F.mse_loss(theta_s.float(), theta_star.float())
        loss_task_value = 0.0  # no diffusion forward in distill mode
        remove_metrics = {
            "train/theta_s_norm": _tensor_norm(theta_s),
            "train/theta_star_norm": _tensor_norm(theta_star),
            "train/distill_cosine": float(
                F.cosine_similarity(theta_s.float(), theta_star.float(), dim=0).detach()
            ),
        }

        apply_optimizer_step(ctx, config, step, s, loss_remove, loss_task_value, remove_metrics, pbar)

        if (step + 1) % config.save_interval == 0:
            run_eval_checkpoint(ctx, config, step + 1, offload_diffusion=True)

    ctx.recorder.close()
    print("UnHype training complete.")
