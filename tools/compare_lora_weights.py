"""Compare two LoRA checkpoints in weight space.

Why weight space and not metrics
--------------------------------
Some questions about a training run are about the *update rule*, not about what the model ends up
generating — and for those, eval metrics are a blunt and expensive instrument. exp178 asks whether
``erase_esd_eta`` is a reparameterization of ``retention_weight``: the AdamW update depends only on
the ratio ``r = eta / w`` at initialisation, so two runs with the same ratio should land on the
*same weights*, not merely on similar nudity rates.

That comparison is only legal because our frame_replace runs share a deterministic training stream:
``global_seed`` is set, ``lora_dropout: 0.0``, and evaluation draws from an isolated
``torch.Generator`` per prompt (``zml/unlearn/eval.py``) so it never perturbs the global RNG. Two
runs with matching seed, dataset, ``gradient_accumulation_steps`` and timestep range therefore see
byte-identical data order, timesteps and noise, and any weight difference is attributable to the
hyperparameters under test. **Check those fields match before trusting a number out of this tool** —
it compares whatever it is given and cannot tell a legal comparison from a meaningless one.

Two scales, both read off the data rather than guessed:
  * a *trajectory* distance — one run against itself at a different step — is how far training moves
    on its own, and is the ceiling below which "these are the same run" has to sit;
  * a *known-different* distance — two runs that genuinely differ — is what a real effect looks like.

Runs on CPU in seconds: safetensors only, no model is instantiated.

Run:
    # two checkpoints
    uv run python tools/compare_lora_weights.py A/frame_replace_lora_step40 B/frame_replace_lora_step40
    # two runs across matched steps
    uv run python tools/compare_lora_weights.py A/outputs B/outputs --steps 20,40,60
    # calibration: one run against itself
    uv run python tools/compare_lora_weights.py A/outputs A/outputs --steps 20 --ref-steps 40
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

ADAPTER_FILENAME = "adapter_model.safetensors"
CHECKPOINT_GLOB = "frame_replace_lora_step*"
LORA_A_KEY = "lora_A"
LORA_B_KEY = "lora_B"
COMPONENTS = ("B", "A", "all", "delta")
# Tensors this far apart in relative L2 are the same checkpoint to within float noise; used only to
# label the printed verdict, never to gate anything.
IDENTICAL_REL_L2 = 1e-6


@dataclass(frozen=True)
class TensorDiff:
    """One LoRA tensor compared across two checkpoints."""

    name: str
    rel_l2: float          # ||a - b|| / ||a||
    cosine: float          # cos(a, b), flattened
    norm_ratio: float      # ||b|| / ||a||


@dataclass(frozen=True)
class CheckpointDiff:
    """Aggregate over every LoRA tensor in a checkpoint pair.

    ``rel_l2`` and ``cosine`` are computed on the *concatenation* of all tensors, not averaged over
    per-tensor values: the adapter is one point in parameter space, and a mean over tensors would
    let a few tiny matrices outvote the ones carrying the update.
    """

    label: str
    n_tensors: int
    rel_l2: float
    cosine: float
    norm_ratio: float
    norm_a: float
    norm_b: float
    per_tensor: tuple[TensorDiff, ...]

    @property
    def identical(self) -> bool:
        return self.rel_l2 < IDENTICAL_REL_L2

    def worst(self, k: int) -> tuple[TensorDiff, ...]:
        return tuple(sorted(self.per_tensor, key=lambda d: d.rel_l2, reverse=True)[:k])


def resolve_checkpoint(path: Path, step: int | None) -> Path:
    """Accept a checkpoint dir, a run/outputs dir plus a step, or a path to the safetensors itself."""
    if path.is_file():
        return path
    if step is not None:
        candidate = path / f"frame_replace_lora_step{step}" / ADAPTER_FILENAME
        if not candidate.exists():
            available = sorted(p.name for p in path.glob(CHECKPOINT_GLOB))
            raise SystemExit(
                f"no checkpoint for step {step} under {path}\n"
                f"  available: {', '.join(available) if available else '(none)'}"
            )
        return candidate
    candidate = path / ADAPTER_FILENAME
    if not candidate.exists():
        raise SystemExit(
            f"{path} is not a checkpoint dir (no {ADAPTER_FILENAME}). Pass --steps to select one "
            f"from a run's outputs dir, or point at the checkpoint directly."
        )
    return candidate


def _select(names: list[str], component: str) -> list[str]:
    """Which tensors to compare.

    ``lora_B`` is the default and is the only contamination-free choice. PEFT initialises A randomly
    and B at exactly zero, so B *is* the learned update while A is dominated by an init that every
    run of a given seed shares. Measured on exp080's lr grid at step 20: ||A|| is 21.19 / 21.27 /
    21.50 / 22.27 across a 10x learning-rate range (i.e. essentially constant), while ||B|| is
    0.73 / 1.51 / 2.99 / 6.14 — near-linear in lr. Comparing full weights buries the signal under
    ~93% shared constant and compresses every distance toward zero.

    ``delta`` is the most faithful but slowest: the actual weight change applied to the base model
    is the product B@A, which is what the forward pass sees.
    """
    if component == "all":
        return sorted(names)
    if component == "delta":
        return sorted(n for n in names if LORA_B_KEY in n)
    key = {"B": LORA_B_KEY, "A": LORA_A_KEY}[component]
    return sorted(n for n in names if key in n)


def _delta(state: dict[str, torch.Tensor], b_name: str) -> torch.Tensor:
    """B@A for one module — the weight change the base model actually sees."""
    a_name = b_name.replace(LORA_B_KEY, LORA_A_KEY)
    if a_name not in state:
        raise SystemExit(f"{b_name} has no matching {a_name}; cannot form the B@A delta.")
    return state[b_name].double() @ state[a_name].double()


def compare(a_path: Path, b_path: Path, label: str, component: str = "B") -> CheckpointDiff:
    """Aggregate distance between two adapters, accumulated tensor by tensor.

    The aggregate is the exact concatenated-vector distance — sums of squares and of the dot
    product are additive across tensors — but never materialises the concatenation. That matters
    for ``--component delta``, where each B@A product is a full base-weight-shaped matrix and the
    concatenation would run to billions of elements.
    """
    a, b = load_file(str(a_path)), load_file(str(b_path))
    selected = _select(list(a), component)
    if not selected:
        raise SystemExit(f"no tensors matched --component {component} in {a_path}")
    if missing := sorted(set(selected) - set(b)):
        # A rank or target-module change makes the comparison meaningless rather than merely partial.
        raise SystemExit(
            f"{len(missing)} tensors exist in only one checkpoint (e.g. {missing[:4]}). These "
            f"adapters have different shapes/targets and are not comparable."
        )

    per_tensor: list[TensorDiff] = []
    sq_a = sq_b = sq_diff = dot = 0.0
    for name in selected:
        ta = _delta(a, name) if component == "delta" else a[name].double()
        tb = _delta(b, name) if component == "delta" else b[name].double()
        if ta.shape != tb.shape:
            raise SystemExit(f"shape mismatch on {name}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        ta, tb = ta.flatten(), tb.flatten()
        sq_a += (ta @ ta).item()
        sq_b += (tb @ tb).item()
        dot += (ta @ tb).item()
        sq_diff += ((ta - tb) @ (ta - tb)).item()
        per_tensor.append(TensorDiff(name, *_metrics(ta, tb)))

    norm_a, norm_b = math.sqrt(sq_a), math.sqrt(sq_b)
    rel_l2 = math.sqrt(sq_diff) / norm_a if norm_a else math.inf
    cosine = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    norm_ratio = norm_b / norm_a if norm_a else math.inf
    return CheckpointDiff(label, len(selected), rel_l2, cosine, norm_ratio,
                          norm_a, norm_b, tuple(per_tensor))


def _metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Relative L2, cosine similarity and norm ratio for one flattened pair.

    Reductions are done in float64. The full adapter is ~8.3M elements, and accumulating that dot
    product in float32 costs ~0.2% — enough to report cosine 1.0018 for a checkpoint against
    itself, which is both impossible and larger than the differences this tool exists to resolve.

    LoRA B matrices are zero at init and stay exactly zero in tensors the run never touched, so a
    zero reference norm is expected, not an error — it reports as 0 distance when both are zero.
    """
    a, b = a.double(), b.double()
    norm_a, norm_b = a.norm().item(), b.norm().item()
    if norm_a == 0.0 and norm_b == 0.0:
        return 0.0, 1.0, 1.0
    rel_l2 = (a - b).norm().item() / norm_a if norm_a else math.inf
    cosine = (a @ b).item() / (norm_a * norm_b) if norm_a and norm_b else 0.0
    norm_ratio = norm_b / norm_a if norm_a else math.inf
    return rel_l2, cosine, norm_ratio


def print_diff(diff: CheckpointDiff, worst: int) -> None:
    verdict = "  <- identical to float noise" if diff.identical else ""
    print(f"{diff.label}")
    print(f"  tensors      {diff.n_tensors}")
    print(f"  rel L2       {diff.rel_l2:.6g}{verdict}")
    print(f"  cosine       {diff.cosine:.8f}")
    print(f"  norm ratio   {diff.norm_ratio:.6f}   (||b||/||a||)")
    print(f"  norms        ||a||={diff.norm_a:.6g}  ||b||={diff.norm_b:.6g}")
    if worst and not diff.identical:
        print(f"  worst {worst} tensors by rel L2:")
        for t in diff.worst(worst):
            print(f"    {t.rel_l2:>10.6g}  cos {t.cosine:>9.6f}  {t.name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LoRA checkpoints in weight space (relative L2 + cosine).")
    parser.add_argument("a", type=Path, help="checkpoint dir, run outputs dir, or .safetensors")
    parser.add_argument("b", type=Path, help="the same, for the other side")
    parser.add_argument("--steps", default=None,
                        help="comma-separated steps to compare, when a/b are run outputs dirs")
    parser.add_argument("--ref-steps", default=None,
                        help="steps to read from b instead of --steps; use to compare one run "
                             "against itself at a different step (the trajectory-distance scale)")
    parser.add_argument("--component", choices=COMPONENTS, default="B",
                        help="which tensors to compare. 'B' (default) is the zero-initialised "
                             "half and the only uncontaminated signal; 'delta' uses the true B@A "
                             "weight change (slower); 'A'/'all' are dominated by shared init.")
    parser.add_argument("--worst", type=int, default=5,
                        help="show this many most-divergent tensors per pair (0 to suppress)")
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")] if args.steps else [None]
    ref_steps = [int(s) for s in args.ref_steps.split(",")] if args.ref_steps else steps
    if len(ref_steps) != len(steps):
        raise SystemExit(f"--ref-steps has {len(ref_steps)} entries but --steps has {len(steps)}")

    for step, ref_step in zip(steps, ref_steps):
        a_file = resolve_checkpoint(args.a, step)
        b_file = resolve_checkpoint(args.b, ref_step)
        label = f"step {step} vs {ref_step}" if step is not None else f"{args.a.name} vs {args.b.name}"
        label = f"{label}   [{args.component}]"
        print_diff(compare(a_file, b_file, label, args.component), args.worst)


if __name__ == "__main__":
    main()
