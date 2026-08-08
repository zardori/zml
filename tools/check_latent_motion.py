"""Report temporal variation inside a frame_replace dataset's latents, per region.

A frame_replace target teaches "given this prompt, produce this clip". If the edited region of the
target is temporally flat, the model learns to freeze on the concept prompt — which is exactly what
exp080 did (concept motion -87%..-99% against base 0.686, unrelated untouched) after training on
exp061 triples whose donor fill was a single repeated frame.

This measures that directly on the saved latents, so it needs no VAE, no GPU and no decode:

- ``safe``: mean |x[i+1] - x[i]| over consecutive frames *outside* ``edited_latent_mask``
- ``edit``: the same *inside* it
- ``ratio``: edit / safe — about 1.0 when the fill carries the safe segment's motion, about 0.0
  when it is a frozen copy

Run it on a rebuilt dataset to confirm the fill actually moves, and on the dataset it replaces to
see the before/after. It also catches the failure mode a donor map cannot: if the *source* clip was
static in the safe region, a mirrored fill of it is still static, and the rebuild bought nothing.

    uv run python tools/check_latent_motion.py --metadata <meta.json> --latents-dir <dir>
"""

import argparse
import json
from pathlib import Path

import torch

# Below this, a region is flat enough that the model is being taught a still image rather than motion.
FROZEN_RATIO = 0.15


def region_motion(latent: torch.Tensor, mask: list[bool], inside: bool) -> float:
    """Mean absolute frame-to-frame difference over the frames selected by ``mask``.

    ``latent`` is ``(B, C, F, H, W)``. Only consecutive pairs where *both* frames fall on the
    requested side of the mask are counted, so the seam itself never contributes.
    """
    idx = [i for i, m in enumerate(mask) if m == inside]
    pairs = [(a, b) for a, b in zip(idx, idx[1:]) if b == a + 1]
    if not pairs:
        return 0.0
    diffs = [(latent[:, :, b] - latent[:, :, a]).abs().mean().item() for a, b in pairs]
    return sum(diffs) / len(diffs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N entries")
    args = parser.parse_args()

    latents_dir = Path(args.latents_dir)
    entries = json.load(open(args.metadata))[: args.limit]

    print(f"{'seed':>6} {'safe':>9} {'edit':>9} {'ratio':>7}  verdict")
    ratios = []
    for e in entries:
        latent = torch.load(latents_dir / e["latent_path"], map_location="cpu")
        mask = e.get("edited_latent_mask") or e["concept_latent_mask"]
        safe = region_motion(latent, mask, inside=False)
        edit = region_motion(latent, mask, inside=True)
        ratio = edit / safe if safe else 0.0
        ratios.append(ratio)
        verdict = "FROZEN" if ratio < FROZEN_RATIO else "moves"
        print(f"{e['seed']:>6} {safe:>9.4f} {edit:>9.4f} {ratio:>7.2f}  {verdict}")

    frozen = sum(1 for r in ratios if r < FROZEN_RATIO)
    mean = sum(ratios) / len(ratios) if ratios else 0.0
    print(f"\n{len(ratios)} entries | mean edit/safe ratio {mean:.2f} | {frozen} frozen (<{FROZEN_RATIO})")
    if frozen:
        print("A frozen edited region teaches the model to emit a still image on the concept prompt.")


if __name__ == "__main__":
    main()
