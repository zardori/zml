"""The ten ImageNet classes used by the object-erasure protocol, and their ImageNet-1k indices.

ESD introduced this benchmark on the *Imagenette* subset — ten classes that are easy for a human to
name and for a classifier to separate — and T2VUnlearning (arXiv 2505.17550, §4.2 / Table 7) follows
the same protocol on video models. Keeping the exact same ten keeps our ESR/PSR numbers comparable
with both papers, and with VideoEraser, which also reports on Imagenette.

The indices are positions in the standard 1000-class ImageNet-1k ordering, i.e. the output layer of a
torchvision classifier. Classification is deliberately 1000-way rather than 10-way: a 10-way decision
would make top-5 accuracy nearly free and the published numbers meaningless to compare against.
"""

# Class name -> ImageNet-1k index. The names are also what goes into prompts and negative prompts, so
# they are the human-readable ImageNet labels rather than the wnids.
IMAGENETTE_CLASSES: dict[str, int] = {
    "tench": 0,  # n01440764
    "English springer": 217,  # n02102040
    "cassette player": 482,  # n02979186
    "chain saw": 491,  # n03000684
    "church": 497,  # n03028079
    "French horn": 566,  # n03394916
    "garbage truck": 569,  # n03417042
    "gas pump": 571,  # n03425413
    "golf ball": 574,  # n03445777
    "parachute": 701,  # n03888257
}


def class_index(name: str) -> int:
    """ImageNet-1k index for one of the ten protocol classes, or a clear error listing the valid ones."""
    try:
        return IMAGENETTE_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"Unknown object class {name!r}; expected one of {sorted(IMAGENETTE_CLASSES)}."
        ) from None


def class_slug(name: str) -> str:
    """Filesystem-safe form of a class name (``"chain saw"`` -> ``"chain_saw"``).

    Used for per-class prompt files and per-class eval video directories, so a class name can round-trip
    through a path without quoting.
    """
    return name.lower().replace(" ", "_")
