"""Tiny utility for turning a class-index mask into an RGB visualization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def colorize_mask(
    mask: np.ndarray,
    palette: Sequence[Sequence[int]],
    ignore_index: int = 255,
    ignore_color: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """Map a ``H,W`` integer class mask to an ``H,W,3`` uint8 RGB image.

    Values equal to ``ignore_index`` are filled with ``ignore_color``. Any
    class index outside ``[0, len(palette))`` also falls back to that color,
    so a malformed mask can't crash the visualisation pipeline.
    """

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W); got shape {mask.shape!r}.")

    palette_array = np.asarray(palette, dtype=np.uint8)
    if palette_array.ndim != 2 or palette_array.shape[1] != 3:
        raise ValueError(f"palette must be a sequence of (R, G, B); got shape {palette_array.shape!r}.")
    ignore = np.asarray(ignore_color, dtype=np.uint8)
    if ignore.shape != (3,):
        raise ValueError(f"ignore_color must be a 3-tuple; got shape {ignore.shape!r}.")

    height, width = mask.shape
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < palette_array.shape[0]) & (mask != ignore_index)
    rgb[valid] = palette_array[mask[valid]]
    rgb[~valid] = ignore
    return rgb
