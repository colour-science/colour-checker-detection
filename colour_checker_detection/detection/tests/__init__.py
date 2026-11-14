"""Shared helpers for detection tests."""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayInt


# Updated OpenCV and inference backends can move detected corners by one pixel.
TOLERANCE_CONTOUR_PIXELS = 1
# End-to-end image sampling remains within one 8-bit code value.
TOLERANCE_SWATCH_COLOURS = 1 / 255


def sort_contour_vertices(contours: ArrayLike) -> NDArrayInt:
    """Sort contour vertices without changing their coordinate pairs."""
    return np.array([sorted(contour.tolist()) for contour in np.asarray(contours)])
