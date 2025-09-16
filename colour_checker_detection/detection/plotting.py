"""
Plotting
========

Visualization utilities for colour checker detection results.
"""

from __future__ import annotations

import typing

import cv2
import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import (
        NDArrayReal,
        Tuple,
    )

from colour.plotting import CONSTANTS_COLOUR_STYLE, plot_image

from colour_checker_detection.detection.common import (
    DataDetectionColourChecker,
    DataSegmentationColourCheckers,
    as_int32_array,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_detection_results",
]


def plot_detection_results(
    colour_checkers_data: Tuple[DataDetectionColourChecker, ...],
    swatches_horizontal: int,
    swatches_vertical: int,
    segmentation_data: DataSegmentationColourCheckers | None = None,
    image: NDArrayReal | None = None,
) -> None:
    """
    Visualize colour checker detection results.

    This function provides consistent visualization across all detection methods
    (inference, segmentation, and templated).

    Parameters
    ----------
    colour_checkers_data
        Tuple of detected colour checker data objects.
    swatches_horizontal
        Number of horizontal swatch columns.
    swatches_vertical
        Number of vertical swatch rows.
    segmentation_data
        Optional segmentation data containing swatches, clusters, and segmented image.
        Required for visualizing intermediate segmentation results.
    image
        Optional original image for drawing contours overlay.
        Required if segmentation_data is provided.

    Notes
    -----
    -   Generates 2 plots per colour checker if segmentation_data is None
        (inference method).
    -   Generates 4 additional plots if segmentation_data is provided
        (segmentation/templated methods).

    Examples
    --------
    >>> from colour_checker_detection import detect_colour_checkers_segmentation
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> results = detect_colour_checkers_segmentation(image, additional_data=True)
    >>> plot_detection_results(results, 6, 4)
    ... # doctest: +SKIP
    """

    # Part 1: Plot colour checkers with masked swatches and extracted colours
    for colour_checker_data in colour_checkers_data:
        # Mask swatches by setting them to 0
        colour_checker = np.copy(colour_checker_data.colour_checker)
        for swatch_mask in colour_checker_data.swatch_masks:
            colour_checker[
                swatch_mask[0] : swatch_mask[1],
                swatch_mask[2] : swatch_mask[3],
                ...,
            ] = 0

        # Plot detected colour checker with masked swatches
        plot_image(
            CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(colour_checker),
        )

        # Plot extracted swatch colours as grid
        plot_image(
            CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(
                np.reshape(
                    colour_checker_data.swatch_colours,
                    [swatches_vertical, swatches_horizontal, 3],
                )
            ),
        )

    # Part 2: Plot segmentation results (if available)
    if segmentation_data is not None:
        # Plot segmented/thresholded image
        plot_image(
            segmentation_data.segmented_image,
            text_kwargs={"text": "Segmented Image", "color": "black"},
        )

        # Plot swatches and clusters overlay on original image
        if image is not None:
            image_c = np.copy(image)

            # Draw swatches in magenta
            cv2.drawContours(
                image_c,
                [as_int32_array(s) for s in segmentation_data.swatches],
                -1,
                (1, 0, 1),
                3,
            )

            # Draw clusters in cyan
            cv2.drawContours(
                image_c,
                [as_int32_array(c) for c in segmentation_data.clusters],
                -1,
                (0, 1, 1),
                3,
            )

            plot_image(
                CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(image_c),
                text_kwargs={"text": "Swatches & Clusters", "color": "white"},
            )
