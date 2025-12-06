"""
Colour Checker Detection - generate_template
=======================================

Generates a template for a colour checker.

-  :attr:`Template`
-  :func:`are_three_collinear`
-  :func:`generate_template`

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import permutations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from colour_checker_detection.detection.common import is_quadrilateral

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Template",
    "generate_template",
    "load_template",
]


@dataclass
class Template:
    """
    Template dataclass for colour checker structure representation.

    Parameters
    ----------
    swatch_centroids
        Centroids of the swatches.
    colours
        Colours of the swatches.
    correspondences
        Possible correspondences between the reference swatches and the detected ones.
    width
        Width of the template.
    height
        Height of the template.
    """

    swatch_centroids: np.ndarray
    colours: np.ndarray
    correspondences: list
    width: int
    height: int


def load_template(template_path: str) -> Template:
    """
    Load a template from a structured NPZ file.

    Parameters
    ----------
    template_path
        Full path to the NPZ template file.

    Returns
    -------
    :class:`Template`
        Loaded template object.
    """
    template_file_path = template_path

    # Load the NPZ file
    with np.load(template_file_path) as data:
        swatch_centroids = data["swatch_centroids"]
        colours = data["colours"]
        correspondences = data["correspondences"].tolist()  # Convert back to list
        width = int(data["width"])
        height = int(data["height"])

    return Template(swatch_centroids, colours, correspondences, width, height)


def generate_template(
    swatch_centroids: np.ndarray,
    colours: np.ndarray,
    name: str,
    width: int,
    height: int,
    show: bool = False,
    output_directory: str | None = None,
) -> Template:
    """
    Generate a template from colour checker structure.

    Parameters
    ----------
    swatch_centroids
        Centroids of the swatches.
    colours
        Colours of the swatches.
    name
        Name of the template.
    width
        Width of the template.
    height
        Height of the template.
    show
        Whether to show visualizations of the template.
    output_directory
        Directory to save the template JSON file. If None, saves in the
        templates directory.

    Returns
    -------
    :class:`Template`
        Generated template object.
    """
    template = Template(swatch_centroids, colours, [], width, height)

    valid_correspondences = []
    for correspondence in permutations(range(len(swatch_centroids)), 4):
        points = swatch_centroids[list(correspondence)]
        centroid = np.mean(points, axis=0)
        angle = np.array(
            [np.arctan2((pt[1] - centroid[1]), (pt[0] - centroid[0])) for pt in points]
        )
        # Account for the border from pi to -pi
        angle = np.append(angle[np.argmin(angle) :], angle[: np.argmin(angle)])
        angle_difference = np.diff(angle)

        if np.all(angle_difference > 0) and is_quadrilateral(points):
            valid_correspondences.append(list(correspondence))

    # Sort by area as a means to reach promising combinations earlier
    valid_correspondences = sorted(
        valid_correspondences,
        key=lambda x: cv2.contourArea(template.swatch_centroids[list(x)]),
        reverse=True,
    )
    template.correspondences = valid_correspondences

    if output_directory is None:
        output_directory = os.path.dirname(__file__)

    template_file_path = os.path.join(output_directory, f"template_{name}.npz")

    n_correspondences = len(template.correspondences)

    if n_correspondences == 0:
        correspondences_array = np.empty((0, 4), dtype=np.int32)
    else:
        correspondences_array = np.array(template.correspondences, dtype=np.int32)

    np.savez_compressed(
        template_file_path,
        swatch_centroids=template.swatch_centroids.astype(np.float32),
        colours=template.colours.astype(np.float32),
        correspondences=correspondences_array,
        width=np.int32(template.width),
        height=np.int32(template.height),
        n_correspondences=np.int32(n_correspondences),
    )

    if show:
        template_adjacency_matrix = np.zeros(
            (len(swatch_centroids), len(swatch_centroids))
        )
        for i, pt1 in enumerate(swatch_centroids):
            for j, pt2 in enumerate(swatch_centroids):
                if i != j:
                    template_adjacency_matrix[i, j] = np.linalg.norm(pt1 - pt2)
                else:
                    template_adjacency_matrix[i, j] = np.inf

        dist = np.max(np.min(template_adjacency_matrix, axis=0)) * 1.2
        template_graph = template_adjacency_matrix < dist

        image = np.zeros((height, width))
        plt.scatter(*swatch_centroids.T, s=15)
        for nr, pt in enumerate(swatch_centroids):
            plt.annotate(str(nr), pt, fontsize=10, color="white")

        for r, row in enumerate(template_graph):
            for c, col in enumerate(row):
                if col == 1:
                    cv2.line(
                        image,
                        swatch_centroids[r],
                        swatch_centroids[c],
                        (255, 255, 255),
                        thickness=2,
                    )
        plt.imshow(image, cmap="gray")

    return template
