"""
Colour Checker Detection - generate_template
=======================================

Generates a template for a colour checker.

-  :attr:`Template`
-  :func:`are_three_collinear`
-  :func:`generate_template`

"""

import json
import os
from dataclasses import dataclass
from itertools import combinations, permutations

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Template:
    """
    Template dataclass.

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


def are_three_collinear(points: np.ndarray) -> bool:
    """
    Check if three points are collinear.

    Parameters
    ----------
    points
        Points to check.

    Returns
    -------
    bool
        True if the points are collinear, False otherwise.
    """
    combined_ranks = 0
    for pts in combinations(points, 3):
        matrix = np.column_stack((pts, np.ones(len(pts))))
        combined_ranks += np.linalg.matrix_rank(matrix)
    return combined_ranks != 12


def generate_template(
    swatch_centroids: np.ndarray,
    colours: np.ndarray,
    name: str,
    width: int,
    height: int,
    visualize: bool = False,
):
    """
    Generate a template.

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
    visualize
        Whether to save visualizations of the template.

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

        if np.all(angle_difference > 0) and are_three_collinear(points):
            valid_correspondences.append(list(correspondence))

    # Sort by area as a means to reach promising combinations earlier
    valid_correspondences = sorted(
        valid_correspondences,
        key=lambda x: cv2.contourArea(template.swatch_centroids[list(x)]),
        reverse=True,
    )
    template.correspondences = valid_correspondences

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"template_{name}.json"
        ),
        "w",
    ) as f:
        template.swatch_centroids = template.swatch_centroids.tolist()
        template.colours = template.colours.tolist()
        json.dump(template.__dict__, f, indent=2)

    if visualize:
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
                    )  # pyright: ignore
        plt.imshow(image, cmap="gray")

        plt.savefig(f"template_{name}.png", bbox_inches="tight")
