"""
Colour Checker Detection - ColorChecker Classic Template
========================================================

Defines the template for the *X-Rite ColorChecker Classic* and
*CALIBRITE COLORCHECKER PASSPORT PHOTO 2* colour side.

References
----------
-   :cite:`CALIBRITE2024`

Attributes
----------
-   :attr:`colour_checker_detection.detection.templates.template_colorchecker_classic.\
CENTROIDS`
-   :attr:`colour_checker_detection.detection.templates.template_colorchecker_classic.\
COLOURS`
-   :attr:`colour_checker_detection.detection.templates.template_colorchecker_classic.\
NAME`
-   :attr:`colour_checker_detection.detection.templates.template_colorchecker_classic.\
WIDTH`
-   :attr:`colour_checker_detection.detection.templates.template_colorchecker_classic.\
HEIGHT`
"""

from __future__ import annotations

import numpy as np

from colour_checker_detection.detection.templates import generate_template

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CENTROIDS",
    "COLOURS",
    "NAME",
    "WIDTH",
    "HEIGHT",
]

CENTROIDS: np.ndarray = np.array(
    [
        [51, 56],
        [192, 56],
        [333, 56],
        [474, 56],
        [615, 56],
        [756, 56],
        [51, 205],
        [192, 205],
        [333, 205],
        [474, 205],
        [615, 205],
        [756, 205],
        [51, 354],
        [192, 354],
        [333, 354],
        [474, 354],
        [615, 354],
        [756, 354],
        [51, 503],
        [192, 503],
        [333, 503],
        [474, 503],
        [615, 503],
        [756, 503],
    ],
    dtype=int,
)
"""
Centroids of the *CALIBRITE COLORCHECKER PASSPORT PHOTO 2* colour side
swatches.
"""

COLOURS: np.ndarray = np.array(
    [
        [0.17355167, 0.07874029, 0.05326058],
        [0.55946176, 0.27734355, 0.21194777],
        [0.10509124, 0.18955202, 0.32693865],
        [0.10506442, 0.15021316, 0.05221047],
        [0.22885963, 0.21350031, 0.42346758],
        [0.11449231, 0.50663347, 0.41229432],
        [0.74499115, 0.20172072, 0.0325174],
        [0.0606182, 0.10259253, 0.38373146],
        [0.56055825, 0.08072134, 0.11432307],
        [0.10983077, 0.04254067, 0.13682661],
        [0.32967574, 0.49495612, 0.04886544],
        [0.7689789, 0.35655545, 0.02534346],
        [0.0225082, 0.04870543, 0.28081679],
        [0.0444356, 0.29068277, 0.06458335],
        [0.44636923, 0.03676343, 0.0406788],
        [0.83803037, 0.57175305, 0.01273052],
        [0.52392518, 0.07924915, 0.28656418],
        [0.0, 0.23415773, 0.37506175],
        [0.87919095, 0.88476747, 0.8349529],
        [0.58443959, 0.59212352, 0.58458201],
        [0.35767777, 0.36706043, 0.36528718],
        [0.19008669, 0.19086038, 0.1898278],
        [0.08593528, 0.08873843, 0.08978779],
        [0.03135966, 0.03149993, 0.03231098],
    ],
    dtype=float,
)
"""
Colours of the *X-Rite ColorChecker Classic* and *CALIBRITE COLORCHECKER
PASSPORT PHOTO 2* colour side swatches.
"""

NAME: str = "colorchecker_classic"
"""
Name of the *X-Rite ColorChecker Classic* template.
"""

WIDTH: int = 810
"""
Width of the *X-Rite ColorChecker Classic* template
scaled such that swatches are roughly 100x100 pixels.
"""

HEIGHT: int = 560
"""
Height of the *X-Rite ColorChecker Classic* template
scaled such that swatches are roughly 100x100 pixels.
"""

if __name__ == "__main__":
    generate_template(CENTROIDS, COLOURS, NAME, WIDTH, HEIGHT)
