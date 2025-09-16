"""
Colour Checker Detection - ColorChecker Creative Enhancement Template
====================================================================

Defines the template for the *X-Rite ColorChecker Creative Enhancement*
140-patch colour checker and similar gray-scale patterns.

References
----------
-   :cite:`CALIBRITE2024`

Attributes
----------
-   :attr:`colour_checker_detection.detection.templates.\
template_colorchecker_creative_enhancement.CENTROIDS`
-   :attr:`colour_checker_detection.detection.templates.\
template_colorchecker_creative_enhancement.COLOURS`
-   :attr:`colour_checker_detection.detection.templates.\
template_colorchecker_creative_enhancement.NAME`
-   :attr:`colour_checker_detection.detection.templates.\
template_colorchecker_creative_enhancement.WIDTH`
-   :attr:`colour_checker_detection.detection.templates.\
template_colorchecker_creative_enhancement.HEIGHT`
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
        [46, 50],
        [175, 50],
        [304, 50],
        [433, 50],
        [562, 50],
        [691, 50],
        [820, 50],
        [949, 50],
        [178, 185],
        [338, 185],
        [498, 185],
        [658, 185],
        [818, 185],
        [178, 355],
        [338, 355],
        [498, 355],
        [658, 355],
        [818, 355],
        [46, 490],
        [175, 490],
        [304, 490],
        [433, 490],
        [562, 490],
        [691, 490],
        [820, 490],
        [949, 490],
    ],
    dtype=int,
)
"""
Centroids of the *CALIBRITE COLORCHECKER PASSPORT PHOTO 2* gray side
swatches.
"""

COLOURS: np.ndarray = np.array(
    [
        [0.67923814, 0.06980343, 0.07945908],
        [1.0, 0.31823821, 0.03654352],
        [0.96112153, 0.6609519, 0.01170877],
        [0.0, 0.41704516, 0.11468953],
        [0.0, 0.36410292, 0.5490582],
        [0.01927884, 0.21480208, 0.64918505],
        [0.31240024, 0.14104544, 0.41277012],
        [0.5978367, 0.07613614, 0.22527837],
        [0.58305055, 0.58271855, 0.58193234],
        [0.56645735, 0.61388531, 0.61728672],
        [0.51232862, 0.60808644, 0.61548368],
        [0.4780242, 0.61108785, 0.62536069],
        [0.44239077, 0.63319999, 0.65610075],
        [0.66795228, 0.55539626, 0.58511335],
        [0.6545383, 0.57009509, 0.59099094],
        [0.58305055, 0.58271855, 0.58193234],
        [0.53814518, 0.59996143, 0.63089161],
        [0.49082519, 0.60725164, 0.67226831],
        [0.03082092, 0.03117947, 0.03215021],
        [0.03593636, 0.03756943, 0.03883438],
        [0.04465723, 0.0477725, 0.04860736],
        [0.052435, 0.05585386, 0.05590433],
        [0.46330457, 0.46845033, 0.46399649],
        [0.58285503, 0.58974152, 0.58586303],
        [0.72908483, 0.73813111, 0.73902925],
        [0.91373491, 0.90723776, 0.87534055],
    ],
    dtype=float,
)
"""
Colours of the *CALIBRITE COLORCHECKER PASSPORT PHOTO 2* gray side
swatches.
"""

NAME: str = "colorchecker_creative_enhancement"
"""
Name of the *X-Rite ColorChecker Creative Enhancement* template.
"""

WIDTH: int = 1000
"""
Width of the *X-Rite ColorChecker Creative Enhancement* template
scaled such that swatches are roughly 100x100 pixels.
"""

HEIGHT: int = 540
"""
Height of the *X-Rite ColorChecker Creative Enhancement* template
scaled such that swatches are roughly 100x100 pixels.
"""

if __name__ == "__main__":
    generate_template(CENTROIDS, COLOURS, NAME, WIDTH, HEIGHT)
