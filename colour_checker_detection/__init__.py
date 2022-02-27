"""
Colour - Checker Detection
==========================

Colour checker detection algorithms for *Python*.

Subpackages
-----------
-   detection : Colour checker detection.
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import subprocess  # nosec

import colour

from .detection import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    SETTINGS_SEGMENTATION_COLORCHECKER_SG,
    colour_checkers_coordinates_segmentation,
    extract_colour_checkers_segmentation,
    detect_colour_checkers_segmentation,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "SETTINGS_SEGMENTATION_COLORCHECKER_SG",
    "colour_checkers_coordinates_segmentation",
    "extract_colour_checkers_segmentation",
    "detect_colour_checkers_segmentation",
]

RESOURCES_DIRECTORY: str = os.path.join(os.path.dirname(__file__), "resources")
EXAMPLES_RESOURCES_DIRECTORY: str = os.path.join(
    RESOURCES_DIRECTORY, "colour-checker-detection-examples-datasets"
)
TESTS_RESOURCES_DIRECTORY: str = os.path.join(
    RESOURCES_DIRECTORY, "colour-checker-detection-tests-datasets"
)

__application_name__ = "Colour - Checker Detection"

__major_version__ = "0"
__minor_version__ = "1"
__change_version__ = "3"
__version__ = ".".join(
    (__major_version__, __minor_version__, __change_version__)
)

try:
    _version = (
        subprocess.check_output(  # nosec
            ["git", "describe"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.STDOUT,
        )
        .strip()
        .decode("utf-8")
    )
except Exception:
    _version = __version__

colour.utilities.ANCILLARY_COLOUR_SCIENCE_PACKAGES[
    "colour-checker-detection"
] = _version
colour.utilities.ANCILLARY_RUNTIME_PACKAGES["opencv"] = cv2.__version__

del _version

# TODO: Remove legacy printing support when deemed appropriate.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass
