"""
Colour - Checker Detection
==========================

Colour checker detection algorithms for *Python*.

Subpackages
-----------
-   detection : Colour checker detection.
"""

from __future__ import annotations

import contextlib
import os
import subprocess

import colour
import cv2
import numpy as np

from .detection import (
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC,
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI,
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    SETTINGS_SEGMENTATION_COLORCHECKER_NANO,
    SETTINGS_SEGMENTATION_COLORCHECKER_SG,
    detect_colour_checkers_inference,
    detect_colour_checkers_segmentation,
    inferencer_default,
    segmenter_default,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC",
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI",
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "SETTINGS_SEGMENTATION_COLORCHECKER_NANO",
    "SETTINGS_SEGMENTATION_COLORCHECKER_SG",
    "detect_colour_checkers_inference",
    "detect_colour_checkers_segmentation",
    "inferencer_default",
    "segmenter_default",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")
ROOT_RESOURCES_EXAMPLES: str = os.path.join(
    ROOT_RESOURCES, "colour-checker-detection-examples-datasets"
)
ROOT_RESOURCES_TESTS: str = os.path.join(
    ROOT_RESOURCES, "colour-checker-detection-tests-datasets"
)

__all__ += ["ROOT_RESOURCES", "ROOT_RESOURCES_EXAMPLES", "ROOT_RESOURCES_TESTS"]

__application_name__ = "Colour - Checker Detection"

__major_version__ = "0"
__minor_version__ = "2"
__change_version__ = "1"
__version__ = ".".join((__major_version__, __minor_version__, __change_version__))

try:
    _version = (
        subprocess.check_output(
            ["git", "describe"],  # noqa: S603, S607
            cwd=os.path.dirname(__file__),
            stderr=subprocess.STDOUT,
        )
        .strip()
        .decode("utf-8")
    )
except Exception:
    _version = __version__

colour.utilities.ANCILLARY_COLOUR_SCIENCE_PACKAGES[  # pyright: ignore
    "colour-checker-detection"
] = _version
colour.utilities.ANCILLARY_RUNTIME_PACKAGES["opencv"] = cv2.__version__  # pyright: ignore

del _version

# TODO: Remove legacy printing support when deemed appropriate.
with contextlib.suppress(TypeError):
    np.set_printoptions(legacy="1.13")
