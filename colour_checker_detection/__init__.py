# -*- coding: utf-8 -*-
"""
Colour - Checker Detection
==========================

Colour checker detection algorithms for *Python*.

Subpackages
-----------
-   detection : Colour checker detection.
"""

from __future__ import absolute_import

import cv2
import numpy as np
import os
import subprocess  # nosec

import colour

from .detection import (colour_checkers_coordinates_segmentation,
                        extract_colour_checkers_segmentation,
                        detect_colour_checkers_segmentation)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2018-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'colour_checkers_coordinates_segmentation',
    'extract_colour_checkers_segmentation',
    'detect_colour_checkers_segmentation'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')
EXAMPLES_RESOURCES_DIRECTORY = os.path.join(
    RESOURCES_DIRECTORY, 'colour-checker-detection-examples-datasets')
TESTS_RESOURCES_DIRECTORY = os.path.join(
    RESOURCES_DIRECTORY, 'colour-checker-detection-tests-datasets')

__application_name__ = 'Colour - Checker Detection'

__major_version__ = '0'
__minor_version__ = '1'
__change_version__ = '2'
__version__ = '.'.join(
    (__major_version__,
     __minor_version__,
     __change_version__))  # yapf: disable

try:
    version = subprocess.check_output(  # nosec
        ['git', 'describe'],
        cwd=os.path.dirname(__file__),
        stderr=subprocess.STDOUT).strip()
    version = version.decode('utf-8')
except Exception:
    version = __version__

colour.utilities.ANCILLARY_COLOUR_SCIENCE_PACKAGES[
    'colour-checker-detection'] = version
colour.utilities.ANCILLARY_RUNTIME_PACKAGES['opencv'] = cv2.__version__

# TODO: Remove legacy printing support when deemed appropriate.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass
