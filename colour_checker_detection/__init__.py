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

import numpy as np
import os

from .detection import (colour_checkers_coordinates_segmentation,
                        extract_colour_checkers_segmentation,
                        detect_colour_checkers_segmentation)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'colour_checkers_coordinates_segmentation',
    'extract_colour_checkers_segmentation',
    'detect_colour_checkers_segmentation'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')
EXAMPLES_RESOURCES_DIRECTORY = os.path.join(
    RESOURCES_DIRECTORY, 'colour-checker-detection-examples-dataset')
TESTS_RESOURCES_DIRECTORY = os.path.join(
    RESOURCES_DIRECTORY, 'colour-checker-detection-tests-dataset')

__application_name__ = 'Colour - Checker Detection'

__major_version__ = '0'
__minor_version__ = '1'
__change_version__ = '0'
__version__ = '.'.join(
    (__major_version__,
     __minor_version__,
     __change_version__))  # yapf: disable

# TODO: Remove legacy printing support when deemed appropriate.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass
