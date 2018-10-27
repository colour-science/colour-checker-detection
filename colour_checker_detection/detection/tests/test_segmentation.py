# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour_checker_detection.detection.segmentation`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import unittest

from colour import read_image
from colour.utilities import tstack

from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
from colour_checker_detection.detection import (
    colour_checkers_coordinates_segmentation,
    extract_colour_checkers_segmentation, detect_colour_checkers_segmentation)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['DETECTION_DIRECTORY', 'TestColourCheckersCoordinatesSegmentation']

DETECTION_DIRECTORY = os.path.join(TESTS_RESOURCES_DIRECTORY,
                                   'colour_checker_detection', 'detection')


class TestColourCheckersCoordinatesSegmentation(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
colour_checkers_coordinates_segmentation` definition unit tests methods.
    """

    def test_colour_checkers_coordinates_segmentation(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
colour_checkers_coordinates_segmentation` definition unit tests methods.
        """

        pass


if __name__ == '__main__':
    unittest.main()
