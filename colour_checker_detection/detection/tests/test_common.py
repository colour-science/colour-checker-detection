# !/usr/bin/env python
"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.common` module.
"""

import glob
import os
import unittest

import numpy as np
from colour import read_image

from colour_checker_detection import ROOT_RESOURCES_TESTS
from colour_checker_detection.detection.common import (
    adjust_image,
    contour_centroid,
    crop_with_rectangle,
    is_square,
    scale_contour,
    swatch_masks,
)
from colour_checker_detection.detection.segmentation import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DETECTION_DIRECTORY",
    "PNG_FILES",
    "TestSwatchMasks",
    "TestAdjustImage",
    "TestCropWithRectangle",
    "TestIsSquare",
    "TestContourCentroid",
    "TestScaleContour",
]

DETECTION_DIRECTORY = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_checker_detection", "detection"
)

PNG_FILES = glob.glob(os.path.join(DETECTION_DIRECTORY, "IMG_19*.png"))


class TestSwatchMasks(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.common.swatch_masks`
    definition unit tests methods.
    """

    def test_swatch_masks(self):
        """
        Define :func:`colour_checker_detection.detection.common.swatch_masks`
        definition unit tests methods.
        """

        np.testing.assert_equal(
            swatch_masks(16, 8, 4, 2, 2),
            np.array(
                [
                    [1, 3, 1, 3],
                    [1, 3, 5, 7],
                    [1, 3, 9, 11],
                    [1, 3, 13, 15],
                    [5, 7, 1, 3],
                    [5, 7, 5, 7],
                    [5, 7, 9, 11],
                    [5, 7, 13, 15],
                ]
            ),
        )


class TestAdjustImage(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.common.adjust_image`
    definition unit tests methods.
    """

    def test_adjust_image(self):
        """
        Define :func:`colour_checker_detection.detection.common.adjust_image`
        definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        image = adjust_image(read_image(PNG_FILES[0]), 1440)
        self.assertEqual(
            image.shape[1],
            SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["working_width"],
        )


class TestCropWithRectangle(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.common.\
crop_with_rectangle` definition unit tests methods.
    """

    def test_crop_with_rectangle(self):
        """
        Define :func:`colour_checker_detection.detection.common.\
crop_with_rectangle` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        image = adjust_image(read_image(PNG_FILES[0]), 1440)

        rectangle = (
            (832.99865723, 473.05020142),
            (209.08610535, 310.13061523),
            -88.35559082,
        )

        np.testing.assert_array_equal(
            crop_with_rectangle(image, rectangle).shape,
            (209, 310, 3),
        )


class TestIsSquare(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.common.is_square`
    definition unit tests methods.
    """

    def test_is_square(self):
        """
        Define :func:`colour_checker_detection.detection.common.is_square`
        definition unit tests methods.
        """

        shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertTrue(is_square(shape))

        shape = np.array([[0, 0.5], [1, 0], [1, 1], [0, 1]])
        self.assertFalse(is_square(shape))
        self.assertTrue(is_square(shape, 0.5))


class TestContourCentroid(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.common.contour_centroid`
    definition unit tests methods.
    """

    def test_contour_centroid(self):
        """
                Define :func:`colour_checker_detection.detection.common.
        contour_centroid` definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(contour_centroid(contour), (0.5, 0.5))


class TestScaleContour(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.common.scale_contour`
    definition unit tests methods.
    """

    def test_scale_contour(self):
        """
        Define :func:`colour_checker_detection.detection.common.scale_contour`
        definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(
            scale_contour(contour, 2),
            np.array([[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]]),
        )

        np.testing.assert_array_equal(
            scale_contour(contour, 0.5),
            np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]),
        )


if __name__ == "__main__":
    unittest.main()
