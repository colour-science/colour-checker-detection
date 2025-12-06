"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.common` module.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from colour import read_image
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import tstack, zeros

from colour_checker_detection import ROOT_RESOURCES_TESTS
from colour_checker_detection.detection.common import (
    approximate_contour,
    as_float32_array,
    cluster_swatches,
    contour_centroid,
    detect_contours,
    filter_clusters,
    is_quadrilateral,
    is_square,
    quadrilateralise_contours,
    reformat_image,
    remove_stacked_contours,
    sample_colour_checker,
    scale_contour,
    swatch_colours,
    swatch_masks,
    transform_image,
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
    "TestSwatchColours",
    "TestReformatImage",
    "TestTransformImage",
    "TestDetectContours",
    "TestIsQuadrilateral",
    "TestIsSquare",
    "TestContourCentroid",
    "TestScaleContour",
    "TestApproximateContour",
    "TestQuadrilateraliseContours",
    "TestRemoveStackedContours",
    "TestClusterSwatches",
    "TestFilterClusters",
    "TestSampleColourChecker",
]

DETECTION_DIRECTORY = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_checker_detection", "detection"
)

PNG_FILES = sorted(glob.glob(os.path.join(DETECTION_DIRECTORY, "IMG_19*.png")))


class TestSwatchMasks:
    """
    Define :func:`colour_checker_detection.detection.common.swatch_masks`
    definition unit tests methods.
    """

    def test_swatch_masks(self) -> None:
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


class TestSwatchColours:
    """
    Define :func:`colour_checker_detection.detection.common.swatch_colours`
    definition unit tests methods.
    """

    def test_swatch_colours(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.swatch_colours`
        definition unit tests methods.
        """

        x = np.linspace(0, 1, 16)
        y = np.linspace(0, 1, 8)
        xx, yy = np.meshgrid(x, y)
        image = tstack([xx, yy, zeros(xx.shape)])

        np.testing.assert_allclose(
            swatch_colours(image, swatch_masks(16, 8, 4, 2, 1)),
            np.array(
                [
                    [0.10000000, 0.21428572, 0.00000000],
                    [0.36666667, 0.21428572, 0.00000000],
                    [0.63333333, 0.21428572, 0.00000000],
                    [0.89999998, 0.21428572, 0.00000000],
                    [0.10000000, 0.78571427, 0.00000000],
                    [0.36666667, 0.78571427, 0.00000000],
                    [0.63333333, 0.78571427, 0.00000000],
                    [0.89999998, 0.78571427, 0.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestReformatImage:
    """
    Define :func:`colour_checker_detection.detection.common.reformat_image`
    definition unit tests methods.
    """

    def test_reformat_image(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.reformat_image`
        definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        path = next(png_file for png_file in PNG_FILES if "1970" in png_file)

        assert (
            reformat_image(read_image(path), 1440).shape[1]
            == SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["working_width"]
        )


class TestTransformImage:
    """
    Define :func:`colour_checker_detection.detection.common.transform_image`
    definition unit tests methods.
    """

    def test_transform_image(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.transform_image`
        definition unit tests methods.
        """

        image = np.reshape(as_float32_array(np.arange(96)), (4, 8, 3))

        np.testing.assert_allclose(
            transform_image(image, np.array([2, 4]), 45, np.array([2, 3])),
            np.array(
                [
                    [
                        [47.68359375, 48.68359375, 49.68359375],
                        [41.15771866, 42.15770721, 43.15771484],
                        [37.69516373, 38.69516754, 39.69516754],
                        [34.04169083, 35.04169464, 36.04168320],
                        [29.82055664, 30.82055473, 31.82055283],
                        [22.41366768, 23.41366577, 24.41366577],
                        [17.44537354, 18.44537163, 19.44537544],
                        [13.23165703, 14.23165894, 15.23165798],
                    ],
                    [
                        [56.25146103, 57.25147247, 58.25146484],
                        [49.13193512, 50.13193512, 51.13193512],
                        [43.10541153, 44.10540771, 45.10540390],
                        [40.26855469, 41.26855469, 42.26855850],
                        [36.38168335, 37.38168335, 38.38168335],
                        [31.61718750, 32.61718750, 33.61718750],
                        [24.64370728, 25.64370728, 26.64370537],
                        [19.65682602, 20.65682983, 21.65682793],
                    ],
                    [
                        [62.66984177, 63.66983414, 64.66983795],
                        [58.19916534, 59.19915771, 60.19916153],
                        [51.70532227, 52.70532227, 53.70532227],
                        [45.44540405, 46.44540405, 47.44540024],
                        [42.06518555, 43.06518555, 44.06518555],
                        [38.61172867, 39.61172485, 40.61172867],
                        [33.82863998, 34.82864380, 35.82864380],
                        [26.57885551, 27.57885933, 28.57886314],
                    ],
                    [
                        [69.03441620, 70.03442383, 71.03441620],
                        [65.24322510, 66.24321747, 67.24322510],
                        [60.53916168, 61.53916168, 62.53915405],
                        [53.50195312, 54.50195312, 55.50195312],
                        [47.67544556, 48.67544174, 49.67544556],
                        [44.27664185, 45.27664185, 46.27664185],
                        [40.54687500, 41.54687500, 42.54687500],
                        [36.09164429, 37.09164429, 38.09164810],
                    ],
                ],
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS * 100,
        )


class TestDetectContours:
    """
    Define :func:`colour_checker_detection.detection.common.detect_contours`
    definition unit tests methods.
    """

    def test_detect_contours(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.detect_contours`
        definition unit tests methods.
        """

        image = zeros((240, 320, 3))
        image[100:140, 50:90] = 1
        image[150:190, 140:180] = 1

        assert len(detect_contours(image)) == 5


class TestIsQuadrilateral:
    """
    Define :func:`colour_checker_detection.detection.common.is_quadrilateral`
    definition unit tests methods.
    """

    def test_is_quadrilateral(self) -> None:
        """
        Test :func:`colour_checker_detection.detection.common.is_quadrilateral`
        definition.
        """

        # Valid quadrilateral (rectangle corners)
        assert is_quadrilateral(np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))

        # Three collinear points (invalid quadrilateral)
        assert not is_quadrilateral(np.array([[0, 0], [5, 0], [10, 0], [0, 10]]))

        # Another valid quadrilateral (irregular but valid)
        assert is_quadrilateral(np.array([[0, 0], [8, 2], [10, 12], [2, 10]]))

        # All points on a line (degenerate case)
        assert not is_quadrilateral(np.array([[0, 0], [2, 0], [4, 0], [6, 0]]))

        # Three points collinear on vertical line
        assert not is_quadrilateral(np.array([[0, 0], [0, 5], [0, 10], [5, 5]]))

        # Template correspondence case
        assert is_quadrilateral(np.array([[192, 56], [756, 56], [756, 503], [51, 503]]))


class TestIsSquare:
    """
    Define :func:`colour_checker_detection.detection.common.is_square`
    definition unit tests methods.
    """

    def test_is_square(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.is_square`
        definition unit tests methods.
        """

        shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert is_square(shape)

        shape = np.array([[0, 0.5], [1, 0], [1, 1], [0, 1]])
        assert not is_square(shape)
        assert is_square(shape, 0.5)


class TestContourCentroid:
    """
    Define :func:`colour_checker_detection.detection.common.contour_centroid`
    definition unit tests methods.
    """

    def test_contour_centroid(self) -> None:
        """
                Define :func:`colour_checker_detection.detection.common.
        contour_centroid` definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(contour_centroid(contour), (0.5, 0.5))


class TestScaleContour:
    """
    Define :func:`colour_checker_detection.detection.common.scale_contour`
    definition unit tests methods.
    """

    def test_scale_contour(self) -> None:
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


class TestApproximateContour:
    """
    Define :func:`colour_checker_detection.detection.common.approximate_contour`
    definition unit tests methods.
    """

    def test_approximate_contour(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.approximate_contour`
        definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [1, 2], [0, 1]])

        np.testing.assert_array_equal(
            approximate_contour(contour, 4),
            np.array([[0, 0], [1, 0], [1, 2], [0, 1]]),
        )

        np.testing.assert_array_equal(
            approximate_contour(contour, 3),
            np.array([[0, 0], [1, 0], [1, 2]]),
        )


class TestQuadrilateraliseContours:
    """
    Define :func:`colour_checker_detection.detection.common.\
quadrilateralise_contours` definition unit tests methods.
    """

    def test_quadrilateralise_contours(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.\
quadrilateralise_contours` definition unit tests methods.
        """

        contours = np.array(
            [
                [[0, 0], [1, 0], [1, 1], [1, 2], [0, 1]],
                [[0, 0], [1, 2], [1, 0], [1, 1], [0, 1]],
            ]
        )

        np.testing.assert_array_equal(
            quadrilateralise_contours(contours),
            np.array(
                [
                    [[0, 0], [1, 0], [1, 2], [0, 1]],
                    [[0, 0], [1, 2], [1, 0], [1, 1]],
                ]
            ),
        )


class TestRemoveStackedContours:
    """
    Define :func:`colour_checker_detection.detection.common.\
remove_stacked_contours` definition unit tests methods.
    """

    def test_remove_stacked_contours(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.\
remove_stacked_contours` definition unit tests methods.
        """

        contours = np.array(
            [
                [[0, 0], [7, 0], [7, 7], [0, 7]],
                [[0, 0], [8, 0], [8, 8], [0, 8]],
                [[0, 0], [10, 0], [10, 10], [0, 10]],
            ]
        )

        np.testing.assert_array_equal(
            remove_stacked_contours(contours),
            np.array([[[0, 0], [7, 0], [7, 7], [0, 7]]]),
        )

        np.testing.assert_array_equal(
            remove_stacked_contours(contours, False),
            np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]]),
        )


class TestClusterSwatches:
    """
    Define :func:`colour_checker_detection.detection.common.cluster_swatches`
    definition unit tests methods.
    """

    def test_cluster_swatches(self) -> None:
        """
        Test :func:`colour_checker_detection.detection.common.cluster_swatches`
        definition.
        """

        image = np.zeros((600, 900, 3))

        # Two separate swatches that should form two clusters
        swatches = np.array(
            [
                [[100, 100], [200, 100], [200, 200], [100, 200]],
                [[300, 100], [400, 100], [400, 200], [300, 200]],
            ],
            dtype=np.int32,
        )
        result = cluster_swatches(image, swatches, 1.5)
        assert result.shape == (2, 4, 2)
        assert result.dtype == np.int32

        # Two overlapping swatches that should form one cluster
        swatches = np.array(
            [
                [[100, 100], [150, 100], [150, 150], [100, 150]],
                [[140, 100], [190, 100], [190, 150], [140, 150]],
            ],
            dtype=np.int32,
        )
        result = cluster_swatches(image, swatches, 2.0)
        assert result.shape[0] == 1
        assert result.dtype == np.int32

        # Empty swatches array
        swatches = np.array([], dtype=np.int32).reshape(0, 4, 2)
        result = cluster_swatches(image, swatches, 1.5)
        assert len(result) == 0


class TestFilterClusters:
    """
    Define :func:`colour_checker_detection.detection.common.filter_clusters`
    definition unit tests methods.
    """

    def test_filter_clusters(self) -> None:
        """
        Test :func:`colour_checker_detection.detection.common.filter_clusters`
        definition.
        """

        # Both clusters contain swatches within range
        clusters = np.array(
            [
                [[0, 0], [200, 0], [200, 200], [0, 200]],
                [[300, 300], [400, 300], [400, 400], [300, 400]],
            ],
            dtype=np.int32,
        )
        swatches = np.array(
            [
                [[50, 50], [100, 50], [100, 100], [50, 100]],
                [[350, 350], [380, 350], [380, 380], [350, 380]],
            ],
            dtype=np.int32,
        )
        result = filter_clusters(clusters, swatches, 1, 2)
        assert result.shape == (2, 4, 2)
        assert result.dtype == np.int32

        # Only first cluster contains swatches
        swatches = np.array(
            [
                [[50, 50], [100, 50], [100, 100], [50, 100]],
            ],
            dtype=np.int32,
        )
        result = filter_clusters(clusters, swatches, 1, 2)
        assert result.shape == (1, 4, 2)

        # No clusters contain required number of swatches
        result = filter_clusters(clusters, swatches, 5, 10)
        assert result.shape == (0, 4, 2)

        # Empty clusters array
        empty_clusters = np.array([], dtype=np.int32).reshape(0, 4, 2)
        result = filter_clusters(empty_clusters, swatches, 1, 2)
        assert result.shape == (0, 4, 2)

        # Empty swatches array
        empty_swatches = np.array([], dtype=np.int32).reshape(0, 4, 2)
        result = filter_clusters(clusters, empty_swatches, 1, 2)
        assert result.shape == (0, 4, 2)


class TestSampleColourChecker:
    """
    Define :func:`colour_checker_detection.detection.common.\
remove_stacked_contours` definition unit tests methods.
    """

    def test_sample_colour_checker(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.common.\
sample_colour_checker` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        path = next(png_file for png_file in PNG_FILES if "1967" in png_file)

        quadrilateral = np.array([[358, 691], [373, 219], [1086, 242], [1071, 713]])
        rectangle = np.array([[1440, 0], [1440, 960], [0, 960], [0, 0]])
        colour_checkers_data = sample_colour_checker(
            read_image(path), quadrilateral, rectangle
        )

        np.testing.assert_allclose(
            colour_checkers_data.swatch_colours,
            np.array(
                [
                    [0.75710917, 0.67630458, 0.47606474],
                    [0.25871587, 0.21974973, 0.16204563],
                    [0.15012611, 0.11881837, 0.07829906],
                    [0.14475887, 0.11828972, 0.07471170],
                    [0.15182742, 0.12059662, 0.07984065],
                    [0.15811475, 0.12584405, 0.07951307],
                    [0.99963307, 0.82756299, 0.53623772],
                    [0.26152441, 0.22938406, 0.16862768],
                    [0.15809630, 0.11951645, 0.07755180],
                    [0.16762769, 0.13303326, 0.08851139],
                    [0.17338796, 0.14148802, 0.08979498],
                    [0.17304046, 0.14195150, 0.09080467],
                    [1.00000000, 0.98902053, 0.67808318],
                    [0.25435534, 0.22063790, 0.15692709],
                    [0.15027192, 0.12475526, 0.07843940],
                    [0.34583551, 0.21429974, 0.11217980],
                    [0.36254194, 0.22595090, 0.11665937],
                    [0.62459683, 0.39098999, 0.24112946],
                    [0.97804743, 1.00000000, 0.86419195],
                    [0.25577253, 0.22349517, 0.15844890],
                    [0.15959230, 0.12591116, 0.08147947],
                    [0.35486832, 0.21910854, 0.11063413],
                    [0.36308041, 0.22740598, 0.12138989],
                    [0.62340593, 0.39334935, 0.24371558],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert colour_checkers_data.swatch_masks.shape == (24, 4)
        assert colour_checkers_data.colour_checker.shape == (960, 1440, 3)
        assert colour_checkers_data.quadrilateral.shape == quadrilateral.shape
