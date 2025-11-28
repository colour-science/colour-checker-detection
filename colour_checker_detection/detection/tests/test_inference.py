"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.inference` module.
"""

from __future__ import annotations

import glob
import os
import platform
import sys

import numpy as np
import pytest
from colour import read_image

from colour_checker_detection import ROOT_RESOURCES_TESTS
from colour_checker_detection.detection import (
    detect_colour_checkers_inference,
    extractor_inference,
    inferencer_default,
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
    "TestInferencerDefault",
    "TestExtractorInference",
    "TestDetectColourCheckersInference",
]

DETECTION_DIRECTORY = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_checker_detection", "detection"
)

PNG_FILES = sorted(glob.glob(os.path.join(DETECTION_DIRECTORY, "IMG_19*.png")))[:-2]


class TestInferencerDefault:
    """
    Define :func:`colour_checker_detection.detection.inference.\
inferencer_default` definition unit tests methods.
    """

    @pytest.mark.skipif(
        platform.system() in ("Windows", "Microsoft", "Linux")
        or sys.version_info >= (3, 14),
        reason="Unit test is only reproducible on macOS and requires Python < 3.14",
    )
    def test_inferencer_default(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.inference.\
inferencer_default` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        shapes = [
            (864, 1280),
            (864, 1280),
            (1280, 864),
            (864, 1280),
        ]

        for i, png_file in enumerate(PNG_FILES):
            results = inferencer_default(png_file)
            assert results[0][0] > 0.85
            assert int(results[0][1]) == 0
            assert results[0][2].shape == shapes[i]


class TestExtractorInference:
    """Define :func:`extractor_inference` definition unit tests methods."""

    @pytest.mark.skipif(
        platform.system() in ("Windows", "Microsoft", "Linux")
        or sys.version_info >= (3, 14),
        reason="Unit test is only reproducible on macOS and requires Python < 3.14",
    )
    def test_extractor_inference(self) -> None:
        """Test :func:`extractor_inference` definition."""

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        image = read_image(PNG_FILES[0])

        inference_results = inferencer_default(PNG_FILES[0])

        extractors_result = extractor_inference(
            image, inference_results, additional_data=True
        )

        assert isinstance(extractors_result, tuple)
        assert len(extractors_result) == 1

        colour_checker_data = extractors_result[0]
        assert hasattr(colour_checker_data, "swatch_colours")
        assert hasattr(colour_checker_data, "colour_checker")
        assert hasattr(colour_checker_data, "swatch_masks")

        assert colour_checker_data.swatch_colours.shape == (24, 3)  # pyright: ignore

        extractors_result_simple = extractor_inference(
            image, inference_results, additional_data=False
        )

        assert isinstance(extractors_result_simple, tuple)
        assert len(extractors_result_simple) == 1

        assert isinstance(extractors_result_simple[0], np.ndarray)
        assert extractors_result_simple[0].shape == (24, 3)

        np.testing.assert_allclose(
            colour_checker_data.swatch_colours,  # pyright: ignore
            extractors_result_simple[0],
            atol=0.0001,
        )


class TestDetectColourCheckersInference:
    """
    Define :func:`colour_checker_detection.detection.inference.\
detect_colour_checkers_inference` definition unit tests methods.
    """

    @pytest.mark.skipif(
        platform.system() in ("Windows", "Microsoft", "Linux")
        or sys.version_info >= (3, 14),
        reason="Unit test is only reproducible on macOS and requires Python < 3.14",
    )
    def test_detect_colour_checkers_inference(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.inference.\
detect_colour_checkers_inference` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        test_swatches = [
            # IMG_1966.png
            (
                np.array(
                    [
                        [0.24863878, 0.15329053, 0.08158918],
                        [0.41507748, 0.25731990, 0.15481426],
                        [0.21601081, 0.19963661, 0.18040785],
                        [0.19686793, 0.17291754, 0.06548586],
                        [0.27357155, 0.20301168, 0.19400775],
                        [0.23356010, 0.28720620, 0.18817304],
                        [0.45823488, 0.23316798, 0.04992789],
                        [0.18078160, 0.16045800, 0.20766605],
                        [0.39627394, 0.16691503, 0.11309760],
                        [0.19422898, 0.11006075, 0.11565600],
                        [0.31493783, 0.29084757, 0.02219770],
                        [0.43131799, 0.25828508, 0.00795346],
                        [0.13812497, 0.12478630, 0.17762889],
                        [0.18720838, 0.24592778, 0.07732303],
                        [0.35516253, 0.13156214, 0.07454837],
                        [0.46337095, 0.32273552, 0.00165196],
                        [0.37116164, 0.15837687, 0.16341914],
                        [0.12098821, 0.20031747, 0.18507589],
                        [0.50464547, 0.41264072, 0.29166794],
                        [0.41424599, 0.33581433, 0.23724470],
                        [0.32868224, 0.26551095, 0.18507436],
                        [0.23896469, 0.18752976, 0.12675764],
                        [0.16701609, 0.13087222, 0.08458477],
                        [0.10941069, 0.07924574, 0.04898232],
                    ]
                ),
            ),
            # IMG_1967.png
            (
                np.array(
                    [
                        [0.36007342, 0.22303678, 0.11766040],
                        [0.62607545, 0.39443627, 0.24180005],
                        [0.33200133, 0.31590021, 0.28866205],
                        [0.30415800, 0.27339226, 0.10521446],
                        [0.41758743, 0.31893715, 0.30788019],
                        [0.34878933, 0.43871346, 0.29159448],
                        [0.67982006, 0.35233310, 0.07041400],
                        [0.27139527, 0.25354654, 0.33075848],
                        [0.62072551, 0.27040577, 0.18629737],
                        [0.30715409, 0.17973351, 0.19184262],
                        [0.48536164, 0.45853454, 0.03277667],
                        [0.65034246, 0.40020591, 0.01576474],
                        [0.19285583, 0.18593574, 0.27413625],
                        [0.28041738, 0.38502172, 0.12292562],
                        [0.55452663, 0.21458797, 0.12545331],
                        [0.72076070, 0.51544499, 0.00525500],
                        [0.57798642, 0.25786015, 0.26852059],
                        [0.17531879, 0.31668669, 0.29529998],
                        [0.74044472, 0.61071527, 0.43872431],
                        [0.62955171, 0.51785052, 0.37301064],
                        [0.51464999, 0.42113122, 0.29825154],
                        [0.37083188, 0.30355468, 0.20928001],
                        [0.26390594, 0.21514489, 0.14332861],
                        [0.16213489, 0.13396774, 0.08086098],
                    ]
                ),
            ),
            # IMG_1968.png
            (
                np.array(
                    [
                        [0.18891563, 0.31932956, 0.29802763],
                        [0.52564883, 0.24632359, 0.24680115],
                        [0.33316478, 0.24978259, 0.08024282],
                        [0.22091143, 0.24506716, 0.10507169],
                        [0.19054136, 0.17549875, 0.24753447],
                        [0.15760350, 0.12744455, 0.09203142],
                        [0.65549982, 0.40460357, 0.02602872],
                        [0.45806545, 0.42677194, 0.05421838],
                        [0.22650817, 0.16784146, 0.13089909],
                        [0.22247627, 0.19619618, 0.20893870],
                        [0.63250530, 0.33052164, 0.07766631],
                        [0.26472476, 0.16740909, 0.08628228],
                        [0.36052877, 0.43975237, 0.29521543],
                        [0.39757040, 0.30876780, 0.29023170],
                        [0.22451068, 0.18822640, 0.11181563],
                        [0.41199884, 0.27477866, 0.17575167],
                        [0.34759662, 0.21948732, 0.12056293],
                        [0.21642110, 0.15858017, 0.09983096],
                        [0.16862650, 0.13624829, 0.09448092],
                        [0.16893627, 0.13745177, 0.09204258],
                        [0.16604233, 0.13258407, 0.09098586],
                        [0.15434082, 0.12374580, 0.08661740],
                        [0.14643972, 0.11655007, 0.07757708],
                        [0.19485806, 0.15879279, 0.11575103],
                    ]
                ),
            ),
            # IMG_1969.png
            (
                np.array(
                    [
                        [0.22001560, 0.12819208, 0.05780617],
                        [0.38588333, 0.23513840, 0.13717623],
                        [0.19886261, 0.18835427, 0.17192243],
                        [0.19373806, 0.17203276, 0.06229085],
                        [0.27893409, 0.21379797, 0.20769426],
                        [0.24462585, 0.30783489, 0.20605047],
                        [0.42575374, 0.21035977, 0.03103235],
                        [0.16033459, 0.14435254, 0.19109112],
                        [0.39280602, 0.16093710, 0.10477450],
                        [0.19482799, 0.11117131, 0.11691989],
                        [0.33916837, 0.31749195, 0.02215747],
                        [0.48302668, 0.29534918, 0.01196240],
                        [0.11179636, 0.10340962, 0.14993328],
                        [0.17667602, 0.23034677, 0.06186982],
                        [0.35338593, 0.12842597, 0.06959131],
                        [0.48558396, 0.34025896, 0.00081675],
                        [0.40566599, 0.17618640, 0.18331584],
                        [0.13100371, 0.23202963, 0.21993101],
                        [0.48221427, 0.37988359, 0.25555688],
                        [0.41229609, 0.32613912, 0.22259885],
                        [0.34035870, 0.27100950, 0.18408188],
                        [0.25901079, 0.20153706, 0.13319436],
                        [0.19520043, 0.15373182, 0.10178117],
                        [0.13309827, 0.10635833, 0.06500074],
                    ]
                ),
            ),
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                detect_colour_checkers_inference(
                    read_image(png_file), additional_data=False
                ),
                test_swatches[i],
                atol=0.0001,
            )

        (
            swatch_colours,
            swatch_masks,
            colour_checker,
            quadrilateral,
        ) = detect_colour_checkers_inference(
            read_image(PNG_FILES[0]), additional_data=True
        )[0].values

        np.testing.assert_allclose(
            swatch_colours,
            test_swatches[0][0],
            atol=0.0001,
        )

        np.testing.assert_array_equal(
            colour_checker.shape[0:2],
            np.array([1008, 1440]),
        )

        np.testing.assert_array_equal(
            swatch_masks,
            np.array(
                [
                    [110, 142, 104, 136],
                    [110, 142, 344, 376],
                    [110, 142, 584, 616],
                    [110, 142, 824, 856],
                    [110, 142, 1064, 1096],
                    [110, 142, 1304, 1336],
                    [362, 394, 104, 136],
                    [362, 394, 344, 376],
                    [362, 394, 584, 616],
                    [362, 394, 824, 856],
                    [362, 394, 1064, 1096],
                    [362, 394, 1304, 1336],
                    [614, 646, 104, 136],
                    [614, 646, 344, 376],
                    [614, 646, 584, 616],
                    [614, 646, 824, 856],
                    [614, 646, 1064, 1096],
                    [614, 646, 1304, 1336],
                    [866, 898, 104, 136],
                    [866, 898, 344, 376],
                    [866, 898, 584, 616],
                    [866, 898, 824, 856],
                    [866, 898, 1064, 1096],
                    [866, 898, 1304, 1336],
                ]
            ),
        )

        assert quadrilateral.shape == (4, 2)
