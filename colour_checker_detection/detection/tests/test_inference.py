# !/usr/bin/env python
"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.inference` module.
"""

import glob
import os
import platform
import sys
import unittest

import numpy as np
from colour import read_image

from colour_checker_detection import ROOT_RESOURCES_TESTS
from colour_checker_detection.detection import (
    detect_colour_checkers_inference,
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
    "TestDetectColourCheckersInference",
]

DETECTION_DIRECTORY = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_checker_detection", "detection"
)

PNG_FILES = sorted(
    glob.glob(os.path.join(DETECTION_DIRECTORY, "IMG_19*.png"))
)[:-2]


class TestInferencerDefault(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.inference.\
inferencer_default` definition unit tests methods.
    """

    def test_inferencer_default(self):
        """
        Define :func:`colour_checker_detection.detection.inference.\
inferencer_default` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        # TODO: Enable when "torch" is available on Python 3.12.
        if sys.version_info[1] > 11:  # noqa: YTT203
            return

        shapes = [
            (864, 1280),
            (864, 1280),
            (1280, 864),
            (864, 1280),
        ]

        for i, png_file in enumerate(PNG_FILES):
            results = inferencer_default(png_file)
            self.assertTrue(results[0][0] > 0.85)
            self.assertEqual(int(results[0][1]), 0)
            self.assertTupleEqual(results[0][2].shape, shapes[i])


class TestDetectColourCheckersInference(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.inference.\
detect_colour_checkers_inference` definition unit tests methods.
    """

    def test_detect_colour_checkers_inference(self):
        """
        Define :func:`colour_checker_detection.detection.inference.\
detect_colour_checkers_inference` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        # TODO: Enable when "torch" is available on Python 3.12.
        if sys.version_info[1] > 11:  # noqa: YTT203
            return

        test_swatches = [
            (
                np.array(
                    [
                        [0.24915336, 0.15333557, 0.08062609],
                        [0.41582590, 0.25704128, 0.15597928],
                        [0.21691470, 0.19983344, 0.18059847],
                        [0.19566718, 0.17250466, 0.06643628],
                        [0.27478909, 0.20288917, 0.19493811],
                        [0.23374973, 0.28814146, 0.19005357],
                        [0.45762375, 0.23317388, 0.05121415],
                        [0.18061841, 0.16022089, 0.20770057],
                        [0.39544231, 0.16719289, 0.11211669],
                        [0.19535244, 0.10992710, 0.11463679],
                        [0.31359121, 0.29133955, 0.02011782],
                        [0.43297896, 0.25808954, 0.00887722],
                        [0.13763773, 0.12528725, 0.17699082],
                        [0.18486719, 0.24581401, 0.07584951],
                        [0.35452405, 0.13150652, 0.07456490],
                        [0.46239638, 0.32247591, 0.00122477],
                        [0.37084514, 0.15799911, 0.16431224],
                        [0.11738659, 0.20026121, 0.18583715],
                        [0.50368768, 0.41160455, 0.29139283],
                        [0.41439033, 0.33571342, 0.23684677],
                        [0.32857990, 0.26543081, 0.18447344],
                        [0.23778303, 0.18715258, 0.12640880],
                        [0.16784327, 0.13128684, 0.08443624],
                        [0.10924070, 0.07945030, 0.04722051],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.35930955, 0.22294356, 0.11652736],
                        [0.62508970, 0.39426908, 0.24249625],
                        [0.33216712, 0.31617990, 0.28927535],
                        [0.30448821, 0.27402756, 0.10357682],
                        [0.41776320, 0.31983960, 0.30784929],
                        [0.34831995, 0.44018656, 0.29236460],
                        [0.68040967, 0.35255539, 0.06841964],
                        [0.27309224, 0.25288051, 0.33048338],
                        [0.62147486, 0.27055049, 0.18659429],
                        [0.30654705, 0.18056440, 0.19198996],
                        [0.48673603, 0.45989344, 0.03273499],
                        [0.65062267, 0.40057424, 0.01651634],
                        [0.19466802, 0.18593474, 0.27454826],
                        [0.28092542, 0.38499087, 0.12309728],
                        [0.55481929, 0.21417859, 0.12555254],
                        [0.72179741, 0.51570368, 0.00593671],
                        [0.57785285, 0.25778547, 0.26881799],
                        [0.17877635, 0.31785583, 0.29541582],
                        [0.74003130, 0.61015379, 0.43858936],
                        [0.62925828, 0.51779926, 0.37218189],
                        [0.51435107, 0.42165166, 0.29885665],
                        [0.37107259, 0.30354372, 0.20976804],
                        [0.26378226, 0.21566097, 0.14350446],
                        [0.16334273, 0.13377619, 0.08050516],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.34787202, 0.22239830, 0.12071132],
                        [0.58308375, 0.37266096, 0.23445724],
                        [0.31696445, 0.30354100, 0.27743283],
                        [0.30365786, 0.27450961, 0.11830053],
                        [0.42217895, 0.32414007, 0.30887246],
                        [0.35737351, 0.43327817, 0.29111093],
                        [0.62362486, 0.32676762, 0.07449088],
                        [0.25730333, 0.23566855, 0.30452645],
                        [0.57681578, 0.25527635, 0.17708671],
                        [0.30016240, 0.18407927, 0.18981223],
                        [0.48437726, 0.45353514, 0.05076985],
                        [0.64548796, 0.39820802, 0.02829487],
                        [0.18757187, 0.17559552, 0.24984352],
                        [0.26328433, 0.35216329, 0.11459546],
                        [0.51652521, 0.20295261, 0.11941541],
                        [0.69001114, 0.49240762, 0.00375420],
                        [0.56554443, 0.25572431, 0.26536795],
                        [0.19376248, 0.31754220, 0.29425311],
                        [0.69671404, 0.56982183, 0.40269485],
                        [0.58879417, 0.48042127, 0.34024647],
                        [0.48349753, 0.39325854, 0.27507150],
                        [0.36126551, 0.29105616, 0.19889101],
                        [0.26881081, 0.21673009, 0.14677900],
                        [0.18107167, 0.14177044, 0.09304354],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.21935888, 0.12829103, 0.05823010],
                        [0.38621023, 0.23481502, 0.13791171],
                        [0.19868790, 0.18798569, 0.17158148],
                        [0.19399667, 0.17182195, 0.06093559],
                        [0.27950001, 0.21304348, 0.20691113],
                        [0.24463765, 0.30781290, 0.20621555],
                        [0.42518276, 0.21057689, 0.03164171],
                        [0.15967241, 0.14403240, 0.19099186],
                        [0.39287907, 0.16042854, 0.10488193],
                        [0.19517671, 0.11038135, 0.11735366],
                        [0.33836949, 0.31697488, 0.02259050],
                        [0.48356858, 0.29514906, 0.01217067],
                        [0.11187637, 0.10295546, 0.15011585],
                        [0.17635491, 0.23033310, 0.06228105],
                        [0.35300460, 0.12854379, 0.06986750],
                        [0.48630539, 0.34021181, 0.00108740],
                        [0.40561464, 0.17577751, 0.18303318],
                        [0.12991810, 0.23211640, 0.21919341],
                        [0.48198688, 0.37970755, 0.25524086],
                        [0.41261905, 0.32600898, 0.22289814],
                        [0.34072644, 0.27106091, 0.18362711],
                        [0.25940242, 0.20172483, 0.13366182],
                        [0.19378240, 0.15367146, 0.10122680],
                        [0.13230942, 0.10608750, 0.06544701],
                    ]
                ),
            ),
        ]

        np.set_printoptions(
            formatter={"float": "{:0.8f}".format}, suppress=True
        )
        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                detect_colour_checkers_inference(read_image(png_file)),
                test_swatches[i],
                atol=0.0001,
            )

        (
            swatch_colours,
            swatch_masks,
            colour_checker,
        ) = detect_colour_checkers_inference(
            read_image(PNG_FILES[0]), additional_data=True
        )[
            0
        ].values

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


if __name__ == "__main__":
    unittest.main()
