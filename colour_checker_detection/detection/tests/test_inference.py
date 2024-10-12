"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.inference` module.
"""

import glob
import os
import platform
import sys

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

PNG_FILES = sorted(glob.glob(os.path.join(DETECTION_DIRECTORY, "IMG_19*.png")))[:-2]


class TestInferencerDefault:
    """
    Define :func:`colour_checker_detection.detection.inference.\
inferencer_default` definition unit tests methods.
    """

    def test_inferencer_default(self):
        """
        Define :func:`colour_checker_detection.detection.inference.\
inferencer_default` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
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
            assert results[0][0] > 0.85
            assert int(results[0][1]) == 0
            assert results[0][2].shape == shapes[i]


class TestDetectColourCheckersInference:
    """
    Define :func:`colour_checker_detection.detection.inference.\
detect_colour_checkers_inference` definition unit tests methods.
    """

    def test_detect_colour_checkers_inference(self):
        """
        Define :func:`colour_checker_detection.detection.inference.\
detect_colour_checkers_inference` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        # TODO: Enable when "torch" is available on Python 3.13.
        if sys.version_info[1] > 12:  # noqa: YTT203
            return

        test_swatches = [
            np.array(
                [
                    [
                        [0.24868909, 0.15350598, 0.08116075],
                        [0.41540268, 0.25717098, 0.15545619],
                        [0.21684574, 0.20027032, 0.18109877],
                        [0.19675766, 0.17302327, 0.06548654],
                        [0.27416429, 0.20367995, 0.19459581],
                        [0.23362409, 0.28895241, 0.18982816],
                        [0.45757887, 0.23318496, 0.05019996],
                        [0.18031126, 0.16013199, 0.20791255],
                        [0.39581251, 0.16763814, 0.11296371],
                        [0.19527651, 0.10987040, 0.11621131],
                        [0.31525213, 0.29152983, 0.02060614],
                        [0.43262833, 0.25884929, 0.00885420],
                        [0.13825633, 0.12534967, 0.17780316],
                        [0.18896411, 0.24588297, 0.07770619],
                        [0.35498002, 0.13155451, 0.07399154],
                        [0.46168804, 0.32344660, 0.00148291],
                        [0.37211949, 0.15806609, 0.16444303],
                        [0.11952692, 0.20040615, 0.18599187],
                        [0.50398535, 0.41233703, 0.29100367],
                        [0.41492999, 0.33637702, 0.23704307],
                        [0.32918635, 0.26559123, 0.18590339],
                        [0.23806982, 0.18820626, 0.12598225],
                        [0.16772291, 0.13166212, 0.08472667],
                        [0.10938574, 0.07928163, 0.04741153],
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [0.36078018, 0.22207782, 0.11697489],
                        [0.62627727, 0.39423054, 0.24255158],
                        [0.33202896, 0.31646109, 0.28833535],
                        [0.30576822, 0.27346721, 0.10463000],
                        [0.41724655, 0.31947500, 0.30804843],
                        [0.34876767, 0.43955785, 0.29208037],
                        [0.67817026, 0.35152173, 0.07043589],
                        [0.27260271, 0.25263956, 0.33063090],
                        [0.62258488, 0.27038357, 0.18698311],
                        [0.30695105, 0.18043391, 0.19091117],
                        [0.48766014, 0.45945448, 0.03280113],
                        [0.65045166, 0.40045342, 0.01603201],
                        [0.19251584, 0.18574022, 0.27414128],
                        [0.28029913, 0.38490412, 0.12232948],
                        [0.55480623, 0.21456560, 0.12517895],
                        [0.72210294, 0.51611030, 0.00603206],
                        [0.57790947, 0.25799227, 0.26868472],
                        [0.17926446, 0.31794888, 0.29533628],
                        [0.74142271, 0.61076266, 0.43976876],
                        [0.63066620, 0.51781225, 0.37297052],
                        [0.51444507, 0.42198676, 0.30036736],
                        [0.37206760, 0.30385801, 0.20999040],
                        [0.26381230, 0.21583907, 0.14343576],
                        [0.16325143, 0.13384773, 0.08024697],
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [0.34837398, 0.22248265, 0.12170179],
                        [0.58388305, 0.37338290, 0.23275897],
                        [0.31933340, 0.30370870, 0.27962670],
                        [0.30756134, 0.27525371, 0.11792573],
                        [0.41984171, 0.32147259, 0.30852041],
                        [0.35473925, 0.42866808, 0.28895292],
                        [0.62471563, 0.32704774, 0.07252023],
                        [0.25692821, 0.23611195, 0.30394992],
                        [0.57950014, 0.25530624, 0.17775886],
                        [0.30231547, 0.18419002, 0.18951628],
                        [0.48620912, 0.45199829, 0.04738132],
                        [0.64306474, 0.39646858, 0.02802419],
                        [0.18743382, 0.17561989, 0.24962272],
                        [0.26411217, 0.35147783, 0.11445957],
                        [0.51501918, 0.20269375, 0.12031434],
                        [0.69076639, 0.49227387, 0.00245023],
                        [0.56558263, 0.25513411, 0.26413113],
                        [0.18770075, 0.31425264, 0.29354358],
                        [0.69720978, 0.56977922, 0.40252122],
                        [0.58962691, 0.48107553, 0.33920413],
                        [0.48512009, 0.39316276, 0.27548853],
                        [0.36088896, 0.29062313, 0.19837178],
                        [0.26906908, 0.21666811, 0.14570932],
                        [0.17873077, 0.14149866, 0.09204219],
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [0.21954346, 0.12827335, 0.05807348],
                        [0.38594031, 0.23523375, 0.13748619],
                        [0.19902036, 0.18813717, 0.17216082],
                        [0.19323272, 0.17206185, 0.06205419],
                        [0.27916723, 0.21370420, 0.20743532],
                        [0.24624889, 0.30755320, 0.20590794],
                        [0.42592812, 0.21029468, 0.03172819],
                        [0.16014631, 0.14433359, 0.19122502],
                        [0.39261296, 0.16085321, 0.10504565],
                        [0.19512193, 0.11113719, 0.11700730],
                        [0.33913988, 0.31777325, 0.02248015],
                        [0.48319355, 0.29541984, 0.01134784],
                        [0.11160174, 0.10332654, 0.15007614],
                        [0.17605095, 0.23046510, 0.06220178],
                        [0.35337761, 0.12851048, 0.06960399],
                        [0.48561531, 0.34026867, 0.00105998],
                        [0.40623522, 0.17625771, 0.18369043],
                        [0.13130365, 0.23233752, 0.21957417],
                        [0.48197183, 0.37976372, 0.25521374],
                        [0.41217238, 0.32606238, 0.22287071],
                        [0.34036109, 0.27108088, 0.18371078],
                        [0.25957564, 0.20170440, 0.13371080],
                        [0.19509768, 0.15370770, 0.10181070],
                        [0.13300470, 0.10627789, 0.06550305],
                    ]
                ]
            ),
        ]

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
