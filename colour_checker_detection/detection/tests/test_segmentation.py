# !/usr/bin/env python
"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.segmentation` module.
"""

import glob
import os
import platform
import unittest

import numpy as np
from colour import read_image
from colour.models import cctf_encoding

from colour_checker_detection import ROOT_RESOURCES_TESTS
from colour_checker_detection.detection import (
    colour_checkers_coordinates_segmentation,
    detect_colour_checkers_segmentation,
    extract_colour_checkers_segmentation,
)
from colour_checker_detection.detection.segmentation import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    adjust_image,
    as_8_bit_BGR_image,
    contour_centroid,
    crop_and_level_image_with_rectangle,
    is_square,
    scale_contour,
    swatch_masks,
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
    "TestAs8BitBGRImage",
    "TestAdjustImage",
    "TestIsSquare",
    "TestContourCentroid",
    "TestScaleContour",
    "TestCropAndLevelImageWithRectangle",
    "TestColourCheckersCoordinatesSegmentation",
    "TestExtractColourCheckersSegmentation",
    "TestDetectColourCheckersSegmentation",
]

DETECTION_DIRECTORY = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_checker_detection", "detection"
)

PNG_FILES = glob.glob(os.path.join(DETECTION_DIRECTORY, "IMG_19*.png"))


class TestSwatchMasks(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
swatch_masks` definition unit tests methods.
    """

    def test_swatch_masks(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
swatch_masks` definition unit tests methods.
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


class TestAs8BitBGRImage(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
as_8_bit_BGR_image` definition unit tests methods.
    """

    def test_as_8_bit_BGR_image(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
as_8_bit_BGR_image` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        image_i = read_image(PNG_FILES[0])
        image_o = as_8_bit_BGR_image(image_i)

        self.assertEqual(image_o.dtype, np.uint8)
        np.testing.assert_array_almost_equal(
            image_o[16, 16, ...],
            (cctf_encoding(image_i[16, 16, ::-1]) * 255).astype(np.uint8),
        )


class TestAdjustImage(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
adjust_image` definition unit tests methods.
    """

    def test_adjust_image(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
adjust_image` definition unit tests methods.
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


class TestIsSquare(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
is_square` definition unit tests methods.
    """

    def test_is_square(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
is_square` definition unit tests methods.
        """

        shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertTrue(is_square(shape))

        shape = np.array([[0, 0.5], [1, 0], [1, 1], [0, 1]])
        self.assertFalse(is_square(shape))
        self.assertTrue(is_square(shape, 0.5))


class TestContourCentroid(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
contour_centroid` definition unit tests methods.
    """

    def test_contour_centroid(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
contour_centroid` definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(contour_centroid(contour), (0.5, 0.5))


class TestScaleContour(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
scale_contour` definition unit tests methods.
    """

    def test_scale_contour(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
scale_contour` definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(
            scale_contour(contour, 2),
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        )

        np.testing.assert_array_equal(
            scale_contour(contour, 0.5),
            np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]),
        )


class TestCropAndLevelImageWithRectangle(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
crop_and_level_image_with_rectangle` definition unit tests methods.
    """

    def test_crop_and_level_image_with_rectangle(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
crop_and_level_image_with_rectangle` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        image = as_8_bit_BGR_image(
            adjust_image(read_image(PNG_FILES[0]), 1440)
        )
        rectangle = (
            (832.99865723, 473.05020142),
            (209.08610535, 310.13061523),
            -88.35559082,
        )

        np.testing.assert_array_equal(
            crop_and_level_image_with_rectangle(image, rectangle).shape,
            (209, 310, 3),
        )


class TestColourCheckersCoordinatesSegmentation(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
colour_checkers_coordinates_segmentation` definition unit tests methods.
    """

    def test_colour_checkers_coordinates_segmentation(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
colour_checkers_coordinates_segmentation` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        colour_checkers_coordinates = [
            (np.array([[640, 333], [795, 333], [795, 438], [640, 438]]),),
            (np.array([[764, 294], [1006, 294], [1006, 652], [764, 652]]),),
            (np.array([[365, 685], [382, 222], [1078, 247], [1061, 710]]),),
            (np.array([[675, 574], [681, 365], [991, 374], [985, 583]]),),
            (np.array([[576, 667], [580, 362], [1038, 367], [1035, 672]]),),
            (np.array([[622, 595], [626, 311], [1052, 317], [1048, 601]]),),
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                colour_checkers_coordinates_segmentation(read_image(png_file)),
                colour_checkers_coordinates[i],
                atol=5,
            )

        (
            colour_checkers,
            clusters,
            swatches,
            segmented_image,
        ) = colour_checkers_coordinates_segmentation(
            read_image(PNG_FILES[0]), additional_data=True
        ).values

        np.testing.assert_allclose(
            colour_checkers,
            colour_checkers_coordinates[0],
            atol=5,
        )

        np.testing.assert_allclose(
            clusters,
            (
                np.array([[627, 482], [783, 482], [783, 580], [627, 580]]),
                np.array([[640, 333], [795, 333], [795, 438], [640, 438]]),
            ),
            atol=5,
        )

        np.testing.assert_allclose(
            swatches,
            (
                ([[696, 562], [713, 562], [713, 579], [696, 579]]),
                ([[629, 561], [647, 561], [647, 579], [629, 579]]),
                ([[760, 536], [782, 536], [782, 558], [760, 558]]),
                ([[732, 535], [754, 535], [754, 557], [732, 557]]),
                ([[705, 535], [727, 535], [727, 557], [705, 557]]),
                ([[678, 535], [699, 535], [699, 557], [678, 557]]),
                ([[650, 556], [650, 534], [672, 534], [672, 556]]),
                ([[760, 508], [782, 508], [782, 530], [760, 530]]),
                ([[732, 508], [754, 508], [754, 530], [732, 530]]),
                ([[705, 508], [727, 508], [727, 529], [705, 529]]),
                ([[678, 529], [678, 507], [700, 507], [700, 529]]),
                ([[650, 507], [672, 507], [672, 529], [650, 529]]),
                ([[697, 485], [714, 485], [714, 502], [697, 502]]),
                ([[630, 502], [630, 484], [647, 484], [647, 502]]),
                ([[745, 414], [766, 414], [766, 434], [745, 434]]),
                ([[719, 434], [719, 413], [740, 413], [740, 434]]),
                ([[669, 413], [689, 413], [689, 433], [669, 433]]),
                ([[643, 413], [663, 413], [663, 433], [643, 433]]),
                ([[695, 436], [695, 412], [714, 412], [714, 436]]),
                ([[771, 409], [771, 388], [792, 388], [792, 409]]),
                ([[746, 408], [746, 388], [766, 388], [766, 408]]),
                ([[720, 388], [740, 388], [740, 408], [720, 408]]),
                ([[669, 407], [669, 387], [689, 387], [689, 407]]),
                ([[773, 383], [773, 366], [794, 366], [794, 383]]),
                ([[746, 383], [746, 364], [766, 364], [766, 383]]),
                ([[695, 362], [715, 362], [715, 382], [695, 382]]),
                ([[670, 362], [690, 362], [690, 382], [670, 382]]),
                ([[644, 361], [664, 361], [664, 382], [644, 382]]),
                ([[771, 337], [792, 337], [792, 358], [771, 358]]),
                ([[746, 337], [766, 337], [766, 357], [746, 357]]),
                ([[720, 337], [740, 337], [740, 357], [720, 357]]),
                ([[695, 357], [695, 336], [715, 336], [715, 357]]),
            ),
            atol=5,
        )

        np.testing.assert_allclose(
            segmented_image.shape,
            (958, 1440),
            atol=5,
        )


class TestExtractColourCheckersSegmentation(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
extract_colour_checkers_segmentation` definition unit tests methods.
    """

    def test_extract_colour_checkers_segmentation(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
extract_colour_checkers_segmentation` definition unit tests methods.
        """

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        colour_checkers_shapes = np.array(
            [
                [(105, 155, 3)],
                [(241, 357, 3)],
                [(463, 696, 3)],
                [(209, 310, 3)],
                [(305, 459, 3)],
                [(284, 426, 3)],
            ]
        )

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                [
                    colour_checker.shape
                    for colour_checker in extract_colour_checkers_segmentation(
                        read_image(png_file)
                    )
                ],
                colour_checkers_shapes[i],
                atol=5,
            )


class TestDetectColourCheckersSegmentation(unittest.TestCase):
    """
    Define :func:`colour_checker_detection.detection.segmentation.\
detect_colour_checkers_segmentation` definition unit tests methods.
    """

    def test_detect_colour_checkers_segmentation(self):
        """
        Define :func:`colour_checker_detection.detection.segmentation.\
detect_colour_checkers_segmentation` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        test_swatches = [
            (
                np.array(
                    [
                        [3.59111789e-01, 2.19585960e-01, 1.13242585e-01],
                        [6.11375859e-01, 3.82348203e-01, 2.35521738e-01],
                        [3.17052209e-01, 3.03401395e-01, 2.80831650e-01],
                        [3.00868150e-01, 2.70973564e-01, 1.04340101e-01],
                        [4.27589523e-01, 3.29751740e-01, 3.23010681e-01],
                        [3.76707634e-01, 4.73432755e-01, 3.22061600e-01],
                        [6.63490287e-01, 3.43289388e-01, 5.26502890e-02],
                        [2.51196615e-01, 2.31823610e-01, 3.09359686e-01],
                        [5.98445258e-01, 2.57178276e-01, 1.76880634e-01],
                        [2.90977047e-01, 1.68648050e-01, 1.85927482e-01],
                        [5.00261083e-01, 4.71408363e-01, 2.62840838e-02],
                        [6.91688578e-01, 4.28901145e-01, 8.98280160e-03],
                        [1.83628775e-01, 1.66755416e-01, 2.56656486e-01],
                        [2.74258081e-01, 3.62026525e-01, 1.12746450e-01],
                        [5.40437937e-01, 2.00225055e-01, 1.18382021e-01],
                        [7.21408678e-01, 5.11678935e-01, 2.46123150e-04],
                        [5.81894909e-01, 2.58429305e-01, 2.72910893e-01],
                        [1.75151369e-01, 3.32180413e-01, 3.14811845e-01],
                        [7.59493128e-01, 6.11989269e-01, 4.25430930e-01],
                        [6.31990565e-01, 5.12169545e-01, 3.60544908e-01],
                        [5.07482000e-01, 4.12817130e-01, 2.88771690e-01],
                        [3.82708522e-01, 3.05064983e-01, 2.10084265e-01],
                        [2.69205284e-01, 2.17579729e-01, 1.47982758e-01],
                        [1.72673535e-01, 1.37549006e-01, 8.70347510e-02],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.52559307, 0.33428539, 0.18499975],
                        [0.88770956, 0.57002551, 0.36210461],
                        [0.47194692, 0.45505569, 0.4253372],
                        [0.44549033, 0.4065853, 0.16505868],
                        [0.62016679, 0.47918434, 0.47222468],
                        [0.53634524, 0.66547438, 0.45442796],
                        [0.94609384, 0.5039193, 0.09566254],
                        [0.37087743, 0.34980679, 0.46413095],
                        [0.86227225, 0.38319543, 0.27081925],
                        [0.42952385, 0.2563723, 0.28148704],
                        [0.7090406, 0.67451328, 0.04574433],
                        [0.96657426, 0.60488708, 0.01830741],
                        [0.26786693, 0.25040147, 0.37929101],
                        [0.39253821, 0.52487103, 0.17380688],
                        [0.76544996, 0.29529122, 0.18075902],
                        [0.99600647, 0.72455925, 0.00180694],
                        [0.81283947, 0.36777656, 0.39310485],
                        [0.2347535, 0.46610054, 0.44507379],
                        [0.99929421, 0.84808996, 0.60653093],
                        [0.87382741, 0.71794711, 0.51701245],
                        [0.70495748, 0.58175102, 0.41833053],
                        [0.53024677, 0.43065902, 0.30339399],
                        [0.37846916, 0.30754826, 0.21584367],
                        [0.23499116, 0.18916327, 0.12443729],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.3604223, 0.22352724, 0.1187593],
                        [0.62719536, 0.39481536, 0.24229516],
                        [0.33165357, 0.3156168, 0.28913772],
                        [0.30472833, 0.27362183, 0.10432951],
                        [0.4168942, 0.3197342, 0.3080623],
                        [0.34742638, 0.4415272, 0.29351243],
                        [0.68119717, 0.35294917, 0.07219157],
                        [0.27226624, 0.25338253, 0.33126962],
                        [0.62066275, 0.2703675, 0.18658477],
                        [0.3068099, 0.17993326, 0.19172294],
                        [0.48657364, 0.45990756, 0.03377367],
                        [0.6523077, 0.40164948, 0.01614917],
                        [0.19164474, 0.18564115, 0.27356568],
                        [0.2793217, 0.3852577, 0.12227942],
                        [0.5540942, 0.2143543, 0.12548569],
                        [0.7213888, 0.5158426, 0.00546947],
                        [0.5781947, 0.2579714, 0.26868656],
                        [0.17703038, 0.31641382, 0.29574612],
                        [0.74389654, 0.6117858, 0.43972403],
                        [0.6297197, 0.5179646, 0.37325087],
                        [0.5145417, 0.42146567, 0.29868528],
                        [0.37115434, 0.3037116, 0.20961109],
                        [0.26532546, 0.21578446, 0.14347702],
                        [0.16331832, 0.13397452, 0.08070049],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.2488954, 0.15317793, 0.08199862],
                        [0.41607243, 0.25705677, 0.15547077],
                        [0.21563996, 0.1995038, 0.18032782],
                        [0.19695863, 0.17280695, 0.06623787],
                        [0.27329034, 0.20329261, 0.19478495],
                        [0.23463742, 0.28968006, 0.18990019],
                        [0.45766518, 0.23289812, 0.05205948],
                        [0.17926256, 0.16008899, 0.20721273],
                        [0.39577234, 0.16690929, 0.11228606],
                        [0.19279528, 0.10992336, 0.1145397],
                        [0.31474733, 0.29151246, 0.01935229],
                        [0.4329992, 0.25889504, 0.0085974],
                        [0.13864459, 0.12494186, 0.17611597],
                        [0.18529654, 0.24500132, 0.07599235],
                        [0.35326675, 0.13125867, 0.07497095],
                        [0.46260715, 0.32276395, 0.00188213],
                        [0.37161955, 0.15807284, 0.16387214],
                        [0.11466587, 0.20015398, 0.18587948],
                        [0.5032432, 0.41186845, 0.2906085],
                        [0.41381165, 0.3353557, 0.23686318],
                        [0.329134, 0.26522735, 0.18421698],
                        [0.23773287, 0.18706763, 0.12534218],
                        [0.16809067, 0.13127366, 0.08450561],
                        [0.10719948, 0.07965489, 0.04696838],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.34730858, 0.22198382, 0.12156568],
                        [0.58284885, 0.37259457, 0.23408672],
                        [0.31723604, 0.30341142, 0.27804196],
                        [0.30376184, 0.27455556, 0.11838572],
                        [0.4213095, 0.32402194, 0.30899087],
                        [0.3594815, 0.43695083, 0.29322192],
                        [0.6236764, 0.32682335, 0.07517024],
                        [0.25809756, 0.23572059, 0.30381173],
                        [0.576985, 0.25527695, 0.17703938],
                        [0.30071345, 0.18336897, 0.18986957],
                        [0.4840834, 0.45427305, 0.05083579],
                        [0.64937377, 0.39936244, 0.0275686],
                        [0.18818328, 0.1758956, 0.24927449],
                        [0.26252288, 0.35170296, 0.11353981],
                        [0.51517504, 0.20295215, 0.11945031],
                        [0.6903043, 0.49266025, 0.00377388],
                        [0.5655643, 0.25562114, 0.26504168],
                        [0.19228458, 0.31727245, 0.29531002],
                        [0.69715834, 0.5699175, 0.40268826],
                        [0.58900243, 0.4805662, 0.34061125],
                        [0.48388335, 0.39298296, 0.27496526],
                        [0.36165974, 0.2906192, 0.19971861],
                        [0.2695179, 0.21655917, 0.14629818],
                        [0.1796263, 0.1427241, 0.09330364],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.22016735, 0.12827504, 0.0579543],
                        [0.38682458, 0.23528327, 0.13792013],
                        [0.19906875, 0.18848994, 0.17247298],
                        [0.19388746, 0.17252938, 0.06136553],
                        [0.2802787, 0.21345483, 0.20764254],
                        [0.24673642, 0.30888048, 0.20729934],
                        [0.42634252, 0.21066168, 0.03271125],
                        [0.16120888, 0.14459828, 0.19191903],
                        [0.39259893, 0.16067706, 0.10573678],
                        [0.19603245, 0.11122748, 0.11785921],
                        [0.3395359, 0.31840178, 0.02238519],
                        [0.48402014, 0.2953641, 0.0108898],
                        [0.1123625, 0.10246556, 0.15069418],
                        [0.17383477, 0.23009333, 0.06224241],
                        [0.35488874, 0.12860598, 0.06979445],
                        [0.48709735, 0.3405046, 0.00158875],
                        [0.40679526, 0.17632408, 0.1855304],
                        [0.13376327, 0.23343821, 0.22016124],
                        [0.482124, 0.3799315, 0.25568947],
                        [0.41329905, 0.32676035, 0.22327706],
                        [0.34211567, 0.27198744, 0.18467663],
                        [0.25739494, 0.2012522, 0.13348623],
                        [0.19565332, 0.15490581, 0.10306674],
                        [0.13583797, 0.10635002, 0.06712713],
                    ]
                ),
            ),
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                detect_colour_checkers_segmentation(read_image(png_file)),
                test_swatches[i],
                atol=0.0001,
            )

        (
            swatch_colours,
            colour_checker_image,
            swatch_masks,
        ) = detect_colour_checkers_segmentation(
            read_image(PNG_FILES[0]), additional_data=True
        )[
            0
        ].values

        np.testing.assert_allclose(
            swatch_colours,
            test_swatches[0][0],
            atol=0.0001,
        )

        np.testing.assert_allclose(
            colour_checker_image.shape[0:2],
            (105, 155),
            atol=5,
        )

        np.testing.assert_allclose(
            swatch_masks,
            (
                ([5, 21, 4, 20]),
                ([5, 21, 30, 46]),
                ([5, 21, 56, 72]),
                ([5, 21, 82, 98]),
                ([5, 21, 108, 124]),
                ([5, 21, 134, 150]),
                ([31, 47, 4, 20]),
                ([31, 47, 30, 46]),
                ([31, 47, 56, 72]),
                ([31, 47, 82, 98]),
                ([31, 47, 108, 124]),
                ([31, 47, 134, 150]),
                ([57, 73, 4, 20]),
                ([57, 73, 30, 46]),
                ([57, 73, 56, 72]),
                ([57, 73, 82, 98]),
                ([57, 73, 108, 124]),
                ([57, 73, 134, 150]),
                ([83, 99, 4, 20]),
                ([83, 99, 30, 46]),
                ([83, 99, 56, 72]),
                ([83, 99, 82, 98]),
                ([83, 99, 108, 124]),
                ([83, 99, 134, 150]),
            ),
            atol=5,
        )


if __name__ == "__main__":
    unittest.main()
