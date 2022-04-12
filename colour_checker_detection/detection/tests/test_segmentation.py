# !/usr/bin/env python
"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.segmentation` module.
"""

import glob
import numpy as np
import os
import platform
import unittest

from colour import read_image
from colour.models import cctf_encoding

from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
from colour_checker_detection.detection.segmentation import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    swatch_masks,
    as_8_bit_BGR_image,
    adjust_image,
    is_square,
    contour_centroid,
    scale_contour,
    crop_and_level_image_with_rectangle,
)
from colour_checker_detection.detection import (
    colour_checkers_coordinates_segmentation,
    extract_colour_checkers_segmentation,
    detect_colour_checkers_segmentation,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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
    TESTS_RESOURCES_DIRECTORY, "colour_checker_detection", "detection"
)

PNG_FILES = glob.glob(os.path.join(DETECTION_DIRECTORY, "*.png"))


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
        np.testing.assert_almost_equal(
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
            (
                np.array(
                    [
                        [640, 333],
                        [795, 333],
                        [795, 437],
                        [640, 437],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [761, 650],
                        [765, 293],
                        [1007, 295],
                        [1003, 652],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [369, 688],
                        [382, 226],
                        [1078, 246],
                        [1065, 707],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [675, 573],
                        [681, 365],
                        [991, 373],
                        [985, 582],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [573, 665],
                        [577, 357],
                        [1039, 364],
                        [1034, 672],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [622, 596],
                        [624, 311],
                        [1051, 315],
                        [1048, 600],
                    ]
                ),
            ),
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                colour_checkers_coordinates_segmentation(read_image(png_file)),
                colour_checkers_coordinates[i],
                rtol=5,
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
            rtol=5,
            atol=5,
        )

        np.testing.assert_allclose(
            clusters,
            np.array([[[985, 582], [675, 573], [681, 365], [991, 373]]]),
            rtol=5,
            atol=5,
        )

        np.testing.assert_allclose(
            swatches,
            [
                [
                    [785, 581],
                    [785, 562],
                    [804, 562],
                    [804, 581],
                ],
                [
                    [696, 579],
                    [696, 562],
                    [713, 562],
                    [713, 579],
                ],
                [
                    [629, 578],
                    [629, 560],
                    [647, 560],
                    [647, 578],
                ],
                [
                    [782, 557],
                    [760, 557],
                    [760, 535],
                    [782, 535],
                ],
                [
                    [754, 557],
                    [732, 557],
                    [732, 535],
                    [754, 535],
                ],
                [
                    [705, 556],
                    [705, 534],
                    [727, 534],
                    [727, 556],
                ],
                [
                    [699, 556],
                    [678, 556],
                    [678, 534],
                    [699, 534],
                ],
                [
                    [672, 556],
                    [650, 556],
                    [650, 534],
                    [672, 534],
                ],
                [
                    [782, 530],
                    [760, 530],
                    [760, 507],
                    [782, 507],
                ],
                [
                    [732, 529],
                    [732, 507],
                    [754, 507],
                    [754, 529],
                ],
                [
                    [705, 529],
                    [705, 507],
                    [727, 507],
                    [727, 529],
                ],
                [
                    [678, 529],
                    [678, 507],
                    [700, 507],
                    [700, 529],
                ],
                [
                    [650, 529],
                    [650, 507],
                    [672, 507],
                    [672, 529],
                ],
                [
                    [647, 501],
                    [630, 501],
                    [630, 484],
                    [647, 484],
                ],
                [
                    [745, 434],
                    [745, 413],
                    [766, 413],
                    [766, 434],
                ],
                [
                    [719, 433],
                    [719, 413],
                    [740, 413],
                    [740, 433],
                ],
                [
                    [669, 433],
                    [669, 413],
                    [689, 413],
                    [689, 433],
                ],
                [
                    [695, 435],
                    [695, 412],
                    [715, 412],
                    [715, 435],
                ],
                [
                    [643, 433],
                    [643, 412],
                    [663, 412],
                    [663, 433],
                ],
                [
                    [771, 409],
                    [771, 388],
                    [792, 388],
                    [792, 409],
                ],
                [
                    [746, 408],
                    [746, 388],
                    [766, 388],
                    [766, 408],
                ],
                [
                    [720, 408],
                    [720, 387],
                    [740, 387],
                    [740, 408],
                ],
                [
                    [669, 407],
                    [669, 387],
                    [689, 387],
                    [689, 407],
                ],
                [
                    [773, 383],
                    [773, 366],
                    [794, 366],
                    [794, 383],
                ],
                [
                    [722, 382],
                    [722, 365],
                    [740, 365],
                    [740, 382],
                ],
                [
                    [746, 382],
                    [746, 363],
                    [766, 363],
                    [766, 382],
                ],
                [
                    [670, 382],
                    [670, 361],
                    [690, 361],
                    [690, 382],
                ],
                [
                    [644, 381],
                    [644, 361],
                    [664, 361],
                    [664, 381],
                ],
                [
                    [792, 357],
                    [771, 357],
                    [771, 337],
                    [792, 337],
                ],
                [
                    [766, 357],
                    [746, 357],
                    [746, 337],
                    [766, 337],
                ],
                [
                    [720, 357],
                    [720, 336],
                    [740, 336],
                    [740, 357],
                ],
                [
                    [695, 356],
                    [695, 336],
                    [715, 336],
                    [715, 356],
                ],
            ],
            rtol=5,
            atol=5,
        )

        np.testing.assert_allclose(
            segmented_image.shape,
            (958, 1440),
            rtol=5,
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

        colour_checkers_shapes = np.array(
            [
                [(209, 310, 3)],
                [(462, 696, 3)],
                [(307, 462, 3)],
                [(285, 426, 3)],
                [(104, 155, 3)],
                [(241, 357, 3)],
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
                rtol=5,
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
                        [3.59135091e-01, 2.19796017e-01, 1.13940984e-01],
                        [6.11350119e-01, 3.82453948e-01, 2.35822365e-01],
                        [3.17555845e-01, 3.03430647e-01, 2.80962378e-01],
                        [3.00975740e-01, 2.70867497e-01, 1.05559312e-01],
                        [4.27512020e-01, 3.29888999e-01, 3.23142976e-01],
                        [3.76991749e-01, 4.73176211e-01, 3.22182775e-01],
                        [6.63555682e-01, 3.43329966e-01, 5.56962527e-02],
                        [2.51527131e-01, 2.32076332e-01, 3.09353679e-01],
                        [5.98495960e-01, 2.57286400e-01, 1.77112564e-01],
                        [2.91366130e-01, 1.68986112e-01, 1.85891747e-01],
                        [5.00188053e-01, 4.71660137e-01, 2.81275827e-02],
                        [6.91424429e-01, 4.29082453e-01, 1.01405950e-02],
                        [1.84073970e-01, 1.66931704e-01, 2.56890744e-01],
                        [2.74631500e-01, 3.61989856e-01, 1.13976061e-01],
                        [5.40477335e-01, 2.00360253e-01, 1.18817203e-01],
                        [7.21281230e-01, 5.11703551e-01, 2.84224603e-04],
                        [5.82268715e-01, 2.58642584e-01, 2.73099512e-01],
                        [1.76658988e-01, 3.32206041e-01, 3.14804256e-01],
                        [7.60072112e-01, 6.12240493e-01, 4.25461680e-01],
                        [6.32735372e-01, 5.12524962e-01, 3.60872835e-01],
                        [5.07732809e-01, 4.13086951e-01, 2.88933039e-01],
                        [3.82734120e-01, 3.05057764e-01, 2.10200354e-01],
                        [2.69534051e-01, 2.17777327e-01, 1.48127720e-01],
                        [1.72729716e-01, 1.37724116e-01, 8.77513140e-02],
                    ],
                ),
            ),
            (
                np.array(
                    [
                        [0.52756390, 0.33409548, 0.18627323],
                        [0.88773686, 0.56965977, 0.36253598],
                        [0.47228992, 0.45506176, 0.42488652],
                        [0.44654600, 0.40695095, 0.16546243],
                        [0.61917967, 0.47957844, 0.47256800],
                        [0.53719630, 0.66512376, 0.45406640],
                        [0.94689820, 0.50445830, 0.10463549],
                        [0.37162787, 0.35053793, 0.46481314],
                        [0.86292750, 0.38395333, 0.27230266],
                        [0.43009907, 0.25669205, 0.28193754],
                        [0.70892240, 0.67454440, 0.05742301],
                        [0.96584123, 0.60484190, 0.02137790],
                        [0.26997070, 0.25065875, 0.37982088],
                        [0.39374420, 0.52499750, 0.17823155],
                        [0.76600980, 0.29578838, 0.18248339],
                        [0.99625840, 0.72449340, 0.00176726],
                        [0.81199276, 0.36806000, 0.39334726],
                        [0.23785806, 0.46591488, 0.44494370],
                        [0.99939430, 0.84789670, 0.60762470],
                        [0.87401450, 0.71789000, 0.51758230],
                        [0.70589740, 0.58195670, 0.41843010],
                        [0.53050977, 0.43094724, 0.30446407],
                        [0.37936595, 0.30739483, 0.21599181],
                        [0.23655598, 0.18945555, 0.12693338],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.36162690, 0.22410667, 0.11878378],
                        [0.62805945, 0.39508832, 0.24347660],
                        [0.33262320, 0.31561825, 0.28910384],
                        [0.30484140, 0.27389738, 0.10699851],
                        [0.41748697, 0.31996697, 0.30815527],
                        [0.34787300, 0.44131935, 0.29316148],
                        [0.68163010, 0.35390508, 0.07533977],
                        [0.27310503, 0.25284675, 0.33129203],
                        [0.61923355, 0.27038336, 0.18663871],
                        [0.30685670, 0.18033665, 0.19198073],
                        [0.48663542, 0.45940048, 0.03741863],
                        [0.65185230, 0.40106082, 0.01718869],
                        [0.19415711, 0.18558015, 0.27506325],
                        [0.27999467, 0.38546097, 0.12410387],
                        [0.55374810, 0.21390040, 0.12673323],
                        [0.72080450, 0.51529040, 0.00619462],
                        [0.57783604, 0.25785333, 0.26879928],
                        [0.18094501, 0.31747428, 0.29599026],
                        [0.74275220, 0.61075540, 0.43984395],
                        [0.62961080, 0.51776063, 0.37280324],
                        [0.51395893, 0.42163070, 0.29926940],
                        [0.37044013, 0.30339270, 0.20930897],
                        [0.26418540, 0.21540071, 0.14412679],
                        [0.16500981, 0.13452393, 0.08174373],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.24899970, 0.15329513, 0.08204897],
                        [0.41600925, 0.25695294, 0.15538970],
                        [0.21601720, 0.19951240, 0.18035220],
                        [0.19743803, 0.17275658, 0.06657911],
                        [0.27315283, 0.20332055, 0.19492690],
                        [0.23477884, 0.28963318, 0.19012354],
                        [0.45784200, 0.23288018, 0.05315211],
                        [0.17953615, 0.16007920, 0.20722930],
                        [0.39575022, 0.16697972, 0.11242444],
                        [0.19297591, 0.11014009, 0.11486403],
                        [0.31489920, 0.29152456, 0.01970659],
                        [0.43349510, 0.25906911, 0.00869834],
                        [0.13871042, 0.12494299, 0.17615323],
                        [0.18534148, 0.24493850, 0.07676452],
                        [0.35320520, 0.13134533, 0.07502046],
                        [0.46276328, 0.32283294, 0.00164528],
                        [0.37163872, 0.15798312, 0.16394606],
                        [0.11516723, 0.20027927, 0.18598069],
                        [0.50357120, 0.41225713, 0.29077667],
                        [0.41387890, 0.33538613, 0.23684000],
                        [0.32938275, 0.26531030, 0.18418522],
                        [0.23757711, 0.18711837, 0.12562586],
                        [0.16830724, 0.13147430, 0.08490208],
                        [0.10761920, 0.07978929, 0.04749516],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.34849316, 0.22257258, 0.12145711],
                        [0.58353340, 0.37294240, 0.23449922],
                        [0.31712460, 0.30358097, 0.27860895],
                        [0.30434608, 0.27504350, 0.11997047],
                        [0.42171988, 0.32460594, 0.30943406],
                        [0.35964364, 0.43669078, 0.29267043],
                        [0.62359790, 0.32662240, 0.07844281],
                        [0.25833362, 0.23585865, 0.30367905],
                        [0.57666445, 0.25537570, 0.17756248],
                        [0.30080100, 0.18376847, 0.19022983],
                        [0.48485094, 0.45427766, 0.05356237],
                        [0.64873170, 0.39921263, 0.02978591],
                        [0.18815327, 0.17610477, 0.24940938],
                        [0.26319380, 0.35211080, 0.11492524],
                        [0.51515460, 0.20293087, 0.11956533],
                        [0.69080410, 0.49294570, 0.00357560],
                        [0.56598730, 0.25593892, 0.26515755],
                        [0.19344516, 0.31749627, 0.29541177],
                        [0.69728480, 0.56969213, 0.40341163],
                        [0.58938280, 0.48067078, 0.34070572],
                        [0.48363292, 0.39276072, 0.27507660],
                        [0.36210418, 0.29088630, 0.20011935],
                        [0.27015495, 0.21669416, 0.14708833],
                        [0.18023512, 0.14293702, 0.09383766],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [0.22030936, 0.12823439, 0.05837110],
                        [0.38676384, 0.23527192, 0.13789266],
                        [0.19912052, 0.18842259, 0.17246534],
                        [0.19406164, 0.17242720, 0.06169559],
                        [0.28042048, 0.21353368, 0.20798405],
                        [0.24619524, 0.30837760, 0.20659286],
                        [0.42645420, 0.21086293, 0.03303316],
                        [0.16146795, 0.14470735, 0.19204976],
                        [0.39259452, 0.16076456, 0.10586262],
                        [0.19669615, 0.11152513, 0.11831117],
                        [0.33953740, 0.31845555, 0.02318342],
                        [0.48390120, 0.29534210, 0.01077500],
                        [0.11324691, 0.10292091, 0.15061907],
                        [0.17446674, 0.23027256, 0.06380728],
                        [0.35519427, 0.12887615, 0.07051251],
                        [0.48733094, 0.34079686, 0.00148997],
                        [0.40726367, 0.17645681, 0.18597038],
                        [0.13456236, 0.23352131, 0.22043468],
                        [0.48234250, 0.38024990, 0.25605250],
                        [0.41350517, 0.32686144, 0.22346000],
                        [0.34250504, 0.27191713, 0.18499231],
                        [0.25733090, 0.20132765, 0.13358471],
                        [0.19622569, 0.15520023, 0.10352947],
                        [0.13602558, 0.10674744, 0.06762378],
                    ]
                ),
            ),
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                detect_colour_checkers_segmentation(read_image(png_file)),
                test_swatches[i],
                rtol=0.0001,
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
            rtol=0.0001,
            atol=0.0001,
        )

        np.testing.assert_allclose(
            colour_checker_image.shape[0:2],
            (209, 310),
            rtol=5,
            atol=5,
        )

        np.testing.assert_allclose(
            swatch_masks,
            [
                [18, 34, 17, 33],
                [18, 34, 69, 85],
                [18, 34, 121, 137],
                [18, 34, 172, 188],
                [18, 34, 224, 240],
                [18, 34, 276, 292],
                [70, 86, 17, 33],
                [70, 86, 69, 85],
                [70, 86, 121, 137],
                [70, 86, 172, 188],
                [70, 86, 224, 240],
                [70, 86, 276, 292],
                [122, 138, 17, 33],
                [122, 138, 69, 85],
                [122, 138, 121, 137],
                [122, 138, 172, 188],
                [122, 138, 224, 240],
                [122, 138, 276, 292],
                [174, 190, 17, 33],
                [174, 190, 69, 85],
                [174, 190, 121, 137],
                [174, 190, 172, 188],
                [174, 190, 224, 240],
                [174, 190, 276, 292],
            ],
            rtol=5,
            atol=5,
        )


if __name__ == "__main__":
    unittest.main()
