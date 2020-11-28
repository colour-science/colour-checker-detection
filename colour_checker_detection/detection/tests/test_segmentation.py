# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour_checker_detection.detection.segmentation`
module.
"""

from __future__ import division, unicode_literals

import glob
import numpy as np
import os
import platform
import unittest

from colour import read_image
from colour.models import cctf_encoding

from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
from colour_checker_detection.detection.segmentation import (
    WORKING_WIDTH, swatch_masks, as_8_bit_BGR_image, adjust_image, is_square,
    contour_centroid, scale_contour, crop_and_level_image_with_rectangle)
from colour_checker_detection.detection import (
    colour_checkers_coordinates_segmentation,
    extract_colour_checkers_segmentation, detect_colour_checkers_segmentation)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2018-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'DETECTION_DIRECTORY', 'PNG_FILES', 'TestSwatchMasks',
    'TestAs8BitBGRImage', 'TestAdjustImage', 'TestIsSquare',
    'TestContourCentroid', 'TestScaleContour',
    'TestCropAndLevelImageWithRectangle',
    'TestColourCheckersCoordinatesSegmentation',
    'TestExtractColourCheckersSegmentation',
    'TestDetectColourCheckersSegmentation'
]

DETECTION_DIRECTORY = os.path.join(TESTS_RESOURCES_DIRECTORY,
                                   'colour_checker_detection', 'detection')

PNG_FILES = glob.glob(os.path.join(DETECTION_DIRECTORY, '*.png'))


class TestSwatchMasks(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
swatch_masks` definition unit tests methods.
    """

    def test_swatch_masks(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
swatch_masks` definition unit tests methods.
        """

        np.testing.assert_equal(
            swatch_masks(16, 8, 4, 2, 2),
            np.array([
                [1, 3, 1, 3],
                [1, 3, 5, 7],
                [1, 3, 9, 11],
                [1, 3, 13, 15],
                [5, 7, 1, 3],
                [5, 7, 5, 7],
                [5, 7, 9, 11],
                [5, 7, 13, 15],
            ]))


class TestAs8BitBGRImage(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
as_8_bit_BGR_image` definition unit tests methods.
    """

    def test_as_8_bit_BGR_image(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
as_8_bit_BGR_image` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) > 0:
            return

        image_i = read_image(PNG_FILES[0])
        image_o = as_8_bit_BGR_image(image_i)

        self.assertEqual(image_o.dtype, np.uint8)
        np.testing.assert_almost_equal(
            image_o[16, 16, ...],
            (cctf_encoding(image_i[16, 16, ::-1]) * 255).astype(np.uint8))


class TestAdjustImage(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
adjust_image` definition unit tests methods.
    """

    def test_adjust_image(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
adjust_image` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) > 0:
            return

        image = adjust_image(read_image(PNG_FILES[0]))
        self.assertEqual(image.shape[1], WORKING_WIDTH)


class TestIsSquare(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
is_square` definition unit tests methods.
    """

    def test_is_square(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
is_square` definition unit tests methods.
        """

        shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertTrue(is_square(shape))

        shape = np.array([[0, 0.5], [1, 0], [1, 1], [0, 1]])
        self.assertFalse(is_square(shape))
        self.assertTrue(is_square(shape, 0.5))


class TestContourCentroid(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
contour_centroid` definition unit tests methods.
    """

    def test_contour_centroid(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
contour_centroid` definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(contour_centroid(contour), (0.5, 0.5))


class TestScaleContour(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
scale_contour` definition unit tests methods.
    """

    def test_scale_contour(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
scale_contour` definition unit tests methods.
        """

        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(
            scale_contour(contour, 2),
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))

        np.testing.assert_array_equal(
            scale_contour(contour, 0.5),
            np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]))


class TestCropAndLevelImageWithRectangle(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
crop_and_level_image_with_rectangle` definition unit tests methods.
    """

    def test_crop_and_level_image_with_rectangle(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
crop_and_level_image_with_rectangle` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) > 0:
            return

        image = as_8_bit_BGR_image(adjust_image(read_image(PNG_FILES[0])))
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
    Defines :func:`colour_checker_detection.detection.segmentation.\
colour_checkers_coordinates_segmentation` definition unit tests methods.
    """

    def test_colour_checkers_coordinates_segmentation(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
colour_checkers_coordinates_segmentation` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) > 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ('Windows', 'Microsoft', 'Linux'):
            return

        colour_checkers_coordinates = np.array([
            [[
                [985, 582],
                [675, 573],
                [681, 365],
                [991, 373],
            ]],
            [[
                [1065, 707],
                [369, 688],
                [382, 226],
                [1078, 246],
            ]],
            [[
                [1033, 674],
                [571, 664],
                [578, 357],
                [1040, 367],
            ]],
            [[
                [1048, 600],
                [622, 595],
                [625, 310],
                [1051, 315],
            ]],
            [[
                [640, 437],
                [640, 333],
                [795, 333],
                [795, 437],
            ]],
            [[
                [763, 651],
                [763, 294],
                [1004, 294],
                [1004, 651],
            ]],
        ])

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                colour_checkers_coordinates_segmentation(read_image(png_file)),
                colour_checkers_coordinates[i],
                rtol=5,
                atol=5,
            )

        colour_checkers, clusters, swatches, segmented_image = (
            colour_checkers_coordinates_segmentation(
                read_image(PNG_FILES[0]), additional_data=True))

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
            [[
                [785, 581],
                [785, 562],
                [804, 562],
                [804, 581],
            ], [
                [696, 579],
                [696, 562],
                [713, 562],
                [713, 579],
            ], [
                [629, 578],
                [629, 560],
                [647, 560],
                [647, 578],
            ], [
                [782, 557],
                [760, 557],
                [760, 535],
                [782, 535],
            ], [
                [754, 557],
                [732, 557],
                [732, 535],
                [754, 535],
            ], [
                [705, 556],
                [705, 534],
                [727, 534],
                [727, 556],
            ], [
                [699, 556],
                [678, 556],
                [678, 534],
                [699, 534],
            ], [
                [672, 556],
                [650, 556],
                [650, 534],
                [672, 534],
            ], [
                [782, 530],
                [760, 530],
                [760, 507],
                [782, 507],
            ], [
                [732, 529],
                [732, 507],
                [754, 507],
                [754, 529],
            ], [
                [705, 529],
                [705, 507],
                [727, 507],
                [727, 529],
            ], [
                [678, 529],
                [678, 507],
                [700, 507],
                [700, 529],
            ], [
                [650, 529],
                [650, 507],
                [672, 507],
                [672, 529],
            ], [
                [647, 501],
                [630, 501],
                [630, 484],
                [647, 484],
            ], [
                [745, 434],
                [745, 413],
                [766, 413],
                [766, 434],
            ], [
                [719, 433],
                [719, 413],
                [740, 413],
                [740, 433],
            ], [
                [669, 433],
                [669, 413],
                [689, 413],
                [689, 433],
            ], [
                [695, 435],
                [695, 412],
                [715, 412],
                [715, 435],
            ], [
                [643, 433],
                [643, 412],
                [663, 412],
                [663, 433],
            ], [
                [771, 409],
                [771, 388],
                [792, 388],
                [792, 409],
            ], [
                [746, 408],
                [746, 388],
                [766, 388],
                [766, 408],
            ], [
                [720, 408],
                [720, 387],
                [740, 387],
                [740, 408],
            ], [
                [669, 407],
                [669, 387],
                [689, 387],
                [689, 407],
            ], [
                [773, 383],
                [773, 366],
                [794, 366],
                [794, 383],
            ], [
                [722, 382],
                [722, 365],
                [740, 365],
                [740, 382],
            ], [
                [746, 382],
                [746, 363],
                [766, 363],
                [766, 382],
            ], [
                [670, 382],
                [670, 361],
                [690, 361],
                [690, 382],
            ], [
                [644, 381],
                [644, 361],
                [664, 361],
                [664, 381],
            ], [
                [792, 357],
                [771, 357],
                [771, 337],
                [792, 337],
            ], [
                [766, 357],
                [746, 357],
                [746, 337],
                [766, 337],
            ], [
                [720, 357],
                [720, 336],
                [740, 336],
                [740, 357],
            ], [
                [695, 356],
                [695, 336],
                [715, 336],
                [715, 356],
            ]],
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
    Defines :func:`colour_checker_detection.detection.segmentation.\
extract_colour_checkers_segmentation` definition unit tests methods.
    """

    def test_extract_colour_checkers_segmentation(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
extract_colour_checkers_segmentation` definition unit tests methods.
        """

        colour_checkers_shapes = np.array([[(209, 310, 3)], [(462, 696, 3)],
                                           [(307, 462, 3)], [(285, 426, 3)],
                                           [(104, 155, 3)], [(241, 357, 3)]])

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                [
                    colour_checker.shape
                    for colour_checker in extract_colour_checkers_segmentation(
                        read_image(png_file))
                ],
                colour_checkers_shapes[i],
                rtol=5,
                atol=5,
            )


class TestDetectColourCheckersSegmentation(unittest.TestCase):
    """
    Defines :func:`colour_checker_detection.detection.segmentation.\
detect_colour_checkers_segmentation` definition unit tests methods.
    """

    def test_detect_colour_checkers_segmentation(self):
        """
    Defines :func:`colour_checker_detection.detection.segmentation.\
detect_colour_checkers_segmentation` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g. when testing
        # the distributed "Python" package.
        if len(PNG_FILES) > 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ('Windows', 'Microsoft', 'Linux'):
            return

        test_swatches = [
            [
                np.array([
                    [0.35780658, 0.21887935, 0.11332743],
                    [0.60980982, 0.38113484, 0.23483638],
                    [0.31662177, 0.30233533, 0.27987430],
                    [0.29982070, 0.26980816, 0.10485601],
                    [0.42581077, 0.32897089, 0.32163341],
                    [0.37569798, 0.47129285, 0.32077679],
                    [0.66186179, 0.34212569, 0.05480295],
                    [0.25033150, 0.23137900, 0.30837602],
                    [0.59662466, 0.25645667, 0.17627259],
                    [0.29039783, 0.16846092, 0.18513382],
                    [0.49874807, 0.47034683, 0.03125394],
                    [0.68811219, 0.42681792, 0.06196242],
                    [0.18321167, 0.16620774, 0.25615072],
                    [0.27359525, 0.36056265, 0.11325250],
                    [0.53876827, 0.19952200, 0.11799413],
                    [0.71935517, 0.50992044, 0.01751254],
                    [0.58001364, 0.25744168, 0.27190885],
                    [0.17621797, 0.32957333, 0.31132333],
                    [0.75804678, 0.61039472, 0.42408299],
                    [0.63104497, 0.51088655, 0.35949697],
                    [0.50608677, 0.41167629, 0.28791533],
                    [0.38130358, 0.30400433, 0.20914981],
                    [0.26827934, 0.21626740, 0.14666692],
                    [0.17270683, 0.13763729, 0.08748060],
                ])
            ],
            [
                np.array([
                    [0.52433683, 0.33144400, 0.18534079],
                    [0.88356274, 0.56643302, 0.36014118],
                    [0.46916827, 0.45241115, 0.42218149],
                    [0.44397496, 0.40450780, 0.16354437],
                    [0.61624905, 0.47667439, 0.46937269],
                    [0.53325879, 0.66112913, 0.45146251],
                    [0.94274028, 0.50129830, 0.10335965],
                    [0.36898370, 0.34836797, 0.46171213],
                    [0.85905409, 0.38124036, 0.27045626],
                    [0.42731814, 0.25481624, 0.28011738],
                    [0.70610078, 0.67137995, 0.06742196],
                    [0.96225287, 0.60184749, 0.11469709],
                    [0.26731991, 0.24864520, 0.37756258],
                    [0.39151317, 0.52242693, 0.17689547],
                    [0.76248281, 0.29347406, 0.18076836],
                    [0.99184341, 0.72099446, 0.12416999],
                    [0.80891458, 0.36559528, 0.39084680],
                    [0.23693798, 0.46332763, 0.44218835],
                    [0.99381354, 0.84404891, 0.60409553],
                    [0.87018780, 0.71427575, 0.51491297],
                    [0.70219160, 0.57902124, 0.41568739],
                    [0.52714890, 0.42840677, 0.30212678],
                    [0.37685584, 0.30561161, 0.21384473],
                    [0.23390055, 0.18772539, 0.12498756],
                ])
            ],
            [
                np.array([
                    [0.35948942, 0.22254193, 0.11769965],
                    [0.62500589, 0.39319473, 0.24176362],
                    [0.33041940, 0.31421030, 0.28743836],
                    [0.30342698, 0.27218129, 0.10535372],
                    [0.41534888, 0.31836052, 0.30678427],
                    [0.34584654, 0.43934008, 0.29126657],
                    [0.67822155, 0.35195730, 0.07526861],
                    [0.27152317, 0.25155353, 0.32954119],
                    [0.61711241, 0.26872086, 0.18529358],
                    [0.30497967, 0.17922753, 0.19080851],
                    [0.48443665, 0.45765189, 0.03925590],
                    [0.64941525, 0.39912232, 0.03292602],
                    [0.19229490, 0.18420265, 0.27310652],
                    [0.27805557, 0.38365905, 0.12331341],
                    [0.55158151, 0.21266312, 0.12505309],
                    [0.71786190, 0.51329133, 0.08042136],
                    [0.57539564, 0.25639470, 0.26721066],
                    [0.17990586, 0.31605843, 0.29452960],
                    [0.74020782, 0.60882968, 0.43749755],
                    [0.62723910, 0.51560845, 0.37135411],
                    [0.51203637, 0.41963052, 0.29762951],
                    [0.36901673, 0.30191905, 0.20830509],
                    [0.26247920, 0.21433495, 0.14289918],
                    [0.16254381, 0.13333125, 0.08074121],
                ])
            ],
            [
                np.array([
                    [0.24804802, 0.15215377, 0.08139494],
                    [0.41473714, 0.25583726, 0.15487005],
                    [0.21525293, 0.19868558, 0.17945636],
                    [0.19657497, 0.17205765, 0.06598838],
                    [0.27235477, 0.20219212, 0.19411806],
                    [0.23402616, 0.28827268, 0.18941107],
                    [0.45635083, 0.23202143, 0.05238228],
                    [0.17876795, 0.15938797, 0.20633563],
                    [0.39455255, 0.16629801, 0.11177821],
                    [0.19202603, 0.10953780, 0.11426256],
                    [0.31350632, 0.29031684, 0.02019586],
                    [0.43217213, 0.25821897, 0.01960076],
                    [0.13790150, 0.12447056, 0.17549193],
                    [0.18469915, 0.24427271, 0.07604639],
                    [0.35192171, 0.13068614, 0.07453623],
                    [0.46155108, 0.32159970, 0.05020197],
                    [0.37046691, 0.15719873, 0.16320057],
                    [0.11435303, 0.19925489, 0.18536782],
                    [0.50210644, 0.41045128, 0.28989755],
                    [0.41277984, 0.33416395, 0.23614804],
                    [0.32812070, 0.26430429, 0.18333192],
                    [0.23652011, 0.18629603, 0.12491600],
                    [0.16748914, 0.13084711, 0.08430425],
                    [0.10719646, 0.07931455, 0.04688601],
                ])
            ],
            [
                np.array([
                    [0.34701525, 0.22172807, 0.12071275],
                    [0.58200400, 0.37141878, 0.23361104],
                    [0.31602135, 0.30266319, 0.27730655],
                    [0.30315578, 0.27378302, 0.11921947],
                    [0.42040456, 0.32280672, 0.30821592],
                    [0.35867045, 0.43478750, 0.29160041],
                    [0.62156950, 0.32537299, 0.07734905],
                    [0.25728022, 0.23487385, 0.30259822],
                    [0.57532181, 0.25459917, 0.17682790],
                    [0.29973022, 0.18279066, 0.18942509],
                    [0.48324637, 0.45263220, 0.05123036],
                    [0.64743875, 0.39792055, 0.02921486],
                    [0.18733215, 0.17565706, 0.24822064],
                    [0.26187537, 0.35087504, 0.11371152],
                    [0.51344352, 0.20189662, 0.11897135],
                    [0.68924993, 0.49155572, 0.09595922],
                    [0.56422316, 0.25521005, 0.26422731],
                    [0.19201479, 0.31554463, 0.29434504],
                    [0.69551957, 0.56754272, 0.40245852],
                    [0.58760926, 0.47907305, 0.33952546],
                    [0.48229859, 0.39153664, 0.27376249],
                    [0.36073251, 0.28959296, 0.19918530],
                    [0.26916516, 0.21564080, 0.14627819],
                    [0.17946655, 0.14203732, 0.09334321],
                ])
            ],
            [
                np.array([
                    [0.21933775, 0.12766601, 0.05760120],
                    [0.38566171, 0.23464573, 0.13707750],
                    [0.19848620, 0.18760381, 0.17195033],
                    [0.19310005, 0.17161982, 0.06117522],
                    [0.27939249, 0.21226332, 0.20707089],
                    [0.24572717, 0.30808876, 0.20597220],
                    [0.42549769, 0.20958095, 0.03217330],
                    [0.16062161, 0.14405722, 0.19146335],
                    [0.39141920, 0.15994439, 0.10535855],
                    [0.19575138, 0.11089487, 0.11773576],
                    [0.33848985, 0.31667756, 0.02133427],
                    [0.48267015, 0.29441540, 0.02091322],
                    [0.11257099, 0.10248654, 0.15001226],
                    [0.17348718, 0.22966481, 0.06309751],
                    [0.35408396, 0.12809606, 0.07010551],
                    [0.48598036, 0.33909363, 0.04208639],
                    [0.40599189, 0.17559977, 0.18494269],
                    [0.13339172, 0.23236492, 0.21942291],
                    [0.48098948, 0.37859672, 0.25556063],
                    [0.41186677, 0.32607698, 0.22257254],
                    [0.34114682, 0.27056572, 0.18399094],
                    [0.25640222, 0.20016887, 0.13265912],
                    [0.19531349, 0.15447677, 0.10262365],
                    [0.13553152, 0.10609387, 0.06718504],
                ])
            ],
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                detect_colour_checkers_segmentation(read_image(png_file)),
                test_swatches[i],
                rtol=0.0001,
                atol=0.0001,
            )

        swatch_colours, colour_checker_image, swatch_masks = (
            detect_colour_checkers_segmentation(
                read_image(PNG_FILES[0]), additional_data=True)[0])

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


if __name__ == '__main__':
    unittest.main()
