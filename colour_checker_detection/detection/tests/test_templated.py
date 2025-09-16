"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.templated` module detection functions.
"""

from __future__ import annotations

import glob
import os
import platform

import numpy as np
from colour import read_image

from colour_checker_detection import ROOT_RESOURCES_TESTS
from colour_checker_detection.detection import (
    detect_colour_checkers_templated,
    extractor_templated,
    segmenter_templated,
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
    "TestSegmenterTemplated",
    "TestExtractorTemplated",
    "TestDetectColourCheckersTemplated",
]

DETECTION_DIRECTORY = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_checker_detection", "detection"
)

PNG_FILES = sorted(glob.glob(os.path.join(DETECTION_DIRECTORY, "IMG_19*.png")))


class TestSegmenterTemplated:
    """
    Define :func:`colour_checker_detection.detection.templated.\
segmenter_templated` definition unit tests methods.
    """

    def test_segmenter_templated(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.templated.\
segmenter_templated` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        colour_checkers_rectangles = [
            # IMG_1966.png
            np.array([[[672, 579], [675, 366], [994, 371], [991, 584]]]),
            # IMG_1967.png
            np.array([[[357, 690], [373, 219], [1086, 244], [1069, 715]]]),
            # IMG_1968.png
            np.array([[[572, 670], [576, 357], [1044, 364], [1040, 676]]]),
            # IMG_1969.png
            np.array([[[616, 603], [619, 309], [1057, 313], [1054, 607]]]),
            # IMG_1970.png
            np.array([[[639, 333], [795, 333], [795, 437], [639, 437]]]),
            # IMG_1971.png
            np.array(
                [
                    [[760, 655], [762, 290], [1010, 291], [1007, 657]],
                    [[422, 677], [425, 261], [660, 263], [657, 679]],
                ]
            ),
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_array_equal(
                segmenter_templated(read_image(png_file), additional_data=False),
                colour_checkers_rectangles[i],
            )

        (
            colour_checkers,
            clusters,
            swatches,
            segmented_image,
        ) = segmenter_templated(read_image(PNG_FILES[0]), additional_data=True).values

        np.testing.assert_array_equal(
            colour_checkers,
            colour_checkers_rectangles[0],
        )

        np.testing.assert_array_equal(
            clusters,
            np.array([[[672, 579], [675, 366], [994, 371], [991, 584]]]),
        )

        np.testing.assert_array_equal(
            swatches,
            np.array(
                [
                    [[887, 532], [886, 575], [930, 576], [931, 532]],
                    [[836, 530], [835, 574], [879, 575], [880, 531]],
                    [[784, 529], [783, 572], [828, 574], [828, 530]],
                    [[733, 528], [732, 571], [776, 572], [777, 529]],
                    [[682, 526], [681, 570], [725, 571], [726, 528]],
                    [[940, 482], [939, 525], [982, 526], [984, 483]],
                    [[888, 480], [887, 524], [932, 525], [933, 481]],
                    [[837, 479], [835, 523], [880, 524], [881, 480]],
                    [[785, 478], [784, 522], [829, 523], [829, 479]],
                    [[734, 478], [733, 521], [777, 522], [778, 479]],
                    [[941, 430], [939, 474], [985, 475], [986, 431]],
                    [[889, 429], [888, 473], [933, 474], [934, 431]],
                    [[838, 429], [837, 472], [881, 473], [882, 430]],
                    [[786, 428], [785, 472], [830, 472], [831, 429]],
                    [[735, 427], [734, 470], [778, 471], [779, 428]],
                    [[683, 426], [683, 470], [727, 471], [728, 427]],
                    [[942, 380], [941, 423], [986, 424], [987, 380]],
                    [[890, 379], [889, 422], [934, 423], [935, 379]],
                    [[839, 378], [838, 421], [882, 422], [883, 379]],
                    [[787, 377], [786, 420], [830, 421], [831, 378]],
                    [[736, 376], [735, 420], [779, 421], [780, 377]],
                    [[685, 376], [684, 419], [727, 420], [728, 376]],
                ],
            ),
        )

        np.testing.assert_array_equal(
            segmented_image.shape,
            (959, 1440),
        )


class TestExtractorTemplated:
    """Define :func:`extractor_templated` definition unit tests methods."""

    def test_extractor_templated(self) -> None:
        """Test :func:`extractor_templated` definition."""

        # Test the extractor function with segmentation data
        image = read_image(PNG_FILES[0])

        # First get segmentation data
        segmentation_data = segmenter_templated(image, additional_data=True)

        # Then use extractor to get colors
        extractors_result = extractor_templated(
            image, segmentation_data, additional_data=True
        )

        # Should return tuple of DataDetectionColourChecker
        assert isinstance(extractors_result, tuple)
        assert len(extractors_result) == 1

        # Check that result has expected attributes
        colour_checker_data = extractors_result[0]
        assert hasattr(colour_checker_data, "swatch_colours")
        assert hasattr(colour_checker_data, "colour_checker")
        assert hasattr(colour_checker_data, "swatch_masks")

        # Check shape of swatch colors (24 swatches, RGB)
        assert colour_checker_data.swatch_colours.shape == (24, 3)

        # Test without additional_data
        extractors_result_simple = extractor_templated(
            image, segmentation_data, additional_data=False
        )

        # Should return tuple of NDArrayFloat when additional_data=False
        assert isinstance(extractors_result_simple, tuple)
        assert len(extractors_result_simple) == 1

        # When additional_data=False, should return just the swatch colors
        assert isinstance(extractors_result_simple[0], np.ndarray)
        assert extractors_result_simple[0].shape == (24, 3)

        # Colors should be similar between both calls
        np.testing.assert_allclose(
            extractors_result[0].swatch_colours,
            extractors_result_simple[0],
            atol=0.0001,
        )


class TestDetectColourCheckersTemplated:
    """
    Define :func:`colour_checker_detection.detection.templated.\
detect_colour_checkers_templated` definition unit tests methods.
    """

    def test_detect_colour_checkers_templated(self) -> None:
        """
        Define :func:`colour_checker_detection.detection.templated.\
detect_colour_checkers_templated` definition unit tests methods.
        """

        # Skipping unit test when "png" files are missing, e.g., when testing
        # the distributed "Python" package.
        if len(PNG_FILES) == 0:
            return

        # TODO: Unit test is only reproducible on "macOs", skipping other OSes.
        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        test_swatches = [
            # IMG_1966.png
            (
                np.array(
                    [
                        [0.24915956, 0.15287110, 0.08159882],
                        [0.41551590, 0.25698534, 0.15530014],
                        [0.21628329, 0.19956540, 0.18045376],
                        [0.19618173, 0.17265946, 0.06624625],
                        [0.27358866, 0.20297317, 0.19456311],
                        [0.23418421, 0.28981102, 0.18981700],
                        [0.45830765, 0.23331086, 0.05202382],
                        [0.17930801, 0.16004612, 0.20712934],
                        [0.39580896, 0.16698070, 0.11200063],
                        [0.19282600, 0.10980366, 0.11475594],
                        [0.31437686, 0.29136035, 0.02001573],
                        [0.43281034, 0.25896379, 0.00915949],
                        [0.13839324, 0.12469792, 0.17636539],
                        [0.18606649, 0.24508719, 0.07567421],
                        [0.35319504, 0.13144426, 0.07481027],
                        [0.46281537, 0.32263476, 0.00187021],
                        [0.37178665, 0.15826064, 0.16372900],
                        [0.11477983, 0.20017651, 0.18604887],
                        [0.50303972, 0.41179717, 0.29019251],
                        [0.41358572, 0.33540562, 0.23694754],
                        [0.32893375, 0.26547331, 0.18385099],
                        [0.23856288, 0.18731910, 0.12602139],
                        [0.16771783, 0.13127713, 0.08464726],
                        [0.10808321, 0.07953004, 0.04714511],
                    ],
                ),
            ),
            # IMG_1967.png
            (
                np.array(
                    [
                        [0.36081576, 0.22396202, 0.11733589],
                        [0.62748152, 0.39514375, 0.24308297],
                        [0.33063054, 0.31587511, 0.28996205],
                        [0.30372787, 0.27424741, 0.10494538],
                        [0.41764253, 0.31940183, 0.30804706],
                        [0.34960267, 0.44142178, 0.29417506],
                        [0.68280101, 0.35389230, 0.07184852],
                        [0.27251157, 0.25320089, 0.33145225],
                        [0.62005484, 0.27033421, 0.18676178],
                        [0.30792719, 0.18030460, 0.19187638],
                        [0.48746303, 0.46047920, 0.03282085],
                        [0.65414560, 0.40173233, 0.01583917],
                        [0.19250122, 0.18560401, 0.27390230],
                        [0.28076768, 0.38508102, 0.12207687],
                        [0.55276263, 0.21404609, 0.12562890],
                        [0.72171789, 0.51569265, 0.00520882],
                        [0.57813776, 0.25853688, 0.26927036],
                        [0.17615536, 0.31684747, 0.29624644],
                        [0.74493927, 0.61261493, 0.44073734],
                        [0.63145453, 0.51879370, 0.37287709],
                        [0.51434940, 0.42190555, 0.29967216],
                        [0.37282884, 0.30393514, 0.21030639],
                        [0.26457760, 0.21623953, 0.14379750],
                        [0.16090113, 0.13422866, 0.08146800],
                    ],
                ),
            ),
            # IMG_1968.png
            (
                np.array(
                    [
                        [0.34668878, 0.22140820, 0.12132180],
                        [0.58335876, 0.37248936, 0.23336188],
                        [0.31770673, 0.30328715, 0.27763316],
                        [0.30484903, 0.27426320, 0.11717703],
                        [0.42068076, 0.32389927, 0.30854851],
                        [0.35896346, 0.43704817, 0.29320297],
                        [0.62393320, 0.32698905, 0.07527813],
                        [0.25698277, 0.23586898, 0.30427641],
                        [0.57752204, 0.25518653, 0.17710596],
                        [0.30067605, 0.18337454, 0.19010259],
                        [0.48468947, 0.45448440, 0.05042161],
                        [0.64951634, 0.39956507, 0.02743427],
                        [0.18916014, 0.17624815, 0.24872999],
                        [0.26294309, 0.35173666, 0.11394175],
                        [0.51496542, 0.20304430, 0.11947616],
                        [0.69045079, 0.49273443, 0.00373580],
                        [0.56573755, 0.25570625, 0.26498970],
                        [0.19238387, 0.31710511, 0.29539549],
                        [0.69802165, 0.57028323, 0.40264180],
                        [0.58908159, 0.48113304, 0.34029976],
                        [0.48443520, 0.39301181, 0.27513951],
                        [0.36241093, 0.29113463, 0.20048413],
                        [0.26974455, 0.21687169, 0.14648052],
                        [0.18037629, 0.14280236, 0.09329061],
                    ],
                ),
            ),
            # IMG_1969.png
            (
                np.array(
                    [
                        [0.13465774, 0.10617381, 0.06685567],
                        [0.19544454, 0.15519440, 0.10257368],
                        [0.25781783, 0.20217295, 0.13446048],
                        [0.34330013, 0.27209586, 0.18505126],
                        [0.41410297, 0.32735473, 0.22432567],
                        [0.48287287, 0.38068679, 0.25660801],
                        [0.13456589, 0.23335326, 0.22020742],
                        [0.40686405, 0.17614609, 0.18559088],
                        [0.48698768, 0.34031433, 0.00158083],
                        [0.35494950, 0.12872738, 0.06966569],
                        [0.17312363, 0.23023728, 0.06198502],
                        [0.11261383, 0.10257933, 0.15067598],
                        [0.48374513, 0.29518697, 0.01138871],
                        [0.33909237, 0.31831315, 0.02223635],
                        [0.19633672, 0.11130971, 0.11785933],
                        [0.39250675, 0.16068922, 0.10591369],
                        [0.16103746, 0.14459959, 0.19214986],
                        [0.42685473, 0.21070191, 0.03301515],
                        [0.24692154, 0.30896127, 0.20742536],
                        [0.28017253, 0.21348022, 0.20757540],
                        [0.19365267, 0.17260496, 0.06150100],
                        [0.19919579, 0.18852948, 0.17243436],
                        [0.38703433, 0.23533984, 0.13788503],
                        [0.22038600, 0.12839238, 0.05798824],
                    ],
                ),
            ),
            # IMG_1970.png
            (
                np.array(
                    [
                        [0.17157966, 0.13732299, 0.08750899],
                        [0.27111420, 0.21819340, 0.14958230],
                        [0.38644472, 0.30523640, 0.21086419],
                        [0.50922370, 0.41260293, 0.28798342],
                        [0.63196236, 0.51280522, 0.36035526],
                        [0.76167721, 0.61181736, 0.42540407],
                        [0.17408487, 0.33436161, 0.31662294],
                        [0.58316070, 0.25924474, 0.27242607],
                        [0.72373080, 0.51145762, 0.00010671],
                        [0.53920835, 0.20007114, 0.11949177],
                        [0.27674118, 0.36214554, 0.11133296],
                        [0.18371764, 0.16747326, 0.25573304],
                        [0.69212395, 0.43016854, 0.00638011],
                        [0.49993438, 0.47209752, 0.02879643],
                        [0.28961363, 0.16841158, 0.18649881],
                        [0.59825498, 0.25672621, 0.17595795],
                        [0.25149420, 0.23129882, 0.31028765],
                        [0.66379333, 0.34333602, 0.05028133],
                        [0.37794197, 0.47511625, 0.32269648],
                        [0.42837596, 0.32986039, 0.32348266],
                        [0.30207154, 0.27007088, 0.10520907],
                        [0.31910795, 0.30330661, 0.28008693],
                        [0.61248827, 0.38259700, 0.23763599],
                        [0.35867560, 0.22025365, 0.11427503],
                    ],
                ),
            ),
            # IMG_1971.png
            (
                np.array(
                    [
                        [0.52547717, 0.33433315, 0.18594886],
                        [0.88797843, 0.57013738, 0.36206961],
                        [0.47263935, 0.45490581, 0.42478570],
                        [0.44510925, 0.40670478, 0.16414125],
                        [0.61924666, 0.47956389, 0.47200805],
                        [0.53733605, 0.66552925, 0.45476383],
                        [0.94637203, 0.50409657, 0.09351142],
                        [0.37122250, 0.35024104, 0.46484435],
                        [0.86249471, 0.38330165, 0.27208051],
                        [0.43018803, 0.25626740, 0.28188750],
                        [0.70788974, 0.67395777, 0.04756298],
                        [0.96569711, 0.60438657, 0.01782006],
                        [0.26837143, 0.25034413, 0.37931514],
                        [0.39343819, 0.52498251, 0.17418264],
                        [0.76596940, 0.29502082, 0.18094559],
                        [0.99596196, 0.72455662, 0.00185360],
                        [0.81230140, 0.36783698, 0.39307770],
                        [0.23451890, 0.46583921, 0.44496056],
                        [0.99919575, 0.84779102, 0.60650510],
                        [0.87385929, 0.71774924, 0.51740980],
                        [0.70527261, 0.58176094, 0.41911286],
                        [0.52998233, 0.43069851, 0.30393562],
                        [0.37740639, 0.30739316, 0.21495330],
                        [0.23335798, 0.18924308, 0.12429100],
                    ],
                ),
            ),
        ]

        for i, png_file in enumerate(PNG_FILES):
            np.testing.assert_allclose(
                detect_colour_checkers_templated(png_file, additional_data=False),
                test_swatches[i],
                atol=0.0001,
            )

        (
            swatch_colours,
            swatch_masks,
            colour_checker,
            quadrilateral,
        ) = detect_colour_checkers_templated(
            read_image(PNG_FILES[0]), additional_data=True
        )[0].values

        np.testing.assert_allclose(
            swatch_colours,
            test_swatches[0][0],
            atol=0.0001,
        )

        np.testing.assert_array_equal(
            colour_checker.shape[0:2],
            np.array([560, 810]),
        )

        np.testing.assert_array_equal(
            swatch_masks,
            np.array([]),
        )

        assert quadrilateral.shape == (4, 2)
