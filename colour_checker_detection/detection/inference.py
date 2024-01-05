"""
Colour Checker Detection - Segmentation
=======================================

Defines the objects for colour checker detection using segmentation:

-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC`
-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_SG`
-   :func:`colour_checker_detection.colour_checkers_coordinates_segmentation`
-   :func:`colour_checker_detection.extract_colour_checkers_segmentation`
-   :func:`colour_checker_detection.detect_colour_checkers_segmentation`

References
----------
-   :cite:`Abecassis2011` : Abecassis, F. (2011). OpenCV - Rotation
    (Deskewing). Retrieved October 27, 2018, from http://felix.abecassis.me/\
2011/10/opencv-rotation-deskewing/
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from colour.hints import (
    Any,
    ArrayLike,
    Dict,
    List,
    NDArrayFloat,
    NDArrayInt,
    Tuple,
    cast,
)
from colour.utilities import (
    MixinDataclassIterable,
    Structure,
    as_float_array,
    as_int_array,
    usage_warning,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)

from colour_checker_detection.detection.common import (
    FLOAT_DTYPE_DEFAULT,
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
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "SETTINGS_SEGMENTATION_COLORCHECKER_SG",
    "DataColourCheckersCoordinatesSegmentation",
    "colour_checkers_coordinates_segmentation",
    "extract_colour_checkers_segmentation",
    "DataDetectColourCheckersSegmentation",
    "detect_colour_checkers_segmentation",
]


SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC: Dict = {
    "aspect_ratio": 1.5,
    "aspect_ratio_minimum": 1.5 * 0.9,
    "aspect_ratio_maximum": 1.5 * 1.1,
    "swatches": 24,
    "swatches_horizontal": 6,
    "swatches_vertical": 4,
    "swatches_count_minimum": int(24 * 0.75),
    "swatches_count_maximum": int(24 * 1.25),
    "swatches_chromatic_slice": slice(0 + 1, 0 + 6 - 1, 1),
    "swatches_achromatic_slice": slice(18 + 1, 18 + 6 - 1, 1),
    "swatch_minimum_area_factor": 200,
    "swatch_contour_scale": 1 + 1 / 3,
    "cluster_contour_scale": 0.975,
    "working_width": 1440,
    "fast_non_local_means_denoising_kwargs": {
        "h": 10,
        "templateWindowSize": 7,
        "searchWindowSize": 21,
    },
    "adaptive_threshold_kwargs": {
        "maxValue": 255,
        "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
        "thresholdType": cv2.THRESH_BINARY,
        "blockSize": int(1440 * 0.015) - int(1440 * 0.015) % 2 + 1,
        "C": 3,
    },
    "interpolation_method": cv2.INTER_CUBIC,
}
if is_documentation_building():  # pragma: no cover
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC = DocstringDict(
        SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC
    )
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.__doc__ = """
Settings for the segmentation of the *X-Rite* *ColorChecker Classic* and
*X-Rite* *ColorChecker Passport*.
"""

SETTINGS_SEGMENTATION_COLORCHECKER_SG: Dict = (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.copy()
)

SETTINGS_SEGMENTATION_COLORCHECKER_SG.update(
    {
        "aspect_ratio": 1.4,
        "aspect_ratio_minimum": 1.4 * 0.9,
        "aspect_ratio_maximum": 1.4 * 1.1,
        "swatches": 140,
        "swatches_horizontal": 14,
        "swatches_vertical": 10,
        "swatches_count_minimum": int(140 * 0.50),
        "swatches_count_maximum": int(140 * 1.5),
        "swatch_minimum_area_factor": 200,
        "swatches_chromatic_slice": slice(48, 48 + 5, 1),
        "swatches_achromatic_slice": slice(115, 115 + 5, 1),
        "swatch_contour_scale": 1 + 1 / 3,
        "cluster_contour_scale": 1,
    }
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_SEGMENTATION_COLORCHECKER_SG = DocstringDict(
        SETTINGS_SEGMENTATION_COLORCHECKER_SG
    )
    SETTINGS_SEGMENTATION_COLORCHECKER_SG.__doc__ = """
Settings for the segmentation of the *X-Rite* *ColorChecker SG**.
"""


@dataclass
class DataColourCheckersCoordinatesSegmentation(MixinDataclassIterable):
    """
    Colour checkers detection data used for plotting, debugging and further
    analysis.

    Parameters
    ----------
    colour_checkers
        Colour checker bounding boxes, i.e., the clusters that have the
        relevant count of swatches.
    clusters
        Detected swatches clusters.
    swatches
        Detected swatches.
    segmented_image
        Thresholded/Segmented image.
    """

    colour_checkers: Tuple[NDArrayInt, ...]
    clusters: Tuple[NDArrayInt, ...]
    swatches: Tuple[NDArrayInt, ...]
    segmented_image: NDArrayFloat


def colour_checkers_coordinates_segmentation(
    image: ArrayLike, additional_data: bool = False, **kwargs: Any
) -> DataColourCheckersCoordinatesSegmentation | Tuple[NDArrayInt, ...]:
    """
    Detect the colour checkers coordinates in given image :math:`image` using
    segmentation.

    This is the core detection definition. The process is a follows:

    -   Input image :math:`image` is converted to a grayscale image
        :math:`image_g`.
    -   Image :math:`image_g` is denoised.
    -   Image :math:`image_g` is thresholded/segmented to image
        :math:`image_s`.
    -   Image :math:`image_s` is eroded and dilated to cleanup remaining noise.
    -   Contours are detected on image :math:`image_s`.
    -   Contours are filtered to only keep squares/swatches above and below
        defined surface area.
    -   Squares/swatches are clustered to isolate region-of-interest that are
        potentially colour checkers: Contours are scaled by a third so that
        colour checkers swatches are expected to be joined, creating a large
        rectangular cluster. Rectangles are fitted to the clusters.
    -   Clusters with an aspect ratio different to the expected one are
        rejected, a side-effect is that the complementary pane of the
        *X-Rite* *ColorChecker Passport* is omitted.
    -   Clusters with a number of swatches close to the expected one are
        kept.

    Parameters
    ----------
    image
        Image to detect the colour checkers in.
    additional_data
        Whether to output additional data.

    Other Parameters
    ----------------
    aspect_ratio
        Colour checker aspect ratio, e.g. 1.5.
    aspect_ratio_minimum
        Minimum colour checker aspect ratio for detection: projective geometry
        might reduce the colour checker aspect ratio.
    aspect_ratio_maximum
        Maximum colour checker aspect ratio for detection: projective geometry
        might increase the colour checker aspect ratio.
    swatches
        Colour checker swatches total count.
    swatches_horizontal
        Colour checker swatches horizontal columns count.
    swatches_vertical
        Colour checker swatches vertical row count.
    swatches_count_minimum
        Minimum swatches count to be considered for the detection.
    swatches_count_maximum
        Maximum swatches count to be considered for the detection.
    swatches_chromatic_slice
        A `slice` instance defining chromatic swatches used to detect if the
        colour checker is upside down.
    swatches_achromatic_slice
        A `slice` instance defining achromatic swatches used to detect if the
        colour checker is upside down.
    swatch_minimum_area_factor
        Swatch minimum area factor :math:`f` with the minimum area :math:`m_a`
        expressed as follows: :math:`m_a = image_w * image_h / s_c / f` where
        :math:`image_w`, :math:`image_h` and :math:`s_c` are respectively the
        image width, height and the swatches count.
    swatch_contour_scale
        As the image is filtered, the swatches area will tend to shrink, the
        generated contours can thus be scaled.
    cluster_contour_scale
        As the swatches are clustered, it might be necessary to adjust the
        cluster scale so that the masks are centred better on the swatches.
    working_width
        Size the input image is resized to for detection.
    fast_non_local_means_denoising_kwargs
        Keyword arguments for :func:`cv2.fastNlMeansDenoising` definition.
    adaptive_threshold_kwargs
        Keyword arguments for :func:`cv2.adaptiveThreshold` definition.
    interpolation_method
        Interpolation method used when resizing the images, `cv2.INTER_CUBIC`
        and `cv2.INTER_LINEAR` methods are recommended.

    Returns
    -------
    :class:`colour_checker_detection.detection.segmentation.\
DataColourCheckersCoordinatesSegmentation` or :class:`tuple`
        Tuple of colour checkers coordinates or
        :class:`DataColourCheckersCoordinatesSegmentation` class
        instance with additional data.

    Notes
    -----
    -   Multiple colour checkers can be detected if presented in ``image``.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> colour_checkers_coordinates_segmentation(image)  # doctest: +ELLIPSIS
    (array([[ 365,  684],
           [ 382,  221],
           [1077,  247],
           [1060,  710]]...)
    """

    image = as_float_array(image, FLOAT_DTYPE_DEFAULT)[..., :3]

    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    image = as_8_bit_BGR_image(
        adjust_image(
            image, settings.working_width, settings.interpolation_method
        )
    )

    width, height = image.shape[1], image.shape[0]
    maximum_area = width * height / settings.swatches
    minimum_area = (
        width
        * height
        / settings.swatches
        / settings.swatch_minimum_area_factor
    )

    # Thresholding/Segmentation.
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_g = cv2.fastNlMeansDenoising(
        image_g, None, **settings.fast_non_local_means_denoising_kwargs
    )
    image_s = cv2.adaptiveThreshold(
        image_g, **settings.adaptive_threshold_kwargs
    )
    # Cleanup.
    kernel = np.ones([3, 3], np.uint8)
    image_c = cv2.erode(image_s, kernel, iterations=1)
    image_c = cv2.dilate(image_c, kernel, iterations=1)

    # Detecting contours.
    contours, _hierarchy = cv2.findContours(
        image_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    # Filtering squares/swatches contours.
    swatches = []
    for contour in contours:
        curve = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True
        )
        if minimum_area < cv2.contourArea(curve) < maximum_area and is_square(
            curve
        ):
            swatches.append(
                as_int_array(cv2.boxPoints(cv2.minAreaRect(curve)))
            )

    # Clustering squares/swatches.
    contours = np.zeros(image.shape, dtype=np.uint8)
    for swatch in [
        as_int_array(scale_contour(swatch, settings.swatch_contour_scale))
        for swatch in swatches
    ]:
        cv2.drawContours(contours, [swatch], -1, [255] * 3, -1)
    contours = cv2.cvtColor(contours, cv2.COLOR_RGB2GRAY)
    contours, _hierarchy = cv2.findContours(
        contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    clusters = [
        as_int_array(
            scale_contour(
                cv2.boxPoints(cv2.minAreaRect(cluster)),
                settings.cluster_contour_scale,
            )
        )
        for cluster in contours
    ]

    # Filtering clusters using their aspect ratio.
    filtered_clusters = []
    for cluster in clusters[:]:
        rectangle = cv2.minAreaRect(cluster)
        width = max(rectangle[1][0], rectangle[1][1])
        height = min(rectangle[1][0], rectangle[1][1])
        ratio = width / height
        if (
            settings.aspect_ratio_minimum
            < ratio
            < settings.aspect_ratio_maximum
        ):
            filtered_clusters.append(as_int_array(cluster))
    clusters = filtered_clusters

    # Filtering swatches within cluster.
    counts = []
    for cluster in clusters:
        count = 0
        for swatch in swatches:
            if (
                cv2.pointPolygonTest(cluster, contour_centroid(swatch), False)
                == 1
            ):
                count += 1
        counts.append(count)

    indexes = np.where(
        np.logical_and(
            as_int_array(counts) >= settings.swatches_count_minimum,
            as_int_array(counts) <= settings.swatches_count_maximum,
        )
    )[0]

    colour_checkers = tuple(clusters[i] for i in indexes)

    if additional_data:
        return DataColourCheckersCoordinatesSegmentation(
            tuple(colour_checkers),
            tuple(clusters),
            tuple(swatches),
            image_c,  # pyright: ignore
        )
    else:
        return colour_checkers


def extract_colour_checkers_segmentation(
    image: ArrayLike, **kwargs: Any
) -> Tuple[NDArrayFloat, ...]:
    """
    Extract the colour checkers sub-images in given image using segmentation.

    Parameters
    ----------
    image
        Image to extract the colours checkers sub-images from.

    Other Parameters
    ----------------
    aspect_ratio
        Colour checker aspect ratio, e.g. 1.5.
    aspect_ratio_minimum
        Minimum colour checker aspect ratio for detection: projective geometry
        might reduce the colour checker aspect ratio.
    aspect_ratio_maximum
        Maximum colour checker aspect ratio for detection: projective geometry
        might increase the colour checker aspect ratio.
    swatches
        Colour checker swatches total count.
    swatches_horizontal
        Colour checker swatches horizontal columns count.
    swatches_vertical
        Colour checker swatches vertical row count.
    swatches_count_minimum
        Minimum swatches count to be considered for the detection.
    swatches_count_maximum
        Maximum swatches count to be considered for the detection.
    swatches_chromatic_slice
        A `slice` instance defining chromatic swatches used to detect if the
        colour checker is upside down.
    swatches_achromatic_slice
        A `slice` instance defining achromatic swatches used to detect if the
        colour checker is upside down.
    swatch_minimum_area_factor
        Swatch minimum area factor :math:`f` with the minimum area :math:`m_a`
        expressed as follows: :math:`m_a = image_w * image_h / s_c / f` where
        :math:`image_w`, :math:`image_h` and :math:`s_c` are respectively the
        image width, height and the swatches count.
    swatch_contour_scale
        As the image is filtered, the swatches area will tend to shrink, the
        generated contours can thus be scaled.
    cluster_contour_scale
        As the swatches are clustered, it might be necessary to adjust the
        cluster scale so that the masks are centred better on the swatches.
    working_width
        Size the input image is resized to for detection.
    fast_non_local_means_denoising_kwargs
        Keyword arguments for :func:`cv2.fastNlMeansDenoising` definition.
    adaptive_threshold_kwargs
        Keyword arguments for :func:`cv2.adaptiveThreshold` definition.
    interpolation_method
        Interpolation method used when resizing the images, `cv2.INTER_CUBIC`
        and `cv2.INTER_LINEAR` methods are recommended.

    Returns
    -------
    :class:`tuple`
        Tuple of colour checkers sub-images.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> extract_colour_checkers_segmentation(image)
    ... # doctest: +SKIP
    (array([[[ 0.17908671,  0.14010708,  0.09243158],
            [ 0.17805016,  0.13058874,  0.09513047],
            [ 0.17175764,  0.13128328,  0.08811688],
            ...,
            [ 0.15934898,  0.13436384,  0.07479276],
            [ 0.17178158,  0.13138185,  0.07703256],
            [ 0.15082785,  0.11866678,  0.07680314]],
    <BLANKLINE>
           [[ 0.16597673,  0.13563241,  0.08780421],
            [ 0.16490564,  0.13110894,  0.08601525],
            [ 0.16939694,  0.12963502,  0.08783565],
            ...,
            [ 0.14708202,  0.12856133,  0.0814603 ],
            [ 0.16883563,  0.12862256,  0.08452422],
            [ 0.16781917,  0.12363558,  0.07361614]],
    <BLANKLINE>
           [[ 0.16326806,  0.13720085,  0.08925959],
            [ 0.16014062,  0.13585283,  0.08104862],
            [ 0.16657823,  0.12889633,  0.08870038],
            ...,
            [ 0.14619341,  0.13086307,  0.07367594],
            [ 0.16302426,  0.13062705,  0.07938427],
            [ 0.16618022,  0.1266259 ,  0.07200021]],
    <BLANKLINE>
           ...,
           [[ 0.1928642 ,  0.14578913,  0.11224515],
            [ 0.18931177,  0.14416392,  0.10288388],
            [ 0.17707473,  0.1436448 ,  0.09188452],
            ...,
            [ 0.16879168,  0.12867133,  0.09001681],
            [ 0.1699731 ,  0.1287041 ,  0.07616285],
            [ 0.17137891,  0.129711  ,  0.07517841]],
    <BLANKLINE>
           [[ 0.19514292,  0.1532704 ,  0.10375113],
            [ 0.18217109,  0.14982903,  0.10452617],
            [ 0.18830594,  0.1469499 ,  0.10896181],
            ...,
            [ 0.18234864,  0.12642328,  0.08047272],
            [ 0.17617388,  0.13000189,  0.06874527],
            [ 0.17108543,  0.13264084,  0.06309374]],
    <BLANKLINE>
           [[ 0.16243187,  0.14983535,  0.08954653],
            [ 0.155507  ,  0.14899652,  0.10273992],
            [ 0.17993385,  0.1498394 ,  0.1099571 ],
            ...,
            [ 0.18079454,  0.1253967 ,  0.07739887],
            [ 0.17239226,  0.13181566,  0.07806754],
            [ 0.17422497,  0.13277327,  0.07513551]]], dtype=float32),)
    """

    image = as_float_array(image, FLOAT_DTYPE_DEFAULT)[..., :3]

    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    image = adjust_image(
        image, settings.working_width, settings.interpolation_method
    )

    colour_checkers = []
    for rectangle in cast(
        List[NDArrayFloat],
        colour_checkers_coordinates_segmentation(image, **settings),
    ):
        colour_checker = crop_and_level_image_with_rectangle(
            image,
            cv2.minAreaRect(rectangle),  # pyright: ignore
            settings.interpolation_method,
        )
        width, height = (colour_checker.shape[1], colour_checker.shape[0])

        if width < height:
            colour_checker = cv2.rotate(
                colour_checker, cv2.ROTATE_90_CLOCKWISE
            )

        colour_checkers.append(colour_checker)

    return tuple(colour_checkers)


@dataclass
class DataDetectColourCheckersSegmentation(MixinDataclassIterable):
    """
    Colour checker swatches data used for plotting, debugging and further
    analysis.

    Parameters
    ----------
    swatch_colours
        Colour checker swatches colours.
    colour_checker_image
        Cropped and levelled Colour checker image.
    swatch_masks
        Colour checker swatches masks.
    """

    swatch_colours: Tuple[NDArrayFloat, ...]
    colour_checker_image: NDArrayFloat
    swatch_masks: Tuple[NDArrayInt, ...]


def detect_colour_checkers_segmentation(
    image: ArrayLike,
    samples: int = 16,
    additional_data: bool = False,
    **kwargs: Any,
) -> Tuple[DataDetectColourCheckersSegmentation | NDArrayFloat, ...]:
    """
    Detect the colour checkers swatches in given image using segmentation.

    Parameters
    ----------
    image : array_like
        Image to detect the colour checkers swatches in.
    samples : int
        Samples count to use to compute the swatches colours. The effective
        samples count is :math:`samples^2`.
    additional_data : bool, optional
        Whether to output additional data.

    Other Parameters
    ----------------
    aspect_ratio
        Colour checker aspect ratio, e.g. 1.5.
    aspect_ratio_minimum
        Minimum colour checker aspect ratio for detection: projective geometry
        might reduce the colour checker aspect ratio.
    aspect_ratio_maximum
        Maximum colour checker aspect ratio for detection: projective geometry
        might increase the colour checker aspect ratio.
    swatches
        Colour checker swatches total count.
    swatches_horizontal
        Colour checker swatches horizontal columns count.
    swatches_vertical
        Colour checker swatches vertical row count.
    swatches_count_minimum
        Minimum swatches count to be considered for the detection.
    swatches_count_maximum
        Maximum swatches count to be considered for the detection.
    swatches_chromatic_slice
        A `slice` instance defining chromatic swatches used to detect if the
        colour checker is upside down.
    swatches_achromatic_slice
        A `slice` instance defining achromatic swatches used to detect if the
        colour checker is upside down.
    swatch_minimum_area_factor
        Swatch minimum area factor :math:`f` with the minimum area :math:`m_a`
        expressed as follows: :math:`m_a = image_w * image_h / s_c / f` where
        :math:`image_w`, :math:`image_h` and :math:`s_c` are respectively the
        image width, height and the swatches count.
    swatch_contour_scale
        As the image is filtered, the swatches area will tend to shrink, the
        generated contours can thus be scaled.
    cluster_contour_scale
        As the swatches are clustered, it might be necessary to adjust the
        cluster scale so that the masks are centred better on the swatches.
    working_width
        Size the input image is resized to for detection.
    fast_non_local_means_denoising_kwargs
        Keyword arguments for :func:`cv2.fastNlMeansDenoising` definition.
    adaptive_threshold_kwargs
        Keyword arguments for :func:`cv2.adaptiveThreshold` definition.
    interpolation_method
        Interpolation method used when resizing the images, `cv2.INTER_CUBIC`
        and `cv2.INTER_LINEAR` methods are recommended.

    Returns
    -------
    :class`tuple`
        Tuple of :class:`DataDetectColourCheckersSegmentation` class
        instances or colour checkers swatches.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> detect_colour_checkers_segmentation(image)  # doctest: +SKIP
    (array([[ 0.361626... ,  0.2241066...,  0.1187837...],
           [ 0.6280594...,  0.3950883...,  0.2434766...],
           [ 0.3326232...,  0.3156182...,  0.2891038...],
           [ 0.3048414...,  0.2738973...,  0.1069985...],
           [ 0.4174869...,  0.3199669...,  0.3081552...],
           [ 0.347873 ...,  0.4413193...,  0.2931614...],
           [ 0.6816301...,  0.3539050...,  0.0753397...],
           [ 0.2731050...,  0.2528467...,  0.3312920...],
           [ 0.6192335...,  0.2703833...,  0.1866387...],
           [ 0.3068567...,  0.1803366...,  0.1919807...],
           [ 0.4866354...,  0.4594004...,  0.0374186...],
           [ 0.6518523...,  0.4010608...,  0.0171886...],
           [ 0.1941571...,  0.1855801...,  0.2750632...],
           [ 0.2799946...,  0.3854609...,  0.1241038...],
           [ 0.5537481...,  0.2139004...,  0.1267332...],
           [ 0.7208045...,  0.5152904...,  0.0061946...],
           [ 0.5778360...,  0.2578533...,  0.2687992...],
           [ 0.1809450...,  0.3174742...,  0.2959902...],
           [ 0.7427522...,  0.6107554...,  0.4398439...],
           [ 0.6296108...,  0.5177606...,  0.3728032...],
           [ 0.5139589...,  0.4216307...,  0.2992694...],
           [ 0.3704401...,  0.3033927...,  0.2093089...],
           [ 0.2641854...,  0.2154007...,  0.1441267...],
           [ 0.1650098...,  0.1345239...,  0.0817437...]], dtype=float32),)
    """

    image = as_float_array(image, FLOAT_DTYPE_DEFAULT)[..., :3]

    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    image = adjust_image(
        image, settings.working_width, settings.interpolation_method
    )

    swatches_h, swatches_v = (
        settings.swatches_horizontal,
        settings.swatches_vertical,
    )

    colour_checkers_colours = []
    colour_checkers_data = []
    for colour_checker in extract_colour_checkers_segmentation(
        image, **settings
    ):
        width, height = colour_checker.shape[1], colour_checker.shape[0]
        masks = swatch_masks(width, height, swatches_h, swatches_v, samples)

        swatch_colours = []
        for mask in masks:
            swatch_colours.append(
                np.mean(
                    colour_checker[mask[0] : mask[1], mask[2] : mask[3], ...],
                    axis=(0, 1),
                )
            )

        # The colour checker might be flipped: The mean standard deviation
        # of some expected normalised chromatic and achromatic neutral
        # swatches is computed. If the chromatic mean is lesser than the
        # achromatic mean, it means that the colour checker is flipped.
        std_means = []
        for slice_ in [
            settings.swatches_chromatic_slice,
            settings.swatches_achromatic_slice,
        ]:
            swatch_std_mean = as_float_array(swatch_colours[slice_])
            swatch_std_mean /= swatch_std_mean[..., 1][..., None]
            std_means.append(np.mean(np.std(swatch_std_mean, 0)))
        if std_means[0] < std_means[1]:
            usage_warning(
                "Colour checker was seemingly flipped,"
                " reversing the samples!"
            )
            swatch_colours = swatch_colours[::-1]

        colour_checkers_colours.append(np.asarray(swatch_colours))
        colour_checkers_data.append((colour_checker, masks))

    if additional_data:
        return tuple(
            DataDetectColourCheckersSegmentation(
                tuple(colour_checkers_colours[i]), *colour_checkers_data[i]
            )
            for i, colour_checker_colours in enumerate(colour_checkers_colours)
        )
    else:
        return tuple(colour_checkers_colours)
