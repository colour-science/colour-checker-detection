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

import cv2
import numpy as np
from dataclasses import dataclass

from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Dict,
    DTypeFloating,
    Floating,
    Integer,
    List,
    Literal,
    NDArray,
    Tuple,
    Type,
    Union,
    cast,
)
from colour.models import cctf_encoding
from colour.utilities import (
    Structure,
    as_float_array,
    as_int_array,
    as_int,
    usage_warning,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2018-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "SETTINGS_SEGMENTATION_COLORCHECKER_SG",
    "FLOAT_DTYPE_DEFAULT",
    "ColourCheckersDetectionData",
    "ColourCheckerSwatchesData",
    "swatch_masks",
    "as_8_bit_BGR_image",
    "adjust_image",
    "is_square",
    "contour_centroid",
    "scale_contour",
    "crop_and_level_image_with_rectangle",
    "colour_checkers_coordinates_segmentation",
    "extract_colour_checkers_segmentation",
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

FLOAT_DTYPE_DEFAULT: Type[DTypeFloating] = np.float32
"""
Dtype used for the computations.
"""


@dataclass
class ColourCheckersDetectionData:
    """
    Colour checkers detection data used for plotting, debugging and further
    analysis.

    Parameters
    ----------
    colour_checkers
        Colour checker bounding boxes, i.e., the. clusters that have the
        relevant count of swatches.
    clusters
        Detected swatches clusters.
    swatches
        Detected swatches.
    segmented_image
        Thresholded/Segmented image.
    """

    colour_checkers: Tuple[NDArray, ...]
    clusters: Tuple[NDArray, ...]
    swatches: Tuple[NDArray, ...]
    segmented_image: NDArray


@dataclass
class ColourCheckerSwatchesData:
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

    swatch_colours: Tuple[NDArray, ...]
    colour_checker_image: NDArray
    swatch_masks: Tuple[NDArray, ...]


def swatch_masks(
    width: Integer,
    height: Integer,
    swatches_h: Integer,
    swatches_v: Integer,
    samples: Integer,
) -> Tuple[NDArray, ...]:
    """
    Returns swatch masks for given image width and height and swatches count.

    Parameters
    ----------
    width
        Image width.
    height
        Image height.
    swatches_h
        Horizontal swatches count.
    swatches_v
        Vertical swatches count.
    samples
        Samples count.

    Returns
    -------
    :class:`tuple`
        Tuple of swatch masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> pprint(swatch_masks(16, 8, 4, 2, 1))  # doctest: +ELLIPSIS
    (array([2, 2, 2, 2]...),
     array([2, 2, 6, 6]...),
     array([ 2,  2, 10, 10]...),
     array([ 2,  2, 14, 14]...),
     array([6, 6, 2, 2]...),
     array([6, 6, 6, 6]...),
     array([ 6,  6, 10, 10]...),
     array([ 6,  6, 14, 14]...))
    """

    samples_half = as_int(samples / 2)

    masks = []
    offset_h = width / swatches_h / 2
    offset_v = height / swatches_v / 2
    for j in np.linspace(offset_v, height - offset_v, swatches_v):
        for i in np.linspace(offset_h, width - offset_h, swatches_h):
            masks.append(
                as_int_array(
                    [
                        j - samples_half,
                        j + samples_half,
                        i - samples_half,
                        i + samples_half,
                    ]
                )
            )

    return tuple(masks)


def as_8_bit_BGR_image(image: ArrayLike) -> NDArray:
    """
    Converts and encodes given linear float *RGB* image to 8-bit *BGR* with
    *sRGB* reverse OETF.

    Parameters
    ----------
    image
        Image to convert.

    Returns
    -------
    :class:`numpy.ndarray`
        Converted image.

    Notes
    -----
    -   In the eventuality where the image is already an integer array, the
        conversion is by-passed.

    Examples
    --------
    >>> from colour.algebra import random_triplet_generator
    >>> prng = np.random.RandomState(4)
    >>> image = list(random_triplet_generator(8, random_state=prng))
    >>> image = np.reshape(image, [4, 2, 3])
    >>> print(image)
    [[[ 0.96702984  0.25298236  0.0089861 ]
      [ 0.54723225  0.43479153  0.38657128]]
    <BLANKLINE>
     [[ 0.97268436  0.77938292  0.04416006]
      [ 0.71481599  0.19768507  0.95665297]]
    <BLANKLINE>
     [[ 0.69772882  0.86299324  0.43614665]
      [ 0.2160895   0.98340068  0.94897731]]
    <BLANKLINE>
     [[ 0.97627445  0.16384224  0.78630599]
      [ 0.00623026  0.59733394  0.8662893 ]]]
    >>> image = as_8_bit_BGR_image(image)
    >>> print(image)
    [[[ 23 137 251]
      [167 176 195]]
    <BLANKLINE>
     [[ 59 228 251]
      [250 122 219]]
    <BLANKLINE>
     [[176 238 217]
      [249 253 128]]
    <BLANKLINE>
     [[229 112 252]
      [239 203  18]]]
    >>> as_8_bit_BGR_image(image)
    array([[[ 23, 137, 251],
            [167, 176, 195]],
    <BLANKLINE>
           [[ 59, 228, 251],
            [250, 122, 219]],
    <BLANKLINE>
           [[176, 238, 217],
            [249, 253, 128]],
    <BLANKLINE>
           [[229, 112, 252],
            [239, 203,  18]]], dtype=uint8)
    """

    image = np.asarray(image)[..., :3]

    if image.dtype == np.uint8:
        return image

    return cv2.cvtColor(
        cast(NDArray, cctf_encoding(image) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )


def adjust_image(
    image: ArrayLike,
    target_width: Integer,
    interpolation_method: Literal[  # type: ignore[misc]
        cv2.INTER_AREA,
        cv2.INTER_BITS,
        cv2.INTER_BITS2,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR,
    ] = cv2.INTER_CUBIC,
) -> NDArray:
    """
    Adjusts given image so that it is horizontal and resizes it to given target
    width.

    Parameters
    ----------
    image
        Image to adjust.
    target_width
        Width the image is resized to.
    interpolation_method
        Interpolation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Resized image.

    Examples
    --------
    >>> from colour.algebra import random_triplet_generator
    >>> prng = np.random.RandomState(4)
    >>> image = list(random_triplet_generator(8, random_state=prng))
    >>> image = np.reshape(image, [2, 4, 3])
    >>> adjust_image(image, 5)  # doctest: +ELLIPSIS
    array([[[ 0.9925325...,  0.2419374..., -0.0139522...],
            [ 0.6174497...,  0.3460756...,  0.3189758...],
            [ 0.7447774...,  0.678666 ...,  0.1652180...],
            [ 0.9476452...,  0.6550805...,  0.2609945...],
            [ 0.6991505...,  0.1623470...,  1.0120867...]],
    <BLANKLINE>
           [[ 0.7269885...,  0.8556784...,  0.4049920...],
            [ 0.2666565...,  1.0401633...,  0.8238320...],
            [ 0.6419699...,  0.5442698...,  0.9082211...],
            [ 0.7894426...,  0.1944301...,  0.7906868...],
            [-0.0526997...,  0.6236685...,  0.8711483...]]], dtype=float32)
    """

    image = as_float_array(image, FLOAT_DTYPE_DEFAULT)[..., :3]

    width, height = image.shape[1], image.shape[0]
    if width < height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height

    ratio = width / target_width

    if np.allclose(ratio, 1):
        return cast(NDArray, image)
    else:
        return cv2.resize(
            image,
            (as_int(target_width), as_int(height / ratio)),
            interpolation=interpolation_method,
        )


def is_square(contour: ArrayLike, tolerance: Floating = 0.015) -> Boolean:
    """
    Returns if given contour is a square.

    Parameters
    ----------
    contour
        Shape to test whether it is a square.
    tolerance
        Tolerance under which the contour is considered to be a square.

    Returns
    -------
    :class:`bool`
        Whether given contour is a square.

    Examples
    --------
    >>> shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> is_square(shape)
    True
    >>> shape = np.array([[0.5, 0], [1, 0], [1, 1], [0, 1]])
    >>> is_square(shape)
    False
    """

    return (
        cv2.matchShapes(
            contour,
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            cv2.CONTOURS_MATCH_I2,
            0.0,
        )
        < tolerance
    )


def contour_centroid(contour: ArrayLike) -> Tuple[Floating, Floating]:
    """
    Returns the centroid of given contour.

    Parameters
    ----------
    contour
        Contour to return the centroid of.

    Returns
    -------
    :class:`tuple`
        Contour centroid.

    Notes
    -----
    -   A :class:`tuple` class is returned instead of a :class:`ndarray` class
        for convenience with *OpenCV*.

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> contour_centroid(contour)
    (0.5, 0.5)
    """

    moments = cv2.moments(contour)
    centroid = (
        moments["m10"] / moments["m00"],
        moments["m01"] / moments["m00"],
    )

    return cast(Tuple[Floating, Floating], centroid)


def scale_contour(contour: ArrayLike, factor: Floating) -> NDArray:
    """
    Scales given contour by given scale factor.

    Parameters
    ----------
    contour
        Contour to scale.
    factor
        Scale factor.

    Returns
    -------
    :class:`numpy.ndarray`
        Scaled contour.

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> scale_contour(contour, 2)
    array([[ 0.,  0.],
           [ 2.,  0.],
           [ 2.,  2.],
           [ 0.,  2.]])
    """

    centroid = as_int_array(contour_centroid(contour))
    scaled_contour = (as_float_array(contour) - centroid) * factor + centroid

    return scaled_contour


def crop_and_level_image_with_rectangle(
    image: ArrayLike,
    rectangle: Tuple[Tuple, Tuple, Floating],
    interpolation_method: Literal[  # type: ignore[misc]
        cv2.INTER_AREA,
        cv2.INTER_BITS,
        cv2.INTER_BITS2,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR,
    ] = cv2.INTER_CUBIC,
):
    """
    Crops and rotates/levels given image using given rectangle.

    Parameters
    ----------
    image
        Image to crop and rotate/level.
    rectangle
        Rectangle used to crop and rotate/level the image.
    interpolation_method
        Interpolation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Cropped and rotated/levelled image.

    References
    ----------
    :cite:`Abecassis2011`

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
    >>> path = os.path.join(TESTS_RESOURCES_DIRECTORY,
    ...                     'colour_checker_detection', 'detection',
    ...                     'IMG_1967.png')
    >>> image = adjust_image(read_image(path), 1440)
    >>> rectangle = (
    ...     (723.29608154, 465.50939941),
    ...     (461.24377441, 696.34759522),
    ...     -88.18692780,
    ... )
    >>> print(image.shape)
    (958, 1440, 3)
    >>> image = crop_and_level_image_with_rectangle(image, rectangle)
    >>> print(image.shape)
    (461, 696, 3)
    """

    image = as_float_array(image, FLOAT_DTYPE_DEFAULT)[..., :3]

    width, height = image.shape[1], image.shape[0]
    width_r, height_r = rectangle[1]
    centroid = contour_centroid(cv2.boxPoints(rectangle))
    angle = rectangle[-1]

    if angle < -45:
        angle += 90
        width_r, height_r = height_r, width_r

    width_r, height_r = as_int_array([width_r, height_r])

    M_r = cv2.getRotationMatrix2D(centroid, angle, 1)

    image_r = cv2.warpAffine(image, M_r, (width, height), interpolation_method)
    image_c = cv2.getRectSubPix(
        image_r, (width_r, height_r), (centroid[0], centroid[1])
    )

    return image_c


def colour_checkers_coordinates_segmentation(
    image: ArrayLike, additional_data: Boolean = False, **kwargs: Any
) -> Union[ColourCheckersDetectionData, Tuple[NDArray, ...]]:
    """
    Detects the colour checkers coordinates in given image :math:`image` using
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
ColourCheckersDetectionData` or :class:`tuple`
        Tuple of colour checkers coordinates or
        :class:`ColourCheckersDetectionData` class instance with additional
        data.

    Notes
    -----
    -   Multiple colour checkers can be detected if presented in ``image``.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
    >>> path = os.path.join(TESTS_RESOURCES_DIRECTORY,
    ...                     'colour_checker_detection', 'detection',
    ...                     'IMG_1967.png')
    >>> image = read_image(path)
    >>> colour_checkers_coordinates_segmentation(image)
    [array([[ 369,  688],
           [ 383,  227],
           [1078,  247],
           [1065,  707]])]
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
            swatches.append(cv2.boxPoints(cv2.minAreaRect(curve)))

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
        return ColourCheckersDetectionData(
            tuple(colour_checkers), tuple(clusters), tuple(swatches), image_c
        )
    else:
        return colour_checkers


def extract_colour_checkers_segmentation(
    image: ArrayLike, **kwargs: Any
) -> Tuple[NDArray, ...]:
    """
    Extracts the colour checkers sub-images in given image using segmentation.

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
    >>> from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
    >>> path = os.path.join(TESTS_RESOURCES_DIRECTORY,
    ...                     'colour_checker_detection', 'detection',
    ...                     'IMG_1967.png')
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
        List[NDArray],
        colour_checkers_coordinates_segmentation(image, **settings),
    ):
        colour_checker = crop_and_level_image_with_rectangle(
            image, cv2.minAreaRect(rectangle), settings.interpolation_method
        )
        width, height = (colour_checker.shape[1], colour_checker.shape[0])

        if width < height:
            colour_checker = cv2.rotate(
                colour_checker, cv2.ROTATE_90_CLOCKWISE
            )

        colour_checkers.append(colour_checker)

    return tuple(colour_checkers)


def detect_colour_checkers_segmentation(
    image: ArrayLike,
    samples: Integer = 16,
    additional_data: Boolean = False,
    **kwargs: Any
) -> Union[Tuple[ColourCheckerSwatchesData, ...], Tuple[NDArray, ...]]:
    """
    Detects the colour checkers swatches in given image using segmentation.

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
        Tuple of :class:`ColourCheckerSwatchesData` class instances or
        colour checkers swatches.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
    >>> path = os.path.join(TESTS_RESOURCES_DIRECTORY,
    ...                     'colour_checker_detection', 'detection',
    ...                     'IMG_1967.png')
    >>> image = read_image(path)
    >>> detect_colour_checkers_segmentation(image)  # doctest: +ELLIPSIS
    (array([[ 0.3616269...,  0.2241066...,  0.1187838...],
           [ 0.6280594...,  0.3950882...,  0.2434766...],
           [ 0.3326231...,  0.3156182...,  0.2891038...],
           [ 0.3048413...,  0.2738974...,  0.1069985...],
           [ 0.4174869...,  0.3199669...,  0.3081552...],
           [ 0.3478729...,  0.4413193...,  0.2931614...],
           [ 0.6816301...,  0.3539050...,  0.0753397...],
           [ 0.2731048...,  0.2528467...,  0.331292 ...],
           [ 0.6192336...,  0.2703833...,  0.1866386...],
           [ 0.3068567...,  0.1803366...,  0.1919807...],
           [ 0.4866353...,  0.4594004...,  0.0374185...],
           [ 0.6518524...,  0.4010608...,  0.0171887...],
           [ 0.1941569...,  0.1855801...,  0.2750632...],
           [ 0.2799947...,  0.385461 ...,  0.1241038...],
           [ 0.5537481...,  0.2139004...,  0.1267332...],
           [ 0.7208043...,  0.5152904...,  0.0061947...],
           [ 0.577836 ...,  0.2578533...,  0.2687992...],
           [ 0.1809449...,  0.3174741...,  0.2959902...],
           [ 0.7427522...,  0.6107554...,  0.439844 ...],
           [ 0.6296108...,  0.5177607...,  0.3728032...],
           [ 0.5139589...,  0.4216308...,  0.2992694...],
           [ 0.3704402...,  0.3033927...,  0.2093090...],
           [ 0.2641854...,  0.2154006...,  0.1441268...],
           [ 0.1650097...,  0.1345238...,  0.0817438...]], dtype=float32),)
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
            swatch_std_mean /= swatch_std_mean[..., 1][..., np.newaxis]
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
            ColourCheckerSwatchesData(
                tuple(colour_checkers_colours[i]), *colour_checkers_data[i]
            )
            for i, colour_checker_colours in enumerate(colour_checkers_colours)
        )
    else:
        return tuple(colour_checkers_colours)
