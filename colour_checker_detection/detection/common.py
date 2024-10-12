"""
Common Utilities
================

Define the common utilities objects that don't fall in any specific category.

References
----------
-   :cite:`Dallas2024` : Dallas, J. (2024). [BUG]: Flipped colour chart.
    https://github.com/colour-science/colour-checker-detection/issues/\
73#issuecomment-1879471360
-   :cite:`Olferuk2019` : Olferuk, A. (2019). How to force approxPolyDP() to
    return only the best 4 corners? - Opencv 2.4.2. https://stackoverflow.com/\
a/55339684/931625
-   :cite:`Walter2022` : Walter, T. (2022). [ENHANCEMENT] Proposal to allow
    detection from different perspectives. Retrieved January 8, 2024, from
    https://github.com/colour-science/colour-checker-detection/issues/60
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from colour.algebra import linear_conversion
from colour.characterisation import CCS_COLOURCHECKERS
from colour.hints import (
    Any,
    ArrayLike,
    Dict,
    DTypeFloat,
    DTypeInt,
    Literal,
    NDArrayFloat,
    NDArrayInt,
    Tuple,
    Type,
    Union,
    cast,
)
from colour.models import XYZ_to_RGB, xyY_to_XYZ
from colour.utilities import (
    MixinDataclassIterable,
    Structure,
    as_array,
    as_float_array,
    as_int_array,
    metric_mse,
    usage_warning,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DTYPE_INT_DEFAULT",
    "DTYPE_FLOAT_DEFAULT",
    "SETTINGS_DETECTION_COLORCHECKER_CLASSIC",
    "SETTINGS_DETECTION_COLORCHECKER_SG",
    "SETTINGS_CONTOUR_DETECTION_DEFAULT",
    "as_int32_array",
    "as_float32_array",
    "swatch_masks",
    "swatch_colours",
    "reformat_image",
    "transform_image",
    "detect_contours",
    "is_square",
    "contour_centroid",
    "scale_contour",
    "approximate_contour",
    "quadrilateralise_contours",
    "remove_stacked_contours",
    "DataDetectionColourChecker",
    "sample_colour_checker",
]

DTYPE_INT_DEFAULT: Type[DTypeInt] = np.int32
"""Default int number dtype."""

DTYPE_FLOAT_DEFAULT: Type[DTypeFloat] = np.float32
"""Default floating point number dtype."""


_COLOURCHECKER = CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
_COLOURCHECKER_VALUES = XYZ_to_RGB(
    xyY_to_XYZ(list(_COLOURCHECKER.data.values())),
    "sRGB",
    _COLOURCHECKER.illuminant,
)
SETTINGS_DETECTION_COLORCHECKER_CLASSIC: Dict = {
    "aspect_ratio": 6 / 4,
    "swatches": 24,
    "swatches_horizontal": 6,
    "swatches_vertical": 4,
    "swatches_chromatic_slice": slice(0 + 1, 0 + 6 - 1, 1),
    "swatches_achromatic_slice": slice(18 + 1, 18 + 6 - 1, 1),
    "working_width": 1440,
    "working_height": int(1440 * 4 / 6),
    "interpolation_method": cv2.INTER_CUBIC,
    "reference_values": _COLOURCHECKER_VALUES,
    "transform": {},
}
if is_documentation_building():  # pragma: no cover
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC = DocstringDict(
        SETTINGS_DETECTION_COLORCHECKER_CLASSIC
    )
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC.__doc__ = """
Settings for the detection of the *X-Rite* *ColorChecker Classic* and
*X-Rite* *ColorChecker Passport*.
"""

SETTINGS_DETECTION_COLORCHECKER_SG: Dict = (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
)

# TODO: Update when "Colour" 0.4.5 is released.
_COLOURCHECKER = CCS_COLOURCHECKERS.get("ColorCheckerSG - After November 2014")
if _COLOURCHECKER is not None:
    _COLOURCHECKER_VALUES = XYZ_to_RGB(
        xyY_to_XYZ(list(_COLOURCHECKER.data.values())),
        "sRGB",
        _COLOURCHECKER.illuminant,
    )
else:
    _COLOURCHECKER_VALUES = None
SETTINGS_DETECTION_COLORCHECKER_SG.update(
    {
        "swatches": 140,
        "swatches_horizontal": 14,
        "swatches_vertical": 10,
        "swatches_chromatic_slice": slice(48, 48 + 5, 1),
        "swatches_achromatic_slice": slice(115, 115 + 5, 1),
        "aspect_ratio": 14 / 10,
        "working_height": int(1440 * 10 / 14),
        "reference_values": _COLOURCHECKER_VALUES,
    }
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_DETECTION_COLORCHECKER_SG = DocstringDict(
        SETTINGS_DETECTION_COLORCHECKER_SG
    )
    SETTINGS_DETECTION_COLORCHECKER_SG.__doc__ = """
Settings for the detection of the *X-Rite* *ColorChecker SG**.
"""

del _COLOURCHECKER, _COLOURCHECKER_VALUES

SETTINGS_CONTOUR_DETECTION_DEFAULT: Dict = {
    "bilateral_filter_iterations": 5,
    "bilateral_filter_kwargs": {"sigmaColor": 5, "sigmaSpace": 5},
    "adaptive_threshold_kwargs": {
        "maxValue": 255,
        "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
        "thresholdType": cv2.THRESH_BINARY,
        "blockSize": int(1440 * 0.015) - int(1440 * 0.015) % 2 + 1,
        "C": 3,
    },
    "convolution_kernel": np.ones([3, 3], np.uint8),
    "convolution_iterations": 1,
}
if is_documentation_building():  # pragma: no cover
    SETTINGS_CONTOUR_DETECTION_DEFAULT = DocstringDict(
        SETTINGS_CONTOUR_DETECTION_DEFAULT
    )
    SETTINGS_CONTOUR_DETECTION_DEFAULT.__doc__ = """
Settings for contour detection.
"""


def as_int32_array(a: ArrayLike) -> NDArrayInt:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using
    `np.int32` :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray` using
        `np.int32` :class:`numpy.dtype`.

    Examples
    --------
    >>> as_int32_array([1.5, 2.5, 3.5])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    return as_int_array(a, dtype=DTYPE_INT_DEFAULT)


def as_float32_array(a: ArrayLike) -> NDArrayFloat:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using
    `np.float32` :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray` using
        `np.float32` :class:`numpy.dtype`.

    Examples
    --------
    >>> as_float32_array([1, 2, 3])  # doctest: +ELLIPSIS
    array([...1...,...2...,...3...]...)
    """

    return as_float_array(a, dtype=DTYPE_FLOAT_DEFAULT)


def swatch_masks(
    width: int,
    height: int,
    swatches_h: int,
    swatches_v: int,
    samples: int,
) -> NDArrayInt:
    """
    Return swatch masks for given image width and height and swatches count.

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
        Sample count.

    Returns
    -------
    :class:`tuple`
        Tuple of swatch masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> pprint(swatch_masks(16, 8, 4, 2, 1))  # doctest: +ELLIPSIS
    array([[ 1,  3,  1,  3],
           [ 1,  3,  5,  7],
           [ 1,  3,  9, 11],
           [ 1,  3, 13, 15],
           [ 5,  7,  1,  3],
           [ 5,  7,  5,  7],
           [ 5,  7,  9, 11],
           [ 5,  7, 13, 15]]...)
    """

    samples_half = max(samples / 2, 1)

    masks = []
    offset_h = width / swatches_h / 2
    offset_v = height / swatches_v / 2
    for j in np.linspace(offset_v, height - offset_v, swatches_v):
        for i in np.linspace(offset_h, width - offset_h, swatches_h):
            masks.append(
                as_int32_array(
                    [
                        j - samples_half,
                        j + samples_half,
                        i - samples_half,
                        i + samples_half,
                    ]
                )
            )

    return as_int32_array(masks)


def swatch_colours(image: ArrayLike, masks: ArrayLike) -> NDArrayFloat:
    """
    Extract the swatch colours from given image using given masks.

    Parameters
    ----------
    image
        Image to extract the swatch colours from.
    masks
        Masks to use to extract the swatch colours from the image.

    Returns
    -------
    :class:`numpy.ndarray`
        Extracted swatch colours.

    Examples
    --------
    >>> from colour.utilities import tstack, zeros
    >>> x = np.linspace(0, 1, 16)
    >>> y = np.linspace(0, 1, 8)
    >>> xx, yy = np.meshgrid(x, y)
    >>> image = tstack([xx, yy, zeros(xx.shape)])
    >>> swatch_colours(image, swatch_masks(16, 8, 4, 2, 1))  # doctest: +ELLIPSIS
    array([[...0.1...,...0.2142...,...0...],
           [...0.3...,...0.2142...,...0...],
           [...0.6...,...0.2142...,...0...],
           [...0.9...,...0.2142...,...0...],
           [...0.1...,...0.7857...,...0...],
           [...0.3...,...0.7857...,...0...],
           [...0.6...,...0.7857...,...0...],
           [...0.9...,...0.7857...,...0...]]...)
    """

    image = as_array(image)
    masks = as_int32_array(masks)

    return as_float32_array(
        [
            np.mean(
                image[mask[0] : mask[1], mask[2] : mask[3], ...],
                axis=(0, 1),
            )
            for mask in masks
        ]
    )


def reformat_image(
    image: ArrayLike,
    target_width: int,
    interpolation_method: Literal[
        cv2.INTER_AREA,  # pyright: ignore
        cv2.INTER_CUBIC,  # pyright: ignore
        cv2.INTER_LANCZOS4,  # pyright: ignore
        cv2.INTER_LINEAR,  # pyright: ignore
        cv2.INTER_LINEAR_EXACT,  # pyright: ignore
        cv2.INTER_MAX,  # pyright: ignore
        cv2.INTER_NEAREST,  # pyright: ignore
        cv2.INTER_NEAREST_EXACT,  # pyright: ignore
        cv2.WARP_FILL_OUTLIERS,  # pyright: ignore
        cv2.WARP_INVERSE_MAP,  # pyright: ignore
    ] = cv2.INTER_CUBIC,
) -> NDArrayInt | NDArrayFloat:
    """
    Reformat given image so that it is horizontal and resizes it to given target
    width.

    Parameters
    ----------
    image
        Image to reformat.
    target_width
        Width the image is resized to.
    interpolation_method
        Interpolation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Reformatted image.

    Examples
    --------
    >>> image = np.reshape(np.arange(24), (2, 4, 3))
    >>> image  # doctest: +ELLIPSIS
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23]]]...)

    # NOTE: Need to use `cv2.INTER_NEAREST_EXACT` or `cv2.INTER_LINEAR_EXACT`
    # for integer images.

    >>> reformat_image(image, 6, interpolation_method=cv2.INTER_LINEAR_EXACT)
    ... # doctest: +ELLIPSIS
    array([[[ 0,  1,  2],
            [ 2,  3,  4],
            [ 4,  5,  6],
            [ 5,  6,  7],
            [ 8,  9, 10],
            [ 9, 10, 11]],
    <BLANKLINE>
           [[ 6,  7,  8],
            [ 8,  9, 10],
            [10, 11, 12],
            [12, 13, 14],
            [14, 15, 16],
            [15, 16, 17]],
    <BLANKLINE>
           [[12, 13, 14],
            [14, 15, 16],
            [16, 17, 18],
            [17, 18, 19],
            [20, 21, 22],
            [21, 22, 23]]]...)
    """

    image = np.asarray(image)

    if image.ndim == 3:
        image = image[..., :3]

    width, height = image.shape[1], image.shape[0]
    if width < height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height

    ratio = width / target_width

    return cv2.resize(  # pyright: ignore
        image,
        (target_width, int(height / ratio)),
        interpolation=interpolation_method,
    )


def transform_image(
    image,
    translation=np.array([0, 0]),
    rotation=0,
    scale=np.array([1, 1]),
    interpolation_method: Literal[
        cv2.INTER_AREA,  # pyright: ignore
        cv2.INTER_CUBIC,  # pyright: ignore
        cv2.INTER_LANCZOS4,  # pyright: ignore
        cv2.INTER_LINEAR,  # pyright: ignore
        cv2.INTER_LINEAR_EXACT,  # pyright: ignore
        cv2.INTER_MAX,  # pyright: ignore
        cv2.INTER_NEAREST,  # pyright: ignore
        cv2.INTER_NEAREST_EXACT,  # pyright: ignore
        cv2.WARP_FILL_OUTLIERS,  # pyright: ignore
        cv2.WARP_INVERSE_MAP,  # pyright: ignore
    ] = cv2.INTER_CUBIC,
) -> NDArrayInt | NDArrayFloat:
    """
    Transform given image using given translation, rotation and scale values.

    The transformation is performed relatively to the image center and in the
    following order:

    1. Scale
    2. Rotation
    3. Translation

    Parameters
    ----------
    image
        Image to transform.
    translation
        Translation value.
    rotation
        Rotation value in degrees.
    scale
        Scale value.
    interpolation_method
        Interpolation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed image.

    Examples
    --------
    >>> image = np.reshape(np.arange(24), (2, 4, 3))
    >>> image  # doctest: +ELLIPSIS
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23]]]...)

    # NOTE: Need to use `cv2.INTER_NEAREST` for integer images.

    >>> transform_image(
    ...     image, translation=np.array([1, 0]), interpolation_method=cv2.INTER_NEAREST
    ... )  # doctest: +ELLIPSIS
    array([[[ 0,  1,  2],
            [ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
    <BLANKLINE>
           [[12, 13, 14],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20]]]...)
    >>> transform_image(
    ...     image, rotation=90, interpolation_method=cv2.INTER_NEAREST
    ... )  # doctest: +ELLIPSIS
    array([[[15, 16, 17],
            [15, 16, 17],
            [15, 16, 17],
            [ 3,  4,  5]],
    <BLANKLINE>
           [[18, 19, 20],
            [18, 19, 20],
            [18, 19, 20],
            [ 6,  7,  8]]]...)
    >>> transform_image(
    ...     image, scale=np.array([2, 0.5]), interpolation_method=cv2.INTER_NEAREST
    ... )  # doctest: +ELLIPSIS
    array([[[ 3,  4,  5],
            [ 6,  7,  8],
            [ 6,  7,  8],
            [ 9, 10, 11]],
    <BLANKLINE>
           [[15, 16, 17],
            [18, 19, 20],
            [18, 19, 20],
            [21, 22, 23]]]...)
    """

    image = as_array(image)

    t_x, t_y = as_float32_array(translation)
    s_x, s_y = as_float32_array(scale)

    center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
    scale_transform = np.array(
        [[s_x, 0, (center_x) * (1 - s_x)], [0, s_y, (center_y) * (1 - s_y)]],
        dtype=np.float32,
    )
    scale_transform = np.vstack((scale_transform, [0, 0, 1]))

    rotation_transform = cv2.getRotationMatrix2D((center_x, center_y), -rotation, 1)
    rotation_transform = np.vstack((rotation_transform, [0, 0, 1]))

    transform = np.dot(rotation_transform, scale_transform)[:2, ...]
    transform += as_float32_array([[0, 0, t_x], [0, 0, t_y]])

    return cast(
        Union[NDArrayInt, NDArrayFloat],
        cv2.warpAffine(
            image,
            transform,
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
            flags=interpolation_method,
        ),
    )


def detect_contours(
    image: ArrayLike, additional_data: bool = False, **kwargs: Any
) -> Tuple[NDArrayInt] | Tuple[Tuple[NDArrayInt], NDArrayInt | NDArrayFloat]:
    """
    Detect the contours of given image using given settings.

    The process is a follows:

    -   Input image :math:`image` is converted to a grayscale image
        :math:`image_g` and normalised to range [0, 1].
    -   Image :math:`image_g` is denoised using multiple bilateral filtering
        passes into image :math:`image_d.`
    -   Image :math:`image_d` is thresholded into image :math:`image_t`.
    -   Image :math:`image_t` is eroded and dilated to cleanup remaining noise
        into image :math:`image_k`.
    -   Contours are detected on image :math:`image_k`

    Parameters
    ----------
    image
        Image to detect the contour of.
    additional_data
        Whether to output additional data.

    Other Parameters
    ----------------
    adaptive_threshold_kwargs
        Keyword arguments for :func:`cv2.adaptiveThreshold` definition.
    bilateral_filter_iterations
        Number of iterations to use for bilateral filtering.
    bilateral_filter_kwargs
        Keyword arguments for :func:`cv2.bilateralFilter` definition.
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.

    Returns
    -------
    :class:`numpy.ndarray`
        Detected image contours.

    Warnings
    --------
    The process and especially the default settings assume that the image has
    been resized to :attr:`SETTINGS_DETECTION_COLORCHECKER_CLASSIC.working_width`
    value!

    Examples
    --------
    >>> from colour.utilities import zeros
    >>> image = zeros([240, 320, 3])
    >>> image[150:190, 140:180] = 1
    >>> len(detect_contours(image))
    3
    """

    settings = Structure(**SETTINGS_CONTOUR_DETECTION_DEFAULT)
    settings.update(**kwargs)

    image_g = np.max(image, axis=-1)

    # Normalisation
    image_g = (
        linear_conversion(image_g, (np.min(image_g), np.max(image_g)), (0, 1)) * 255
    ).astype(np.uint8)

    # Denoising
    image_d = image_g
    for _ in range(settings.bilateral_filter_iterations):
        image_d = cv2.bilateralFilter(image_d, -1, **settings.bilateral_filter_kwargs)

    # Thresholding
    image_t = cv2.adaptiveThreshold(image_d, **settings.adaptive_threshold_kwargs)

    # Erosion / Dilation
    image_k = cv2.erode(
        image_t,
        settings.convolution_kernel,
        iterations=settings.convolution_iterations,
    )
    image_k = cv2.dilate(
        image_k,
        settings.convolution_kernel,
        iterations=settings.convolution_iterations,
    )

    image_k = cast(Union[NDArrayInt, NDArrayFloat], image_k)

    # Detecting contours.
    contours, _hierarchy = cv2.findContours(
        image_k, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    contours = cast(Tuple[NDArrayInt], contours)

    if additional_data:
        return contours, image_k
    else:
        return contours


def is_square(contour: ArrayLike, tolerance: float = 0.015) -> bool:
    """
    Return if given contour is a square.

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
            contour,  # pyright: ignore
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            cv2.CONTOURS_MATCH_I2,
            0.0,
        )
        < tolerance
    )


def contour_centroid(contour: ArrayLike) -> Tuple[float, float]:
    """
    Return the centroid of given contour.

    Parameters
    ----------
    contour
        Contour to return the centroid of.

    Returns
    -------
    :class:`np.ndarray`
        Contour centroid.

    Notes
    -----
    -   A :class:`tuple` class is returned instead of a :class:`ndarray` class
        for convenience with *OpenCV*.

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> contour_centroid(contour)  # doctest: +ELLIPSIS
    (0.5, 0.5)
    """

    contour = as_float32_array(contour)

    moments = cv2.moments(contour)

    centroid = (
        moments["m10"] / moments["m00"],
        moments["m01"] / moments["m00"],
    )

    return centroid


def scale_contour(contour: ArrayLike, factor: ArrayLike) -> NDArrayFloat:
    """
    Scale given contour by given scale factor.

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

    Warnings
    --------
    This definition returns floating point contours!

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> scale_contour(contour, 2)  # doctest: +ELLIPSIS
    array([[...-0.5, ...-0.5],
           [... 1.5, ...-0.5],
           [... 1.5, ... 1.5],
           [...-0.5, ... 1.5]]...)
    """

    contour = as_float32_array(contour)
    factor = as_float32_array(factor)

    centroid = contour_centroid(contour)

    scaled_contour = (contour - centroid) * factor + centroid

    return scaled_contour


def approximate_contour(
    contour: ArrayLike, points: int = 4, iterations: int = 100
) -> NDArrayInt:
    """
    Approximate given contour to have given number of points.

    The process uses binary search to find the best *epsilon* value
    producing a contour approximation with exactly ``points``.

    Parameters
    ----------
    contour
        Contour to approximate.
    points
        Number of points to approximate the contour to.
    iterations
        Maximal number of iterations to perform to approximate the contour.

    Returns
    -------
    :class:`numpy.ndarray`
        Approximated contour.

    References
    ----------
    :cite:`Olferuk2019`

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [1, 2], [0, 1]])
    >>> approximate_contour(contour, 4)  # doctest: +ELLIPSIS
    array([[0, 0],
           [1, 0],
           [1, 2],
           [0, 1]]...)
    """

    contour = as_int32_array(contour)

    i = 0
    low, high = 0, 1

    while True:
        i += 1
        if i > iterations:
            return contour

        center = (low + high) / 2
        approximation = cv2.approxPolyDP(
            contour, center * cv2.arcLength(contour, True), True
        )

        approximation = cast(NDArrayInt, approximation)

        if len(approximation) > points:
            low = (low + high) / 2
        elif len(approximation) < points:
            high = (low + high) / 2
        else:
            return np.squeeze(approximation)


def quadrilateralise_contours(contours: ArrayLike) -> Tuple[NDArrayInt, ...]:
    """
    Convert given to quadrilaterals.

    Parameters
    ----------
    contours
        Contours to convert to quadrilaterals

    Returns
    -------
    :class:`tuple`
        Quadrilateralised contours.

    Examples
    --------
    >>> contours = np.array(
    ...     [
    ...         [[0, 0], [1, 0], [1, 1], [1, 2], [0, 1]],
    ...         [[0, 0], [1, 2], [1, 0], [1, 1], [0, 1]],
    ...     ]
    ... )
    >>> quadrilateralise_contours(contours)  # doctest: +ELLIPSIS
    (array([[0, 0],
           [1, 0],
           [1, 2],
           [0, 1]]...), array([[0, 0],
           [1, 2],
           [1, 0],
           [1, 1]]...))

    """

    return tuple(
        as_int32_array(approximate_contour(contour, 4))
        for contour in contours  # pyright: ignore
    )


def remove_stacked_contours(
    contours: ArrayLike, keep_smallest: bool = True
) -> Tuple[NDArrayInt, ...]:
    """
    Remove amd filter out the stacked contours from given contours keeping
    either the smallest or the largest ones.

    Parameters
    ----------
    contours
        Stacked contours to filter.
    keep_smallest
        Whether to keep the smallest contours.

    Returns
    -------
    :class:`tuple`
        Filtered contours.

    References
    ----------
    :cite:`Walter2022`

    Examples
    --------
    >>> contours = np.array(
    ...     [
    ...         [[0, 0], [7, 0], [7, 7], [0, 7]],
    ...         [[0, 0], [8, 0], [8, 8], [0, 8]],
    ...         [[0, 0], [10, 0], [10, 10], [0, 10]],
    ...     ]
    ... )
    >>> remove_stacked_contours(contours)  # doctest: +ELLIPSIS
    (array([[0, 0],
           [7, 0],
           [7, 7],
           [0, 7]]...)
    >>> remove_stacked_contours(contours, False)  # doctest: +ELLIPSIS
    (array([[ 0,  0],
           [10,  0],
           [10, 10],
           [ 0, 10]]...)
    """

    contours = as_int32_array(contours)

    filtered_contours = []

    for contour in contours:
        centroid = contour_centroid(contour)

        stacked_contours = [
            filtered_contour
            for filtered_contour in filtered_contours
            if cv2.pointPolygonTest(filtered_contour, centroid, False) > 0
        ]

        if not stacked_contours:
            filtered_contours.append(contour)
        else:
            areas = as_float32_array(
                [
                    cv2.contourArea(stacked_contour)
                    for stacked_contour in stacked_contours
                ]
            )

            if keep_smallest:
                result = np.all(cv2.contourArea(contour) < areas)
                index = 0
            else:
                result = np.all(cv2.contourArea(contour) > areas)
                index = -1

            if result:
                stacked_contour = as_int32_array(stacked_contours)[np.argsort(areas)][0]

                index = np.argwhere(
                    np.all(
                        as_int32_array(filtered_contours) == stacked_contour,
                        axis=(1, 2),
                    )
                )[index][0]

                filtered_contours[index] = contour

    return tuple(
        as_int32_array(filtered_contour) for filtered_contour in filtered_contours
    )


@dataclass
class DataDetectionColourChecker(MixinDataclassIterable):
    """
    Colour checker swatches data used for plotting, debugging and further
    analysis.

    Parameters
    ----------
    swatch_colours
        Colour checker swatches colours.
    swatch_masks
        Colour checker swatches masks.
    colour_checker
        Cropped and levelled Colour checker image.
    quadrilateral
        Source quadrilateral where the colour checker has been detected.
    """

    swatch_colours: NDArrayFloat
    swatch_masks: NDArrayInt
    colour_checker: NDArrayFloat
    quadrilateral: NDArrayFloat


def sample_colour_checker(
    image: ArrayLike, quadrilateral, rectangle, samples=32, **kwargs
) -> DataDetectionColourChecker:
    """
    Sample the colour checker using the given source quadrilateral, i.e.,
    detected colour checker in the image, and the given target rectangle.

    Parameters
    ----------
    image
        Image to sample from.
    quadrilateral
        Source quadrilateral where the colour checker has been detected.
    rectangle
        Target rectangle to warp the detected source quadrilateral onto.
    samples
        Sample count to use to sample the swatches colours. The effective
        sample count is :math:`samples^2`.

    Other Parameters
    ----------------
    reference_values
        Reference values for the colour checker of interest.
    swatches_horizontal
        Colour checker swatches horizontal columns count.
    swatches_vertical
        Colour checker swatches vertical row count.
    transform
        Transform to apply to the colour checker image post-detection.
    working_width
        Width the input image is resized to for detection.
    working_height
        Height the input image is resized to for detection.

    Returns
    -------
    :class:`colour_checker.DataDetectionColourChecker`
        Sampling process data.

    References
    ----------
    :cite:`Dallas2024`

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
    >>> quadrilateral = np.array([[358, 691], [373, 219], [1086, 242], [1071, 713]])
    >>> rectangle = np.array([[1440, 0], [1440, 960], [0, 960], [0, 0]])
    >>> colour_checkers_data = sample_colour_checker(image, quadrilateral, rectangle)
    >>> colour_checkers_data.swatch_colours  # doctest: +SKIP
    array([[ 0.75710917,  0.6763046 ,  0.47606474],
           [ 0.25871587,  0.21974973,  0.16204563],
           [ 0.15012611,  0.11881837,  0.07829906],
           [ 0.14475887,  0.11828972,  0.0747117 ],
           [ 0.15182742,  0.12059662,  0.07984065],
           [ 0.15811475,  0.12584405,  0.07951307],
           [ 0.9996331 ,  0.827563  ,  0.5362377 ],
           [ 0.2615244 ,  0.22938406,  0.16862768],
           [ 0.1580963 ,  0.11951645,  0.0775518 ],
           [ 0.16762769,  0.13303326,  0.08851139],
           [ 0.17338796,  0.14148802,  0.08979498],
           [ 0.17304046,  0.1419515 ,  0.09080467],
           [ 1.        ,  0.9890205 ,  0.6780832 ],
           [ 0.25435534,  0.2206379 ,  0.1569271 ],
           [ 0.15027192,  0.12475526,  0.0784394 ],
           [ 0.3458355 ,  0.21429974,  0.1121798 ],
           [ 0.36254194,  0.2259509 ,  0.11665937],
           [ 0.62459683,  0.39099   ,  0.24112946],
           [ 0.97804743,  1.        ,  0.86419195],
           [ 0.25577253,  0.22349517,  0.1584489 ],
           [ 0.1595923 ,  0.12591116,  0.08147947],
           [ 0.35486832,  0.21910854,  0.11063413],
           [ 0.3630804 ,  0.22740598,  0.12138989],
           [ 0.62340593,  0.39334935,  0.24371558]]...)
    >>> colour_checkers_data.swatch_masks.shape
    (24, 4)
    >>> colour_checkers_data.colour_checker.shape
    (960, 1440, 3)
    """

    image = as_array(image)

    settings = Structure(**SETTINGS_DETECTION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    quadrilateral = as_float32_array(quadrilateral)
    rectangle = as_float32_array(rectangle)

    swatches_horizontal = settings.swatches_horizontal
    swatches_vertical = settings.swatches_vertical
    working_width = settings.working_width
    working_height = settings.working_height

    transform = cv2.getPerspectiveTransform(quadrilateral, rectangle)
    colour_checker = cv2.warpPerspective(
        image,
        transform,
        (working_width, working_height),
        flags=settings.interpolation_method,
    )

    if settings.transform:
        colour_checker = transform_image(colour_checker, **settings.transform)

    masks = swatch_masks(
        working_width,
        working_height,
        swatches_horizontal,
        swatches_vertical,
        samples,
    )
    sampled_colours = swatch_colours(colour_checker, masks)

    # TODO: Update when "Colour" 0.4.5 is released.
    if settings.reference_values is None:
        usage_warning(
            "Cannot compute the colour checker orientation because the "
            'reference values are not available! Please update "Colour" to a '
            "version greater-than 0.4.4."
        )
    else:
        reference_mse = metric_mse(settings.reference_values, sampled_colours)
        candidate_quadrilateral = np.copy(quadrilateral)
        for _ in range(3):
            candidate_quadrilateral = np.roll(candidate_quadrilateral, 1, 0)
            transform = cv2.getPerspectiveTransform(
                candidate_quadrilateral,
                rectangle,
            )
            colour_checker_candidate = cv2.warpPerspective(
                image,
                transform,
                (working_width, working_height),
                flags=settings.interpolation_method,
            )

            if settings.transform:
                colour_checker_candidate = transform_image(
                    colour_checker_candidate, **settings.transform
                )

            candidate_sampled_colours = swatch_colours(colour_checker_candidate, masks)
            candidate_mse = metric_mse(
                settings.reference_values, candidate_sampled_colours
            )
            if candidate_mse < reference_mse:
                reference_mse = candidate_mse
                sampled_colours = candidate_sampled_colours
                colour_checker = colour_checker_candidate
                quadrilateral = candidate_quadrilateral

    colour_checker = cast(NDArrayFloat, colour_checker)

    return DataDetectionColourChecker(
        sampled_colours, masks, colour_checker, quadrilateral
    )
