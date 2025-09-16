"""
Colour Checker Detection - Segmentation
=======================================

Define the objects for colour checker detection using segmentation:

-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC`
-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_SG`
-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_NANO`
-   :func:`colour_checker_detection.segmenter_default`
-   :func:`colour_checker_detection.detect_colour_checkers_segmentation`

References
----------
-   :cite:`Abecassis2011` : Abecassis, F. (2011). OpenCV - Rotation
    (Deskewing). Retrieved October 27, 2018, from http://felix.abecassis.me/\
2011/10/opencv-rotation-deskewing/
"""

from __future__ import annotations

import typing

import cv2

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Callable,
        Dict,
        Literal,
        NDArrayFloat,
        NDArrayInt,
        Tuple,
    )

from colour.hints import Any, Dict, NDArrayReal, cast
from colour.io import convert_bit_depth, read_image
from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.utilities import (
    Structure,
    optional,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)

from colour_checker_detection.detection.common import (
    DTYPE_FLOAT_DEFAULT,
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
    SETTINGS_DETECTION_COLORCHECKER_SG,
    DataDetectionColourChecker,
    DataSegmentationColourCheckers,
    as_float32_array,
    as_int32_array,
    cluster_swatches,
    detect_contours,
    filter_clusters,
    is_square,
    quadrilateralise_contours,
    reformat_image,
    remove_stacked_contours,
    sample_colour_checker,
)
from colour_checker_detection.detection.plotting import plot_detection_results

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "SETTINGS_SEGMENTATION_COLORCHECKER_SG",
    "SETTINGS_SEGMENTATION_COLORCHECKER_NANO",
    "DataSegmentationColourCheckers",
    "segmenter_default",
    "extractor_segmentation",
    "detect_colour_checkers_segmentation",
]

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC: Dict = (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
)

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.update(
    {
        "aspect_ratio_minimum": 1.5 * 0.9,
        "aspect_ratio_maximum": 1.5 * 1.1,
        "swatches_count_minimum": int(24 * 0.5),
        "swatches_count_maximum": int(24 * 1.5),
        "swatch_minimum_area_factor": 200,
        "swatch_contour_scale": 1 + 1 / 3,
    }
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC = DocstringDict(
        SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC
    )
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.__doc__ = """
Settings for the segmentation of the *X-Rite* *ColorChecker Classic* and
*X-Rite* *ColorChecker Passport*.
"""

SETTINGS_SEGMENTATION_COLORCHECKER_SG: Dict = SETTINGS_DETECTION_COLORCHECKER_SG.copy()

SETTINGS_SEGMENTATION_COLORCHECKER_SG.update(
    {
        "aspect_ratio_minimum": 1.4 * 0.9,
        "aspect_ratio_maximum": 1.4 * 1.1,
        "swatches_count_minimum": int(140 * 0.50),
        "swatches_count_maximum": int(140 * 1.5),
        "swatch_contour_scale": 1 + 1 / 3,
        "swatch_minimum_area_factor": 200,
    }
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_SEGMENTATION_COLORCHECKER_SG = DocstringDict(
        SETTINGS_SEGMENTATION_COLORCHECKER_SG
    )
    SETTINGS_SEGMENTATION_COLORCHECKER_SG.__doc__ = """
Settings for the segmentation of the *X-Rite* *ColorChecker SG**.
"""

SETTINGS_SEGMENTATION_COLORCHECKER_NANO: Dict = (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.copy()
)

SETTINGS_SEGMENTATION_COLORCHECKER_NANO.update(
    {
        "aspect_ratio_minimum": 1.4 * 0.75,
        "aspect_ratio_maximum": 1.4 * 1.5,
        "swatch_contour_scale": 1 + 1 / 2,
    }
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_SEGMENTATION_COLORCHECKER_NANO = DocstringDict(
        SETTINGS_SEGMENTATION_COLORCHECKER_NANO
    )
    SETTINGS_SEGMENTATION_COLORCHECKER_NANO.__doc__ = """
Settings for the segmentation of the *X-Rite* *ColorChecker Nano**.
"""


@typing.overload
def segmenter_default(
    image: ArrayLike,
    cctf_encoding: Callable = ...,
    apply_cctf_encoding: bool = ...,
    additional_data: Literal[True] = True,
    **kwargs: Any,
) -> DataSegmentationColourCheckers: ...


@typing.overload
def segmenter_default(
    image: ArrayLike,
    cctf_encoding: Callable = ...,
    apply_cctf_encoding: bool = ...,
    *,
    additional_data: Literal[False],
    **kwargs: Any,
) -> NDArrayInt: ...


@typing.overload
def segmenter_default(
    image: ArrayLike,
    cctf_encoding: Callable,
    apply_cctf_encoding: bool,
    additional_data: Literal[False],
    **kwargs: Any,
) -> NDArrayInt: ...


def segmenter_default(
    image: ArrayLike,
    cctf_encoding: Callable = eotf_inverse_sRGB,
    apply_cctf_encoding: bool = True,
    additional_data: bool = False,
    **kwargs: Any,
) -> DataSegmentationColourCheckers | NDArrayInt:
    """
    Detect the colour checker rectangles in specified image using segmentation.

    The process is as follows:

    -   Input image :math:`image` is converted to a grayscale image
        :math:`image_g` and normalised to range [0, 1].
    -   Image :math:`image_g` is denoised using multiple bilateral filtering
        passes into image :math:`image_d.`
    -   Image :math:`image_d` is thresholded into image :math:`image_t`.
    -   Image :math:`image_t` is eroded and dilated to cleanup remaining noise
        into image :math:`image_k`.
    -   Contours are detected on image :math:`image_k`
    -   Contours are filtered to only keep squares/swatches above and below
        defined surface area.
    -   Squares/swatches are clustered to isolate region-of-interest that are
        potentially colour checkers: Contours are scaled by a third so that
        colour checkers swatches are joined, creating a large rectangular
        cluster. Rectangles are fitted to the clusters.
    -   Clusters with an aspect ratio different to the expected one are
        rejected, a side-effect is that the complementary pane of the
        *X-Rite* *ColorChecker Passport* is omitted.
    -   Clusters with a number of swatches close to the expected one are
        kept.

    Parameters
    ----------
    image
        Image to detect the colour checker rectangles from.
    cctf_encoding
        Encoding colour component transfer function / opto-electronic
        transfer function used when converting the image from float to 8-bit.
    apply_cctf_encoding
        Apply the encoding colour component transfer function / opto-electronic
        transfer function.
    additional_data
        Whether to output additional data.

    Other Parameters
    ----------------
    adaptive_threshold_kwargs
        Keyword arguments for :func:`cv2.adaptiveThreshold` definition.
    aspect_ratio
        Colour checker aspect ratio, e.g., 1.5.
    aspect_ratio_minimum
        Minimum colour checker aspect ratio for detection: projective geometry
        might reduce the colour checker aspect ratio.
    aspect_ratio_maximum
        Maximum colour checker aspect ratio for detection: projective geometry
        might increase the colour checker aspect ratio.
    bilateral_filter_iterations
        Number of iterations to use for bilateral filtering.
    bilateral_filter_kwargs
        Keyword arguments for :func:`cv2.bilateralFilter` definition.
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    interpolation_method
        Interpolation method used when resizing the images, `cv2.INTER_CUBIC`
        and `cv2.INTER_LINEAR` methods are recommended.
    reference_values
        Reference values for the colour checker of interest.
    swatch_contour_scale
        As the image is filtered, the swatches area will tend to shrink, the
        generated contours can thus be scaled.
    swatch_minimum_area_factor
        Swatch minimum area factor :math:`f` with the minimum area :math:`m_a`
        expressed as follows: :math:`m_a = image_w * image_h / s_c / f` where
        :math:`image_w`, :math:`image_h` and :math:`s_c` are respectively the
        image width, height and the swatches count.
    swatches
        Colour checker swatches total count.
    swatches_achromatic_slice
        A `slice` instance defining achromatic swatches used to detect if the
        colour checker is upside down.
    swatches_chromatic_slice
        A `slice` instance defining chromatic swatches used to detect if the
        colour checker is upside down.
    swatches_count_maximum
        Maximum swatches count to be considered for the detection.
    swatches_count_minimum
        Minimum swatches count to be considered for the detection.
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
    :class:`colour_checker_detection.DataSegmentationColourCheckers` or \
:class:`np.ndarray`
        Colour checker rectangles and additional data or colour checker
        rectangles only.

    Notes
    -----
    -   Multiple colour checkers can be detected if present in ``image``.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS, segmenter_default
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> segmenter_default(image)  # doctest: +ELLIPSIS
    array([[[ 3...,  6...],
            [ 3...,  2...],
            [10...,  2...],
            [10...,  7...]]]...)
    """

    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    if apply_cctf_encoding:
        image = cctf_encoding(image)

    image = reformat_image(image, settings.working_width, settings.interpolation_method)

    width, height = image.shape[1], image.shape[0]
    minimum_area = (
        width * height / settings.swatches / settings.swatch_minimum_area_factor
    )
    maximum_area = width * height / settings.swatches

    contours, image_k = detect_contours(image, True, **settings)  # pyright: ignore

    # Filtering squares/swatches contours.
    squares = []
    for swatch_contour in quadrilateralise_contours(contours):
        if minimum_area < cv2.contourArea(swatch_contour) < maximum_area and is_square(
            swatch_contour
        ):
            squares.append(  # noqa: PERF401
                as_int32_array(cv2.boxPoints(cv2.minAreaRect(swatch_contour)))
            )

    swatches = as_int32_array(remove_stacked_contours(squares))

    clusters = cluster_swatches(
        as_float32_array(image), swatches, settings.swatch_contour_scale
    )

    # Filtering clusters using their aspect ratio.
    filtered_clusters = []
    for cluster in clusters[:]:
        rectangle = cv2.minAreaRect(cluster)
        width = max(rectangle[1][0], rectangle[1][1])
        height = min(rectangle[1][0], rectangle[1][1])
        ratio = width / height

        if settings.aspect_ratio_minimum < ratio < settings.aspect_ratio_maximum:
            filtered_clusters.append(as_int32_array(cluster))
    clusters = as_int32_array(filtered_clusters)

    # Filtering swatches within cluster.
    rectangles = filter_clusters(
        clusters,
        swatches,
        settings.swatches_count_minimum,
        settings.swatches_count_maximum,
    )

    if additional_data:
        return DataSegmentationColourCheckers(
            rectangles,
            clusters,
            swatches,
            image_k,  # pyright: ignore
        )
    return rectangles


@typing.overload
def extractor_segmentation(
    image: ArrayLike,
    segmentation_data: Any,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    additional_data: Literal[True] = True,
    **kwargs: Any,
) -> tuple[DataDetectionColourChecker, ...]: ...


@typing.overload
def extractor_segmentation(
    image: ArrayLike,
    segmentation_data: Any,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    *,
    additional_data: Literal[False],
    **kwargs: Any,
) -> tuple[NDArrayFloat, ...]: ...


@typing.overload
def extractor_segmentation(
    image: ArrayLike,
    segmentation_data: Any,
    samples: int,
    cctf_decoding: Callable,
    apply_cctf_decoding: bool,
    additional_data: Literal[False],
    **kwargs: Any,
) -> tuple[NDArrayFloat, ...]: ...


def extractor_segmentation(
    image: ArrayLike,
    segmentation_data: Any,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> tuple[DataDetectionColourChecker, ...] | tuple[NDArrayFloat, ...]:
    """
    Extract colour swatches using segmentation-based methods.

    This function takes segmentation data (rectangles/quadrilaterals) and extracts
    colors using the standard geometric sampling approach.

    Parameters
    ----------
    image
        Image to extract colours from.
    segmentation_data
        Segmentation data containing detected rectangles and swatches.
    samples
        Sample count to use to average (mean) the swatches colours. The effective
        sample count is :math:`samples^2`.
    cctf_decoding
        Decoding colour component transfer function / opto-electronic
        transfer function used when converting the image from 8-bit to float.
    apply_cctf_decoding
        Apply the decoding colour component transfer function / opto-electronic
        transfer function.
    additional_data
        Whether to include additional extraction data.

    Other Parameters
    ----------------
    interpolation_method
        Interpolation method used when resizing the images, `cv2.INTER_CUBIC`
        and `cv2.INTER_LINEAR` methods are recommended.
    working_width
        Width the input image is resized to for detection.
    aspect_ratio
        Colour checker aspect ratio, e.g. 1.5.

    Returns
    -------
    :class:`tuple`
        Tuple of detected colour checker data objects.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import (
    ...     ROOT_RESOURCES_TESTS,
    ...     segmenter_default,
    ...     extractor_segmentation,
    ... )
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> segmentation_data = segmenter_default(image, additional_data=True)
    >>> extractor_segmentation(image, segmentation_data)
    ... # doctest: +SKIP
    (array([[ 0.36018878,  0.22291452,  0.11730091],
           [ 0.6256498 ,  0.39444408,  0.241824  ],
           [ 0.33206907,  0.31609666,  0.2886104 ],
           [ 0.30452022,  0.27339727,  0.10480344],
           [ 0.41740698,  0.3191239 ,  0.30785984],
           [ 0.3486532 ,  0.43936515,  0.29126465],
           [ 0.6797462 ,  0.3522702 ,  0.06983876],
           [ 0.27157846,  0.25354025,  0.33056694],
           [ 0.62124234,  0.27033532,  0.18669768],
           [ 0.3070325 ,  0.17978221,  0.19183952],
           [ 0.48548093,  0.45865163,  0.03294417],
           [ 0.6508147 ,  0.4002312 ,  0.01611003],
           [ 0.1930163 ,  0.18572474,  0.2745065 ],
           [ 0.28079677,  0.38511798,  0.12274227],
           [ 0.5547266 ,  0.21451429,  0.12551409],
           [ 0.7207452 ,  0.5150524 ,  0.00542995],
           [ 0.5774373 ,  0.25776303,  0.26855013],
           [ 0.17295608,  0.31638905,  0.2951069 ],
           [ 0.7390023 ,  0.60931826,  0.43826106],
           [ 0.62775826,  0.5176888 ,  0.3718117 ],
           [ 0.51389074,  0.42036685,  0.29863322],
           [ 0.3694671 ,  0.30227602,  0.20832038],
           [ 0.2630599 ,  0.21490613,  0.14286397],
           [ 0.16101658,  0.13380753,  0.08050805]]...),)
    """

    image = cast("NDArrayReal", image)

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    settings = SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.copy()
    settings.update(kwargs)

    working_height = int(settings["working_width"] / settings["aspect_ratio"])
    image = reformat_image(
        image, settings["working_width"], settings["interpolation_method"]
    )

    rectangle = as_int32_array(
        [
            [settings["working_width"], 0],
            [settings["working_width"], working_height],
            [0, working_height],
            [0, 0],
        ]
    )

    colour_checkers_data = []

    if hasattr(segmentation_data, "rectangles"):
        colour_checkers_data.extend(
            sample_colour_checker(image, quadrilateral, rectangle, samples, **settings)
            for quadrilateral in segmentation_data.rectangles
        )
    else:
        colour_checkers_data.extend(
            sample_colour_checker(image, quadrilateral, rectangle, samples, **settings)
            for quadrilateral in segmentation_data
        )

    if additional_data:
        return tuple(colour_checkers_data)

    return tuple(
        colour_checker_data.swatch_colours
        for colour_checker_data in colour_checkers_data
    )


@typing.overload
def detect_colour_checkers_segmentation(
    image: str | ArrayLike,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    segmenter: Callable = ...,
    segmenter_kwargs: dict | None = ...,
    extractor: Callable = ...,
    extractor_kwargs: dict | None = ...,
    show: bool = ...,
    additional_data: Literal[True] = True,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker, ...]: ...


@typing.overload
def detect_colour_checkers_segmentation(
    image: str | ArrayLike,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    segmenter: Callable = ...,
    segmenter_kwargs: dict | None = ...,
    extractor: Callable = ...,
    extractor_kwargs: dict | None = ...,
    show: bool = ...,
    *,
    additional_data: Literal[False],
    **kwargs: Any,
) -> Tuple[NDArrayFloat, ...]: ...


@typing.overload
def detect_colour_checkers_segmentation(
    image: str | ArrayLike,
    samples: int,
    cctf_decoding: Callable,
    apply_cctf_decoding: bool,
    segmenter: Callable,
    segmenter_kwargs: dict | None,
    extractor: Callable,
    extractor_kwargs: dict | None,
    show: bool,
    additional_data: Literal[False],
    **kwargs: Any,
) -> Tuple[NDArrayFloat, ...]: ...


def detect_colour_checkers_segmentation(
    image: str | ArrayLike,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    segmenter: Callable = segmenter_default,
    segmenter_kwargs: dict | None = None,
    extractor: Callable = extractor_segmentation,
    extractor_kwargs: dict | None = None,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker, ...] | Tuple[NDArrayFloat, ...]:
    """
    Detect the colour checkers swatches in specified image using segmentation.

    Parameters
    ----------
    image
        Image (or image path to read the image from) to detect the colour
        checkers swatches from.
    samples
        Sample count to use to average (mean) the swatches colours. The effective
        sample count is :math:`samples^2`.
    cctf_decoding
        Decoding colour component transfer function / opto-electronic
        transfer function used when converting the image from 8-bit to float.
    apply_cctf_decoding
        Apply the decoding colour component transfer function / opto-electronic
        transfer function.
    segmenter
        Callable responsible to segment the image and extract the colour
        checker rectangles.
    segmenter_kwargs
        Keyword arguments to pass to the ``segmenter``.
    extractor
        Callable responsible to extract the colour checker data from the
        segmented rectangles.
    extractor_kwargs
        Keyword arguments to pass to the ``extractor``.
    show
        Whether to show various debug images.
    additional_data
        Whether to output additional data.

    Other Parameters
    ----------------
    adaptive_threshold_kwargs
        Keyword arguments for :func:`cv2.adaptiveThreshold` definition.
    aspect_ratio
        Colour checker aspect ratio, e.g., 1.5.
    aspect_ratio_minimum
        Minimum colour checker aspect ratio for detection: projective geometry
        might reduce the colour checker aspect ratio.
    aspect_ratio_maximum
        Maximum colour checker aspect ratio for detection: projective geometry
        might increase the colour checker aspect ratio.
    bilateral_filter_iterations
        Number of iterations to use for bilateral filtering.
    bilateral_filter_kwargs
        Keyword arguments for :func:`cv2.bilateralFilter` definition.
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    interpolation_method
        Interpolation method used when resizing the images, `cv2.INTER_CUBIC`
        and `cv2.INTER_LINEAR` methods are recommended.
    reference_values
        Reference values for the colour checker of interest.
    swatch_contour_scale
        As the image is filtered, the swatches area will tend to shrink, the
        generated contours can thus be scaled.
    swatch_minimum_area_factor
        Swatch minimum area factor :math:`f` with the minimum area :math:`m_a`
        expressed as follows: :math:`m_a = image_w * image_h / s_c / f` where
        :math:`image_w`, :math:`image_h` and :math:`s_c` are respectively the
        image width, height and the swatches count.
    swatches
        Colour checker swatches total count.
    swatches_achromatic_slice
        A `slice` instance defining achromatic swatches used to detect if the
        colour checker is upside down.
    swatches_chromatic_slice
        A `slice` instance defining chromatic swatches used to detect if the
        colour checker is upside down.
    swatches_count_maximum
        Maximum swatches count to be considered for the detection.
    swatches_count_minimum
        Minimum swatches count to be considered for the detection.
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
    :class`tuple`
        Tuple of :class:`DataDetectionColourChecker` class
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
    >>> detect_colour_checkers_segmentation(image, apply_cctf_decoding=True)
    ... # doctest: +SKIP
    (array([[  1.06804565e-01,   4.08109687e-02,   1.31266015e-02],
           [  3.49683374e-01,   1.29072860e-01,   4.77699488e-02],
           [  9.05181840e-02,   8.14911500e-02,   6.78349435e-02],
           [  7.58600757e-02,   6.07827008e-02,   1.10303042e-02],
           [  1.45751402e-01,   8.31151009e-02,   7.72669688e-02],
           [  1.00159235e-01,   1.62195116e-01,   6.91367984e-02],
           [  4.20038819e-01,   1.01927504e-01,   6.38073916e-03],
           [  6.02441281e-02,   5.23594357e-02,   8.94734785e-02],
           [  3.43939215e-01,   5.94407171e-02,   2.92109884e-02],
           [  7.70443529e-02,   2.71991435e-02,   3.07092462e-02],
           [  2.01171950e-01,   1.77795976e-01,   2.66185054e-03],
           [  3.81304830e-01,   1.33062363e-01,   1.23752898e-03],
           [  3.14101577e-02,   2.89250631e-02,   6.13652393e-02],
           [  6.45340234e-02,   1.22705154e-01,   1.41930664e-02],
           [  2.68294245e-01,   3.78271416e-02,   1.45846475e-02],
           [  4.78512675e-01,   2.28180945e-01,   4.21307486e-04],
           [  2.93216556e-01,   5.40601388e-02,   5.87599911e-02],
           [  2.61003822e-02,   8.16240311e-02,   7.08173811e-02],
           [  5.06426811e-01,   3.29764754e-01,   1.61412135e-01],
           [  3.52356732e-01,   2.30743274e-01,   1.14147641e-01],
           [  2.27152765e-01,   1.47627085e-01,   7.27180764e-02],
           [  1.12553783e-01,   7.42645189e-02,   3.58113647e-02],
           [  5.65953329e-02,   3.79680507e-02,   1.81693807e-02],
           [  2.30163466e-02,   1.61432363e-02,   7.43864896e-03]]...),)
    """

    if segmenter_kwargs is None:
        segmenter_kwargs = {}

    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    swatches_h = settings.swatches_horizontal
    swatches_v = settings.swatches_vertical

    if isinstance(image, str):
        image = read_image(image)
    else:
        image = convert_bit_depth(
            image,
            DTYPE_FLOAT_DEFAULT.__name__,  # pyright: ignore
        )

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    image = cast("NDArrayReal", image)

    image = reformat_image(image, settings.working_width, settings.interpolation_method)

    segmentation_colour_checkers_data = segmenter(
        image, additional_data=True, **{**segmenter_kwargs, **settings}
    )

    extractor_kwargs = cast("Dict[str, Any]", optional(extractor_kwargs, {}))

    colour_checkers_data = list(
        extractor(
            image,
            segmentation_colour_checkers_data,
            samples=samples,
            cctf_decoding=cctf_decoding,
            apply_cctf_decoding=False,
            additional_data=True,
            **{**extractor_kwargs, **kwargs},
        )
    )

    if show:
        plot_detection_results(
            tuple(colour_checkers_data),
            swatches_h,
            swatches_v,
            segmentation_colour_checkers_data,
            image,
        )

    if additional_data:
        return tuple(colour_checkers_data)

    return tuple(
        colour_checker_data.swatch_colours
        for colour_checker_data in colour_checkers_data
    )
