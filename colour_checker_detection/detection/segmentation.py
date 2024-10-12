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

from dataclasses import dataclass

import cv2
import numpy as np
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Dict,
    NDArrayFloat,
    NDArrayInt,
    Tuple,
    Union,
    cast,
)
from colour.io import convert_bit_depth, read_image
from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.plotting import CONSTANTS_COLOUR_STYLE, plot_image
from colour.utilities import (
    MixinDataclassIterable,
    Structure,
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
    as_int32_array,
    contour_centroid,
    detect_contours,
    is_square,
    quadrilateralise_contours,
    reformat_image,
    remove_stacked_contours,
    sample_colour_checker,
    scale_contour,
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
    "SETTINGS_SEGMENTATION_COLORCHECKER_NANO",
    "DataSegmentationColourCheckers",
    "segmenter_default",
    "detect_colour_checkers_segmentation",
]

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC: Dict = (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
)

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.update(
    {
        "aspect_ratio_minimum": 1.5 * 0.9,
        "aspect_ratio_maximum": 1.5 * 1.1,
        "swatches_count_minimum": int(24 * 0.75),
        "swatches_count_maximum": int(24 * 1.25),
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


@dataclass
class DataSegmentationColourCheckers(MixinDataclassIterable):
    """
    Colour checkers detection data used for plotting, debugging and further
    analysis.

    Parameters
    ----------
    rectangles
        Colour checker bounding boxes, i.e., the clusters that have the
        relevant count of swatches.
    clusters
        Detected swatches clusters.
    swatches
        Detected swatches.
    segmented_image
        Segmented image.
    """

    rectangles: NDArrayInt
    clusters: NDArrayInt
    swatches: NDArrayInt
    segmented_image: NDArrayFloat


def segmenter_default(
    image: ArrayLike,
    cctf_encoding: Callable = eotf_inverse_sRGB,
    apply_cctf_encoding: bool = True,
    additional_data: bool = False,
    **kwargs: Any,
) -> DataSegmentationColourCheckers | NDArrayInt:
    """
    Detect the colour checker rectangles in given image :math:`image` using
    segmentation.

    The process is a follows:

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
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> segmenter_default(image)  # doctest: +ELLIPSIS
    array([[[ 358,  691],
            [ 373,  219],
            [1086,  242],
            [1071,  713]]]...)
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
            squares.append(
                as_int32_array(cv2.boxPoints(cv2.minAreaRect(swatch_contour)))
            )

    # Removing stacked squares.
    squares = as_int32_array(remove_stacked_contours(squares))

    # Clustering swatches.
    swatches = [
        scale_contour(square, settings.swatch_contour_scale) for square in squares
    ]
    image_c = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(
        image_c,
        as_int32_array(swatches),  # pyright: ignore
        -1,
        [255] * 3,
        -1,
    )
    image_c = cv2.cvtColor(image_c, cv2.COLOR_RGB2GRAY)

    contours, _hierarchy = cv2.findContours(
        image_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    clusters = as_int32_array(
        [cv2.boxPoints(cv2.minAreaRect(contour)) for contour in contours]
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
    counts = []
    for cluster in clusters:
        count = 0
        for swatch in swatches:
            if cv2.pointPolygonTest(cluster, contour_centroid(swatch), False) == 1:
                count += 1
        counts.append(count)

    indexes = np.where(
        np.logical_and(
            as_int32_array(counts) >= settings.swatches_count_minimum,
            as_int32_array(counts) <= settings.swatches_count_maximum,
        )
    )[0]

    rectangles = clusters[indexes]

    if additional_data:
        return DataSegmentationColourCheckers(
            rectangles,
            clusters,
            squares,
            image_k,  # pyright: ignore
        )
    else:
        return rectangles


def detect_colour_checkers_segmentation(
    image: str | ArrayLike,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    segmenter: Callable = segmenter_default,
    segmenter_kwargs: dict | None = None,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker | NDArrayFloat, ...]:
    """
    Detect the colour checkers swatches in given image using segmentation.

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
    >>> detect_colour_checkers_segmentation(image)  # doctest: +SKIP
    (array([[ 0.360005  ,  0.22310828,  0.11760835],
           [ 0.6258309 ,  0.39448667,  0.24166533],
           [ 0.33198   ,  0.31600377,  0.28866866],
           [ 0.3046006 ,  0.273321  ,  0.10486555],
           [ 0.41751358,  0.31914026,  0.30789137],
           [ 0.34866226,  0.43934596,  0.29126382],
           [ 0.67983997,  0.35236534,  0.06997226],
           [ 0.27118555,  0.25352538,  0.33078724],
           [ 0.62091863,  0.27034152,  0.18652563],
           [ 0.3071613 ,  0.17978874,  0.19181632],
           [ 0.48547146,  0.4585586 ,  0.03294956],
           [ 0.6507678 ,  0.40023172,  0.01607676],
           [ 0.19286253,  0.18585181,  0.27459183],
           [ 0.28054565,  0.38513032,  0.1224441 ],
           [ 0.5545431 ,  0.21436104,  0.12549178],
           [ 0.72068894,  0.51493925,  0.00548734],
           [ 0.5772921 ,  0.2577179 ,  0.2685553 ],
           [ 0.17289193,  0.3163792 ,  0.2950853 ],
           [ 0.7394083 ,  0.60953134,  0.4383072 ],
           [ 0.6281671 ,  0.51759964,  0.37215686],
           [ 0.51360977,  0.42048824,  0.2985709 ],
           [ 0.36953217,  0.30218402,  0.20827036],
           [ 0.26286703,  0.21493268,  0.14277342],
           [ 0.16102524,  0.13381621,  0.08047409]]...),)

    """

    if segmenter_kwargs is None:
        segmenter_kwargs = {}

    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    swatches_h = settings.swatches_horizontal
    swatches_v = settings.swatches_vertical
    working_width = settings.working_width
    working_height = int(working_width / settings.aspect_ratio)

    if isinstance(image, str):
        image = read_image(image)
    else:
        image = convert_bit_depth(
            image,
            DTYPE_FLOAT_DEFAULT.__name__,  # pyright: ignore
        )

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    image = cast(Union[NDArrayInt, NDArrayFloat], image)

    image = reformat_image(image, settings.working_width, settings.interpolation_method)

    rectangle = as_int32_array(
        [
            [working_width, 0],
            [working_width, working_height],
            [0, working_height],
            [0, 0],
        ]
    )

    segmentation_colour_checkers_data = segmenter(
        image, additional_data=True, **{**segmenter_kwargs, **settings}
    )

    colour_checkers_data = []
    for quadrilateral in segmentation_colour_checkers_data.rectangles:
        colour_checkers_data.append(
            sample_colour_checker(image, quadrilateral, rectangle, samples, **settings)
        )

        if show:
            colour_checker = np.copy(colour_checkers_data[-1].colour_checker)
            for swatch_mask in colour_checkers_data[-1].swatch_masks:
                colour_checker[
                    swatch_mask[0] : swatch_mask[1],
                    swatch_mask[2] : swatch_mask[3],
                    ...,
                ] = 0

            plot_image(
                CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(colour_checker),
            )

            plot_image(
                CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(
                    np.reshape(
                        colour_checkers_data[-1].swatch_colours,
                        [swatches_v, swatches_h, 3],
                    )
                ),
            )

    if show:
        plot_image(
            segmentation_colour_checkers_data.segmented_image,
            text_kwargs={"text": "Segmented Image", "color": "black"},
        )

        image_c = np.copy(image)

        cv2.drawContours(
            image_c,
            segmentation_colour_checkers_data.swatches,
            -1,
            (1, 0, 1),
            3,
        )
        cv2.drawContours(
            image_c,
            segmentation_colour_checkers_data.clusters,
            -1,
            (0, 1, 1),
            3,
        )

        plot_image(
            CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(image_c),
            text_kwargs={"text": "Swatches & Clusters", "color": "white"},
        )

    if additional_data:
        return tuple(colour_checkers_data)
    else:
        return tuple(
            colour_checker_data.swatch_colours
            for colour_checker_data in colour_checkers_data
        )
