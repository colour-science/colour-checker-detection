"""
Colour Checker Detection - Templated
=====================================

Defines the objects to perform templated (warped perspective) colour checker detection:

-   :func:`colour_checker_detection.detect_colour_checkers_templated`
-   :func:`colour_checker_detection.segmenter_templated`
-   :func:`colour_checker_detection.extractor_templated`
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from colour import read_image
from colour.hints import (
    Any,
    ArrayLike,
    NDArrayFloat,
    NDArrayInt,
    NDArrayReal,
    cast,
)

if TYPE_CHECKING:
    from collections.abc import Callable
from colour.io import convert_bit_depth
from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.utilities import (
    Structure,
    optional,
    required,
    usage_warning,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from colour_checker_detection.detection.common import (
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    DataDetectionColourChecker,
    as_float32_array,
    as_int32_array,
    cluster_swatches,
    contour_centroid,
    detect_contours,
    filter_clusters,
    reformat_image,
    remove_stacked_contours,
)
from colour_checker_detection.detection.plotting import plot_detection_results
from colour_checker_detection.detection.segmentation import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    DataSegmentationColourCheckers,
)
from colour_checker_detection.detection.templates import (
    PATH_TEMPLATE_COLORCHECKER_CLASSIC,
    load_template,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC",
    "WarpingData",
    "segmenter_templated",
    "extractor_templated",
    "detect_colour_checkers_templated",
]

SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC: dict = (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.copy()
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC = DocstringDict(
        SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC
    )
    SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC.__doc__ = """
Settings for the templated detection of the *X-Rite* *ColorChecker Classic*.
"""
SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC.update(
    {
        "contour_approximation_factor": 0.1,
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 5,
        "transformation_cost_threshold": 10.0,
        "swatches_chromatic_slice": slice(0, 18),
        "swatches_achromatic_slice": slice(18, 24),
    }
)

TEMPLATE_COLORCHECKER_CLASSIC = load_template(PATH_TEMPLATE_COLORCHECKER_CLASSIC)
"""Default ColorChecker Classic template for templated detection."""


@dataclass
class WarpingData:
    """
    Data class for storing the results of the correspondence finding.

    Parameters
    ----------
    cluster_id
        The index of the cluster that was used for the correspondence.
    cost
        The cost of the transformation, which means the average distance of the
        warped point from the reference template point.
    transformation
        The transformation matrix to warp the cluster to the template.
    """

    cluster_id: int = -1
    cost: float = np.inf
    transformation: np.ndarray | None = field(default=None)


@typing.overload
def segmenter_templated(
    image: ArrayLike,
    cctf_encoding: Callable = ...,
    apply_cctf_encoding: bool = ...,
    additional_data: Literal[True] = True,
    **kwargs: Any,
) -> DataSegmentationColourCheckers: ...


@typing.overload
def segmenter_templated(
    image: ArrayLike,
    cctf_encoding: Callable = ...,
    apply_cctf_encoding: bool = ...,
    *,
    additional_data: Literal[False],
    **kwargs: Any,
) -> NDArrayInt: ...


@typing.overload
def segmenter_templated(
    image: ArrayLike,
    cctf_encoding: Callable,
    apply_cctf_encoding: bool,
    additional_data: Literal[False],
    **kwargs: Any,
) -> NDArrayInt: ...


@required("scikit-learn")  # pyright: ignore
def segmenter_templated(
    image: ArrayLike,
    cctf_encoding: Callable = eotf_inverse_sRGB,
    apply_cctf_encoding: bool = True,
    additional_data: bool = False,
    **kwargs: Any,
) -> DataSegmentationColourCheckers | NDArrayInt:
    """
    Detect the colour checker rectangles, clusters and swatches in specified image
    using segmentation with advanced filtering.

    The process is as follows:
        1. Input image is converted to a grayscale image and normalised to range [0, 1].
        2. Image is denoised using multiple bilateral filtering passes.
        3. Image is thresholded.
        4. Image is eroded and dilated to cleanup remaining noise.
        5. Contours are detected.
        6. Contours are filtered to only keep squares/swatches above and below defined
           surface area, moreover they have to resemble a convex quadrilateral.
           Additionally, squareness, area, aspect ratio and orientation are used as
           features to remove any remaining outlier contours.
        7. Stacked contours are removed.
        8. Swatches are clustered to isolate region-of-interest that are potentially
           colour checkers: Contours are scaled by a third so that colour checkers
           swatches are joined, creating a large rectangular cluster. Rectangles
           are fitted to the clusters.
        9. Clusters with a number of swatches close to the expected one are kept.

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
        Colour checker aspect ratio, e.g. 1.5.
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
    contour_approximation_factor
        Approximation factor for the Douglas-Peucker polygon approximation algorithm.
        It controls how aggressively contours are simplified, expressed as a fraction
        of the contour's perimeter. Lower values (e.g., 0.01) preserve more detail,
        higher values (e.g., 0.1) simplify more aggressively.
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    dbscan_eps
        DBSCAN epsilon parameter defining the maximum distance between two samples
        for them to be considered in the same neighborhood. Lower values create
        tighter clusters. Default is 0.5.
    dbscan_min_samples
        DBSCAN minimum samples parameter defining the number of samples in a
        neighborhood for a point to be considered a core point. Default is 5.
    transformation_cost_threshold
        Cost threshold for early termination of transformation search. If a
        transformation achieves an average distance below this threshold, the search
        stops immediately. Lower values require better matches. Default is 10.0.
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
    :class:`colour_checker_detection.DataSegmentationColourCheckers`
    or :class:`np.ndarray`
    Colour checker rectangles and additional data or colour checker rectangles only.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS, segmenter_templated
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> segmenter_templated(image)  # doctest: +ELLIPSIS
    array([[[ 357,  690],
            [ 373,  219],
            [1086,  244],
            [1069,  715]]], dtype=int32)
    """

    from sklearn.cluster import DBSCAN  # noqa: PLC0415

    settings = Structure(**SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    if apply_cctf_encoding:
        image = cctf_encoding(image)

    image = reformat_image(image, settings.working_width, settings.interpolation_method)

    image = cast("NDArrayFloat", image)

    contours, image_k = detect_contours(image, True, **settings)  # pyright: ignore

    # Filter contours using multiple features: area, convexity, squareness,
    # aspect ratio, and orientation
    width, height = image.shape[1], image.shape[0]
    minimum_area = (
        width * height / settings.swatches / settings.swatch_minimum_area_factor
    )
    maximum_area = width * height / settings.swatches

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    squares = []
    features = []

    for contour in contours:
        curve = cv2.approxPolyDP(
            as_int32_array(contour),
            settings.contour_approximation_factor
            * cv2.arcLength(as_int32_array(contour), True),
            True,
        )

        area = cv2.contourArea(curve)

        if minimum_area < area < maximum_area and len(curve) == 4:
            swatch = curve.reshape(-1, 2)
            squares.append(swatch)

            squareness = cv2.matchShapes(swatch, square, cv2.CONTOURS_MATCH_I2, 0.0)
            bbox = cv2.boundingRect(swatch)
            aspect_ratio = float(bbox[2]) / bbox[3]
            features.append([squareness, area, aspect_ratio])

    if squares:
        features_array = np.array(features)
        features_std = np.std(features_array, axis=0)
        features_std[features_std == 0] = 1.0  # Avoid division by zero
        features_normalized = (
            features_array - np.mean(features_array, axis=0)
        ) / features_std

        clustering = DBSCAN(
            eps=settings.dbscan_eps,
            min_samples=settings.dbscan_min_samples,
        ).fit(features_normalized)
        mask = clustering.labels_ != -1

        # CRITICAL: If DBSCAN removes everything, keep original swatches
        if np.sum(mask) == 0:
            mask = np.ones(len(squares), dtype=bool)

        squares = np.array(squares)[mask]

    squares = (
        as_int32_array(squares)
        if len(squares) > 0
        else np.empty((0, 4, 2), dtype=DTYPE_INT_DEFAULT)
    )

    swatches = as_int32_array(remove_stacked_contours(squares))

    clusters = cluster_swatches(image, swatches, settings.swatch_contour_scale)

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
def extractor_templated(
    image: ArrayLike,
    segmentation_data: DataSegmentationColourCheckers,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    additional_data: Literal[True] = True,
    **kwargs: Any,
) -> tuple[DataDetectionColourChecker, ...]: ...


@typing.overload
def extractor_templated(
    image: ArrayLike,
    segmentation_data: DataSegmentationColourCheckers,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    *,
    additional_data: Literal[False],
    **kwargs: Any,
) -> tuple[NDArrayFloat, ...]: ...


@typing.overload
def extractor_templated(
    image: ArrayLike,
    segmentation_data: DataSegmentationColourCheckers,
    samples: int,
    cctf_decoding: Callable,
    apply_cctf_decoding: bool,
    additional_data: Literal[False],
    **kwargs: Any,
) -> tuple[NDArrayFloat, ...]: ...


def extractor_templated(
    image: ArrayLike,
    segmentation_data: DataSegmentationColourCheckers,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> tuple[DataDetectionColourChecker, ...] | tuple[NDArrayFloat, ...]:
    """
    Extract colour swatches using template-based perspective transformation.

    This function takes segmentation data and extracts colors using template matching
    with perspective transformation. This extractor should be used when the colour
    checker is not facing the camera straight.

    The process is as follows:

    1.  The swatches are converted to centroids and used to filter clusters to only
        keep the ones that contain the expected number of swatches. Moreover, the
        centroids are grouped by the clusters.
    2.  The centroids are ordered within their group to enforce the same ordering as
        the template, which is important to extract the transformation, since OpenCV's
        perspective transform is not invariant to the ordering of the points.
    3.  The best transformation is determined by finding the transformation that
        minimizes the average distance of the warped points from the reference
        template points.
    4.  The image is warped using the determined transformation.
    5.  The colours are extracted from the warped image using a sampling window
        around the centroids.
    6.  The colours are corrected if the chromatic swatches have a lower standard
        deviation than the achromatic swatches.

    Parameters
    ----------
    image
        Image to extract the colour checker swatches and colours from.
    segmentation_data
        Segmentation colour checkers data from the segmenter.
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
        Whether to output additional data.

    Other Parameters
    ----------------
    template
        Template defining the swatches structure, which is exploited to find the best
        correspondences between template and detected swatches, which yield the
        optimal transformation. If not provided, defaults to built-in ColorChecker
        Classic template.
    residual_threshold
        Maximum allowed residual between detected and template colours.
        Higher values are more permissive. Default is 0.3 (30%).
    adaptive_threshold_kwargs
        Keyword arguments for :func:`cv2.adaptiveThreshold` definition.
    aspect_ratio
        Colour checker aspect ratio, e.g. 1.5.
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
    contour_approximation_factor
        Approximation factor for the Douglas-Peucker polygon approximation algorithm.
        It controls how aggressively contours are simplified, expressed as a fraction
        of the contour's perimeter. Lower values (e.g., 0.01) preserve more detail,
        higher values (e.g., 0.1) simplify more aggressively.
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    dbscan_eps
        DBSCAN epsilon parameter defining the maximum distance between two samples
        for them to be considered in the same neighborhood. Lower values create
        tighter clusters. Default is 0.5.
    dbscan_min_samples
        DBSCAN minimum samples parameter defining the number of samples in a
        neighborhood for a point to be considered a core point. Default is 5.
    transformation_cost_threshold
        Cost threshold for early termination of transformation search. If a
        transformation achieves an average distance below this threshold, the search
        stops immediately. Lower values require better matches. Default is 10.0.
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
    >>> from colour_checker_detection import (
    ...     ROOT_RESOURCES_TESTS,
    ...     segmenter_templated,
    ...     extractor_templated,
    ... )
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> segmentation_data = segmenter_templated(image, additional_data=True)
    >>> extractor_templated(image, segmentation_data)  # doctest: +SKIP
    (array([[ 0.36081576,  0.22396202,  0.11733589],
           [ 0.6274815 ,  0.39514375,  0.24308297],
           [ 0.33063054,  0.3158751 ,  0.28996205],
           [ 0.30372787,  0.2742474 ,  0.10494538],
           [ 0.41764253,  0.31940183,  0.30804706],
           [ 0.34960267,  0.44142178,  0.29417506],
           [ 0.682801  ,  0.3538923 ,  0.07184852],
           [ 0.27251157,  0.2532009 ,  0.33145225],
           [ 0.62005484,  0.2703342 ,  0.18676178],
           [ 0.3079272 ,  0.1803046 ,  0.19187638],
           [ 0.48746303,  0.4604792 ,  0.03282085],
           [ 0.6541456 ,  0.40173233,  0.01583917],
           [ 0.19250122,  0.185604  ,  0.2739023 ],
           [ 0.28076768,  0.38508102,  0.12207687],
           [ 0.5527626 ,  0.21404609,  0.1256289 ],
           [ 0.7217179 ,  0.51569265,  0.00520882],
           [ 0.57813776,  0.25853688,  0.26927036],
           [ 0.17615536,  0.31684747,  0.29624644],
           [ 0.74493927,  0.6126149 ,  0.44073734],
           [ 0.6314545 ,  0.5187937 ,  0.3728771 ],
           [ 0.5143494 ,  0.42190555,  0.29967216],
           [ 0.37282884,  0.30393514,  0.21030639],
           [ 0.2645776 ,  0.21623953,  0.1437975 ],
           [ 0.16090113,  0.13422866,  0.081468  ]], dtype=float32),)
    """

    template = kwargs.pop("template", TEMPLATE_COLORCHECKER_CLASSIC)
    residual_threshold = kwargs.pop("residual_threshold", 0.3)

    settings = Structure(**SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    image = reformat_image(image, settings.working_width, settings.interpolation_method)

    image = cast("NDArrayFloat", image)

    all_centroids = np.array(
        [contour_centroid(swatch) for swatch in segmentation_data.swatches]
    )

    clustered_centroids = []
    for cluster in segmentation_data.clusters:
        mask = np.array(
            [
                cv2.pointPolygonTest(cluster, tuple(centroid), False) == 1
                for centroid in all_centroids
            ]
        )
        centroids_in_cluster = all_centroids[mask] if np.any(mask) else np.array([])
        clustered_centroids.append(centroids_in_cluster)

    nr_expected_swatches = len(template.swatch_centroids)
    filtered_centroids = [
        as_int32_array(centroids)
        for centroids in clustered_centroids
        if nr_expected_swatches / 3 <= len(centroids) <= nr_expected_swatches
    ]

    clustered_centroids = filtered_centroids or []

    ordered_clustered_centroids = []

    for centroids in clustered_centroids:
        if len(centroids) < 4:
            ordered_clustered_centroids.append(centroids)
            continue

        cluster_centroid = np.mean(centroids, axis=0)

        angles = np.array(
            [
                np.arctan2(pt[1] - cluster_centroid[1], pt[0] - cluster_centroid[0])
                for pt in centroids
            ]
        )

        quadrant_points = [[] for _ in range(4)]

        for i, angle in enumerate(angles):
            quadrant = int((angle + np.pi) / (np.pi / 2)) % 4
            quadrant_points[quadrant].append((i, centroids[i]))

        ordered_points = []
        for quadrant in quadrant_points:
            if quadrant:
                furthest_idx = max(
                    quadrant,
                    key=lambda x: float(np.linalg.norm(x[1] - cluster_centroid)),
                )[0]
                ordered_points.append(centroids[furthest_idx])

        if len(ordered_points) != 4:
            ordered_points = centroids[:4] if len(centroids) >= 4 else centroids

        ordered_clustered_centroids.append(np.array(ordered_points))

    starting_pts = ordered_clustered_centroids

    warping_data = [
        WarpingData(cluster_id) for cluster_id in range(len(clustered_centroids))
    ]
    best_global_cost = np.inf

    for cluster_id, (cluster, cluster_pts) in enumerate(
        zip(clustered_centroids, starting_pts, strict=False)
    ):
        if best_global_cost < settings.transformation_cost_threshold:
            break

        for correspondence in template.correspondences:
            transformation = cv2.getPerspectiveTransform(
                cluster_pts.astype(np.float32),
                template.swatch_centroids[list(correspondence)].astype(np.float32),
            )
            warped_pts = cv2.perspectiveTransform(
                cluster[None, :, :].astype(np.float32), transformation
            ).reshape(-1, 2)

            cost_matrix = distance_matrix(warped_pts, template.swatch_centroids)
            row_id, col_id = linear_sum_assignment(cost_matrix)  # pyright: ignore
            cost = float(np.sum(cost_matrix[row_id, col_id]) / len(cluster))

            if cost < warping_data[cluster_id].cost:
                warping_data[cluster_id].cost = cost
                warping_data[cluster_id].transformation = transformation

                best_global_cost = min(best_global_cost, cost)

                if cost < settings.transformation_cost_threshold:
                    break

    best_warping_data = min(warping_data, key=lambda x: x.cost)
    transformation = best_warping_data.transformation

    if transformation is None:
        error = "No valid transformation found for colour checker detection."
        raise ValueError(error)

    warped_image = as_float32_array(
        cv2.warpPerspective(image, transformation, (template.width, template.height))
    )

    sample_radius = samples // 2
    h, w = warped_image.shape[:2]
    colours = np.zeros((len(template.swatch_centroids), 3), dtype=DTYPE_FLOAT_DEFAULT)

    for i, centroid in enumerate(template.swatch_centroids):
        x, y = int(centroid[0]), int(centroid[1])

        x_min = max(0, x - sample_radius)
        x_max = min(w, x + sample_radius)
        y_min = max(0, y - sample_radius)
        y_max = min(h, y + sample_radius)

        window = warped_image[y_min:y_max, x_min:x_max]
        colours[i] = np.mean(window.reshape(-1, 3), axis=0)

    swatch_colours = as_float32_array(colours)

    chromatic_std = np.mean(
        [np.std(c) for c in swatch_colours[settings.swatches_chromatic_slice]]
    )
    achromatic_std = np.mean(
        [np.std(c) for c in swatch_colours[settings.swatches_achromatic_slice]]
    )
    if chromatic_std < achromatic_std:
        usage_warning("Colour checker was seemingly flipped, reversing the samples!")
        swatch_colours = swatch_colours[::-1]

    residual = [
        np.abs(r - m)
        for r, m in zip(template.colours, np.array(swatch_colours), strict=False)
    ]
    if np.max(residual) > residual_threshold:
        usage_warning(
            f"Colour accuracy warning: max residual {np.max(residual):.3f} exceeds "
            f"threshold {residual_threshold}. The detected colours may be inaccurate. "
            "Check if the colour checker is properly lit and not occluded."
        )

    # Use the cluster corresponding to the best transformation
    best_cluster = (
        segmentation_data.clusters[best_warping_data.cluster_id]
        if 0 <= best_warping_data.cluster_id < len(segmentation_data.clusters)
        else np.array([])
    )

    colour_checkers_data = [
        DataDetectionColourChecker(
            swatch_colours,
            np.array([]),
            warped_image,
            best_cluster,
        )
    ]

    if additional_data:
        return tuple(colour_checkers_data)

    return tuple(
        colour_checker_data.swatch_colours
        for colour_checker_data in colour_checkers_data
    )


@typing.overload
def detect_colour_checkers_templated(
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
) -> tuple[DataDetectionColourChecker, ...]: ...


@typing.overload
def detect_colour_checkers_templated(
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
) -> tuple[NDArrayFloat, ...]: ...


@typing.overload
def detect_colour_checkers_templated(
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
) -> tuple[NDArrayFloat, ...]: ...


def detect_colour_checkers_templated(
    image: str | ArrayLike,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    segmenter: Callable = segmenter_templated,
    segmenter_kwargs: dict | None = None,
    extractor: Callable = extractor_templated,
    extractor_kwargs: dict | None = None,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> tuple[DataDetectionColourChecker, ...] | tuple[NDArrayFloat, ...]:
    """
    Detect the colour checkers swatches in specified image using templated methods.

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
        Keyword arguments to pass to the ``segmenter``. Can include 'template'
        as str (NPZ file path to template) or Template object.
        If 'template' not provided, defaults to built-in ColorChecker Classic template.
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
    greedy_heuristic : float, optional
        Heuristic threshold for early stopping in transformation search.
        Default is 2.0.
    validation_threshold : float, optional
        Threshold for colour validation.
        Default is 0.5.

    Returns
    -------
    :class:`tuple`
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
    >>> detect_colour_checkers_templated(image, apply_cctf_decoding=True)
    ... # doctest: +SKIP
    (array([[  1.07537337e-01,   4.11238223e-02,   1.31721459e-02],
           [  3.52024108e-01,   1.29535466e-01,   4.84532639e-02],
           [  9.00324881e-02,   8.14048126e-02,   6.83287457e-02],
           [  7.53633380e-02,   6.11113459e-02,   1.10184597e-02],
           [  1.46142766e-01,   8.32280964e-02,   7.74866268e-02],
           [  1.01110630e-01,   1.63705498e-01,   7.03689680e-02],
           [  4.23571885e-01,   1.02802373e-01,   6.50439737e-03],
           [  6.09256141e-02,   5.20781092e-02,   8.99062678e-02],
           [  3.42755497e-01,   5.93896434e-02,   2.91827880e-02],
           [  7.73139372e-02,   2.73966864e-02,   3.07213869e-02],
           [  2.03338221e-01,   1.79222777e-01,   2.65911571e-03],
           [  3.85695517e-01,   1.34022757e-01,   1.26003276e-03],
           [  3.15370820e-02,   2.88631991e-02,   6.08412772e-02],
           [  6.47268444e-02,   1.22600473e-01,   1.40322614e-02],
           [  2.66343951e-01,   3.76947522e-02,   1.44897113e-02],
           [  4.79563773e-01,   2.28976935e-01,   4.33672598e-04],
           [  2.93841749e-01,   5.43766469e-02,   5.89662455e-02],
           [  2.67328490e-02,   8.21092799e-02,   7.17147887e-02],
           [  5.15012801e-01,   3.33238840e-01,   1.63575962e-01],
           [  3.56554657e-01,   2.31821850e-01,   1.14737533e-01],
           [  2.27919579e-01,   1.48821548e-01,   7.34134316e-02],
           [  1.14748545e-01,   7.51190484e-02,   3.64632085e-02],
           [  5.69365770e-02,   3.84297743e-02,   1.82996020e-02],
           [  2.28971709e-02,   1.62528455e-02,   7.39292800e-03]]...),)
    """

    if segmenter_kwargs is None:
        segmenter_kwargs = {}

    settings = Structure(**SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    swatches_h = settings.swatches_horizontal
    swatches_v = settings.swatches_vertical

    segmenter_kwargs = segmenter_kwargs.copy()
    template = load_template(
        optional(
            segmenter_kwargs.pop("template", None), PATH_TEMPLATE_COLORCHECKER_CLASSIC
        )
    )

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
        image,
        additional_data=True,
        **{**segmenter_kwargs, **settings, "template": template},
    )

    extractor_kwargs = cast("dict[str, Any]", optional(extractor_kwargs, {}))

    colour_checkers_data = list(
        extractor(
            image,
            segmentation_colour_checkers_data,
            samples=samples,
            cctf_decoding=cctf_decoding,
            apply_cctf_decoding=False,
            additional_data=True,
            template=template,
            residual_threshold=0.3,
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
