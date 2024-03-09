"""
Colour Checker Detection - Segmentation
=======================================

Defines the objects for colour checker detection using segmentation:

-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC`
-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_SG`
-   :attr:`colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_NANO`
-   :func:`colour_checker_detection.plot_contours`
-   :func:`colour_checker_detection.plot_swatches_and_clusters`
-   :func:`colour_checker_detection.plot_colours`
-   :func:`colour_checker_detection.plot_colours_warped`
-   :func:`colour_checker_detection.segmenter_default`
-   :func:`colour_checker_detection.segmenter_warped`
-   :func:`colour_checker_detection.extractor_default`
-   :func:`colour_checker_detection.extractor_warped`
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
    is_string,
    usage_warning,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

from colour_checker_detection.detection.common import (
    DTYPE_FLOAT_DEFAULT,
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
    SETTINGS_DETECTION_COLORCHECKER_SG,
    DataDetectionColourChecker,
    as_int32_array,
    contour_centroid,
    detect_contours,
    is_convex_quadrilateral,
    is_square,
    largest_convex_quadrilateral,
    quadrilateralise_contours,
    reformat_image,
    remove_stacked_contours,
    sample_colour_checker,
    scale_contour,
)
from colour_checker_detection.detection.templates.generate_template import Template

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
    "WarpingData",
    "filter_contours",
    "filter_contours_multifeature",
    "cluster_swatches",
    "filter_clusters",
    "filter_clusters_by_swatches",
    "group_swatches",
    "order_centroids",
    "determine_best_transformation",
    "extract_colours",
    "correct_flipped",
    "check_residuals",
    "plot_contours",
    "plot_swatches_and_clusters",
    "plot_colours",
    "plot_colours_warped",
    "segmenter_default",
    "segmenter_warped",
    "extractor_default",
    "extractor_warped",
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
        "greedy_heuristic": 10,
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
    transformation: np.ndarray = None  # pyright: ignore


def filter_contours(
    image: NDArrayFloat,
    contours: ArrayLike,
    swatches: int,
    swatch_minimum_area_factor: float,
) -> NDArrayInt:
    """
    Filter the contours first by area and whether then by squareness.

    Parameters
    ----------
    image
        The image containing the contours. Only used for its shape.
    contours
        The contours from which to filter the swatches.
    swatches
        The expected number of swatches.
    swatch_minimum_area_factor
        The minimum area factor of the smallest swatch.

    Returns
    -------
    NDArrayInt
        The filtered contours.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import filter_contours
    >>> image = np.zeros((600, 900, 3))
    >>> contours = [
    ...     [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...     [[300, 100], [400, 100], [400, 200], [300, 200], [250, 100]],
    ...     [[200, 100], [600, 100], [600, 400], [200, 400]],
    ... ]
    >>> filter_contours(image, contours, 24, 200)
    array([[[100, 100],
            [200, 100],
            [200, 200],
            [100, 200]]], dtype=int32)
    """
    width, height = image.shape[1], image.shape[0]
    minimum_area = width * height / swatches / swatch_minimum_area_factor
    maximum_area = width * height / swatches
    squares = []
    for swatch_contour in quadrilateralise_contours(contours):
        if minimum_area < cv2.contourArea(swatch_contour) < maximum_area and is_square(
            swatch_contour
        ):
            squares.append(
                as_int32_array(cv2.boxPoints(cv2.minAreaRect(swatch_contour)))
            )
    return as_int32_array(squares)


def filter_contours_multifeature(
    image: NDArrayFloat,
    contours: NDArrayInt | Tuple[NDArrayInt],
    swatches: int,
    swatch_minimum_area_factor: float,
) -> NDArrayInt:
    """
    Filter the contours first by area and whether they are roughly a convex
    quadrilateral and afterwards by multiple features namely squareness,
    area, aspect ratio and orientation.

    Parameters
    ----------
    image
        The image containing the contours. Only used for its shape.
    contours
        The contours from which to filter the swatches.
    swatches
        The expected number of swatches.
    swatch_minimum_area_factor
        The minimum area factor of the smallest swatch.

    Returns
    -------
    NDArrayInt
        The filtered contours.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import filter_contours_multifeature
    >>> image = np.zeros((600, 900, 3))
    >>> contours = [
    ...     [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...     [[200, 200], [300, 200], [300, 300], [200, 300]],
    ...     [[300, 300], [400, 300], [400, 400], [300, 400]],
    ...     [[400, 400], [500, 400], [500, 500], [400, 500]],
    ...     [[500, 500], [600, 500], [600, 600], [500, 600]],
    ...     [[300, 100], [400, 100], [400, 200], [300, 200], [250, 100]],
    ...     [[200, 100], [600, 100], [600, 400], [200, 400]],
    ... ]
    >>> filter_contours_multifeature(image, contours, 24, 200)
    array([[[[100, 100]],
    <BLANKLINE>
            [[200, 100]],
    <BLANKLINE>
            [[200, 200]],
    <BLANKLINE>
            [[100, 200]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[200, 200]],
    <BLANKLINE>
            [[300, 200]],
    <BLANKLINE>
            [[300, 300]],
    <BLANKLINE>
            [[200, 300]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[300, 300]],
    <BLANKLINE>
            [[400, 300]],
    <BLANKLINE>
            [[400, 400]],
    <BLANKLINE>
            [[300, 400]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[400, 400]],
    <BLANKLINE>
            [[500, 400]],
    <BLANKLINE>
            [[500, 500]],
    <BLANKLINE>
            [[400, 500]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[500, 500]],
    <BLANKLINE>
            [[600, 500]],
    <BLANKLINE>
            [[600, 600]],
    <BLANKLINE>
            [[500, 600]]]], dtype=int32)
    """
    width, height = image.shape[1], image.shape[0]
    minimum_area = width * height / swatches / swatch_minimum_area_factor
    maximum_area = width * height / swatches

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    squares = []
    features = []
    for contour in contours:
        curve = cv2.approxPolyDP(
            as_int32_array(contour),
            0.01 * cv2.arcLength(as_int32_array(contour), True),
            True,
        )
        if minimum_area < cv2.contourArea(
            curve
        ) < maximum_area and is_convex_quadrilateral(curve):
            squares.append(largest_convex_quadrilateral(curve)[0])
            squareness = cv2.matchShapes(
                squares[-1], square, cv2.CONTOURS_MATCH_I2, 0.0
            )
            area = cv2.contourArea(squares[-1])
            aspect_ratio = (
                float(cv2.boundingRect(squares[-1])[2])
                / cv2.boundingRect(squares[-1])[3]
            )
            orientation = cv2.minAreaRect(squares[-1])[-1]
            features.append([squareness, area, aspect_ratio, orientation])

    if squares:
        features = np.array(features)
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        clustering = DBSCAN().fit(features)
        mask = clustering.labels_ != -1
        squares = np.array(squares)[mask]

    return squares  # pyright: ignore


def cluster_swatches(
    image: NDArrayFloat, swatches: NDArrayInt, swatch_contour_scale: float
) -> NDArrayInt:
    """
    Determine the clusters of swatches by expanding the swatches and
    fitting rectangles to overlapping swatches.

    Parameters
    ----------
    image
        The image containing the swatches. Only used for its shape.
    swatches
        The swatches to cluster.
    swatch_contour_scale
        The scale by which to expand the swatches.

    Returns
    -------
    NDArrayInt
        The clusters of swatches.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import cluster_swatches
    >>> image = np.zeros((600, 900, 3))
    >>> swatches = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[300, 100], [400, 100], [400, 200], [300, 200]],
    ...     ]
    ... )
    >>> cluster_swatches(image, swatches, 1.5)
    array([[[275,  75],
            [425,  75],
            [425, 225],
            [275, 225]],
    <BLANKLINE>
           [[ 75,  75],
            [225,  75],
            [225, 225],
            [ 75, 225]]], dtype=int32)
    """
    scaled_swatches = [
        scale_contour(swatch, swatch_contour_scale) for swatch in swatches
    ]
    image_c = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(
        image_c,
        as_int32_array(scaled_swatches),  # pyright: ignore
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
    return clusters


def filter_clusters(
    clusters: NDArrayInt, aspect_ratio_minimum: float, aspect_ratio_maximum: float
) -> NDArrayInt:
    """
    Filter the clusters by the expected aspect ratio.

    Parameters
    ----------
    clusters
        The clusters to filter.
    aspect_ratio_minimum
        The minimum aspect ratio.
    aspect_ratio_maximum
        The maximum aspect ratio.

    Returns
    -------
    NDArrayInt
        The filtered clusters.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import filter_clusters
    >>> clusters = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[300, 100], [400, 100], [400, 300], [300, 300]],
    ...     ]
    ... )
    >>> filter_clusters(clusters, 0.9, 1.1)
    array([[[100, 100],
            [200, 100],
            [200, 200],
            [100, 200]]], dtype=int32)
    """
    filtered_clusters = []
    for cluster in clusters[:]:
        rectangle = cv2.minAreaRect(cluster)
        width = max(rectangle[1][0], rectangle[1][1])
        height = min(rectangle[1][0], rectangle[1][1])
        ratio = width / height

        if aspect_ratio_minimum < ratio < aspect_ratio_maximum:
            filtered_clusters.append(as_int32_array(cluster))
    return as_int32_array(filtered_clusters)


def filter_clusters_by_swatches(
    clusters: NDArrayInt,
    swatches: NDArrayInt,
    swatches_count_minimum: int,
    swatches_count_maximum: int,
) -> NDArrayInt:
    """
    Filter the clusters by the number of swatches they contain.

    Parameters
    ----------
    clusters
        The clusters to filter.
    swatches
        The swatches to filter by.
    swatches_count_minimum
        The minimum number of swatches.
    swatches_count_maximum
        The maximum number of swatches.

    Returns
    -------
    NDArrayInt
        The filtered clusters.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import filter_clusters_by_swatches
    >>> clusters = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[300, 100], [400, 100], [400, 300], [300, 300]],
    ...     ]
    ... )
    >>> swatches = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...     ]
    ... )
    >>> filter_clusters_by_swatches(clusters, swatches, 1, 4)
    array([[[100, 100],
            [200, 100],
            [200, 200],
            [100, 200]]])
    """
    counts = []
    for cluster in clusters:
        count = 0
        for swatch in swatches:
            if cv2.pointPolygonTest(cluster, contour_centroid(swatch), False) == 1:
                count += 1
        counts.append(count)

    indexes = np.where(
        np.logical_and(
            as_int32_array(counts) >= swatches_count_minimum,
            as_int32_array(counts) <= swatches_count_maximum,
        )
    )[0]

    rectangles = clusters[indexes]
    return rectangles


def group_swatches(
    clusters: NDArrayInt, swatches: NDArrayInt, template: Template
) -> NDArrayInt:
    """
    Transform the swatches into centroids and groups the swatches by cluster.
    Also, removes clusters that do not contain the expected number of swatches.

    Parameters
    ----------
    clusters
        The clusters to group the swatches by.
    swatches
        The swatches to group.
    template
        The template that contains the expected number of swatches

    Returns
    -------
    NDArrayInt
        The clustered swatch centroids.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import group_swatches
    >>> from colour_checker_detection.detection.templates.generate_template import (
    ...     Template,
    ... )
    >>> template = Template(None, None, None, None, None)
    >>> template.swatch_centroids = np.array(
    ...     [
    ...         [150, 150],
    ...         [300, 100],
    ...     ]
    ... )
    >>> clusters = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[300, 100], [400, 100], [400, 300], [300, 300]],
    ...     ]
    ... )
    >>> swatches = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...     ]
    ... )
    >>> group_swatches(clusters, swatches, template)
    array([[[150, 150],
            [150, 150]]], dtype=int32)
    """

    clustered_centroids = []
    for cluster in clusters:
        centroids_in_cluster = []
        for swatch in swatches:
            centroid = contour_centroid(swatch)
            if cv2.pointPolygonTest(cluster, centroid, False) == 1:
                centroids_in_cluster.append(centroid)
        clustered_centroids.append(np.array(centroids_in_cluster))

    nr_expected_swatches = len(template.swatch_centroids)
    clustered_centroids = as_int32_array(
        [
            as_int32_array(centroids)
            for centroids in clustered_centroids
            if nr_expected_swatches / 3 <= len(centroids) <= nr_expected_swatches
        ]
    )
    return clustered_centroids


def order_centroids(clustered_centroids: NDArrayInt) -> NDArrayInt:
    """
    Determine the outermost points of the clusters to use as starting
    points for the transformation.

    Parameters
    ----------
    clustered_centroids
        The centroids of all swatches grouped by cluster.

    Returns
    -------
    NDArrayInt
        The starting points for the transformation.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import order_centroids
    >>> clustered_centroids = np.array(
    ...     [
    ...         [[200, 100], [100, 100], [200, 200], [100, 200]],
    ...     ]
    ... )
    >>> order_centroids(clustered_centroids)
    array([[[100, 100],
            [200, 100],
            [200, 200],
            [100, 200]]], dtype=int32)
    """
    starting_pts = []
    for centroids_in_cluster in clustered_centroids:
        cluster_centroid = np.mean(centroids_in_cluster, axis=0)

        distances = np.zeros(len(centroids_in_cluster))
        angles = np.zeros(len(centroids_in_cluster))
        for i, centroid in enumerate(centroids_in_cluster):
            distances[i] = np.linalg.norm(centroid - cluster_centroid)
            angles[i] = np.arctan2(
                (centroid[1] - cluster_centroid[1]), (centroid[0] - cluster_centroid[0])
            )

        bins = np.linspace(
            np.nextafter(np.float32(-np.pi), -np.pi - 1),
            np.nextafter(np.float32(np.pi), np.pi + 1),
            num=5,
            endpoint=True,
        )
        bin_indices = np.digitize(angles, bins)

        cluster_starting_pts = []
        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                bin_distances = distances[bin_mask]
                max_index = np.argmax(bin_distances)
                cluster_starting_pts.append(centroids_in_cluster[bin_mask][max_index])
            else:
                cluster_starting_pts = None
                break

        starting_pts.append(np.array(cluster_starting_pts))
    return as_int32_array(starting_pts)


def determine_best_transformation(
    template: Template,
    clustered_centroids: NDArrayInt,
    starting_pts: NDArrayInt,
    greedy_heuristic: float,
) -> NDArrayFloat:
    """
    Determine the best transformation to warp the clustered centroids to the template.
    This is achieved by brute forcing through possible correspondences and calculating
    the distance of the warped points to the template points. Some gains are achieved
    by employing a greedy heuristic to stop the search early if a good enough
    correspondence is found.

    Parameters
    ----------
    template
        The template to which we want to transform the clustered centroids
    clustered_centroids
        The centroids of the clusters that are to be transformed to the template.
    starting_pts
        The points of the cluster that are used to find initial correspondences
        in the template.
    greedy_heuristic
        The heuristic to stop the search early.

    Returns
    -------
    NDArrayFloat
        The transformation matrix to warp the clustered centroids to the template.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import determine_best_transformation
    >>> from colour_checker_detection.detection.templates.generate_template import (
    ...     Template,
    ... )
    >>> template = Template(None, None, None, None, None)
    >>> template.swatch_centroids = np.array(
    ...     [
    ...         [100, 100],
    ...         [200, 100],
    ...         [200, 200],
    ...         [100, 200],
    ...     ]
    ... )
    >>> template.correspondences = np.array(
    ...     [
    ...         (0, 1, 2, 3),
    ...     ]
    ... )
    >>> clustered_centroids = np.array(
    ...     [
    ...         [[200, 100], [100, 100], [200, 200], [100, 200]],
    ...     ],
    ...     dtype=np.float32,
    ... )
    >>> starting_pts = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...     ],
    ...     dtype=np.float32,
    ... )
    >>> determine_best_transformation(template, clustered_centroids, starting_pts, 10)
    array([[  1.00000000e+00,   3.39609879e-32,  -4.26325641e-14],
           [  2.13162821e-16,   1.00000000e+00,  -4.26325641e-14],
           [  2.13162821e-18,   1.93870456e-34,   1.00000000e+00]])
    """
    warping_data = [
        WarpingData(cluster_id) for cluster_id in range(len(clustered_centroids))
    ]
    for cluster_id, (cluster, cluster_pts) in enumerate(
        zip(clustered_centroids, starting_pts)
    ):
        for correspondence in template.correspondences:
            transformation = cv2.getPerspectiveTransform(
                cluster_pts.astype(np.float32),
                template.swatch_centroids[list(correspondence)].astype(np.float32),
            )
            warped_pts = cv2.perspectiveTransform(
                cluster[None, :, :].astype(np.float32), transformation
            ).reshape(-1, 2)

            cost_matrix = distance_matrix(warped_pts, template.swatch_centroids)
            row_id, col_id = linear_sum_assignment(cost_matrix)
            cost = np.sum(cost_matrix[row_id, col_id]) / len(cluster)

            if cost < warping_data[cluster_id].cost:
                warping_data[cluster_id].cost = cost
                warping_data[cluster_id].transformation = transformation
                if cost < greedy_heuristic:
                    break
    unique_warping_data = []
    for _ in range(len(clustered_centroids)):
        unique_warping_data.append(min(warping_data, key=lambda x: x.cost))

    transformation = min(unique_warping_data, key=lambda x: x.cost).transformation
    return transformation


def extract_colours(warped_image: ArrayLike, template: Template) -> NDArrayFloat:
    """
    Extract the swatch colours from the warped image utilizing the template centroids.

    Parameters
    ----------
    warped_image
        The warped image.
    template
        The template providing the centroids.

    Returns
    -------
    NDArrayFloat
        The swatch colours.

    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> from colour_checker_detection.detection.templates.generate_template import (
    ...     Template,
    ... )
    >>> from colour_checker_detection.detection import extract_colours
    >>> template = Template(None, None, None, None, None)
    >>> template.swatch_centroids = np.array(
    ...     [
    ...         [100, 100],
    ...         [200, 100],
    ...         [200, 200],
    ...         [100, 200],
    ...     ]
    ... )
    >>> warped_image = np.zeros((600, 900, 3))
    >>> warped_image[100:200, 100:200] = 0.2
    >>> extract_colours(warped_image, template)
    array([[ 0.05,  0.05,  0.05],
           [ 0.05,  0.05,  0.05],
           [ 0.05,  0.05,  0.05],
           [ 0.05,  0.05,  0.05]])
    """
    swatch_colours = []

    for swatch_center in template.swatch_centroids:
        swatch_slice = warped_image[  # pyright: ignore
            swatch_center[1] - 20 : swatch_center[1] + 20,
            swatch_center[0] - 20 : swatch_center[0] + 20,
        ]
        swatch_colours += [np.mean(swatch_slice, axis=(0, 1)).tolist()]  # pyright: ignore
    return np.array(swatch_colours)


def correct_flipped(swatch_colours: NDArrayFloat) -> NDArrayFloat:
    """
    Reorder the swatch colours if the colour checker was flipped.

    Parameters
    ----------
    swatch_colours
        The swatch colours.

    Returns
    -------
    NDArrayFloat
        The reordered swatch colours.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import correct_flipped
    >>> swatch_colours = np.array(
    ...     [
    ...         [0, 0, 0],
    ...         [1, 0.5, 0.1],
    ...     ]
    ... )
    >>> correct_flipped(swatch_colours)
    array([[ 1. ,  0.5,  0.1],
           [ 0. ,  0. ,  0. ]])
    """
    chromatic_std = np.std(swatch_colours[0])
    achromatic_std = np.std(swatch_colours[-1])
    if chromatic_std < achromatic_std:
        usage_warning("Colour checker was seemingly flipped, reversing the samples!")
        swatch_colours = swatch_colours[::-1]
    return swatch_colours


def check_residuals(swatch_colours: NDArrayFloat, template: Template) -> NDArrayFloat:
    """
    Check the residuals between the template and the swatch colours.

    Parameters
    ----------
    swatch_colours
        The swatch colours.
    template
        The template to compare to.

    Returns
    -------
    NDArrayFloat
        The swatch colours or none if the residuals are too high.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection.detection import check_residuals
    >>> from colour_checker_detection.detection.templates.generate_template import (
    ...     Template,
    ... )
    >>> template = Template(None, None, None, None, None)
    >>> template.colours = np.array(
    ...     [
    ...         [0, 0, 0],
    ...         [1, 0.5, 0.1],
    ...     ]
    ... )
    >>> swatch_colours = np.array(
    ...     [
    ...         [0, 0, 0],
    ...         [1, 0.5, 0.1],
    ...     ]
    ... )
    >>> check_residuals(swatch_colours, template)
    array([[ 0. ,  0. ,  0. ],
           [ 1. ,  0.5,  0.1]])
    """
    residual = [
        np.abs(r - m) for r, m in zip(template.colours, np.array(swatch_colours))
    ]
    if np.max(residual) > 0.5:
        usage_warning(
            "Colour seems wrong, either calibration is very bad or checker "
            "was not detected correctly."
            "Make sure the checker is not occluded and try again!"
        )
        swatch_colours = np.array([])
    return swatch_colours


def plot_contours(image: ArrayLike, contours: NDArrayInt | Tuple[NDArrayInt]):
    """
    Plot the image and marks the detected contours

    Parameters
    ----------
    image
        The image with the colour checker.
    contours
        The contours to highlight.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection import plot_contours
    >>> image = np.zeros((600, 900, 3))
    >>> contours = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[300, 100], [400, 100], [400, 200], [300, 200]],
    ...     ]
    ... )
    >>> plot_contours(image, contours)
    """
    image_contours = np.copy(image)
    cv2.drawContours(
        image_contours,
        contours,  # pyright: ignore
        -1,
        (0, 1, 0),
        5,
    )
    plot_image(
        image_contours,
        text_kwargs={"text": "Contours", "color": "Green"},
    )


def plot_swatches_and_clusters(
    image: ArrayLike, swatches: NDArrayInt, clusters: NDArrayInt
):
    """
    Plot the image and marks the swatches and clusters.

    Parameters
    ----------
    image
        The image with the colour checker.
    swatches
        The swatches to display.
    clusters
        The clusters to display.

    Examples
    --------
    >>> import numpy as np
    >>> from colour_checker_detection import plot_swatches_and_clusters
    >>> image = np.zeros((600, 900, 3))
    >>> swatches = np.array(
    ...     [
    ...         [[100, 100], [200, 100], [200, 200], [100, 200]],
    ...         [[300, 100], [400, 100], [400, 200], [300, 200]],
    ...     ]
    ... )
    >>> clusters = np.array(
    ...     [
    ...         [[50, 50], [500, 50], [500, 500], [50, 500]],
    ...     ]
    ... )
    >>> plot_swatches_and_clusters(image, swatches, clusters)
    """
    image_swatches = np.copy(image)
    cv2.drawContours(
        image_swatches,
        swatches,  # pyright: ignore
        -1,
        (1, 0, 1),
        5,
    )
    cv2.drawContours(
        image_swatches,
        clusters,  # pyright: ignore
        -1,
        (0, 1, 1),
        5,
    )
    plot_image(
        CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(image_swatches),
        text_kwargs={"text": "Swatches & Clusters", "color": "Red"},
    )


def plot_colours(
    colour_checkers_data: list[DataDetectionColourChecker],
    swatches_vertical: int,
    swatches_horizontal: int,
):
    """
    Plot the warped image with the swatch colours annotated.

    Parameters
    ----------
    colour_checkers_data
        The colour checkers data.
    swatches_vertical
        The number of vertical swatches.
    swatches_horizontal
        The number of horizontal swatches.

    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> from colour_checker_detection.detection.common import DataDetectionColourChecker
    >>> from colour_checker_detection import plot_colours
    >>> colour_checkers_data = DataDetectionColourChecker(None, None, None, None)
    >>> colour_checkers_data.colour_checker = np.zeros((600, 900, 3))
    >>> colour_checkers_data.colour_checker[100:200, 100:200] = 0.2
    >>> colour_checkers_data.swatch_masks = [
    ...     [100, 200, 100, 200],
    ...     [300, 400, 100, 200],
    ... ]
    >>> colour_checkers_data.swatch_colours = np.array(
    ...     [
    ...         [0, 0, 0],
    ...         [1, 0.5, 0.1],
    ...     ]
    ... )
    >>> plot_colours([colour_checkers_data], 2, 1)
    """
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
                [swatches_vertical, swatches_horizontal, 3],
            )
        ),
    )


def plot_colours_warped(
    warped_image: ArrayLike, template: Template, swatch_colours: NDArrayFloat
):
    """
    Plot the warped image with the swatch colours annotated.

    Parameters
    ----------
    warped_image
        The warped image.
    template
        The template corresponding to the colour checker.
    swatch_colours
        The swatch colours.

    Examples
    --------
    >>> import os
    >>> import json
    >>> import colour_checker_detection.detection.templates.template_colour
    >>> import numpy as np
    >>> from colour_checker_detection import (
    ...     ROOT_DETECTION_TEMPLATES,
    ...     Template,
    ...     plot_colours_warped,
    ... )
    >>> template = Template(
    ...     **json.load(
    ...         open(os.path.join(ROOT_DETECTION_TEMPLATES, "template_colour.pkl"), "r")
    ...     )
    ... )
    >>> warped_image = np.zeros((600, 900, 3))
    >>> swatch_colours = np.array([np.random.rand(3) for _ in range(24)])
    >>> plot_colours_warped(warped_image, template, swatch_colours)
    """
    annotated_image = np.copy(warped_image)

    for i, swatch_center in enumerate(template.swatch_centroids):
        top_left = (int(swatch_center[0] - 20), int(swatch_center[1] - 20))
        bottom_right = (int(swatch_center[0] + 20), int(swatch_center[1] + 20))

        cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)

        swatch_colour = swatch_colours[i]

        if swatch_colour.dtype in (np.float32, np.float64):
            swatch_colour = (swatch_colour * 255).astype(np.uint8)

        cv2.putText(
            annotated_image,
            str(swatch_colour),
            top_left,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    plot_image(
        CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(annotated_image),
        text_kwargs={"text": "Warped Image", "color": "red"},
    )


def segmenter_default(
    image: ArrayLike,
    cctf_encoding: Callable = eotf_inverse_sRGB,
    apply_cctf_encoding: bool = True,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> DataSegmentationColourCheckers | NDArrayInt:
    """
    Detect the colour checker rectangles in given image :math:`image` using
    segmentation.

    The process is as follows:

    1.  Input image :math:`image` is converted to a grayscale image
        :math:`image_g` and normalised to range [0, 1].
    2.  Image :math:`image_g` is denoised using multiple bilateral filtering
        passes into image :math:`image_d.`
    3.  Image :math:`image_d` is thresholded into image :math:`image_t`.
    4.  Image :math:`image_t` is eroded and dilated to cleanup remaining noise
        into image :math:`image_k`.
    5.  Contours are detected on image :math:`image_k`
    6.  Contours are filtered to only keep squares/swatches above and below
        defined surface area.
    7.  Squares/swatches are clustered to isolate region-of-interest that are
        potentially colour checkers: Contours are scaled by a third so that
        colour checkers swatches are joined, creating a large rectangular
        cluster. Rectangles are fitted to the clusters.
    8.  Clusters with an aspect ratio different to the expected one are
        rejected, a side-effect is that the complementary pane of the
        *X-Rite* *ColorChecker Passport* is omitted.
    9.  Clusters with a number of swatches close to the expected one are
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
    show
        Whether to show various debug images.
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
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    greedy_heuristic
        The heuristic to stop the search for transformations early,
        if warped extractor is used.
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

    image = cast(NDArrayFloat, image)

    contours, image_k = detect_contours(image, True, **settings)  # pyright: ignore

    if show:
        plot_image(image_k, text_kwargs={"text": "Segmented Image", "color": "black"})
        plot_contours(image, contours)

    squares = filter_contours(
        image, contours, settings.swatches, settings.swatch_minimum_area_factor
    )

    swatches = remove_stacked_contours(squares)

    clusters = cluster_swatches(image, swatches, settings.swatch_contour_scale)

    clusters = filter_clusters(
        clusters, settings.aspect_ratio_minimum, settings.aspect_ratio_maximum
    )

    if show:
        plot_swatches_and_clusters(image, swatches, clusters)

    rectangles = filter_clusters_by_swatches(
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
    else:
        return rectangles


def segmenter_warped(
    image: ArrayLike,
    cctf_encoding: Callable = eotf_inverse_sRGB,
    apply_cctf_encoding: bool = True,
    show: bool = False,
    additional_data: bool = True,
    **kwargs: Any,
) -> DataSegmentationColourCheckers | NDArrayInt:
    """
    Detect the colour checker rectangles, clusters and swatches in given image
    :math:`image` using segmentation.

    The process is as follows:
        1. Input image :math:`image` is converted to a grayscale image :math:`image_g`
        and normalised to range [0, 1].
        2. Image :math:`image_g` is denoised using multiple bilateral filtering passes
        into image :math:`image_d.`
        3. Image :math:`image_d` is thresholded into image :math:`image_t`.
        4. Image :math:`image_t` is eroded and dilated to cleanup remaining noise into
        image :math:`image_k`.
        5. Contours are detected on image :math:`image_k`
        6. Contours are filtered to only keep squares/swatches above and below defined
        surface area, moreover they have
           to resemble a convex quadrilateral. Additionally, squareness, area, aspect
           ratio and orientation are used as
           features to remove any remaining outlier contours.
        7. Stacked contours are removed.
        8. Swatches are clustered to isolate region-of-interest that are potentially
        colour checkers: Contours are
           scaled by a third so that colour checkers swatches are joined, creating a
           large rectangular cluster. Rectangles
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
    show
        Whether to show various debug images.
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
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    greedy_heuristic
        The heuristic to stop the search for transformations early, if warped extractor
        is used.
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

    Notes
    -----
    -   Since the warped_extractor does not work of the rectangles, additionaldata is
        true by default.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS, segmenter_warped
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> segmenter_warped(image)  # doctest: +ELLIPSIS
    DataSegmentationColourCheckers(rectangles=array([[[ 694, 1364],...)
    """
    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    if apply_cctf_encoding:
        image = cctf_encoding(image)

    image = cast(NDArrayFloat, image)

    contours, image_k = detect_contours(image, True, **settings)  # pyright: ignore

    if show:
        plot_image(image_k, text_kwargs={"text": "Segmented Image", "color": "black"})
        plot_contours(image, contours)

    squares = filter_contours_multifeature(
        image, contours, settings.swatches, settings.swatch_minimum_area_factor
    )

    swatches = remove_stacked_contours(squares, keep_smallest=False)

    clusters = cluster_swatches(image, swatches, settings.swatch_contour_scale)

    if show:
        plot_swatches_and_clusters(image, swatches, clusters)

    rectangles = filter_clusters_by_swatches(
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
    else:
        return rectangles


def extractor_default(
    image: ArrayLike,
    segmentation_colour_checkers_data: DataSegmentationColourCheckers,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker | NDArrayFloat, ...]:
    """
    Extract the colour checker swatches and colours from given image using the previous
    segmentation.
    Default extractor expects the colour checker to be facing the camera straight.

    Parameters
    ----------
    image
        Image to extract the colour checker swatches and colours from.
    segmentation_colour_checkers_data
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
    show
        Whether to show various debug images.
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
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    greedy_heuristic
        The heuristic to stop the search for transformations early, if warped extractor
        is used.
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
    ...     segmenter_default,
    ...     extractor_default,
    ... )
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> segmentation_colour_checkers_data = segmenter_default(
    ...     image, additional_data=True
    ... )
    >>> extractor_default(
    ...     image, segmentation_colour_checkers_data
    ... )  # doctest: +ELLIPSIS
    (array([[ 0.36000502,  0.22310828,  0.11760838],
           [ 0.62583095,  0.39448658,  0.24166538],
           [ 0.33197987,  0.31600383,  0.28866863],
           [ 0.30460072,  0.27332103,  0.10486546],
           [ 0.4175137 ,  0.3191403 ,  0.30789143],
           [ 0.34866208,  0.43934605,  0.29126382],
           [ 0.6798398 ,  0.35236537,  0.06997224],
           [ 0.27118534,  0.25352538,  0.3307873 ],
           [ 0.6209186 ,  0.27034152,  0.18652563],
           [ 0.30716118,  0.1797888 ,  0.19181633],
           [ 0.48547122,  0.45855856,  0.03294946],
           [ 0.6507675 ,  0.40023163,  0.01607687],
           [ 0.19286261,  0.18585184,  0.27459192],
           [ 0.28054565,  0.3851303 ,  0.12244403],
           [ 0.554543  ,  0.21436104,  0.1254918 ],
           [ 0.7206889 ,  0.51493937,  0.00548728],
           [ 0.5772922 ,  0.25771797,  0.2685552 ],
           [ 0.1728921 ,  0.3163792 ,  0.2950853 ],
           [ 0.7394083 ,  0.60953134,  0.43830705],
           [ 0.6281669 ,  0.5175997 ,  0.37215674],
           [ 0.51360977,  0.42048815,  0.298571  ],
           [ 0.36953208,  0.30218396,  0.20827033],
           [ 0.26286718,  0.21493256,  0.14277342],
           [ 0.16102536,  0.13381618,  0.08047408]], dtype=float32),)
    """
    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    image = cast(Union[NDArrayInt, NDArrayFloat], image)
    image = reformat_image(image, settings.working_width, settings.interpolation_method)

    working_width = settings.working_width
    working_height = int(working_width / settings.aspect_ratio)

    rectangle = as_int32_array(
        [
            [working_width, 0],
            [working_width, working_height],
            [0, working_height],
            [0, 0],
        ]
    )

    colour_checkers_data = []
    for quadrilateral in segmentation_colour_checkers_data.rectangles:
        colour_checkers_data.append(
            sample_colour_checker(image, quadrilateral, rectangle, samples, **settings)
        )

        if show:
            plot_colours(
                colour_checkers_data,
                settings.swatches_vertical,
                settings.swatches_horizontal,
            )
    if additional_data:
        return tuple(colour_checkers_data)
    else:
        return tuple(
            colour_checker_data.swatch_colours
            for colour_checker_data in colour_checkers_data
        )


def extractor_warped(
    image: ArrayLike,
    segmentation_colour_checkers_data: DataSegmentationColourCheckers,
    template: Template,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker | NDArrayFloat, ...]:
    """
    Extract the colour checker swatches and colours from given image using the previous
    segmentation.
    This extractor should be used when the colour checker is not facing the camera
    straight.

    The process is as follows:
        1. The swatches are converted to centroids and used to filter clusters to only
        keep the ones that contain the
           expected number of swatches. Moreover, the centroids are grouped by the
           clusters.
        2. The centroids are ordered within their group to enforce the same ordering as
           the template, which is
           important to extract the transformation, since openCV's perspective transform
           is not invariant to the
           ordering of the points.
        3. The best transformation is determined by finding the transformation that
           minimizes the average distance of
           the warped points from the reference template points.
        4. The image is warped using the determined transformation.
        5. The colours are extracted from the warped image using a 20x20 pixel window
           around the centroids.
        6. The colours are corrected if the chromatic swatches have a lower standard
           deviation than the achromatic
           swatches.

    Parameters
    ----------
    image
        Image to extract the colour checker swatches and colours from.
    segmentation_colour_checkers_data
        Segmentation colour checkers data from the segmenter.
    template
        Template defining the swatches structure, which is exploited to find the best
        correspondences between template
        and detected swatches, which yield the optimal transformation.
    cctf_decoding
        Decoding colour component transfer function / opto-electronic
        transfer function used when converting the image from 8-bit to float.
    apply_cctf_decoding
        Apply the decoding colour component transfer function / opto-electronic
        transfer function.
    show
        Whether to show various debug images.
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
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
    greedy_heuristic
        The heuristic to stop the search for transformations early, if warped extractor
        is used.
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
    >>> import json
    >>> from colour import read_image
    >>> import colour_checker_detection.detection.templates.template_colour
    >>> from colour_checker_detection import (
    ...     ROOT_RESOURCES_TESTS,
    ...     ROOT_DETECTION_TEMPLATES,
    ...     Template,
    ...     segmenter_warped,
    ...     extractor_warped,
    ... )
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> template = Template(
    ...     **json.load(
    ...         open(
    ...             os.path.join(ROOT_DETECTION_TEMPLATES, "template_colour.json"), "r"
    ...         )
    ...     )
    ... )
    >>> segmentation_colour_checkers_data = segmenter_warped(image)
    >>> extractor_warped(
    ...     image, segmentation_colour_checkers_data, template
    ... )  # doctest: +SKIP
    (array([    0.36087,     0.22405,     0.11797]), ...
    """
    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    image = cast(NDArrayFloat, image)

    clustered_centroids = group_swatches(
        segmentation_colour_checkers_data.clusters,
        segmentation_colour_checkers_data.swatches,
        template,
    )

    starting_pts = order_centroids(clustered_centroids)

    transformation = determine_best_transformation(
        template, clustered_centroids, starting_pts, settings.greedy_heuristic
    )

    warped_image = cv2.warpPerspective(
        image, transformation, (template.width, template.height)
    )  # pyright: ignore

    swatch_colours = extract_colours(warped_image, template)

    swatch_colours = correct_flipped(swatch_colours)

    swatch_colours = check_residuals(swatch_colours, template)

    colour_checkers_data = DataDetectionColourChecker(
        swatch_colours,
        np.array([]),
        warped_image,
        segmentation_colour_checkers_data.clusters,
    )

    if show and swatch_colours is not None:
        plot_colours_warped(warped_image, template, swatch_colours)

    if additional_data:
        return tuple(colour_checkers_data)
    else:
        return tuple(swatch_colours)


def detect_colour_checkers_segmentation(
    image: str | ArrayLike,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    segmenter: Callable = segmenter_default,
    segmenter_kwargs: dict | None = None,
    extractor: Callable = extractor_default,
    extractor_kwargs: dict | None = None,
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
        Callable responsible to extract the colour checker swatches and colours from the
        image.
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
    convolution_iterations
        Number of iterations to use for the erosion / dilation process.
    convolution_kernel
        Convolution kernel to use for the erosion / dilation process.
        greedy_heuristic
            The heuristic to stop the search for transformations early, if warped
            extractor is used.
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
    ...     detect_colour_checkers_segmentation,
    ... )
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
    if extractor_kwargs is None:
        extractor_kwargs = {}

    settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    if is_string(image):
        image = read_image(cast(str, image))
    else:
        image = convert_bit_depth(
            image,
            DTYPE_FLOAT_DEFAULT.__name__,  # pyright: ignore
        )

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    image = cast(Union[NDArrayInt, NDArrayFloat], image)

    image = reformat_image(image, settings.working_width, settings.interpolation_method)

    segmentation_colour_checkers_data = segmenter(
        image, additional_data=True, show=show, **{**segmenter_kwargs, **settings}
    )

    colour_checkers_data = extractor(
        image,
        segmentation_colour_checkers_data,
        show=show,
        additional_data=additional_data,
        **{**extractor_kwargs, **settings},
    )

    return colour_checkers_data
