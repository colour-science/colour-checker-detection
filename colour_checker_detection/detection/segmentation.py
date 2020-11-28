# -*- coding: utf-8 -*-
"""
Colour Checker Detection - Segmentation
=======================================

Defines objects for colour checker detection using segmentation:

-   :func:`colour_checkers_coordinates_segmentation`
-   :func:`extract_colour_checkers_segmentation`
-   :func:`detect_colour_checkers_segmentation`

References
----------
-   :cite:`Abecassis2011` : Abecassis, F. (2011). OpenCV - Rotation
    (Deskewing). Retrieved October 27, 2018, from http://felix.abecassis.me/\
2011/10/opencv-rotation-deskewing/
"""

from __future__ import division, unicode_literals

import cv2
import numpy as np
from collections import namedtuple

from colour.models import cctf_decoding, cctf_encoding
from colour.utilities import as_float_array, as_int_array, as_int

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2018-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ASPECT_RATIO', 'SWATCHES_HORIZONTAL', 'SWATCHES_VERTICAL', 'SWATCHES',
    'SWATCH_MINIMUM_AREA_FACTOR', 'WORKING_WIDTH',
    'ColourCheckersDetectionData', 'ColourCheckerSwatchesData', 'swatch_masks',
    'as_8_bit_BGR_image', 'adjust_image', 'is_square', 'contour_centroid',
    'scale_contour', 'crop_and_level_image_with_rectangle',
    'colour_checkers_coordinates_segmentation',
    'extract_colour_checkers_segmentation',
    'detect_colour_checkers_segmentation'
]

ASPECT_RATIO = 1.5
"""
Colour checker aspect ratio.

ASPECT_RATIO : numeric
"""

SWATCHES_HORIZONTAL = 6
"""
Colour checker horizontal swatches count.

SWATCHES_HORIZONTAL : int
"""

SWATCHES_VERTICAL = 4
"""
Colour checker vertical swatches count.

SWATCHES_VERTICAL : int
"""

SWATCHES = SWATCHES_HORIZONTAL * SWATCHES_VERTICAL
"""
Colour checker total swatches count.

SWATCHES : int
"""

SWATCH_MINIMUM_AREA_FACTOR = 200
"""
Swatch minimum area factor :math:`f` with the minimum area :math:`m_a`
expressed as follows: :math:`m_a = image_w * image_h / s_c / f` where
:math:`image_w`, :math:`image_h` and :math:`s_c` are respectively the image
width, height and the swatches count.

SWATCH_MINIMUM_AREA_FACTOR : numeric
"""

WORKING_WIDTH = 1440
"""
Width processed images are resized to.

WORKING_WIDTH : int
"""


class ColourCheckersDetectionData(
        namedtuple(
            'ColourCheckersDetectionData',
            ('colour_checkers', 'clusters', 'swatches', 'segmented_image'))):
    """
    Colour checkers detection data used for plotting, debugging and further
    analysis.

    Parameters
    ----------
    colour_checkers : array_like
        Colour checker bounding boxes, i.e., the. clusters that have the
        relevant count of swatches.
    clusters : array_like
        Detected swatches clusters.
    swatches : array_like
        Detected swatches.
    segmented_image : numeric or array_like
        Thresholded/Segmented image.
    """


class ColourCheckerSwatchesData(
        namedtuple(
            'ColourCheckerSwatchesData',
            ('swatch_colours', 'colour_checker_image', 'swatch_masks'))):
    """
    Colour checker swatches data used for plotting, debugging and further
    analysis.

    Parameters
    ----------
    swatch_colours : array_like
        Colour checker swatches colours.
    colour_checker_image : array_like
        Cropped and levelled Colour checker image.
    swatch_masks : array_like
        Colour checker swatches masks.
    """


def swatch_masks(width, height, swatches_h, swatches_v, samples):
    """
    Returns swatch masks for given image width and height and swatches count.

    Parameters
    ----------
    width : int
        Image width.
    height : height
        Image height.
    swatches_h : int
        Horizontal swatches count.
    swatches_v : int
        Vertical swatches count.
    samples : int
        Samples count.

    Returns
    -------
    list
        List of swatch masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> pprint(swatch_masks(16, 8, 4, 2, 1))  # doctest: +ELLIPSIS
    [array([2, 2, 2, 2]...),
     array([2, 2, 6, 6]...),
     array([ 2,  2, 10, 10]...),
     array([ 2,  2, 14, 14]...),
     array([6, 6, 2, 2]...),
     array([6, 6, 6, 6]...),
     array([ 6,  6, 10, 10]...),
     array([ 6,  6, 14, 14]...)]
    """

    samples = as_int(samples / 2)

    masks = []
    offset_h = width / swatches_h / 2
    offset_v = height / swatches_v / 2
    for j in np.linspace(offset_v, height - offset_v, swatches_v):
        for i in np.linspace(offset_h, width - offset_h, swatches_h):
            masks.append(
                as_int_array(
                    [j - samples, j + samples, i - samples, i + samples]))

    return masks


def as_8_bit_BGR_image(image):
    """
    Converts and encodes given linear float *RGB* image to 8-bit *BGR* with
    *sRGB* reverse OETF.

    Parameters
    ----------
    image : array_like
        Image to convert.

    Returns
    -------
    ndarray
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

    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    return cv2.cvtColor((cctf_encoding(image) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def adjust_image(image, target_width=WORKING_WIDTH):
    """
    Adjusts given image so that it is horizontal and resizes it to given target
    width.

    Parameters
    ----------
    image : array_like
        Image to adjust.
    target_width : int, optional
        Width the image is resized to.

    Returns
    -------
    ndarray
        Resized image.

    Examples
    --------
    >>> from colour.algebra import random_triplet_generator
    >>> prng = np.random.RandomState(4)
    >>> image = list(random_triplet_generator(8, random_state=prng))
    >>> image = np.reshape(image, [2, 4, 3])
    >>> adjust_image(image, 5)  # doctest: +ELLIPSIS
    array([[[ 0.9925326...,  0.2419374..., -0.0139522...],
            [ 0.6174496...,  0.3460755...,  0.3189758...],
            [ 0.7447774...,  0.6786660...,  0.1652180...],
            [ 0.9476451...,  0.6550805...,  0.2609945...],
            [ 0.6991505...,  0.1623470...,  1.0120867...]],
    <BLANKLINE>
           [[ 0.7269885...,  0.8556784...,  0.4049920...],
            [ 0.2666564...,  1.0401633...,  0.8238320...],
            [ 0.6419699...,  0.5442698...,  0.9082210...],
            [ 0.7894426...,  0.1944301...,  0.7906868...],
            [-0.0526997...,  0.6236684...,  0.8711482...]]])
    """

    width, height = image.shape[1], image.shape[0]
    if width < height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height

    ratio = width / target_width

    if np.allclose(ratio, 1):
        return image
    else:
        return cv2.resize(
            image, (as_int(target_width), as_int(height / ratio)),
            interpolation=cv2.INTER_CUBIC)


def is_square(contour, tolerance=0.015):
    """
    Returns if given contour is a square.

    Parameters
    ----------
    contour : array_like
        Shape to test whether it is a square.
    tolerance : numeric, optional
        Tolerance under which the contour is considered to be a square.

    Returns
    -------
    bool
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

    return cv2.matchShapes(contour, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
                           cv2.CONTOURS_MATCH_I2, 0.0) < tolerance


def contour_centroid(contour):
    """
    Returns the centroid of given contour.

    Parameters
    ----------
    contour : array_like
        Contour to return the centroid of.

    Returns
    -------
    tuple
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
    centroid = np.array(
        [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])

    return centroid[0], centroid[1]


def scale_contour(contour, factor):
    """
    Scales given contour by given scale factor.

    Parameters
    ----------
    contour : array_like
        Contour to scale.
    factor : numeric
        Scale factor.

    Returns
    -------
    ndarray
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


def crop_and_level_image_with_rectangle(image, rectangle):
    """
    Crops and rotates/levels given image using given rectangle.

    Parameters
    ----------
    image : array_like
        Image to crop and rotate/level.
    rectangle : tuple
        Rectangle used to crop and rotate/level the image.

    Returns
    -------
    ndarray
        Cropped and rotated/levelled image.

    References
    ----------
    :cite:`Abecassis2011`

    Notes
    -----
    -   ``image`` is expected to be an unsigned 8-bit sRGB encoded image.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
    >>> path = os.path.join(TESTS_RESOURCES_DIRECTORY,
    ...                     'colour_checker_detection', 'detection',
    ...                     'IMG_1967.png')
    >>> image = as_8_bit_BGR_image(adjust_image(read_image(path)))
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

    width, height = image.shape[1], image.shape[0]
    width_r, height_r = rectangle[1]
    centroid = as_int_array(contour_centroid(cv2.boxPoints(rectangle)))
    centroid = centroid[0], centroid[1]
    angle = rectangle[-1]

    if angle < -45:
        angle += 90
        width_r, height_r = height_r, width_r

    width_r, height_r = as_int_array([width_r, height_r])

    M_r = cv2.getRotationMatrix2D(centroid, angle, 1)

    image_r = cv2.warpAffine(image, M_r, (width, height), cv2.INTER_CUBIC)
    image_c = cv2.getRectSubPix(image_r, (width_r, height_r),
                                (centroid[0], centroid[1]))

    return image_c


def colour_checkers_coordinates_segmentation(image, additional_data=False):
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
    -   Clusters with a number of swatches close to :attr:`SWATCHES` are
        kept.

    Parameters
    ----------
    image : array_like
        Image to detect the colour checkers in.
    additional_data : bool, optional
        Whether to output additional data.

    Returns
    -------
    list or ColourCheckersDetectionData
        List of colour checkers coordinates or
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
    >>> colour_checkers_coordinates_segmentation(image)  # doctest: +ELLIPSIS
    [array([[1065,  707],
           [ 369,  688],
           [ 382,  226],
           [1078,  246]]...)]
    """

    image = as_8_bit_BGR_image(adjust_image(image, WORKING_WIDTH))

    width, height = image.shape[1], image.shape[0]
    maximum_area = width * height / SWATCHES
    minimum_area = width * height / SWATCHES / SWATCH_MINIMUM_AREA_FACTOR

    block_size = as_int(WORKING_WIDTH * 0.015)
    block_size = block_size - block_size % 2 + 1

    # Thresholding/Segmentation.
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_g = cv2.fastNlMeansDenoising(image_g, None, 10, 7, 21)
    image_s = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, block_size, 3)
    # Cleanup.
    kernel = np.ones((3, 3), np.uint8)
    image_c = cv2.erode(image_s, kernel, iterations=1)
    image_c = cv2.dilate(image_c, kernel, iterations=1)

    # Detecting contours.
    contours, _hierarchy = cv2.findContours(image_c, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)

    # Filtering squares/swatches contours.
    swatches = []
    for contour in contours:
        curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),
                                 True)
        if minimum_area < cv2.contourArea(curve) < maximum_area and is_square(
                curve):
            swatches.append(
                as_int_array(cv2.boxPoints(cv2.minAreaRect(curve))))

    # Clustering squares/swatches.
    clusters = np.zeros(image.shape, dtype=np.uint8)
    for swatch in [
            as_int_array(scale_contour(swatch, 1 + 1 / 3))
            for swatch in swatches
    ]:
        cv2.drawContours(clusters, [swatch], -1, [255] * 3, -1)
    clusters = cv2.cvtColor(clusters, cv2.COLOR_RGB2GRAY)
    clusters, _hierarchy = cv2.findContours(clusters, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    clusters = [
        as_int_array(
            scale_contour(cv2.boxPoints(cv2.minAreaRect(cluster)), 0.975))
        for cluster in clusters
    ]

    # Filtering clusters using their aspect ratio.
    filtered_clusters = []
    for cluster in clusters[:]:
        rectangle = cv2.minAreaRect(cluster)
        width = max(rectangle[1][0], rectangle[1][1])
        height = min(rectangle[1][0], rectangle[1][1])
        ratio = width / height
        if ASPECT_RATIO * 0.9 < ratio < ASPECT_RATIO * 1.1:
            filtered_clusters.append(cluster)
    clusters = filtered_clusters

    # Filtering swatches within cluster.
    counts = []
    for cluster in clusters:
        count = 0
        for swatch in swatches:
            if cv2.pointPolygonTest(cluster, contour_centroid(swatch),
                                    False) == 1:
                count += 1
        counts.append(count)
    counts = np.array(counts)
    indexes = np.where(
        np.logical_and(counts >= SWATCHES * 0.75,
                       counts <= SWATCHES * 1.25))[0].tolist()

    colour_checkers = [clusters[i] for i in indexes]

    if additional_data:
        return ColourCheckersDetectionData(colour_checkers, clusters, swatches,
                                           image_c)
    else:
        return colour_checkers


def extract_colour_checkers_segmentation(image):
    """
    Extracts the colour checkers sub-images in given image using segmentation.

    Parameters
    ----------
    image : array_like
        Image to extract the colours checkers sub-images from.

    Returns
    -------
    list
        List of colour checkers sub-images.

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
    [array([[[ 86, 104, 113],
            [ 89, 102, 118],
            [ 88, 101, 117],
            ...,
            [ 79, 101, 114],
            [ 76, 101, 114],
            [ 79,  98, 110]],
    <BLANKLINE>
           [[ 84, 104, 112],
            [ 85, 102, 115],
            [ 84, 101, 115],
            ...,
            [ 80, 101, 110],
            [ 79, 101, 112],
            [ 78,  98, 112]],
    <BLANKLINE>
           [[ 84, 102, 112],
            [ 82, 102, 112],
            [ 82, 101, 113],
            ...,
            [ 81, 100, 109],
            [ 80, 100, 110],
            [ 79, 100, 113]],
    <BLANKLINE>
           ...,
           [[ 89, 105, 117],
            [ 90, 106, 120],
            [ 86, 106, 117],
            ...,
            [ 84, 100, 109],
            [ 83, 100, 111],
            [ 80, 100, 114]],
    <BLANKLINE>
           [[ 89, 106, 116],
            [ 91, 107, 121],
            [ 89, 106, 119],
            ...,
            [ 81,  99, 113],
            [ 79, 100, 115],
            [ 75, 100, 114]],
    <BLANKLINE>
           [[ 84, 108, 117],
            [ 89, 108, 117],
            [ 91, 107, 117],
            ...,
            [ 79,  98, 117],
            [ 77, 100, 117],
            [ 73, 101, 116]]], dtype=uint8)]
    """

    image = as_8_bit_BGR_image(adjust_image(image, WORKING_WIDTH))

    colour_checkers = []
    for colour_checker in colour_checkers_coordinates_segmentation(image):
        colour_checker = crop_and_level_image_with_rectangle(
            image, cv2.minAreaRect(colour_checker))
        width, height = (colour_checker.shape[1], colour_checker.shape[0])

        if width < height:
            colour_checker = cv2.rotate(colour_checker,
                                        cv2.ROTATE_90_CLOCKWISE)

        colour_checkers.append(colour_checker)

    return colour_checkers


def detect_colour_checkers_segmentation(image,
                                        samples=16,
                                        additional_data=False):
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

    Returns
    -------
    list
        List of colour checkers swatches or :class:`ColourCheckerSwatchesData`
        class instances.

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
    [array([[ 0.3594894...,  0.2225419...,  0.1176996...],
           [ 0.6250058...,  0.3931947...,  0.2417636...],
           [ 0.3304194...,  0.3142103...,  0.2874383...],
           [ 0.3034269...,  0.2721812...,  0.1053537...],
           [ 0.4153488...,  0.3183605...,  0.3067842...],
           [ 0.3458465...,  0.4393400...,  0.2912665...],
           [ 0.6782215...,  0.3519573...,  0.0752686...],
           [ 0.2715231...,  0.2515535...,  0.3295411...],
           [ 0.6171124...,  0.2687208...,  0.1852935...],
           [ 0.3049796...,  0.1792275...,  0.1908085...],
           [ 0.4844366...,  0.4576518...,  0.0392559...],
           [ 0.6494152...,  0.3991223...,  0.0329260...],
           [ 0.1922949...,  0.1842026...,  0.2731065...],
           [ 0.2780555...,  0.3836590...,  0.1233134...],
           [ 0.5515815...,  0.2126631...,  0.1250530...],
           [ 0.7178619...,  0.5132913...,  0.0804213...],
           [ 0.5753956...,  0.2563947...,  0.2672106...],
           [ 0.1799058...,  0.3160584...,  0.2945296...],
           [ 0.7402078...,  0.6088296...,  0.4374975...],
           [ 0.6272391...,  0.5156084...,  0.3713541...],
           [ 0.5120363...,  0.4196305...,  0.2976295...],
           [ 0.3690167...,  0.3019190...,  0.2083050...],
           [ 0.2624792...,  0.2143349...,  0.1428991...],
           [ 0.1625438...,  0.1333312...,  0.0807412...]])]
    """

    image = adjust_image(image, WORKING_WIDTH)

    swatches_h, swatches_v = SWATCHES_HORIZONTAL, SWATCHES_VERTICAL

    colour_checkers_colours = []
    colour_checkers_data = []
    for colour_checker in extract_colour_checkers_segmentation(image):
        colour_checker = cctf_decoding(
            as_float_array(colour_checker[..., ::-1]) / 255)
        width, height = (colour_checker.shape[1], colour_checker.shape[0])
        masks = swatch_masks(width, height, swatches_h, swatches_v, samples)

        swatch_colours = []
        for i, mask in enumerate(masks):
            swatch_colours.append(
                np.mean(
                    colour_checker[mask[0]:mask[1], mask[2]:mask[3], ...],
                    axis=(0, 1)))

        # Colour checker could be in reverse order.
        swatch_neutral_colours = swatch_colours[18:23]
        is_reversed = False
        for i, swatch, in enumerate(swatch_neutral_colours[:-1]):
            if np.mean(swatch) < np.mean(swatch_neutral_colours[i + 1]):
                is_reversed = True
                break

        if is_reversed:
            swatch_colours = swatch_colours[::-1]

        swatch_colours = np.asarray(swatch_colours)

        colour_checkers_colours.append(swatch_colours)
        colour_checkers_data.append((colour_checker, masks))

    if additional_data:
        return [
            ColourCheckerSwatchesData(colour_checkers_colours[i],
                                      *colour_checkers_data[i])
            for i, colour_checker_colours in enumerate(colour_checkers_colours)
        ]
    else:
        return colour_checkers_colours
