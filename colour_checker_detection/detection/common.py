"""
Common Utilities
================

Defines the common utilities objects that don't fall in any specific category.
"""

from __future__ import annotations

import cv2
import numpy as np
from colour.hints import (
    ArrayLike,
    DTypeFloat,
    Literal,
    NDArrayFloat,
    NDArrayInt,
    Tuple,
    Type,
    cast,
)
from colour.utilities import (
    as_float_array,
    as_int,
    as_int_array,
    as_int_scalar,
    orient,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "FLOAT_DTYPE_DEFAULT",
    "swatch_masks",
    "adjust_image",
    "crop_with_rectangle",
    "is_square",
    "contour_centroid",
    "scale_contour",
    "approximate_contour",
]


FLOAT_DTYPE_DEFAULT: Type[DTypeFloat] = np.float32
"""Dtype used for the computations."""


def swatch_masks(
    width: int,
    height: int,
    swatches_h: int,
    swatches_v: int,
    samples: int,
) -> Tuple[NDArrayInt, ...]:
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


def adjust_image(
    image: ArrayLike,
    target_width: int,
    interpolation_method: Literal[
        cv2.INTER_AREA,  # pyright: ignore
        cv2.INTER_BITS,  # pyright: ignore
        cv2.INTER_BITS2,  # pyright: ignore
        cv2.INTER_CUBIC,  # pyright: ignore
        cv2.INTER_LANCZOS4,  # pyright: ignore
        cv2.INTER_LINEAR,  # pyright: ignore
    ] = cv2.INTER_CUBIC,
) -> NDArrayInt | NDArrayFloat:
    """
    Adjust given image so that it is horizontal and resizes it to given target
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
    >>> image = np.arange(24).reshape([2, 4, 3])
    >>> adjust_image(image, 5)  # doctest: +SKIP
    array([[[ -0.18225056,   0.8177495 ,   1.8177495 ],
            [  1.8322501 ,   2.83225   ,   3.83225   ],
            [  4.5       ,   5.5       ,   6.5       ],
            [  7.1677475 ,   8.167748  ,   9.167748  ],
            [  9.182249  ,  10.182249  ,  11.182249  ]],
    <BLANKLINE>
           [[ 11.817749  ,  12.81775   ,  13.817749  ],
            [ 13.83225   ,  14.832251  ,  15.832251  ],
            [ 16.5       ,  17.5       ,  18.5       ],
            [ 19.16775   ,  20.167747  ,  21.167747  ],
            [ 21.182247  ,  22.18225   ,  23.182251  ]]], dtype=float32)
    """

    image = np.asarray(image)

    if image.ndim == 3:
        image = image[..., :3]

    width, height = image.shape[1], image.shape[0]
    if width < height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height

    ratio = width / target_width

    if np.allclose(ratio, 1):
        return cast(NDArrayInt | NDArrayFloat, image)
    else:
        return cv2.resize(  # pyright: ignore
            image,
            (as_int_scalar(target_width), as_int_scalar(height / ratio)),
            interpolation=interpolation_method,
        )


def crop_with_rectangle(
    image: ArrayLike,
    rectangle: Tuple[Tuple, Tuple, float],
    interpolation_method: Literal[
        cv2.INTER_AREA,  # pyright: ignore
        cv2.INTER_BITS,  # pyright: ignore
        cv2.INTER_BITS2,  # pyright: ignore
        cv2.INTER_CUBIC,  # pyright: ignore
        cv2.INTER_LANCZOS4,  # pyright: ignore
        cv2.INTER_LINEAR,  # pyright: ignore
    ] = cv2.INTER_CUBIC,
) -> NDArrayFloat:
    """
    Crop and rotate/level given image using given rectangle.

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
    >>> from colour_checker_detection import ROOT_RESOURCES_TESTS
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = adjust_image(read_image(path), 1440)
    >>> rectangle = (
    ...     (723.29608154, 465.50939941),
    ...     (461.24377441, 696.34759522),
    ...     -88.18692780,
    ... )
    >>> print(image.shape)
    (959, 1440, 3)
    >>> image = crop_with_rectangle(image, rectangle)
    >>> print(image.shape)
    (461, 696, 3)
    """

    image = as_float_array(image, FLOAT_DTYPE_DEFAULT)[..., :3]

    width, height = image.shape[1], image.shape[0]
    width_r, height_r = rectangle[1]
    centroid = contour_centroid(cv2.boxPoints(rectangle))
    angle = rectangle[-1]

    width_r, height_r = as_int_array([width_r, height_r])

    M_r = cv2.getRotationMatrix2D(centroid, angle, 1)

    image_r = cv2.warpAffine(image, M_r, (width, height), interpolation_method)
    image_c = cv2.getRectSubPix(
        image_r, (width_r, height_r), (centroid[0], centroid[1])
    )

    if image_c.shape[0] > image_c.shape[1]:
        image_c = orient(image_c, "90 CW")

    return image_c  # pyright: ignore


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

    moments = cv2.moments(contour)  # pyright: ignore

    centroid = (
        moments["m10"] / moments["m00"],
        moments["m01"] / moments["m00"],
    )

    return cast(Tuple[float, float], centroid)


def scale_contour(contour: ArrayLike, factor: float) -> NDArrayFloat:
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

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> scale_contour(contour, 2)  # doctest: +ELLIPSIS
    array([[...-0.5, ...-0.5],
           [... 1.5, ...-0.5],
           [... 1.5, ... 1.5],
           [...-0.5, ... 1.5]])
    """

    centroid = as_float_array(contour_centroid(contour))

    scaled_contour = (as_float_array(contour) - centroid) * factor + centroid

    return scaled_contour


# https://stackoverflow.com/a/55339684/931625
def approximate_contour(contour, n_corners=4):
    """
    Binary searches best `epsilon` value to force contour
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    """

    n_iter, max_iter = 0, 200
    lb, ub = 0, 1

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub) / 2
        approximation = cv2.approxPolyDP(
            contour, k * cv2.arcLength(contour, True), True
        )

        if len(approximation) > n_corners:
            lb = (lb + ub) / 2
        elif len(approximation) < n_corners:
            ub = (lb + ub) / 2
        else:
            return approximation
