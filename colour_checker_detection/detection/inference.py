"""
Colour Checker Detection - Inference
====================================

Define the objects for colour checker detection using inference based on
*Ultralytics YOLOv8* machine learning model.

-   :attr:`colour_checker_detection.SETTINGS_INFERENCE_COLORCHECKER_CLASSIC`
-   :func:`colour_checker_detection.inferencer_default`
-   :func:`colour_checker_detection.detect_colour_checkers_inference`
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

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
from colour.io import convert_bit_depth, read_image, write_image
from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.plotting import CONSTANTS_COLOUR_STYLE, plot_image
from colour.utilities import (
    Structure,
    as_int_scalar,
)
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)

from colour_checker_detection.detection.common import (
    DTYPE_FLOAT_DEFAULT,
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
    DataDetectionColourChecker,
    as_int32_array,
    quadrilateralise_contours,
    sample_colour_checker,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC",
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI",
    "PATH_INFERENCE_SCRIPT_DEFAULT",
    "inferencer_default",
    "INFERRED_CLASSES",
    "detect_colour_checkers_inference",
]


SETTINGS_INFERENCE_COLORCHECKER_CLASSIC: Dict = (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC = DocstringDict(
        SETTINGS_INFERENCE_COLORCHECKER_CLASSIC
    )
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC.__doc__ = """
Settings for the inference of the *X-Rite* *ColorChecker Classic*.
"""
SETTINGS_INFERENCE_COLORCHECKER_CLASSIC.update(
    {
        "aspect_ratio": 1000 / 700,
        "working_height": int(1440 / (1000 / 700)),
        "transform": {
            "translation": np.array([0, 0]),
            "rotation": 0,
            "scale": np.array([1.0, 1.05]),
        },
        "inferred_class": "ColorCheckerClassic24",
        "inferred_confidence": 0.85,
    }
)

SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI: Dict = (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
)
if is_documentation_building():  # pragma: no cover
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI = DocstringDict(
        SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI
    )
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI.__doc__ = """
Settings for the inference of the *X-Rite* *ColorChecker Classic Mini*.
"""
SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI.update(
    {
        "aspect_ratio": 1000 / 585,
        "working_height": int(1440 / (1000 / 585)),
        "transform": {
            "translation": np.array([0, 0]),
            "rotation": 0,
            "scale": np.array([1.15, 1.0]),
        },
        "inferred_class": "ColorCheckerSG",
        "inferred_confidence": 0.85,
    }
)


PATH_INFERENCE_SCRIPT_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "inference.py"
)
"""
Path to the default inference script.

Warnings
--------
The default script is provided under the terms of the
*GNU Affero General Public License v3.0* as it uses the *Ultralytics YOLOv8*
API which is incompatible with the *BSD-3-Clause*.
"""


def inferencer_default(
    image: str | ArrayLike,
    cctf_encoding: Callable = eotf_inverse_sRGB,
    apply_cctf_encoding: bool = True,
    show: bool = False,
) -> NDArrayInt | NDArrayFloat:
    """
    Predict the colour checker rectangles in given image using
    *Ultralytics YOLOv8*.

    Parameters
    ----------
    image
        Image (or image path to read the image from) to detect the colour
        checker rectangles from.
    cctf_encoding
        Encoding colour component transfer function / opto-electronic
        transfer function used when converting the image from float to 8-bit.
    apply_cctf_encoding
        Apply the encoding colour component transfer function / opto-electronic
        transfer function.
    show
        Whether to show various debug images.

    Returns
    -------
    :class:`np.ndarray`
        Array of inference results as rows of confidence, class, and mask.

    Warnings
    --------
    This definition sub-processes to a script licensed under the terms of the
    *GNU Affero General Public License v3.0* as it uses the *Ultralytics YOLOv8*
    API which is incompatible with the *BSD-3-Clause*.

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
    >>> results = inferencer_default(path)  # doctest: +SKIP
    >>> results.shape  # doctest: +SKIP
    (1, 3)
    >>> results[0][0]  # doctest: +SKIP
    array(0.9708795...)
    >>> results[0][1]  # doctest: +SKIP
    array(0.0...)
    >>> results[0][2].shape  # doctest: +SKIP
    (864, 1280)
    """

    temp_directory = tempfile.mkdtemp()

    try:
        if not isinstance(image, str):
            input_image = os.path.join(temp_directory, "input-image.png")

            if apply_cctf_encoding:
                image = cctf_encoding(image)

            write_image(image, input_image, "uint8")
        else:
            input_image = image

        output_results = os.path.join(temp_directory, "output-results.npz")
        subprocess.call(
            [  # noqa: S603
                sys.executable,
                PATH_INFERENCE_SCRIPT_DEFAULT,
                "--input",
                input_image,
                "--output",
                output_results,
            ]
            + (["--show"] if show else [])
        )
        results = np.load(output_results, allow_pickle=True)["results"]
    finally:
        shutil.rmtree(temp_directory)

    return results


INFERRED_CLASSES: Dict = {0: "ColorCheckerClassic24"}
"""Inferred classes."""


def detect_colour_checkers_inference(
    image: str | ArrayLike,
    samples: int = 32,
    cctf_decoding=eotf_sRGB,
    apply_cctf_decoding: bool = False,
    inferencer: Callable = inferencer_default,
    inferencer_kwargs: dict | None = None,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker | NDArrayFloat, ...]:
    """
    Detect the colour checkers swatches in given image using inference.

    Parameters
    ----------
    image
        Image (or image path to read the image from) to detect the colour
        checker rectangles from.
    samples
        Sample count to use to average (mean) the swatches colours. The effective
        sample count is :math:`samples^2`.
    cctf_decoding
        Decoding colour component transfer function / opto-electronic
        transfer function used when converting the image from 8-bit to float.
    apply_cctf_decoding
        Apply the decoding colour component transfer function / opto-electronic
        transfer function.
    inferencer
        Callable responsible to make predictions on the image and extract the
        colour checker rectangles.
    inferencer_kwargs
        Keyword arguments to pass to the ``inferencer``.
    show
        Whether to show various debug images.
    additional_data
        Whether to output additional data.

    Other Parameters
    ----------------
    aspect_ratio
        Colour checker aspect ratio, e.g., 1.5.
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
    >>> detect_colour_checkers_inference(image)  # doctest: +SKIP
    (array([[ 0.3602327 ,  0.22158547,  0.11813926],
           [ 0.62800723,  0.39357048,  0.24196433],
           [ 0.3284166 ,  0.31669423,  0.28818974],
           [ 0.3072932 ,  0.2744136 ,  0.10451803],
           [ 0.4204691 ,  0.31953654,  0.30901137],
           [ 0.34471545,  0.44057423,  0.29297924],
           [ 0.678418  ,  0.35242617,  0.06670552],
           [ 0.27259055,  0.2535471 ,  0.32912973],
           [ 0.6190633 ,  0.27043283,  0.18543543],
           [ 0.30721852,  0.18180828,  0.19161244],
           [ 0.4858081 ,  0.46007228,  0.03085822],
           [ 0.6499356 ,  0.4018961 ,  0.01579806],
           [ 0.19425018,  0.18621376,  0.27193058],
           [ 0.27500305,  0.38600868,  0.1245231 ],
           [ 0.55459476,  0.21477987,  0.12434786],
           [ 0.71898675,  0.5149239 ,  0.00561224],
           [ 0.5787967 ,  0.25837064,  0.2693373 ],
           [ 0.1743919 ,  0.31709513,  0.29550385],
           [ 0.7383609 ,  0.60645705,  0.43850273],
           [ 0.62609893,  0.5172464 ,  0.36816722],
           [ 0.5117422 ,  0.4191487 ,  0.3013721 ],
           [ 0.36412936,  0.2987345 ,  0.20754097],
           [ 0.26675388,  0.21421173,  0.14176223],
           [ 0.15856811,  0.13483825,  0.07938566]], dtype=float32),)
    """

    if inferencer_kwargs is None:
        inferencer_kwargs = {}

    settings = Structure(**SETTINGS_INFERENCE_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    swatches_horizontal = settings.swatches_horizontal
    swatches_vertical = settings.swatches_vertical
    working_width = settings.working_width
    working_height = settings.working_height

    results = inferencer(image, **inferencer_kwargs)

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

    rectangle = as_int32_array(
        [
            [0, 0],
            [0, working_height],
            [working_width, working_height],
            [working_width, 0],
        ]
    )

    colour_checkers_data = []
    for result_confidence, result_class, result_mask in results:
        if result_confidence < settings.inferred_confidence:
            continue

        if settings.inferred_class != INFERRED_CLASSES[int(result_class)]:
            continue

        mask = cv2.resize(
            result_mask,
            image.shape[:2][::-1],
            interpolation=cv2.INTER_BITS,
        )

        contours, _hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for quadrilateral in quadrilateralise_contours(contours):
            colour_checkers_data.append(
                sample_colour_checker(
                    image, quadrilateral, rectangle, samples, **settings
                )
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
                    CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(
                        colour_checker
                    ),
                    text_kwargs={
                        "text": (
                            f"Class: "
                            f'"{INFERRED_CLASSES[as_int_scalar(result_class)]}", '
                            f"Confidence : {result_confidence:.3f}"
                        )
                    },
                )

                plot_image(
                    CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(
                        np.reshape(
                            colour_checkers_data[-1].swatch_colours,
                            [swatches_vertical, swatches_horizontal, 3],
                        )
                    ),
                )

    if additional_data:
        return tuple(colour_checkers_data)
    else:
        return tuple(
            colour_checker_data.swatch_colours
            for colour_checker_data in colour_checkers_data
        )
