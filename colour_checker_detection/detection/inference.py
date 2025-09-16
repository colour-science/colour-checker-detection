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
import typing

import cv2
import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Callable,
        Dict,
        Literal,
        NDArrayFloat,
        Tuple,
    )

from colour.hints import NDArrayReal, cast
from colour.io import convert_bit_depth, read_image, write_image
from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.utilities import (
    Structure,
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
    reformat_image,
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
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC",
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI",
    "PATH_INFERENCE_SCRIPT_DEFAULT",
    "inferencer_default",
    "extractor_inference",
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
) -> NDArrayReal:
    """
    Predict the colour checker rectangles in specified image using
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
        subprocess.call(  # noqa: S603
            [
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


def extractor_inference(
    image: ArrayLike,
    inference_data: Any,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    inferred_confidence: float = 0.85,
    working_width: int = 1440,
    additional_data: bool = False,
    **kwargs: Any,
) -> tuple[DataDetectionColourChecker, ...] | tuple[NDArrayFloat, ...]:
    """
    Extract colour swatches using inference-based methods.

    This function takes inference data (bounding boxes/contours) and extracts
    colors using ML-guided sampling approach.

    Parameters
    ----------
    image
        Image to extract colours from.
    inference_data
        Inference data containing detected contours and confidence scores.
    samples
        Sample count to use to average (mean) the swatches colours. The effective
        sample count is :math:`samples^2`.
    cctf_decoding
        Decoding colour component transfer function / opto-electronic
        transfer function used when converting the image from 8-bit to float.
    apply_cctf_decoding
        Apply the decoding colour component transfer function / opto-electronic
        transfer function.
    inferred_confidence
        Minimum confidence threshold for inference results.
    working_width
        Working width for image processing.
    additional_data
        Whether to include additional extraction data.

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
    ...     inferencer_default,
    ...     extractor_inference,
    ... )
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_TESTS,
    ...     "colour_checker_detection",
    ...     "detection",
    ...     "IMG_1967.png",
    ... )
    >>> image = read_image(path)
    >>> inference_data = inferencer_default(image)  # doctest: +SKIP
    >>> extractor_inference(image, inference_data)  # doctest: +SKIP
    (array([[ 0.36007342,  0.22303678,  0.1176604 ],
           [ 0.62607545,  0.39443627,  0.24180005],
           [ 0.33200133,  0.3159002 ,  0.28866205],
           [ 0.304158  ,  0.27339226,  0.10521446],
           [ 0.41758743,  0.31893715,  0.3078802 ],
           [ 0.34878933,  0.43871346,  0.29159448],
           [ 0.67982006,  0.3523331 ,  0.070414  ],
           [ 0.27139527,  0.25354654,  0.33075848],
           [ 0.6207255 ,  0.27040577,  0.18629737],
           [ 0.3071541 ,  0.17973351,  0.19184262],
           [ 0.48536164,  0.45853454,  0.03277667],
           [ 0.65034246,  0.4002059 ,  0.01576474],
           [ 0.19285583,  0.18593574,  0.27413625],
           [ 0.28041738,  0.38502172,  0.12292562],
           [ 0.5545266 ,  0.21458797,  0.12545331],
           [ 0.7207607 ,  0.515445  ,  0.005255  ],
           [ 0.5779864 ,  0.25786015,  0.2685206 ],
           [ 0.17531879,  0.3166867 ,  0.29529998],
           [ 0.7404447 ,  0.61071527,  0.4387243 ],
           [ 0.6295517 ,  0.5178505 ,  0.37301064],
           [ 0.51465   ,  0.42113122,  0.29825154],
           [ 0.37083188,  0.30355468,  0.20928001],
           [ 0.26390594,  0.21514489,  0.1433286 ],
           [ 0.16213489,  0.13396774,  0.08086098]], dtype=float32),)
    """
    image = cast("NDArrayReal", image)

    if apply_cctf_decoding:
        image = cctf_decoding(image)

    settings = SETTINGS_INFERENCE_COLORCHECKER_CLASSIC.copy()
    settings.update(kwargs)

    working_height = int(working_width / settings["aspect_ratio"])
    image = reformat_image(image, working_width, settings["interpolation_method"])

    rectangle = as_int32_array(
        [
            [0, 0],
            [0, working_height],
            [working_width, working_height],
            [working_width, 0],
        ]
    )

    colour_checkers_data = []

    for result_confidence, _result_class, result_mask in inference_data:
        if result_confidence < inferred_confidence:
            continue

        mask = cv2.resize(
            result_mask,
            image.shape[:2][::-1],
            interpolation=cv2.INTER_BITS,
        )

        contours, _hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        colour_checkers_data.extend(
            sample_colour_checker(image, quadrilateral, rectangle, samples, **settings)
            for quadrilateral in quadrilateralise_contours(contours)
        )

    if additional_data:
        return tuple(colour_checkers_data)

    return tuple(
        colour_checker_data.swatch_colours
        for colour_checker_data in colour_checkers_data
    )


INFERRED_CLASSES: Dict = {0: "ColorCheckerClassic24"}
"""Inferred classes."""


@typing.overload
def detect_colour_checkers_inference(
    image: str | ArrayLike,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    inferencer: Callable = ...,
    inferencer_kwargs: dict | None = ...,
    extractor: Callable = ...,
    extractor_kwargs: dict | None = ...,
    show: bool = ...,
    additional_data: Literal[True] = True,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker, ...]: ...


@typing.overload
def detect_colour_checkers_inference(
    image: str | ArrayLike,
    samples: int = ...,
    cctf_decoding: Callable = ...,
    apply_cctf_decoding: bool = ...,
    inferencer: Callable = ...,
    inferencer_kwargs: dict | None = ...,
    extractor: Callable = ...,
    extractor_kwargs: dict | None = ...,
    show: bool = ...,
    *,
    additional_data: Literal[False],
    **kwargs: Any,
) -> Tuple[NDArrayFloat, ...]: ...


@typing.overload
def detect_colour_checkers_inference(
    image: str | ArrayLike,
    samples: int,
    cctf_decoding: Callable,
    apply_cctf_decoding: bool,
    inferencer: Callable,
    inferencer_kwargs: dict | None,
    extractor: Callable,
    extractor_kwargs: dict | None,
    show: bool,
    additional_data: Literal[False],
    **kwargs: Any,
) -> Tuple[NDArrayFloat, ...]: ...


def detect_colour_checkers_inference(
    image: str | ArrayLike,
    samples: int = 32,
    cctf_decoding: Callable = eotf_sRGB,
    apply_cctf_decoding: bool = False,
    inferencer: Callable = inferencer_default,
    inferencer_kwargs: dict | None = None,
    extractor: Callable = extractor_inference,
    extractor_kwargs: dict | None = None,
    show: bool = False,
    additional_data: bool = False,
    **kwargs: Any,
) -> Tuple[DataDetectionColourChecker, ...] | Tuple[NDArrayFloat, ...]:
    """
    Detect the colour checkers swatches in specified image using inference.

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
    extractor
        Callable responsible to extract the colour checker data from the
        inference results.
    extractor_kwargs
        Keyword arguments to pass to the ``extractor``.
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
    >>> detect_colour_checkers_inference(image, apply_cctf_decoding=True)
    ... # doctest: +SKIP
    (array([[  1.06845371e-01,   4.07868698e-02,   1.31441019e-02],
           [  3.50147694e-01,   1.29043385e-01,   4.78129089e-02],
           [  9.05092135e-02,   8.14050063e-02,   6.78419247e-02],
           [  7.56733939e-02,   6.07803836e-02,   1.10923871e-02],
           [  1.45852566e-01,   8.30223709e-02,   7.72851035e-02],
           [  1.00246392e-01,   1.61679372e-01,   6.92804456e-02],
           [  4.20029789e-01,   1.01915061e-01,   6.43535238e-03],
           [  6.03192598e-02,   5.23659140e-02,   8.95039141e-02],
           [  3.43519300e-01,   5.94778359e-02,   2.91121379e-02],
           [  7.70977959e-02,   2.71929316e-02,   3.07033304e-02],
           [  2.01079920e-01,   1.77687049e-01,   2.65044416e-03],
           [  3.80813688e-01,   1.33050218e-01,   1.23625272e-03],
           [  3.13875042e-02,   2.89476123e-02,   6.11585006e-02],
           [  6.44312650e-02,   1.22649640e-01,   1.42761841e-02],
           [  2.68237919e-01,   3.78787108e-02,   1.45872077e-02],
           [  4.78466213e-01,   2.28658482e-01,   4.09228291e-04],
           [  2.93730289e-01,   5.41352853e-02,   5.86953908e-02],
           [  2.68954877e-02,   8.18221569e-02,   7.10325763e-02],
           [  5.08168578e-01,   3.31246942e-01,   1.61783472e-01],
           [  3.54471445e-01,   2.30975851e-01,   1.14853390e-01],
           [  2.28191391e-01,   1.48212209e-01,   7.24881589e-02],
           [  1.13720678e-01,   7.50572383e-02,   3.62149812e-02],
           [  5.70064113e-02,   3.80622372e-02,   1.82537530e-02],
           [  2.30789520e-02,   1.61857158e-02,   7.48640532e-03]...),)
    """

    if inferencer_kwargs is None:
        inferencer_kwargs = {}

    settings = Structure(**SETTINGS_INFERENCE_COLORCHECKER_CLASSIC)
    settings.update(**kwargs)

    swatches_horizontal = settings.swatches_horizontal
    swatches_vertical = settings.swatches_vertical

    inference_results = inferencer(image, **inferencer_kwargs)

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

    if extractor_kwargs is None:
        extractor_kwargs = {}

    colour_checkers_data = list(
        extractor(
            image,
            inference_results,
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
            swatches_horizontal,
            swatches_vertical,
        )

    if additional_data:
        return tuple(colour_checkers_data)

    return tuple(
        colour_checker_data.swatch_colours
        for colour_checker_data in colour_checkers_data
    )
