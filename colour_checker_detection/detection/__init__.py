"""
Detection
=========

Colour checker detection algorithms and utilities.

This subpackage provides common utilities, segmentation-based detection
methods, and machine learning inference approaches for identifying colour
checkers in images.
"""

from .common import (
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    SETTINGS_CONTOUR_DETECTION_DEFAULT,
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
    SETTINGS_DETECTION_COLORCHECKER_SG,
    DataDetectionColourChecker,
    approximate_contour,
    as_float32_array,
    as_int32_array,
    contour_centroid,
    detect_contours,
    is_square,
    quadrilateralise_contours,
    reformat_image,
    remove_stacked_contours,
    sample_colour_checker,
    scale_contour,
    swatch_colours,
    swatch_masks,
    transform_image,
)

# isort: split

from .inference import (
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC,
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI,
    detect_colour_checkers_inference,
    inferencer_default,
)

# isort: split

from .segmentation import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    SETTINGS_SEGMENTATION_COLORCHECKER_NANO,
    SETTINGS_SEGMENTATION_COLORCHECKER_SG,
    detect_colour_checkers_segmentation,
    segmenter_default,
)

__all__ = [
    "DTYPE_FLOAT_DEFAULT",
    "DTYPE_INT_DEFAULT",
    "SETTINGS_CONTOUR_DETECTION_DEFAULT",
    "SETTINGS_DETECTION_COLORCHECKER_CLASSIC",
    "SETTINGS_DETECTION_COLORCHECKER_SG",
    "DataDetectionColourChecker",
    "approximate_contour",
    "as_float32_array",
    "as_int32_array",
    "contour_centroid",
    "detect_contours",
    "is_square",
    "quadrilateralise_contours",
    "reformat_image",
    "remove_stacked_contours",
    "sample_colour_checker",
    "scale_contour",
    "swatch_colours",
    "swatch_masks",
    "transform_image",
]
__all__ += [
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC",
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI",
    "detect_colour_checkers_inference",
    "inferencer_default",
]
__all__ += [
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "SETTINGS_SEGMENTATION_COLORCHECKER_NANO",
    "SETTINGS_SEGMENTATION_COLORCHECKER_SG",
    "detect_colour_checkers_segmentation",
    "segmenter_default",
]
