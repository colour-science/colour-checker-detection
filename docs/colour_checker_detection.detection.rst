Detection
=========

Inference
---------

``colour_checker_detection``

.. currentmodule:: colour_checker_detection

.. autosummary::
    :toctree: generated/

    detect_colour_checkers_inference
    inferencer_default
    extractor_inference
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC
    SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI

Segmentation
------------

``colour_checker_detection``

.. currentmodule:: colour_checker_detection

.. autosummary::
    :toctree: generated/

    detect_colour_checkers_segmentation
    segmenter_default
    extractor_segmentation
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC
    SETTINGS_SEGMENTATION_COLORCHECKER_NANO
    SETTINGS_SEGMENTATION_COLORCHECKER_SG

``colour_checker_detection.detection``

.. currentmodule:: colour_checker_detection.detection

.. autosummary::
    :toctree: generated/

    DataSegmentationColourCheckers

Templated
---------

``colour_checker_detection``

.. currentmodule:: colour_checker_detection

.. autosummary::
    :toctree: generated/

    detect_colour_checkers_templated
    segmenter_templated
    extractor_templated
    SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC

``colour_checker_detection.detection``

.. currentmodule:: colour_checker_detection.detection

.. autosummary::
    :toctree: generated/

    WarpingData

Templates
---------

``colour_checker_detection.detection``

.. currentmodule:: colour_checker_detection.detection.templates

.. autosummary::
    :toctree: generated/

    Template
    generate_template
    load_template
    PATH_TEMPLATE_COLORCHECKER_CLASSIC
    PATH_TEMPLATE_COLORCHECKER_CREATIVE_ENHANCEMENT

Plotting
--------

``colour_checker_detection``

.. currentmodule:: colour_checker_detection

.. autosummary::
    :toctree: generated/

    plot_detection_results

Common Utilities
----------------

``colour_checker_detection.detection``

.. currentmodule:: colour_checker_detection.detection

.. autosummary::
    :toctree: generated/

    approximate_contour
    as_float32_array
    as_int32_array
    contour_centroid
    DataDetectionColourChecker
    detect_contours
    DTYPE_FLOAT_DEFAULT
    DTYPE_INT_DEFAULT
    is_square
    quadrilateralise_contours
    reformat_image
    remove_stacked_contours
    sample_colour_checker
    scale_contour
    cluster_swatches
    filter_clusters
    SETTINGS_CONTOUR_DETECTION_DEFAULT
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC
    SETTINGS_DETECTION_COLORCHECKER_SG
    swatch_colours
    swatch_masks
    transform_image
