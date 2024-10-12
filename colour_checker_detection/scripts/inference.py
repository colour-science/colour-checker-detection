#!/usr/bin/env python
"""
Colour Checker Detection - Inference
====================================

Define the scripts for colour checker detection using inference based on
*Ultralytics YOLOv8* machine learning model.

Warnings
--------
This script is provided under the terms of the
*GNU Affero General Public License v3.0* as it uses the *Ultralytics YOLOv8*
API which is incompatible with the *BSD-3-Clause*.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter

import click
import cv2
import numpy as np
from colour import read_image
from colour.hints import List, Literal, NDArray, Tuple
from colour.io import convert_bit_depth

__author__ = "Colour Developers"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = (
    "GNU Affero General Public License v3.0 - "
    "https://www.gnu.org/licenses/agpl-3.0.en.html"
)
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


__all__ = [
    "ROOT_REPOSITORY",
    "URL_BASE",
    "URL_MODEL_FILE_DEFAULT",
    "inference",
    "segmentation",
]


logger = logging.getLogger(__name__)

ROOT_REPOSITORY: str = os.environ.get(
    "COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY",
    os.path.join(
        os.path.expanduser("~"),
        ".colour-science",
        "colour-checker-detection",
    ),
)
"""Root of the local repository to download the hosted models to."""

URL_BASE: str = "https://huggingface.co/colour-science/colour-checker-detection-models"
"""URL of the remote repository to download the models from."""

URL_MODEL_FILE_DEFAULT: str = (
    f"{URL_BASE}/resolve/main/models/colour-checker-detection-l-seg.pt"
)
"""URL for the default segmentation model."""


def inference(
    source: str | Path | NDArray,
    model: YOLO,  # noqa: F821 # pyright: ignore
    show: bool = False,
    **kwargs,
) -> List[Tuple[NDArray, NDArray, NDArray]]:
    """
    Run the inference on the provided source.

    Parameters
    ----------
    source
        Source of the image to make predictions on. Accepts all source types
        accepted by the *YOLOv8* model.
    model
        The model to use for the inference.
    show
        Whether to show the inference results on the image.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for the *YOLOv8* segmentation method.

    Returns
    -------
    :class:`list`
        Inference results.
    """

    data = []

    for result in model(source, show=show, **kwargs):
        show and cv2.waitKey(0) == ord("n")

        if result.boxes is None:
            continue

        if result.masks is None:
            continue

        data_boxes = result.boxes.data
        data_masks = result.masks.data

        for i in range(data_boxes.shape[0]):
            data.append(
                (
                    data_boxes[i, 4].cpu().numpy(),
                    data_boxes[i, 5].cpu().numpy(),
                    data_masks[i].data.cpu().numpy(),
                )
            )

            if np.any(data[-1][-1]):
                logging.debug(
                    'Found a "%s" class object with "%s" confidence.',
                    data[-1][1],
                    data[-1][0],
                )
            else:
                logging.warning("No objects were detected!")

    return data


@click.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input file to run the segmentation model on.",
)
@click.option(
    "--output",
    help="Output file to write the segmentation results to.",
)
@click.option(
    "--model",
    "model",
    type=click.Path(exists=True),
    help='Segmentation model file to load. Default to the "colour-science" model '
    'hosted on "HuggingFace". It will be downloaded if not cached already.',
)
@click.option(
    "--show/--no-show",
    default=False,
    help="Whether to show the segmentation results.",
)
@click.option(
    "--logging-level",
    "logging_level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def segmentation(
    input: str,  # noqa: A002
    output: str | None = None,
    model: str | None = None,
    show: bool = False,
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> NDArray:
    """
    Run the segmentation model on the given input file and save the results to
    given output file.

    Parameters
    ----------
    input
        Input file to run the segmentation model on.
    output
        Output file to write the segmentation results to.
    model
        Segmentation model file to load. Default to the *colour-science8 model
        hosted on *HuggingFace*. It will be downloaded if not cached already.
    show
        Whether to show the segmentation results.
    logging_level
        Set the logging level.

    Returns
    -------
    :class:`numpy.ndarray`
        Inference results.
    """

    from ultralytics import YOLO
    from ultralytics.utils.downloads import download

    time_start = perf_counter()

    logging.getLogger().setLevel(getattr(logging, logging_level.upper()))

    if model is None:
        model = os.path.join(ROOT_REPOSITORY, os.path.basename(URL_MODEL_FILE_DEFAULT))
        logging.debug('Using "%s" default model.', model)
        if not os.path.exists(model):
            logging.info('Downloading "%s" model...', URL_MODEL_FILE_DEFAULT)
            download(URL_MODEL_FILE_DEFAULT, ROOT_REPOSITORY)

    if input.endswith((".npy", ".npz")):
        logging.debug('Reading "%s" serialised array...', input)
        source = np.load(input)
    else:
        logging.debug('Reading "%s" image...', input)
        source = convert_bit_depth(
            read_image(input)[..., :3],
            np.uint8.__name__,  # pyright: ignore
        )

    # NOTE: YOLOv8 expects "BGR" arrays.
    results = np.array(inference(source[..., ::-1], YOLO(model), show), dtype=object)

    if output is None:
        output = f"{input}.npz"

    np.savez(output, results=results)

    logging.debug('Total segmentation time: "%s" seconds.', perf_counter() - time_start)

    return results


if __name__ == "__main__":
    logging.basicConfig()

    segmentation()
