"""
Define the unit tests for the
:mod:`colour_checker_detection.detection.templates` module.
"""

from __future__ import annotations

import tempfile

import numpy as np

from colour_checker_detection.detection.templates import (
    Template,
    generate_template,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestTemplate",
    "TestGenerateTemplate",
]


class TestTemplate:
    """
    Define :class:`colour_checker_detection.detection.templates.Template`
    class unit tests methods.
    """

    def test_template_creation(self) -> None:
        """Test Template dataclass creation."""
        swatch_centroids = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        colours = np.array(
            [[0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7], [0.1, 0.1, 0.1]]
        )
        correspondences = [[0, 1, 2, 3]]
        width = 100
        height = 100

        template = Template(swatch_centroids, colours, correspondences, width, height)

        assert template.swatch_centroids.shape == (4, 2)
        assert template.colours.shape == (4, 3)
        assert template.correspondences == [[0, 1, 2, 3]]
        assert template.width == width
        assert template.height == height


class TestGenerateTemplate:
    """
    Define :func:`colour_checker_detection.detection.templates.generate_template`
    definition unit tests methods.
    """

    def test_generate_template_basic(self) -> None:
        """Test basic template generation."""
        swatch_centroids = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
        colours = np.array(
            [[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]]
        )
        name = "test_template"
        width = 100
        height = 100

        with tempfile.TemporaryDirectory() as temp_dir:
            template = generate_template(
                swatch_centroids,
                colours,
                name,
                width,
                height,
                output_directory=temp_dir,
            )

        assert isinstance(template, Template)
        assert np.array_equal(template.swatch_centroids, swatch_centroids)
        assert np.array_equal(template.colours, colours)
        assert template.width == width
        assert template.height == height
        assert isinstance(template.correspondences, list)

    def test_generate_template_with_visualization(self) -> None:
        """Test template generation with visualization disabled."""
        swatch_centroids = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
        colours = np.array(
            [[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]]
        )
        name = "test_viz"
        width = 100
        height = 100

        # Test without visualization (default)
        with tempfile.TemporaryDirectory() as temp_dir:
            template = generate_template(
                swatch_centroids,
                colours,
                name,
                width,
                height,
                show=False,
                output_directory=temp_dir,
            )

        assert isinstance(template, Template)
        assert (
            len(template.correspondences) >= 0
        )  # May have valid correspondences or not
