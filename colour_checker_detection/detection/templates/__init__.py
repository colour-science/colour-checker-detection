"""
Templates
=========

Template system for colour checker structure recognition and
correspondence finding in warped perspective detection.
"""

from __future__ import annotations

import os

from colour_checker_detection.detection.templates.generate_template import (
    Template,
    generate_template,
    load_template,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Template",
    "generate_template",
    "load_template",
]


ROOT_TEMPLATES: str = os.path.dirname(__file__)

PATH_TEMPLATE_COLORCHECKER_CLASSIC: str = os.path.join(
    ROOT_TEMPLATES, "template_colorchecker_classic.npz"
)
"""
Path to the *X-Rite ColorChecker Classic* 24-patch template NPZ file.
"""

PATH_TEMPLATE_COLORCHECKER_CREATIVE_ENHANCEMENT: str = os.path.join(
    ROOT_TEMPLATES, "template_colorchecker_creative_enhancement.npz"
)
"""
Path to the *X-Rite ColorChecker Creative Enhancement* 140-patch template NPZ file.
"""

__all__ += [
    "ROOT_TEMPLATES",
    "PATH_TEMPLATE_COLORCHECKER_CLASSIC",
    "PATH_TEMPLATE_COLORCHECKER_CREATIVE_ENHANCEMENT",
]
