"""
Requirements Utilities
======================

Define the requirements utilities objects.
"""

from __future__ import annotations

import colour.utilities

__author__ = "Colour Developers"
__copyright__ = "Copyright 2018 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_sklearn_installed",
]


def is_sklearn_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *scikit-learn* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *scikit-learn* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *scikit-learn* is installed.

    Raises
    ------
    :class:`ImportError`
        If *scikit-learn* is not installed.
    """

    try:  # pragma: no cover
        import sklearn  # noqa: F401, PLC0415
    except ImportError as exception:  # pragma: no cover
        if raise_exception:
            error = (
                f'"scikit-learn" related API features are not available: "{exception}".'
            )

            raise ImportError(error) from exception

        return False
    else:
        return True


colour.utilities.requirements.REQUIREMENTS_TO_CALLABLE.update(
    {
        "scikit-learn": is_sklearn_installed,
    }
)
