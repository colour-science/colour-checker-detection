"""
Colour - Checker Detection - Setup
==================================
"""

import codecs
from setuptools import setup

packages = [
    "colour_checker_detection",
    "colour_checker_detection.detection",
    "colour_checker_detection.detection.tests",
]

package_data = {
    "": ["*"],
    "colour_checker_detection": [
        "examples/*",
        "resources/colour-checker-detection-examples-datasets/*",
        "resources/colour-checker-detection-tests-datasets/*",
    ],
}

install_requires = [
    "colour-science>=0.4.2",
    "imageio>=2,<3",
    "numpy>=1.20,<2",
    "opencv-python>=4,<5",
    "scipy>=1.7,<2",
    "typing-extensions>=4,<5",
]

extras_require = {
    "development": [
        "biblib-simple",
        "black",
        "blackdoc",
        "coverage",
        "coveralls",
        "flake8",
        "flynt",
        "invoke",
        "jupyter",
        "mypy",
        "pre-commit",
        "pydata-sphinx-theme",
        "pydocstyle",
        "pytest",
        "pytest-cov",
        "pyupgrade",
        "restructuredtext-lint",
        "sphinx>=4,<5",
        "sphinxcontrib-bibtex",
        "toml",
        "twine",
    ],
    "plotting": ["matplotlib>=3.4,!=3.5.0,!=3.5.1"],
    "read-the-docs": [
        "matplotlib>=3.4,!=3.5.0,!=3.5.1",
        "pydata-sphinx-theme",
        "sphinxcontrib-bibtex",
    ],
}

setup(
    name="colour-checker-detection",
    version="0.1.4",
    description="Colour checker detection with Python",
    long_description=codecs.open("README.rst", encoding="utf8").read(),
    author="Colour Developers",
    author_email="colour-developers@colour-science.org",
    maintainer="Colour Developers",
    maintainer_email="colour-developers@colour-science.org",
    url="https://www.colour-science.org/",
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9,<3.12",
)
