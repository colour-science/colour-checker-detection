Colour - Checker Detection
==========================

.. start-badges

|actions| |coveralls| |codacy| |version|

.. |actions| image:: https://img.shields.io/github/actions/workflow/status/colour-science/colour-checker-detection/.github/workflows/continuous-integration-quality-unit-tests.yml?branch=develop&style=flat-square
    :target: https://github.com/colour-science/colour-checker-detection/actions
    :alt: Develop Build Status
.. |coveralls| image:: http://img.shields.io/coveralls/colour-science/colour-checker-detection/develop.svg?style=flat-square
    :target: https://coveralls.io/r/colour-science/colour-checker-detection
    :alt: Coverage Status
.. |codacy| image:: https://img.shields.io/codacy/grade/c543bc30229347cdaea00aadd3f79499/develop.svg?style=flat-square
    :target: https://app.codacy.com/gh/colour-science/colour-checker-detection
    :alt: Code Grade
.. |version| image:: https://img.shields.io/pypi/v/colour-checker-detection.svg?style=flat-square
    :target: https://pypi.org/project/colour-checker-detection
    :alt: Package Version

.. end-badges

A `Python <https://www.python.org>`__ package implementing various colour
checker detection algorithms and related utilities.

It is open source and freely available under the
`BSD-3-Clause <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

..  image:: https://raw.githubusercontent.com/colour-science/colour-checker-detection/master/docs/_static/ColourCheckerDetection_001.png

.. contents:: **Table of Contents**
    :backlinks: none
    :depth: 2

.. sectnum::

Features
--------

The following colour checker detection algorithms are implemented:

-   Segmentation
-   Machine learning inference via `Ultralytics YOLOv8 <https://github.com/ultralytics/ultralytics>`__

    -   The model is published on `HuggingFace <https://huggingface.co/colour-science/colour-checker-detection-models>`__,
        and was trained on a purposely constructed `dataset <https://huggingface.co/datasets/colour-science/colour-checker-detection-dataset>`__.
    -   The model has only been trained on *ColorChecker Classic 24* images and
        will not work with *ColorChecker Nano* or *ColorChecker SG* images.
    -   Inference is performed by a script licensed under the terms of the
        *GNU Affero General Public License v3.0* as it uses the
        *Ultralytics YOLOv8* API which is incompatible with the
        *BSD-3-Clause*.

Examples
^^^^^^^^

Various usage examples are available from the
`examples directory <https://github.com/colour-science/colour-checker-detection/tree/master/colour_checker_detection/examples>`__.

User Guide
----------

Installation
^^^^^^^^^^^^

Because of their size, the resources dependencies needed to run the various
examples and unit tests are not provided within the Pypi package. They are
separately available as
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__
when cloning the
`repository <https://github.com/colour-science/colour-checker-detection>`__.

Primary Dependencies
~~~~~~~~~~~~~~~~~~~~

**Colour - Checker Detection** requires various dependencies in order to run:

- `python >= 3.10, < 3.14 <https://www.python.org/download/releases>`__
- `colour-science >= 4.5 <https://pypi.org/project/colour-science>`__
- `imageio >= 2, < 3 <https://imageio.github.io>`__
- `numpy >= 1.24, < 3 <https://pypi.org/project/numpy>`__
- `opencv-python >= 4, < 5 <https://pypi.org/project/opencv-python>`__
- `scipy >= 1.10, < 2 <https://pypi.org/project/scipy>`__

Secondary Dependencies
~~~~~~~~~~~~~~~~~~~~~~

- `click >= 8, < 9 <https://pypi.org/project/click>`__
- `ultralytics >= 8, < 9 <https://pypi.org/project/ultralytics>`__

Pypi
~~~~

Once the dependencies are satisfied, **Colour - Checker Detection** can be installed from
the `Python Package Index <http://pypi.python.org/pypi/colour-checker-detection>`__ by
issuing this command in a shell::

    pip install --user colour-checker-detection

The overall development dependencies are installed as follows::

    pip install --user 'colour-checker-detection[development]'

Contributing
^^^^^^^^^^^^

If you would like to contribute to `Colour - Checker Detection <https://github.com/colour-science/colour-checker-detection>`__,
please refer to the following `Contributing <https://www.colour-science.org/contributing>`__
guide for `Colour <https://github.com/colour-science/colour>`__.

Bibliography
^^^^^^^^^^^^

The bibliography is available in the repository in
`BibTeX <https://github.com/colour-science/colour-checker-detection/blob/develop/BIBLIOGRAPHY.bib>`__
format.

API Reference
-------------

The main technical reference `Colour - Checker Detection <https://github.com/colour-science/colour-checker-detection>`__
is the `API Reference <https://colour-checker-detection.readthedocs.io/en/latest/reference.html>`__.

Code of Conduct
---------------

The *Code of Conduct*, adapted from the `Contributor Covenant 1.4 <https://www.contributor-covenant.org/version/1/4/code-of-conduct.html>`__,
is available on the `Code of Conduct <https://www.colour-science.org/code-of-conduct>`__ page.

Contact & Social
----------------

The *Colour Developers* can be reached via different means:

- `Email <mailto:colour-developers@colour-science.org>`__
- `Facebook <https://www.facebook.com/python.colour.science>`__
- `Github Discussions <https://github.com/colour-science/colour-checker-detection/discussions>`__
- `Gitter <https://gitter.im/colour-science/colour>`__
- `Twitter <https://twitter.com/colour_science>`__

About
-----

| **Colour - Checker Detection** by Colour Developers
| Copyright 2018 Colour Developers – `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour-checker-detection <https://github.com/colour-science/colour-checker-detection>`__
