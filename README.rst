Colour - Checker Detection
==========================

.. start-badges

|travis| |coveralls| |codacy| |version|

.. |travis| image:: https://img.shields.io/travis/colour-science/colour-checker-detection/develop.svg?style=flat-square
    :target: https://travis-ci.org/colour-science/colour-checker-detection
    :alt: Develop Build Status
.. |coveralls| image:: http://img.shields.io/coveralls/colour-science/colour-checker-detection/develop.svg?style=flat-square
    :target: https://coveralls.io/r/colour-science/colour-checker-detection
    :alt: Coverage Status
.. |codacy| image:: https://img.shields.io/codacy/grade/984900e3a85e40239a0f8f633dd1ebcb/develop.svg?style=flat-square
    :target: https://www.codacy.com/app/colour-science/colour-checker-detection
    :alt: Code Grade
.. |version| image:: https://img.shields.io/pypi/v/colour-checker-detection.svg?style=flat-square
    :target: https://pypi.python.org/pypi/colour-checker-detection
    :alt: Package Version

.. end-badges


A `Python <https://www.python.org/>`_ package implementing various colour
checker detection algorithms and related utilities.

It is open source and freely available under the
`New BSD License <https://opensource.org/licenses/BSD-3-Clause>`_ terms.

..  image:: https://raw.githubusercontent.com/colour-science/colour-checker-detection/master/docs/_static/ColourCheckerDetection_001.png

.. contents:: **Table of Contents**
    :backlinks: none
    :depth: 3

.. sectnum::

Features
--------

The following colour checker detection algorithms are implemented:

-   Segmentation

Installation
------------

Because of their size, the resources dependencies needed to run the various
examples and unit tests are not provided within the Pypi package. They are
separately available as
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_
when cloning the
`repository <https://github.com/colour-science/colour-checker-detection>`_.

Primary Dependencies
^^^^^^^^^^^^^^^^^^^^

**Colour - Checker Detection** requires various dependencies in order to run:

-  `Python 2.7 <https://www.python.org/download/releases/>`_ or
   `Python 3.7 <https://www.python.org/download/releases/>`_
-  `Colour Science <https://www.colour-science.org>`_
-  `opencv-python <https://pypi.org/project/opencv-python/>`_

Pypi
^^^^

Once the dependencies satisfied, **Colour - Checker Detection** can be installed from
the `Python Package Index <http://pypi.python.org/pypi/colour-checker-detection>`_ by
issuing this command in a shell::

	pip install colour-checker-detection

The tests suite dependencies are installed as follows::

    pip install 'colour-checker-detection[tests]'

The documentation building dependencies are installed as follows::

    pip install 'colour-checker-detection[docs]'

The overall development dependencies are installed as follows::

    pip install 'colour-checker-detection[development]'

Usage
-----

API
^^^

The main reference for `Colour - Checker Detection <https://github.com/colour-science/colour-checker-detection>`_
is the `Colour - Checker Detection Manual <https://colour-checker-detection.readthedocs.io/en/latest/manual.html>`_.

Examples
^^^^^^^^

Various usage examples are available from the
`examples directory <https://github.com/colour-science/colour-checker-detection/tree/master/colour_checker_detection/examples>`_.

Contributing
------------

If you would like to contribute to `Colour - Checker Detection <https://github.com/colour-science/colour-checker-detection>`_,
please refer to the following `Contributing <https://www.colour-science.org/contributing/>`_
guide for `Colour <https://github.com/colour-science/colour>`_.

Bibliography
------------

The bibliography is available in the repository in
`BibTeX <https://github.com/colour-science/colour-checker-detection/blob/develop/BIBLIOGRAPHY.bib>`_
format.

About
-----

| **Colour - Checker Detection** by Colour Developers
| Copyright © 2018-2019 – Colour Developers – `colour-science@googlegroups.com <colour-science@googlegroups.com>`_
| This software is released under terms of New BSD License: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour-checker-detection <https://github.com/colour-science/colour-checker-detection>`_
