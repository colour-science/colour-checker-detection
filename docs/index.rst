Colour - Checker Detection
==========================

A `Python <https://www.python.org>`__ package implementing various colour
checker detection algorithms and related utilities.

It is open source and freely available under the
`BSD-3-Clause <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

..  image:: https://raw.githubusercontent.com/colour-science/colour-checker-detection/master/docs/_static/ColourCheckerDetection_001.png

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

.. toctree::
    :maxdepth: 2

    user-guide

API Reference
-------------

.. toctree::
    :maxdepth: 2

    reference

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
