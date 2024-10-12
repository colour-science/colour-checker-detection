Installation Guide
==================

Because of their size, the resources dependencies needed to run the various
examples and unit tests are not provided within the Pypi package. They are
separately available as
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__
when cloning the
`repository <https://github.com/colour-science/colour-checker-detection>`__.

Primary Dependencies
--------------------

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
----

Once the dependencies are satisfied, **Colour - Checker Detection** can be installed from
the `Python Package Index <http://pypi.python.org/pypi/colour-checker-detection>`__ by
issuing this command in a shell::

    pip install --user colour-checker-detection

The tests suite dependencies are installed as follows::

    pip install --user 'colour-checker-detection[tests]'

The documentation building dependencies are installed as follows::

    pip install --user 'colour-checker-detection[docs]'

The overall development dependencies are installed as follows::

    pip install --user 'colour-checker-detection[development]'
