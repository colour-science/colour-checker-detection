# -*- coding: utf-8 -*-
import codecs
from setuptools import setup

packages = \
['colour_checker_detection',
 'colour_checker_detection.detection',
 'colour_checker_detection.detection.tests']

package_data = \
{'': ['*'],
 'colour_checker_detection': ['examples/*',
                              'resources/colour-checker-detection-examples-datasets/*',
                              'resources/colour-checker-detection-tests-datasets/*']}

install_requires = \
['colour-science>=0.3.16,<0.4.0', 'opencv-python>=4,<5']

extras_require = \
{'development': ['biblib-simple',
                 'coverage',
                 'coveralls',
                 'flake8',
                 'invoke',
                 'jupyter',
                 'matplotlib',
                 'mock',
                 'nose',
                 'pre-commit',
                 'pytest',
                 'restructuredtext-lint',
                 'sphinx<=3.1.2',
                 'sphinx_rtd_theme',
                 'sphinxcontrib-bibtex',
                 'toml',
                 'twine',
                 'yapf==0.23'],
 'read-the-docs': ['mock', 'numpy', 'sphinxcontrib-bibtex']}

setup(
    name='colour-checker-detection',
    version='0.1.2',
    description='Colour checker detection with Python',
    long_description=codecs.open('README.rst', encoding='utf8').read(),
    author='Colour Developers',
    author_email='colour-developers@colour-science.org',
    maintainer='Colour Developers',
    maintainer_email='colour-developers@colour-science.org',
    url='https://www.colour-science.org/',
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.6,<4.0',
)
