#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
from setuptools import setup

sys.path.insert(0, "src/nirhiss")
from version import __version__


long_description = \
    """
nirHiss is a python package to reduce JWST NIRISS observations.

Read the documentation at https://adina.feinste.in/nirHiss

Changes to v0.0.1rc1 (2022-08-06):
*
"""

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='nirhiss',
    version=__version__,
    license='MIT',
    author='Adina D. Feinstein',
    author_email='adina.d.feinstein@gmail.com',
    packages=[
        'nirhiss',
        ],
    include_package_data=True,
    url='http://github.com/afeinstein20/nirhiss',
    description='For reducing JWST NIRISS observations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['README.md', 'LICENSE']},
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        ],
    )
