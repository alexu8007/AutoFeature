#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
from setuptools import setup, find_packages

# Package meta-data
NAME = 'autofeature'
DESCRIPTION = 'Automated Feature Engineering Framework'
URL = 'https://github.com/user/autofeature'
EMAIL = 'author@example.com'
AUTHOR = 'AutoFeature Team'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = '0.1.0'

# Required packages
REQUIRED = [
    'numpy>=1.19.0',
    'pandas>=1.0.0',
    'scikit-learn>=0.24.0',
    'scipy>=1.5.0',
    'matplotlib>=3.3.0',
    'seaborn>=0.11.0',
]

# Optional packages
EXTRAS = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'flake8>=3.8.0',
        'black>=20.8b1',
        'isort>=5.0.0',
        'mypy>=0.800',
        'sphinx>=3.0.0',
        'sphinx-rtd-theme>=0.5.0',
    ],
    'nlp': [
        'nltk>=3.5',
        'spacy>=3.0.0',
    ],
}

# Get long description from README.md
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
) 