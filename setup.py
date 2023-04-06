#!/usr/bin/env python

import os
import setuptools
from typing import List


def readme() -> str:
    with open('README.rst') as f:
        return f.read()


def _readlines(filename, filebase=''):
    with open(os.path.join(filebase, filename)) as f:
        lines = f.readlines()
    return lines


def get_requirements() -> List[str]:
    requirements = _readlines('requirements.txt')
    if 'READTHEDOCS' in os.environ:
        requirements.extend(get_rtd_requirements())
    return requirements


def get_rtd_requirements() -> List[str]:
    requirements = _readlines('requirements-rtd.txt')
    return requirements


def get_version() -> str:
    """Version number is centrally located in the file called VERSION"""
    with open('VERSION') as f:
        version = f.read().strip()
    return version


setuptools.setup(
    name='cellbender',
    version=get_version(),
    description='A software package for eliminating technical artifacts from '
                'high-throughput single-cell RNA sequencing (scRNA-seq) data',
    long_description=readme(),
    long_description_content_type='text/x-rst',
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords='scRNA-seq bioinformatics',
    url='http://github.com/broadinstitute/CellBender',
    author='Stephen Fleming, Mehrtash Babadi',
    license='BSD (3-Clause)',
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    extras_require={
        "dev": ["pytest", "scikit-learn"],
        "docs": get_rtd_requirements(),
    },
    entry_points={
        'console_scripts': ['cellbender=cellbender.base_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)
