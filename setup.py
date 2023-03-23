#!/usr/bin/env python

import os
import setuptools


def readme() -> str:
    with open('README.rst') as f:
        return f.read()


def get_requirements() -> str:
    filebase = os.path.dirname(__file__)

    def _readlines(filename):
        with open(os.path.join(filebase, filename)) as f:
            lines = f.readlines()
        return lines

    requirements = _readlines('REQUIREMENTS.txt')
    if 'READTHEDOCS' in os.environ:
        requirements.extend(_readlines('REQUIREMENTS-RTD.txt'))
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
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['cellbender=cellbender.base_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)
