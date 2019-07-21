#!/usr/bin/env python

import os
import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements_filename():
    if 'READTHEDOCS' in os.environ:
        return "REQUIREMENTS-RTD.txt"
    else:
        return "REQUIREMENTS.txt"


install_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), get_requirements_filename()))
]

setuptools.setup(
    name='cellbender',
    version='0.1',
    description='A software package for pre-processing and denoising '
                'scRNA-seq data',
    long_description=readme(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research'
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
      'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords='scRNA-seq bioinformatics',
    url='http://github.com/broadinstitute/CellBender',
    author='Stephen Fleming',
    author_email='sfleming@broadinstitute.org',
    license='MIT',
    packages=['cellbender'],
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['cellbender=cellbender.base_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)
