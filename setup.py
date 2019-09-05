#!/usr/bin/env python

import os
import sys

from distutils.core import setup, Extension
from Cython.Build import cythonize


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

extra_compile_args = ["-std=c++11"]
extra_link_args = ['-std=c++11']

if sys.platform == "darwin":
    extra_compile_args.append("-mmacosx-version-min=10.9")
    extra_link_args.append("-mmacosx-version-min=10.9")

extensions = [
    Extension(
        'cellbender.sampling.fingerprint_sampler',
        sources=['proto/chimera/sources/sampling/fingerprint_sampler.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
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
    ext_modules=cythonize(extensions),
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['cellbender=cellbender.base_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)
