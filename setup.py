from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='cellbender',
      version='0.1',
      description='Principled scRNA-seq data analysis using variational '
                  'inference within a probabilistic model',
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
      install_requires=[
          'numpy',
          'scipy',
          'tables',
          'torch',
          'torchvision',
          'pyro-ppl==0.3.0',
          'scikit-learn',
          'matplotlib',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points={
          'console_scripts': ['cellbender=cellbender.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)
