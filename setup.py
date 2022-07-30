import os

from setuptools import find_packages
from setuptools import setup

##################################################
# Dependencies
##################################################

PYTEST_VERSION_ = '7.1.2'

# Packages required in 'production'
REQUIRED = [
    'audioread == 2.1.9',
    'numpy == 1.21.6',
    'scipy == 1.7.3',
    'librosa == 0.9.2',
    'h5py == 3.7.0',
    'networkx == 2.6.3',
    'requests == 2.28.1',
    'intervaltree == 3.1.0',
    'sox == 1.4.1',
    'PGet == 0.5.1',
    'numba == 0.56.0'
]

# Packages required for dev/ci enrionment
EXTRAS = {
    'dev': [
        'click == 8.1.3',
        f"pytest == {PYTEST_VERSION_}",
        'pytest-runner == 6.0',
        'pytest-cov == 3.0.0',
        'requests_mock == 1.9.3',
        'Sphinx == 5.1.1',
        'sphinx-rtd-theme == 1.0.0',
        'pytest-benchmark == 3.4.1'
    ],
    'ci': [
        'flake8 == 4.0.1',
        'flake8-quotes == 3.3.1'
    ],
}

# Packages required for testing
TESTS = [
    f"pytest == {PYTEST_VERSION_}",
    'requests_mock == 1.9.3'
]

##################################################
# Description
##################################################

DESCRIPTION = 'Audiomate is a library for working with audio datasets.'

root = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(root, 'README.md')

# Import the README and use it as the long-description.
try:
    with open(readme_path, encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

##################################################
# SETUP
##################################################

setup(name='audiomate',
      version='6.0.0',
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ynop/audiomate',
      download_url='https://github.com/ynop/audiomate/releases',
      author='Matthias Buechi, Andreas Ahlenstorf',
      author_email='buec@zhaw.ch',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Scientific/Engineering :: Human Machine Interfaces'
      ],
      keywords='audio music sound corpus dataset',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      install_requires=REQUIRED,
      include_package_data=True,
      zip_safe=False,
      test_suite='tests',
      extras_require=EXTRAS,
      setup_requires=['pytest-runner'],
      tests_require=TESTS,
      entry_points={}
      )
