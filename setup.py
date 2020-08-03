import os

from setuptools import find_packages
from setuptools import setup

##################################################
# Dependencies
##################################################

PYTEST_VERSION_ = '5.3.5'

# Packages required in 'production'
REQUIRED = [
    'audioread == 2.1.8',
    'numpy == 1.18.1',
    'scipy == 1.4.1',
    'librosa == 0.7.2',
    'h5py == 2.10.0',
    'networkx == 2.4',
    'requests == 2.23.0',
    'intervaltree == 3.0.2',
    'sox == 1.3.7',
    'PGet == 0.5.0',
    'numba == 0.49.1'
]

# Packages required for dev/ci enrionment
EXTRAS = {
    'dev': [
        'click==7.0',
        'pytest==%s' % (PYTEST_VERSION_,),
        'pytest-runner==5.2',
        'pytest-cov==2.8.1',
        'requests_mock==1.7.0',
        'Sphinx==2.4.4',
        'sphinx-rtd-theme==0.4.3',
        'pytest-benchmark==3.2.3',
    ],
    'ci': [
        'flake8==3.7.9',
        'flake8-quotes==2.1.1'
    ],
}

# Packages required for testing
TESTS = [
    'pytest==%s' % (PYTEST_VERSION_,),
    'requests_mock==1.7.0'
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
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
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
