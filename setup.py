import os

from setuptools import find_packages
from setuptools import setup

##################################################
# Dependencies
##################################################

PYTEST_VERSION_ = '4.0.0'

# Packages required in 'production'
REQUIRED = [
    'llvmlite == 0.29.0',
    'audioread == 2.1.6',
    'numpy == 1.16.2',
    'scipy == 1.2.1',
    'librosa == 0.6.3',
    'h5py == 2.9.0',
    'networkx == 2.2',
    'beautifulsoup4 == 4.7.1',
    'lxml == 4.3.2',
    'requests == 2.21.0',
    'intervaltree == 3.0.2',
]

# Packages required for dev/ci enrionment
EXTRAS = {
    'dev': [
        'click==6.7',
        'pytest==%s' % (PYTEST_VERSION_,),
        'pytest-runner==3.0',
        'pytest-cov==2.5.1',
        'requests_mock==1.4.0',
        'Sphinx==1.8.5',
        'sphinx-rtd-theme==0.4.3',
        'pytest-benchmark==3.1.1',
    ],
    'ci': [
        'flake8==3.6.0',
        'flake8-quotes==0.12.1'
    ],
}

# Packages required for testing
TESTS = [
    'pytest==%s' % (PYTEST_VERSION_,),
    'requests_mock==1.4.0'
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
      version='4.0.0',
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
          'Programming Language :: Python :: 3 :: Only',
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
