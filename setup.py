import os

from setuptools import find_packages
from setuptools import setup

##################################################
# Dependencies
##################################################

PYTEST_VERSION_ = '4.0.0'

# Packages required in 'production'
REQUIRED = [
    'audioread >= 2.1.0',
    'numpy >= 1.14.0',
    'scipy >= 1.1.0',
    'librosa >= 0.6.0',
    'h5py >= 2.7.1',
    'networkx >= 2.0',
    'beautifulsoup4 >= 4.6.0',
    'lxml >= 4.1.1',
    'requests >= 2.18.4'
]

# Packages required for dev/ci enrionment
EXTRAS = {
    'dev': [
        'click==6.7',
        'pytest==%s' % (PYTEST_VERSION_,),
        'pytest-runner==3.0',
        'pytest-cov==2.5.1',
        'requests_mock==1.4.0',
        'Sphinx==1.6.5',
        'sphinx-rtd-theme==0.2.5b1'
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
      version='3.0.0',
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
