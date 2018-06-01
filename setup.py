from setuptools import find_packages
from setuptools import setup

PYTEST_VERSION_ = '3.3.0'

setup(name='audiomate',
      version='1.0.0',
      description='Audiomate is a library for working with audio datasets.',
      long_description='Audiomate is a library for easy access to audio datasets. '
                       'It provides the datastructures for accessing/loading different datasets in a generic way. '
                       'This should ease the use of audio datasets for example for machine learning tasks.',
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
      install_requires=[
          'audioread >= 2.1.0',
          'numpy >= 1.14.0',
          'scipy >= 1.1.0',
          'librosa >= 0.6.0',
          'h5py >= 2.7.1',
          'networkx >= 2.0',
          'beautifulsoup4 >= 4.6.0',
          'lxml >= 4.1.1',
          'requests >= 2.18.4'
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='tests',
      extras_require={
          'dev': [
              'click==6.7',
              'pytest==%s' % (PYTEST_VERSION_,),
              'pytest-runner==3.0',
              'pytest-cov==2.5.1',
              'requests_mock==1.4.0',
              'Sphinx==1.6.5',
              'sphinx-rtd-theme==0.2.5b1'
          ],
          'ci': ['flake8==3.5.0', 'flake8-quotes==0.12.1'],
      },
      setup_requires=['pytest-runner'],
      tests_require=[
          'pytest==%s' % (PYTEST_VERSION_,),
          'requests_mock==1.4.0'
      ],
      entry_points={
      }
      )
