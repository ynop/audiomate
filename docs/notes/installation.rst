Installation
============

Install the latest stable version::

    pip install audiomate

Install the latest development version::

    pip install git+https://github.com/ynop/audiomate.git

Dependencies
------------

**sox**

For parts of the functionality (e.g. audio format conversion) `sox <http://sox.sourceforge.net>`_ is used. In order to use it, you have to install sox.

.. code-block:: bash

   # macos
   brew install sox

   # with support for specific formats
   brew install sox --with-lame --with-flac --with-libvorbis

   # linux
   apt-get install sox

   # anaconda for macOS/windows/linux:
   conda install -c conda-forge sox