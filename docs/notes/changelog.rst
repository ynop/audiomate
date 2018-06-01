Changelog
=========

Next Version
------------


v1.0.0
------

**Breaking Changes**

* The (pre)processing module has moved to :mod:`audiomate.processing`. It now supports online processing in chunks.
  For this purpose a pipeline step can require context.
  The pipeline automatically buffers data, until enough frames are ready.

**New Features**

* Added downloader (:class:`audiomate.corpus.io.FreeSpokenDigitDownloader`) and
  reader (:class:`audiomate.corpus.io.FreeSpokenDigitReader`) for the
  `Free-Spoken-Digit-Dataset <https://github.com/Jakobovski/free-spoken-digit-dataset>`_.


v0.1.0
------

Initial release
