Changelog
=========

Next Version
------------

**Breaking Changes**

* Update various readers to use the correct label-list identifiers as defined
  in :ref:`data-mapping`.

**New Features**

* Added downloader (:class:`audiomate.corpus.io.TatoebaDownloader`) and
  reader (:class:`audiomate.corpus.io.TatoebaReader`) for the
  `Tatoeba platform <https://tatoeba.org/>`_.

* Added downloader (:class:`audiomate.corpus.io.CommonVoiceDownloader`) and
  reader (:class:`audiomate.corpus.io.CommonVoiceReader`) for the
  `Common Voice Corpus <https://voice.mozilla.org/>`_.

* Added processing steps :class:`audiomate.processing.pipeline.AvgPool` and
  :class:`audiomate.processing.pipeline.VarPool` for computing average and variance over
  a given number of sequential frames.

* Added constants for common label-list identifiers/keys in :mod:`audiomate.corpus.assets`.

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
