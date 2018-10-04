Changelog
=========

Next Version
------------

**Breaking Changes**

* Moved label-encoding to its own module (:mod:`audiomate.encoding`).
  It now provides the processing of full corpora and store it in containers.

**New Features**

* Added processing steps for computing Onset-Strength (:class:`audiomate.processing.pipeline.OnsetStrength`))
  and Tempogram (:class:`audiomate.processing.pipeline.Tempogram`)).

* Introduced :class:`audiomate.corpus.validation` module, that is used to validate a corpus.

* Added reader (:class:`audiomate.corpus.io.SWCReader`) for the
  `SWC corpus <https://audiomate.readthedocs.io/en/latest/documentation/indirect_support.html>`_.
  But it only works for the prepared corpus.

* Added function (:func:`audiomate.corpus.utils.label_cleaning.merge_consecutive_labels_with_same_values`)
  for merging consecutive labels with the same value

* Added downloader (:class:`audiomate.corpus.io.GtzanDownloader`) for the
  `GTZAN Music/Speech <https://marsyasweb.appspot.com/download/data_sets/>`_.

**Non-Breaking Changes**

* Create container base class (:class:`audiomate.corpus.assets.Container`), that can be used to store arbitrary data
  per utterance. The :class:`audiomate.corpus.assets.FeatureContainer` is now an extension of the container,
  that provides functionality especially for features.

**Fixes**

* [`#58 <https://github.com/ynop/audiomate/issues/58>`_] Keep track of number of samples per frame and between frames.
  Now the correct values will be stored in a Feature-Container, if the processor implements it correctly.

v2.0.0
------

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

* Added downloader (:class:`audiomate.corpus.io.MusanDownloader`) for the
  `Musan Corpus <http://www.openslr.org/17/>`_.

* Added constants for common label-list identifiers/keys in :mod:`audiomate.corpus`.

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
