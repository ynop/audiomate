Changelog
=========

Next Version
------------

**Breaking Changes**

**New Features**

* Added writer (:class:`audiomate.corpus.io.NvidiaJasperWriter`) for
  `NVIDIA Jasper <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper>`_.

**Fixes**

v5.0.0
------

**Breaking Changes**

* Changed :class:`audiomate.corpus.validation.InvalidItemsResult` to use it not only for Utterances, but also for Tracks for example.

* Refactoring and addition of splitting functions in the :class:`audiomate.corpus.subset.Splitter`.

**New Features**

* Added :class:`audiomate.corpus.validation.TrackReadValidator` to check for corrupt audio tracks/files.

* Added reader (:class:`audiomate.corpus.io.FluentSpeechReader`) for the
  `Fluent Speech Commands Dataset <http://www.fluent.ai/research/fluent-speech-commands/>`_.

* Added functions to check for contained tracks and issuers (:meth:`audiomate.corpus.CorpusView.contains_track`, :meth:`audiomate.corpus.CorpusView.contains_issuer`).

* Multiple options for controlling the behavior of the :class:`audiomate.corpus.io.KaldiWriter`.

* Added writer (:class:`audiomate.corpus.io.Wav2LetterWriter`) for the
  `wav2letter engine <https://github.com/facebookresearch/wav2letter/>`_.

* Added module with functions to read/write sclite trn files (:mod:`audiomate.formats.trn`).

**Fixes**

* Improved performance of Tuda-Reader (:class:`audiomate.corpus.io.TudaReader`).

* Added wrapper for the ```audioread.audio_open``` function (:mod:`audiomate.utils.audioread`) to cache available
  backends. This speeds up audioopen operations a lot.

* Performance improvements, especially for importing utterances, merging, subviews.

v4.0.1
------

**Fixes**

* Fix :class:`audiomate.corpus.io.CommonVoiceReader` to use correct file-extension of the audio files.

v4.0.0
------

**Breaking Changes**

* For utterances and labels ``-1`` was used for representing that the end is the same as the end of the parent utterance/track.
  In order to prevent ``-1`` checks in different methods/places ``float('inf')`` is now used.
  This makes it easier to implement stuff like label overlapping.

* :class:`audiomate.annotations.LabelList` is now backed by an interval-tree instead of a simple list. Therefore the labels have no fixed order anymore. The interval-tree provides functionality for operations like merging, splitting, finding overlaps with much lower code complexity.

* Removed module :mod:`audiomate.annotations.label_cleaning`, since those methods are available on :class:`audiomate.annotations.LabelList` directly.

**New Features**

* Added reader (:class:`audiomate.corpus.io.RouenReader`) and
  downloader (:class:`audiomate.corpus.io.RouenDownloader`) for the
  `LITIS Rouen Audio scene dataset <https://sites.google.com/site/alainrakotomamonjy/home/audio-scene>`_.

* Added downloader (:class:`audiomate.corpus.io.AEDDownloader`) for the
  `Acoustic Event Dataset <https://data.vision.ee.ethz.ch/cvl/ae_dataset/>`_.

* [`#69 <https://github.com/ynop/audiomate/issues/69>`_] Method to get labels within range: :meth:`audiomate.annotations.LabelList.labels_in_range`.

* [`#68 <https://github.com/ynop/audiomate/issues/68>`_] Add convenience method to create Label-List with list of label values: :meth:`audiomate.annotations.LabelList.with_label_values`.

* [`#61 <https://github.com/ynop/audiomate/issues/61>`_] Added function to split utterances of a corpus into multiple utterances with a maximal duration:
  :meth:`audiomate.corpus.CorpusView.split_utterances_to_max_time`.

* Add functions to check for overlap between labels: :meth:`audiomate.annotations.Label.do_overlap` and
  :meth:`audiomate.annotations.Label.overlap_duration`.

* Add function to merge equal labels that overlap within a label-list:
  :meth:`audiomate.annotations.LabelList.merge_overlapping_labels`.

* Added reader (:class:`audiomate.corpus.io.AudioMNISTReader`) and
  downloader (:class:`audiomate.corpus.io.AudioMNISTDownloader`) for the
  `AudioMNIST dataset <https://github.com/soerenab/AudioMNIST>`_.


**Fixes**

* [`#76 <https://github.com/ynop/audiomate/issues/76>`_][`#77 <https://github.com/ynop/audiomate/issues/77>`_][`#78 <https://github.com/ynop/audiomate/issues/78>`_] Multiple fixes on KaldiWriter


v3.0.0
------

**Breaking Changes**

* Moved label-encoding to its own module (:mod:`audiomate.encoding`).
  It now provides the processing of full corpora and store it in containers.

* Moved :class:`audiomate.feeding.PartitioningFeatureIterator` to the :mod:`audiomate.feeding` module.

* Added :class:`audiomate.containers.AudioContainer` to store audio tracks
  in a single file. All container classes are now in a separate module
  :mod:`audiomate.containers`.

* Corpus now contains Tracks not Files anymore. This makes it possible to
  different kinds of audio sources. Audio from a file is now included using
  :class:`audiomate.tracks.FileTrack`. New is the
  :class:`audiomate.tracks.ContainerTrack`, which reads data stored in
  a container.

* The :class:`audiomate.corpus.io.DefaultReader` and the
  :class:`audiomate.corpus.io.DefaultWriter` now load and store tracks,
  that are stored in a container.

* All functionality regarding labels was moved to its own module
  :mod:`audiomate.annotations`.

* The class :class:`audiomate.tracks.Utterance` was moved to the tracks module.

**New Features**

* Introducing the :mod:`audiomate.feeding` module. It provides different tools for accessing container data.
  Via a :class:`audiomate.feeding.Dataset` data can be accessed by indices.
  With a :class:`audiomate.feeding.DataIterator` one can easily iterate over data, such as frames.

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

* Added :meth:`audiomate.corpus.assets.Label.tokenized` to get a list of tokens from a label.
  It basically splits the value and trims whitespace.

* Added methods on :class:`audiomate.corpus.CorpusView`, :class:`audiomate.corpus.assets.Utterance`
  and :class:`audiomate.corpus.assets.LabelList` to get a set of occurring tokens.

* Added :class:`audiomate.encoding.TokenOrdinalEncoder` to encode labels of an utterance
  by mapping every token of the label to a number.

* Create container base class (:class:`audiomate.corpus.assets.Container`), that can be used to store arbitrary data
  per utterance. The :class:`audiomate.corpus.assets.FeatureContainer` is now an extension of the container,
  that provides functionality especially for features.

* Added functions to split utterances and label-lists into multiple parts.
  (:meth:`audiomate.corpus.assets.Utterance.split`, :meth:`audiomate.corpus.assets.LabelList.split`)

* Added :class:`audiomate.processing.pipeline.AddContext` to add context to frames,
  using previous and subsequent frames.

* Added reader (:class:`audiomate.corpus.io.MailabsReader`) and
  downloader (:class:`audiomate.corpus.io.MailabsDownloader`) for the
  `M-AILABS Speech Dataset <http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/>`_.

**Fixes**

* [`#58 <https://github.com/ynop/audiomate/issues/58>`_] Keep track of number of samples per frame and between frames.
  Now the correct values will be stored in a Feature-Container, if the processor implements it correctly.

* [`#72 <https://github.com/ynop/audiomate/issues/72>`_] Fix bug, when reading samples from utterance,
  using a specific duration, while the utterance end is not defined.

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
