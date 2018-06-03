.. _data-mapping:

Data Mapping
============

Since we want to have a consistent abstraction of different formats and datasets,
it is important that all data and information is mapped correctly into the python classes.

Issuer
------

The issuer holds information about the source of the audio content.
Depending on the audio content different attributes are important.
Therefore different types of issuers can be used.

Speech
    For audio content that mainly contains spoken content the :class:`audiomate.corpus.assets.Speaker` has to be used.
    This is most common for datasets regarding speech recognition/synthesis etc.

Music
    For audio content that contains music, the :class:`audiomate.corpus.assets.Artist` has to be used.

Labels
------

In the corpus data structures an utterance can have multiple label-lists. In order to access a label-list a key is used.

.. code-block:: python

    utterance = ...
    label_list = utterance.label_lists['word-transcription']

The used key should be consistent for all datasets. Audio data can be categorized on different levels of abstraction.
On top every  dataset/corpus should contain a label-list **domain**, that classifies the content in one of the following classes:

    * speech
    * music
    * noise

For every of these categories different label-list maybe defined.

speech
^^^^^^

word-transcript
    Non-aligned transcription of speech.

word-transcript-raw
    Non-aligned transcription of speech. Used for unprocessed transcriptions (e.g. containing punctuation, ...).

word-transcript-aligned
    Aligned transcription of speech. The begin and end of the words is defined.
    Every word is a single label in the label-list.

phone-transcript
    Non-aligned transcription of phones.

phone-transcript-aligned
    Aligned transcription of phones. Begin and end of phones is defined.

music
^^^^^

genre
    The genre of the music.

noise
^^^^^

sound-class
    Labels defining any sound-event, acoustic-scene, environmental noise, ...
    e.g. siren, dog_bark, train, car, snoring ...


This list isn't complete. Please open an issue for any additional domains/classes that maybe needed.
