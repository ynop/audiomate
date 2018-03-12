.. _section_default_format:

Default Format
==============

This describes, how a corpus with the default format is saved on disk. Every corpus is a folder with a bunch of files.

**files.txt**

This file contains a list of every audio file in the corpus. Every file is identified by a unique id.
Every line in the file contains the mapping from file-id to the file-path for a single file. The filepath is the path to the audio file relative to the corpus folder.

.. code-block:: bash

    <recording-id> <wav-file-path>

Example:

.. code-block:: bash

    2014-03-17-09-45-16_Kinect-Beam train/2014-03-17-09-45-16_Kinect-Beam.wav
    2014-03-17-09-45-16_Realtek train/2014-03-17-09-45-16_Realtek.wav
    2014-03-17-09-45-16_Yamaha train/2014-03-17-09-45-16_Yamaha.wav
    2014-03-17-10-26-07_Realtek train/2014-03-17-10-26-07_Realtek.wav


**utterances.txt**

This file contains all utterances in the corpus. An utterance is a part of a file (A file can contain one or more utterances).
Every line in this file defines a single utterance, which consists of utterance-id, file-id, start and end. Start and end are measured in seconds within the file.
If end is -1 it is considered to be the end of the file (If the utterance is the full length of the file, start and end are 0/-1).

.. code-block:: bash

    <utterance-id> <recording-id> <start> <end>

Example:

.. code-block:: bash

    1_hello 2014-03-17-09-45-16_Kinect-Beam
    1_hello_sam 2014-03-17-09-45-16_Realtek 0 -1
    2_this_is 2014-03-17-09-45-16_Yamaha 0 5
    3_goto 2014-03-17-09-45-16_Yamaha 5 -1

**utt_issuers.txt**

This file contains the mapping from utterance to issuers, which gives the information who/what is the origin of a given utterance (e.g. the speaker).
Every line contains one mapping from utterance-id to issuer-id.

.. code-block:: bash

    <utterance-id> <issuer-id>

Example:

.. code-block:: bash

    1_hello marc
    1_hello_sam marc
    2_this_is sam
    3_goto jenny

**labels_[x].txt**

There can be multiple label-lists in a corpus (e.g. text-transcription, raw-text-transcription - with punctuation, audio classification type, ...).
Every label-list is saved in a separate file with the prefix *labels_*.
A single file contains labels of a specific type for all utterances. A label-list of an utterance can contain one or more labels (e.g. in a text segmentation every word could be a label).
A label optionally can have a start and end time (in seconds within the utterance). For labels without start/end defined 0/-1 is set.
Every line in the file defines one label. The labels are stored in order per utterance (e.g. 1. word, 2. word, 3. word, ...).
Optionally addtional meta-information can be stored per label. This has to be a json string in square brackets.

.. code-block:: bash

    <utterance-id> <start> <end> <label-value> [<label-meta>]

Example:

.. code-block:: bash

    1_hello 0 -1 hi
    1_hello 0 -1 this
    1_hello 0 -1 is
    1_hello_sam 0 -1 hello
    1_hello_sam 0 -1 sam
    2_this_is 0 -1 this
    2_this_is 0 -1 is [{"prio": 3}]
    2_this_is 0 -1 me [{"stress": true}]
    3_goto 0 -1 go
    3_goto 0 -1 to
    3_goto 0 -1 the
    3_goto 0 -1 mall

**features.txt**

Contains a list of stored features. A corpus can have different feature containers. Every container contains the features of all utterances of a given type (e.g. MFCC features).
A feature container is a h5py file which contains a dataset per utterance. Every line contains one container of features.

.. code-block:: bash

    <feature-name> <relative-path>

Example:

.. code-block:: bash

    mfcc mfcc_features
    fbank fbank_features
