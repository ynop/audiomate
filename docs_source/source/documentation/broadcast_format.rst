Broadcast Format
================

The broadcast format is basically the same as :ref:`section_default_format`, except it uses another format to store labels.
This format is meant for data where not many utterances are given, but with a lot of labels. So instead to have all labels per label-list in one file,
a label-file per utterance is used.

**labels.txt**

This files defines where to find the effective label files. It stores the label-file path per utterance. Additionaly a label-list-id can be given, if there are multiple label-lists per utterance.

.. code-block:: bash

    <utt-id> <label-file-path> <label-list-idx>

Example:

.. code-block:: bash

    utt-1 files/a/labels.txt
    utt-2 files/b/music.txt music
    utt-2 files/b/jingles.txt jingles
    utt-3 files/c/trailers.txt

**[label-file].txt**

The label files reference by the *labels.txt* are in the following format. It contains the start and end in seconds.
The values are **Tab-separated**.

.. code-block:: bash

    <start> <end>   <value>

Example:

.. code-block:: bash

    0	40  hallo
    40.5    100 velo
    102.4   109.2   auto
