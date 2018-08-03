Indirectly Supported Corpora
============================

Some corpora are hard to integrate directly, e.g. due to necessary preprocessing steps.
Here the easiest way to load such corpora is described.

Spoken Wikipedia Corpus
-----------------------

The swc corpus (https://nats.gitlab.io/swc/) is a collection of read wikipedia articles.
These audio files are not exactly transcribed.
Therefore the creators of the corpus prepared a lot of code to extract proper transcriptions.
After running their tools the corpus is in the kaldi format and therefore can be read using audiomate.

1. Download the corpus at https://nats.gitlab.io/swc/

After extraction there should be folder (named with the language of the downloaded corpus).
Every subfolder corresponds to a single wikipedia article.

2. Checkout the code of the swc corpus from https://bitbucket.org/natsuhh/swc

3. Convert audio files to wav format

For every subfolder (article) run the ``prepare_audio.py`` script from the swc-code.

.. code-block:: bash

    # Execute in the folder where all article subdirs are
    for article_dir in ./*; do python3 /path/to/swc-code/prepare_audio.py $article_dir; done

4. Build Java Tools

Switch to the ``Aligner`` folder in the swc code and run maven.
You may have to correct the maryTTS dependencies in the pom.xml (Remove the -SWC suffix),
if there is an error when running maven.

.. code-block:: bash

    mvn package

4. Run Kaldi Snippet Extraction

With the following command all the files required for kaldi are created in the given folder (/output/path).

.. code-block:: bash

    java -jar /path/to/swc-code/Aligner/target/Aligner.jar extractsnippets kaldi /dev/null /path/to/articles/ /output/path

5. Fix the wav.scp with correct audio paths.

At last the relative audio paths have to be set in the ``wav.scp`` file.
For that the following script has to be called with two arguments.
First is the path to the ``wav.scp`` file.
Second is the relative path from the ``wav.scp`` file to the directory with the article folders.

.. code-block:: python

    import sys
    import os
    import re

    scp_file = sys.argv[1]
    article_dir = sys.argv[2]

    pattern = re.compile(r'.*articles/(.*)/audio.*')

    entries = []

    with open(scp_file, 'r') as f:
        for line in f:
            parts = line.strip().split(sep=' ', maxsplit=1)
            idx = parts[0]
            match = pattern.match(parts[1])

            if match is not None:
                article_id = match.group(1)
                path = os.path.join(article_dir, article_id, 'audio.wav')
                entries.append('{} {}'.format(idx, path))

    with open(scp_file, 'w') as f:
        f.write('\n'.join(entries))
