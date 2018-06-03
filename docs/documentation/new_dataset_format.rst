Add Dataset/Format
==================

In this section it is described how to a downloader, reader or writer for a new dataset or another corpus format.
The implementation is pretty straight-forward. For examples checkout some of the existing implementations at
:mod:`audiomate.corpus.io`.

.. IMPORTANT::

    * Use the same name (``type()`` method) for downloader/reader/writer.
    * Import your components in :mod:`audiomate.corpus.io.__init___`. So all components are available from the io module.
    * Checkout :ref:`data-mapping` on what and how to add info/data when reading a corpus.

Corpus Downloader
-----------------
If we aim to load some specific dataset/corpus, a downloader can be implemented,
if it is possible to automate the whole download process. First we create a new class that inherits from
:class:`audiomate.corpus.io.CorpusDownloader`. There we have to implement two methods.
The ``type`` method just has to return a string with the name of the new dataset/format.
The ``_download`` method will do the heavy work of download all the files to the path ``target_path``.

.. code-block:: python

    from audiomate.corpus.io import base


    class MyDownloader(base.CorpusDownloader):

        @classmethod
        def type(cls):
            return 'MyDataset'

        def _download(self, target_path):
            # Download the data to target_path





Corpus Reader
-------------

The reader is the one component that is mostly used. Either for a specific dataset/corpus or a custom format,
a reader is most likely to be required. First we create a new class that inherits from
:class:`audiomate.corpus.io.CorpusReader`. There we have to implement three methods.
The ``type`` method just has to return a string with the name of the new dataset/format.
The ``_check_for_missing_files`` method can be used to check if the given path is a valid input.
For example if the format/dataset requires some specific meta-files it can be check here if they are available.
Finally in the ``_load`` method the actual loading is done and the loaded corpus is returned.

.. code-block:: python

    from audiomate.corpus.io import base


    class MyReader(base.CorpusReader):

        @classmethod
        def type(cls):
            return 'MyDataset'

        def _check_for_missing_files(self, path):
            # Check the path for missing files that are required to read with this reader.
            # Return a list of missing files
            return []

        def _load(self, path):
            # Create a new corpus
            corpus = audiomate.Corpus(path=path)

            # Create files ...
            corpus.new_file(file_path, file_idx)

            # Issuers ...
            issuer = assets.Speaker(issuer_idx)
            corpus.import_issuers(issuer)

            # Utterances with labels ...
            utterance = corpus.new_utterance(file_idx, file_idx, issuer_idx)
            utterance.set_label_list(assets.LabelList(idx='transcription', labels=[
                assets.Label(str(digit))
            ]))

            return corpus



Corpus Writer
-------------

A writer is only useful for custom formats. For a specific dataset a writer is most likely not needed.
First we create a new class that inherits from :class:`audiomate.corpus.io.CorpusWriter`.
There we have to implement two methods.
The ``type`` method just has to return a string with the name of the new dataset/format.
The ``_save`` method does the serialization of the given corpus to the given path.


.. code-block:: python

    from audiomate.corpus.io import base


    class DefaultWriter(base.CorpusWriter):

        @classmethod
        def type(cls):
            return 'MyDataset'

        def _save(self, corpus, path):
            # Do the serialization
