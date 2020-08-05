# AUDIOMATE

[![PyPI](https://img.shields.io/pypi/v/audiomate.svg)](https://pypi.python.org/pypi/audiomate)
[![Build Status](https://travis-ci.com/ynop/audiomate.svg?branch=master)](https://travis-ci.com/ynop/audiomate)
[![Documentation Status](https://readthedocs.org/projects/audiomate/badge/?version=latest)](https://audiomate.readthedocs.io/en/latest/?badge=latest)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/ynop/audiomate/?ref=repository-badge)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02135/status.svg)](https://doi.org/10.21105/joss.02135)

Audiomate is a library for easy access to audio datasets.
It provides the datastructures for accessing/loading different datasets in a generic way.
This should ease the use of audio datasets for example for machine learning tasks.

```python
import audiomate
from audiomate.corpus import io

# Download a dataset
esc_downloader = io.ESC50Downloader()
esc_downloader.download('/local/path')

# Load and work with the dataset
esc50 = audiomate.Corpus.load('/local/path', reader='esc-50')

# e.g. Read the audio signal and the label of specific sample/utterance
utterance = esc50.utterances['1-100032-A-0']
samples = utterance.read_samples()
label_list = utterance.label_lists[audiomate.corpus.LL_SOUND_CLASS]

for label in label_list:
  print(label.start, label.value)
```

Furthermore it provides tools for interacting with datasets
(validation, splitting, subsets, merge, filter), extracting features,
feeding samples for training ML models and more.

* [Documentation](https://audiomate.readthedocs.io)
* [Examples](https://github.com/ynop/audiomate/tree/master/examples)
* [Changelog](https://audiomate.readthedocs.io/en/latest/notes/changelog.html)

Currently supported datasets:
* [Acoustic Event Dataset](https://arxiv.org/pdf/1604.07160.pdf)
* [AudioMNIST](https://github.com/soerenab/AudioMNIST)
* [Mozilla Common Voice](https://voice.mozilla.org/)
* [ESC-50](https://github.com/karoldvl/ESC-50)
* [Fluent Speech Commands](http://www.fluent.ai/research/fluent-speech-commands/)
* [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
* [German Distant Speech Corpus](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html)
* [Google Speech Commands](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)
* [GTZAN](http://marsyas.info/downloads/datasets.html)
* [LibriSpeech](https://www.openslr.org/12/)
* [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)
* [MUSAN](http://www.openslr.org/17/)
* [LITIS Rouen Audio scene dataset](https://sites.google.com/site/alainrakotomamonjy/home/audio-scene)
* [Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/)
* [Tatoeba](https://tatoeba.org/)
* [TIMIT](https://github.com/philipperemy/timit)
* [Urbansound8k](http://urbansounddataset.weebly.com/urbansound8k.html)
* [Voxforge](http://www.voxforge.org/de)

Currently supported formats:
* [Kaldi](http://kaldi-asr.org/)
* [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech)
* [Wav2Letter](https://github.com/facebookresearch/wav2letter)
* [NVIDIA Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper)
* [Custom Formats](https://audiomate.readthedocs.io/en/latest/documentation/formats.html)

## Installation

```sh
pip install audiomate
```

Install the latest development version:

```sh
pip install git+https://github.com/ynop/audiomate.git
```

### Dependencies

#### sox
For parts of the functionality (e.g. audio format conversion) [sox](http://sox.sourceforge.net) is used. In order to use it, you have to install sox.

```sh
# macos
brew install sox

# with support for specific formats
brew install sox --with-lame --with-flac --with-libvorbis

# linux
apt-get install sox

# anaconda for macOS/windows/linux:
conda install -c conda-forge sox
```

## Development

### Prerequisites

* [A supported version of Python > 3.5](https://docs.python.org/devguide/index.html#status-of-python-branches)

It's recommended to use a virtual environment when developing audiomate.
To create one, execute the following command in the project's root directory:

```
python -m venv .
```

To install audiomate and all it's dependencies, execute:

```
pip install -e .
```

### Running the test suite

```
pip install -e .[dev]
pytest
```

With PyCharm you might have to change the default test runner. Otherwise, it might only suggest to use nose. To do so, go to File > Settings > Tools > Python Integrated Tools (on the Mac it's PyCharm > Preferences > Settings > Tools > Python Integrated Tools) and change the test runner to py.test.

### Benchmarks

In order to check the runtime of specific parts, ``pytest-benchmark`` is used. Benchmarks are normal test functions, but call the benchmark fixture for the code under test.

To run benchmarks:

```
# Run all
pytest bench

# Specific benchmark
pytest bench/corpus/test_merge_corpus.py
```

To compare between different runs:

```
pytest-benchmark compare
```

### Editing the Documentation

The documentation is written in [reStructuredText](http://docutils.sourceforge.net/rst.html) and transformed into various output formats with the help of [Sphinx](http://www.sphinx-doc.org/).

* [Syntax reference reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html)
* [Sphinx-specific additions to reStructuredText](http://www.sphinx-doc.org/en/stable/markup/index.html)

To generate the documentation, execute:

```
pip install -e .[dev]
cd docs
make html
```

The generated files are written to `docs/_build/html`.

### Versions

Versions is handled using [bump2version](https://github.com/c4urself/bump2version). To bump the version:

```
bump2version [major,minor,patch,release,num]
```

In order to directly go to a final relase version (skip .dev/.rc/...):

```
bump2version [major,minor,patch] --new-version x.x.x
```

### Release

Commands to create a new release on pypi.

```
rm -rf build
rm -rf dist

python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
