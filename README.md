# AUDIOMATE

[![PyPI](https://img.shields.io/pypi/v/audiomate.svg)](https://pypi.python.org/pypi/audiomate)
[![Run Status](https://api.shippable.com/projects/5a1d31821e6eda0700091230/badge?branch=master)](https://app.shippable.com/github/ynop/audiomate)
[![Coverage Badge](https://api.shippable.com/projects/5a1d31821e6eda0700091230/coverageBadge?branch=master)](https://app.shippable.com/github/ynop/audiomate)
[![Documentation Status](https://readthedocs.org/projects/audiomate/badge/?version=latest)](https://audiomate.readthedocs.io/en/latest/?badge=latest)

Audiomate is a library for easy access to audio datasets.
It provides the datastructures for accessing/loading different datasets in a generic way.
This should ease the use of audio datasets for example for machine learning tasks.

Documentation: https://audiomate.readthedocs.io  
Examples: https://github.com/ynop/audiomate/tree/master/examples

Currently supported datasets:
* [Acoustic Event Dataset](https://data.vision.ee.ethz.ch/cvl/ae_dataset/)
* [Mozilla Common Voice](https://voice.mozilla.org/)
* [ESC-50](https://github.com/karoldvl/ESC-50)
* [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
* [German Distant Speech Corpus](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html)
* [Google Speech Commands](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)
* [GTZAN](https://marsyasweb.appspot.com/download/data_sets/)
* [M-AILABS Speech Dataset](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/)
* [MUSAN](http://www.openslr.org/17/)
* [Tatoeba](https://tatoeba.org/)
* [TIMIT](https://github.com/philipperemy/timit)
* [Urbansound8k](http://urbansounddataset.weebly.com/urbansound8k.html)
* [Voxforge](http://www.voxforge.org/de)

Currently supported formats:
* [Kaldi](http://kaldi-asr.org/)
* [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech)
* [Custom Formats](https://ynop.github.io/audiomate/documentation/formats.html)

Indirectly supported datasets ([Details](https://audiomate.readthedocs.io/en/v1.0.0/documentation/indirect_support.html)):
* [Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/)

## Installation

```sh
pip install audiomate
```

Install the latest development version:

```sh
pip install git+https://github.com/ynop/audiomate.git
```

## Development

### Prerequisites

* [A supported version of Python 3](https://docs.python.org/devguide/index.html#status-of-python-branches)

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
python setup.py test
```

With PyCharm you might have to change the default test runner. Otherwise, it might only suggest to use nose. To do so, go to File > Settings > Tools > Python Integrated Tools (on the Mac it's PyCharm > Preferences > Settings > Tools > Python Integrated Tools) and change the test runner to py.test.

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


