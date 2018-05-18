# AUDIOMATE

[![Run Status](https://api.shippable.com/projects/5a1d31821e6eda0700091230/badge?branch=master)](https://app.shippable.com/github/ynop/audiomate)
[![Coverage Badge](https://api.shippable.com/projects/5a1d31821e6eda0700091230/coverageBadge?branch=master)](https://app.shippable.com/github/ynop/audiomate)

Audiomate is a library for easy access to audio datasets. It provides the datastructures for accessing/loading different datasets in a generic way. This should ease the use of audio datasets for example for machine learning tasks.

Documentation: https://ynop.github.io/audiomate/
Examples: https://github.com/ynop/audiomate/tree/master/examples

## Installation

Install the latest development version:

```sh
pip install git+https://github.com/ynop/audiomate.git
```

## Development

### Prerequisites

* [A supported version of Python 3](https://docs.python.org/devguide/index.html#status-of-python-branches)

It's recommended to use a virtual environment when developing audiomate. To create one, execute the following command in the project's root directory:

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
bump2version [major,minor,patch]
```


