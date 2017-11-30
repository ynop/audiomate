# PINGU

Pingu is a library for easy access to audio datasets. It provides the datastructures for accessing/loading different datasets in a generic way. This should ease the use of audio datasets for example for machine learning tasks.

Documentation: https://ynop.github.io/pingu/

## Installation

Install the latest development version:

```sh
pip install git+https://github.com/ynop/pingu.git
```

## Example
Example for loading a corpus and using the FramedSignalGrabber to retrieve the audio signal in frames.

```python
>>> ds = Corpus.load('/path/to/the/dataset', loader='default')
>>> grabber = FramedSignalGrabber(ds, label_list_idx='music', frame_length=400, hop_size=160)
>>>
>>> # Every frame contains the actual signal and a vector defining the active labels
>>> for frame in grabber:
>>>     print(frame)
(array([-0.00317392,  0.00866726,  0.01651051, ..., -0.01336711,
   -0.01263466, -0.01232948], dtype=float32), array([ 0.,  0.,  1.,  0.], dtype=float32))
...
```

## Development

### Prerequisites

* [A supported version of Python 3](https://docs.python.org/devguide/index.html#status-of-python-branches)

It's recommended to use a virtual environment when developing Pingu. To create one, execute the following command in the project's root directory:

```
python -m venv .
```

To install Pingu and all it's dependencies, execute:

```
pip install -e .
```

### Running the test suite

```
pip install -e .[test]
python setup.py test
```

### Editing the Documentation

The documentation is written in [reStructuredText](http://docutils.sourceforge.net/rst.html) and transformed into various output formats with the help of [Sphinx](http://www.sphinx-doc.org/).

* [Syntax reference reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html)
* [Sphinx-specific additions to reStructuredText](http://www.sphinx-doc.org/en/stable/markup/index.html)

To generate the documentation, execute:

```
pip install -e .[docs]
cd docs
make html
```

The generated files are written to `docs/_build/html`.
