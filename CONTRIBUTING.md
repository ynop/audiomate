# Contributing to audiomate

If you would like to contribute code or documentation to audiomate you can do so through GitHub by forking the repository and starting a pull request. Please follow the guidelines in this document when preparing the pull request. If you need help compiling the code, head to the [README](README.md) for quick instructions. All the details are outlined in the [separate documentation](docs).

## Code Conventions and Housekeeping

Please make every effort to follow the existing conventions and style in order to keep the code as readable as possible. This makes it easier to review your contribution and reduces the amount of work necessary before a merge.

* When writing a commit message please follow [these conventions](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html). If you are addressing an existing issue, add `Fixes GH-XXXX` at the end of the commit message (where `XXXX` denotes the issue number).
* Run `flake8` to ensure that the formatting of your code matches the project's code style. Audiomate's code is formatted according to [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/), but the maximum line length is 100 characters and strings have to be delimited by single quotes.
* Add docstrings to the public API in accordance with [PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/).
* Cover your changes with unit tests.
* Before opening the pull request, rebase it onto the HEAD of the current master (or the relevant target branch).
