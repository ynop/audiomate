#!/bin/bash

set -uex

REPO="https://github.com/ynop/pingu.git"
TEMP_DIR="pages-clone"

# Generate contents
cd docs
make clean html
cd ../

# Prepare a temporary directory, clone everything into it
rm -rf "$TEMP_DIR"
git clone "$REPO" "$TEMP_DIR"
cd "$TEMP_DIR"

# Remove old contents, copy new contents into place, commit and push everything
git checkout gh-pages
rm -rf *
cp -R ../docs/_build/html/* .

git add .
git add -u
git commit -m "Update website"
git push origin gh-pages

# Clean up
cd ../
rm -rf "$TEMP_DIR"
echo "Done!"