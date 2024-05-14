#! /usr/bin/bash

echo "Building OpenCB package"
hatch version b
FOLDER="dist_beta/"
hatch build -t wheel $FOLDER
echo "Built package: opencb-$VERSION.whl"
cd ..
echo "Build complete"
echo "Building Sphinx documentation"
make html
echo "Documentation built"