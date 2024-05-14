#! /usr/bin/bash

echo "Building OpenCB package"
hatch version b
FOLDER="dist_beta/"
hatch build -t wheel $FOLDER
cd $FOLDER
NAME=$(ls -t | head -n1)
VERSION=$(echo $NAME | cut -d'-' -f2)
mv "$NAME" "opencb-$VERSION.whl"
echo "Built package: opencb-$VERSION.whl"
cd ..
echo "Build complete"
echo "Building Sphinx documentation"
make html
echo "Documentation built"