#! /usr/bin/bash
## for use in a development environment
echo "Building Silky package"
hatch version b
FOLDER="dist_beta/"
rm -f $FOLDER*.whl
rm -f $FOLDER*.tar.gz
hatch build -t wheel $FOLDER
echo "Built package!"
