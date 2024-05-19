#! /usr/bin/bash
## for use in a development environment
echo "Building OpenCB package"
hatch version b
FOLDER="dist_beta/"
VERSION = (hatch build -t wheel $FOLDER)
echo "Built package: $VERSION"