#! /usr/bin/bash
## for use in a development environment
echo "Building OpenCB package"
hatch version b
FOLDER="dist_beta/"
hatch build -t wheel $FOLDER
echo "Built package!"