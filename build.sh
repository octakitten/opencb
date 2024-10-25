#! /usr/bin/bash
## for use in a development environment
source .venv/bin/activate
pip install hatch
echo "Building OpenCB package"
hatch version b
FOLDER="dist_beta/"
rm -f $FOLDER*.whl
rm -f $FOLDER*.tar.gz
hatch build -t wheel $FOLDER
echo "Built package!"
