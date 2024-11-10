pip uninstall silky -y
pip install ./dist_beta/*.whl
python3 tests/testing.py
