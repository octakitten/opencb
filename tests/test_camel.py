import sys
from pathlib import Path

#print(sys.path)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#print(sys.path)

import cantor
from cantor import camel
import numpy as np

def test_camel_initialization():
    c = camel()
    assert c.width == 255
    assert c.height == 255
    assert c.depth == 32
    assert c.output01 == (32,32,20)
    assert c.output02 == (32,96,20)
    assert c.output03 == (96,32,20)
    assert c.output04 == (96,96,20)
    assert c.output01_thresh_positive >= -255
    assert c.output01_thresh_positive <= 255
    assert c.layer0.shape == (255,255,32)
    assert c.layer0[200, 100, 21] == 0
    assert c.propensity >= 1
    assert c.propensity <= 255
    assert c.personality3.all() <= c.propensity
    assert c.personality7.all() >= -c.propensity
    return

def test_byop():
    c = camel()
    personalities = (1, 2, 3, 4, 5, 6, 7, 8)
    c.byop(personalities)
    assert c.personality1 == 1
    assert c.personality2 == 2
    assert c.personality3 == 3
    assert c.personality4 == 4
    assert c.personality5 == 5
    assert c.personality6 == 6
    assert c.personality7 == 7
    assert c.personality8 == 8
    return

def test_update():
    c = camel()
    input_image = np.zeros((255,255))
    result = c.update(input_image)
    assert result >= 1
    result2 = c.update('lmao')
    assert result2 == -1
    result3 = c.update(19)
    assert result3 == -1
    result4 = c.update(True)
    assert result4 == -1
    return


def run_all_tests():
    test_camel_initialization()
    test_byop()
    test_update()
    return

run_all_tests()