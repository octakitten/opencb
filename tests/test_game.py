import sys
from pathlib import Path

#print(sys.path)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#print(sys.path)

import cantor
from cantor import game
from cantor import find_food_02
from cantor import horse
import numpy as np

def test_find_food_02_initialization():
    test_w = 255
    test_h = 255
    m = horse()
    g = cantor.find_food_02(test_w, test_h)
    assert g.width == test_w
    assert g.height == test_h
    return

test_find_food_02_initialization()