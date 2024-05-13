'''import sys
from pathlib import Path

#print(sys.path)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#print(sys.path)

import cantor'''
from opencb.src.opencb.utilities.game import game
from opencb.src.opencb.utilities.game import find_food_02
from opencb.src.opencb.models.horse import horse
import numpy as np

def test_find_food_02_initialization():
    test_w = 255
    test_h = 255
    m = horse()
    g = find_food_02(m)
    assert g.width == test_w
    assert g.height == test_h
    return

test_find_food_02_initialization()