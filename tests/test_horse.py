import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cantor import horse
import torch

def test_horse_initialization():
    g = horse()
    assert g.width == 1024
    assert g.height == 1024
    assert g.depth == 32
    assert g.output01 == (32,32,20)
    assert g.output02 == (32,96,20)
    assert g.output03 == (96,32,20)
    assert g.output04 == (96,96,20)
    assert g.output01_thresh_positive >= -255
    assert g.output01_thresh_positive <= 255
    assert g.layer0.shape == (255,255,32)
    assert g.layer0[200, 100, 21] == 0
    assert g.pos_propensity >= 1
    assert g.pos_propensity <= 255
    assert g.neg_propensity >= -255
    assert g.neg_propensity <= -1
    assert g.personality3.all() <= g.pos_propensity
    assert g.personality7.all() >= g.neg_propensity
    return

def test_byop():
    g = horse()
    personalities = (1, 2, 3, 4, 5, 6, 7, 8)
    thresholds = (1, 2, 3, 4, 5, 6, 7, 8)
    g.byop(personalities, thresholds)
    assert g.personality1 == 1
    assert g.personality2 == 2
    assert g.personality3 == 3
    assert g.personality4 == 4
    assert g.personality5 == 5
    assert g.personality6 == 6
    assert g.personality7 == 7
    assert g.personality8 == 8
    assert g.output01_thresh_positive == 1
    assert g.output01_thresh_negative == 2
    assert g.output02_thresh_positive == 3
    assert g.output02_thresh_negative == 4
    assert g.output03_thresh_positive == 5
    assert g.output03_thresh_negative == 6
    assert g.output04_thresh_positive == 7
    assert g.output04_thresh_negative == 8
    return

def test_update():
    g = horse()
    input_image = torch.zeros((255,255))
    result = g.update(input_image)
    assert result >= 1
    result2 = g.update('lmao')
    assert result2 == -1
    result3 = g.update(19)
    assert result3 == -1
    result4 = g.update(True)
    assert result4 == -1
    return


def run_all_tests():
    test_horse_initialization()
    test_byop()
    test_update()
    return

run_all_tests()