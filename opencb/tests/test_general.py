'''import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
'''
from opencb.src.opencb.models.general import general
import torch

def test_general_initialization():
    g = general()
    g.create(1024, 1024, 32, 4)
    assert g.width == 1024
    assert g.height == 1024
    assert g.depth == 32
    assert g.num_controls == 4
    assert g.controls[0][0] >= 1
    assert g.controls[0][0] <= 1024
    assert g.thresholds_pos[0] >= -255
    assert g.thresholds_pos[0] <= 255
    assert g.layer0.shape == (1024,1024,32)
    assert g.layer0[200, 100, 21] == 0
    assert g.pos_propensity >= 1
    assert g.pos_propensity <= 255
    assert g.neg_propensity >= -255
    assert g.neg_propensity <= -1
    assert g.personality3.all() <= g.pos_propensity
    assert g.personality7.all() >= g.neg_propensity
    return

def test_update():
    g = general()
    g.create(256, 256, 32, 4)
    input_image = torch.zeros((256,256))
    result = g.update(input_image)
    assert result[0] == True or result[0] == False or result[0] == -1
    result2 = g.update('lmao')
    assert result2 == -1
    result3 = g.update(19)
    assert result3 == -1
    result4 = g.update(True)
    assert result4 == -1
    return

def test_permute():
    g = general()
    g.create(256, 256, 32, 4)
    g2 = general()
    g2.copy(g)
    g2.permute(degree = 2)
    assert g2.layer0.shape == (256, 256, 32)
    assert g2.layer0[200, 100, 21] <= 255
    assert g2.controls[0][0] == g.controls[0][0]
    assert g2.thresholds_pos[0] >= -255
    assert g2.thresholds_pos[0] <= 255
    


def run_all_tests():
    test_general_initialization()
    test_update()
    test_permute()
    return

run_all_tests()