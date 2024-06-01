from opencb.models.general import general_dev
import torch
import numpy as np

def test_general_dev_initialization():
    model = general_dev()
    model.create(5, 6, 7, 8, 9, 10)
    model.save(path='tests/saved_models/test')
    model2 = general_dev()
    model2.load(path='tests/saved_models/test')
    assert model.width == model2.width
    assert model.height == model2.height
    assert model.depth == model2.depth
    assert model.num_controls == model2.num_controls
    assert model.num_sensations == model2.num_sensations
    assert model.num_personality_layers == model2.num_personality_layers
    assert model.layer0.shape == model2.layer0.shape
    assert model.layer1.shape == model2.layer1.shape
    assert model.layer2.shape == model2.layer2.shape
    assert model.layer3.shape == model2.layer3.shape
    assert model.layer4.shape == model2.layer4.shape
    assert model.emotion1.shape == model2.emotion1.shape
    assert model.emotion8.shape == model2.emotion8.shape
    assert model.personalty5.shape == model2.personalty5.shape
    assert model.personality17.shape == model2.personality17.shape
    model.create(11, 12, 13, 14, 15, 16)
    model2.copy(model)
    assert model.width == model2.width
    assert model.height == 12
    assert model2.depth == 13
    assert model.personality16.shape == model2.personality16.shape
    assert model2.layer0[0, 0, 0] == 0
    
    