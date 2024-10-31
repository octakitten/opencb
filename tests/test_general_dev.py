from opencb.models.general_dev import general_dev
import torch
import numpy as np
import tensorplus as tp

def test_general_dev_initialization():
    print("Init model")
    model = general_dev()
    print("Create model")
    model.create(5, 6, 7, 8, 9)
    print("Save model")
    model.save(path='tests/saved_models/test')
    print("Create model2")
    model2 = general_dev()
    print("Load model 1 to model 2")
    model2.load(path='tests/saved_models/test')
    print("Assertions...")

    assert model.size == model2.size
    assert model.num_controls == model2.num_controls
    assert model.num_sensations == model2.num_sensations
    size1 = 0
    size2 = 0
    tp.size(model.neurons, size1)
    tp.size(model2.neurons, size2)
    assert size1 == size2
    tp.size(model.signals_neg, size1)
    tp.size(model2.signals_neg, size2)
    assert size1 == size2
    tp.size(model.emotion1, size1)
    tp.size(model2.emotion1, size2)
    assert size1 == size2
    tp.size(model.personality17, size1)
    tp.size(model2.personality17, size2)
    assert size1 == size2
    print("Create model1 again")
    model.create(10, 11, 12, 13, 14)
    print("Copy model 1 to model 2")
    model2.copy(model)
    print("more assertions...")
    assert model.image_size == model2.image_size
    assert model.size == 10
    assert model2.image_size == 11
    tp.size(model.personality16, size1)
    tp.size(model2.personality16, size2)
    assert size1 == size2
    assert model2.neurons.get(0) == 0
    print("Done!")
    return

test_general_dev_initialization()
