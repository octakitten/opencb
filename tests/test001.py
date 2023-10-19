import numpy as np
import os

import numpy as np
import cantor

cantor01 = cantor.src.models.neural_space_8depth_4action_greyscale()
for i in range(100):
    input_image_fake = np.random.randint(0, 255, (256,256))
    next_action = cantor01.update(input_image_fake)
    print(next_action)

cantor02 = cantor.src.models.neural_space_16depth_1action_greyscale()
a, b = 50, 50
w, h = 255, 255
for i in range(100):
    input_image_real = cantor.src.utilities.screenshot(a, b, w, h)
    next_action = cantor02.update(input_image_real)
    if next_action == 2:
        print(next_action)
        a += 1
    elif next_action == 3:
        print(next_action)
        a -= 1
    else:
        print(next_action)
