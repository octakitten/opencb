import numpy as np
import models

cantor = models.neural_space_8depth_4action_greyscale()
for i in range(100):
    input_image_fake = np.random.randint(0, 255, (256,256))
    next_action = cantor.update(input_image_fake)
    print(next_action)
