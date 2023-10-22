import numpy as np
import sys
from pathlib import Path

print(sys.path)
sys.path.insert(0, str(Path(__file__).parent.parent))
print(sys.path)

import cantor
from cantor.src.models.d8a4gs import d8a4gs
from cantor.src.models.d16a1gs import d16a1gs
from cantor.src.utilities.screen import screen

'''
cantor01 = d8a4gs()
for i in range(100):
    input_image_fake = np.random.randint(0, 255, (256,256))
    next_action = cantor01.update(input_image_fake)
    print(next_action)
'''

cantor02 = d16a1gs()
a, b = 50, 50
w, h = 255, 255
for i in range(3):
    input_image_real = screen.screenshot_desktop(a, b, w, h)
    next_action = cantor02.update(input_image_real)
    if next_action == 2:
        print(next_action)
        a += 1
    elif next_action == 3:
        print(next_action)
        a -= 1
    else:
        print(next_action)
