from PIL import ImageGrab
from PIL import features
import numpy as np

class screen():

    def screenshot(a, b, w, h):
        # a,b = top left corner of the box
        # w,h = width and height of the box
        # returns a numpy array
        val = -1
        if(features.check_feature("xcb")):
            im = ImageGrab.grab((a, b, a+w, b+h))
            val = np.array(im).astype(np.uint8)[:,:,0]
        
        
        return val