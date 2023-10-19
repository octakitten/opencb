from PIL import ImageGrab
import numpy as np

def screenshot(a, b, w, h):
    # a,b = top left corner of the box
    # w,h = width and height of the box
    # returns a numpy array
    box = (a,b,a+w,b+h)
    im = ImageGrab.grab(box)
    return np.array(im)