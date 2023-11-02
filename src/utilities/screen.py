from PIL import Image
from PIL import ImageGrab
from PIL import features
import numpy as np
import torch
import subprocess


class screen():

    def __init__():
        return

    def snap_desktop(a, b, w, h):
        # a,b = top left corner of the box
        # w,h = width and height of the box
        # returns a numpy array
        val = -1
        

        # this needs to be set to True but we're leaving it on False for now
        if(features.check_feature("xcb") == False):
            # if xcb is enabled, use ffmpeg with xcbgrab to take a screenshot of the desktop
            command = [
            'ffmpeg',
            '-s', f'{w}x{h}',
            '-framerate', '25',
            '-f', 'x11grab',
            '-i', ':0.0+{a},{b}',
            '-vframes', '1',
            'snap.png'
            ]
            subprocess.run(command)
            val = np.array(Image.open('snap.png'))
        else:
            # otherwise, use PIL's ImageGrab to take a screenshot of the desktop, 
            # which will probably default to gnome-screenshot
            im = ImageGrab.grab(bbox =(a, b, a+w, b+h))
            val = np.array(im).astype(np.uint8)[:,:,0]
        
        return val
    
    def save(val, name):
        # save a numpy array or torch tensor as a png image
        # name should not have an extension
        if (isinstance(val, np.ndarray) == True):
            img = Image.fromarray(val)
            img.save(name, 'PNG')
            return 1
        elif (torch.is_tensor(val) == True):
            img = Image.fromarray(val.numpy())
            grey = img.convert('L')
            grey.save(name, 'PNG')
            return 2
        else:
            return -1
    
    
    
