import glob
import os.path
import sys

import numpy as np
import skimage.io
from PIL import Image

# Using the descriptions in the worldview pdf, try to roughly
# make the channels correspond to visible light channels
transform = np.array([[ 0,  0,  0, .4, .4, .2, 0, 0],
                      [ 0, .2, .5, .3,  0,  0, 0, 0],
                      [.5, .5,  0,  0,  0,  0, 0, 0]]).T

def print_min_max(arr):
    """
    For debugging purposes, print out the min/max of each channel
    """
    print("Image shape: %s" % str(arr.shape))
    for channel in range(arr.shape[2]):
        print(channel, np.min(arr[:, :, channel]), np.max(arr[:, :, channel]))

def display_image(path, save_path=None):
    raw = skimage.io.imread(path)
    print_min_max(raw)
    rgb = np.tensordot(raw, transform, 1) / 2048 * 256
    if np.max(rgb) < 0.8 * 256:
        # Brighten up the image a bit, since these images are dark AF
        rgb = rgb / np.max(rgb) * 0.8 * 256
    rgb = np.array(rgb, dtype=np.int8)
    im = Image.fromarray(rgb, "RGB")
    im.show()
    if save_path:
        im.save(save_path)


if __name__ == '__main__':
    path = sys.argv[1]
    if len(sys.argv) > 2:
        save_path = sys.argv[2]
        display_image(path, save_path)
    else:
        display_image(path)
        
