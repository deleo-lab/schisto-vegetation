import glob
import os.path
import sys

import numpy as np
import skimage.io
from PIL import Image

# Using the descriptions in the worldview pdf, try to roughly
# make the channels correspond to visible light channels
transform = np.array([[ 0,  0,  0,  0,  0,  1, 0, 0],
                      [ 0,  0,  1,  0,  0,  0, 0, 0],
                      [ 0,  1,  0,  0,  0,  0, 0, 0]]).T

def print_min_max(arr):
    """
    For debugging purposes, print out the min/max of each channel
    """
    print("Image shape: %s" % str(arr.shape))
    for channel in range(arr.shape[2]):
        print(channel, np.min(arr[:, :, channel]), np.max(arr[:, :, channel]))

def convert_image(path):
    raw = skimage.io.imread(path)
    if raw.shape[0] == 8 and raw.shape[2] != 8:
        raw = np.transpose(raw, axes=(1, 2, 0))
        print("Transposed 8 bands from 0th dimension to 2nd dimension")
    print_min_max(raw)
    rgb = np.tensordot(raw, transform, 1) / 2047 * 255
    if np.max(rgb) < 0.8 * 255:
        # Brighten up the image a bit, since these images are dark AF
        rgb = rgb / np.max(rgb) * 0.8 * 255
    rgb = np.array(rgb, dtype=np.int8)
    im = Image.fromarray(rgb, "RGB")
    return im

def convert_directory(source, dest):
    files = glob.glob(os.path.join(source, "*.tif"))
    for f in files:
        filename = os.path.split(f)[1]
        rgb_home = os.path.join(dest, filename)
        if os.path.exists(rgb_home):
            print("Skipping %s: %s already exists" % (f, rgb_home))
        else:
            print("Converting %s" % f)
            im = convert_image(f)
            im.save(rgb_home)

def display_image(path, save_path=None):
    im = convert_image(path)
    im.show()
    if save_path:
        im.save(save_path)
    return im


if __name__ == '__main__':
    """
    A program to display and possibly convert satellite images.

    Command line is:
      ~ <input> [output]

    If input is a file, a single file is processed.  It will be displayed,
    and if output is present, the file will be written to output.

    If input is a directory, output is necessary.  
    In that case, each image in the directory will be converted.  
    Images will not be displayed.
    """
    path = sys.argv[1]
    if os.path.isdir(path):
        if len(sys.argv) == 2:
            print("To convert an entire directory, please give a directory for output")
            sys.exit(-1)
        dest = sys.argv[2]
        if not os.path.exists(dest):
            os.makedirs(dest)
        elif os.path.isfile(dest):
            print("%s exists and is a regular file" % dest)
            sys.exit(-1)
        convert_directory(path, dest)
    else:
        im = display_image(path)
        if len(sys.argv) > 2:
            save_path = sys.argv[2]
            im.save(save_path)
