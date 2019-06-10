import re
import sys

import numpy as np
from PIL import Image
import skimage.io
from skimage.transform import warp, ProjectiveTransform

def find_edges(raw):
    shape = raw.shape
    return np.asarray([[0, 0], [shape[0], 0],
                       [shape[0], shape[1]], [0, shape[1]]])

if __name__ == '__main__':
    base_tif = sys.argv[1]
    # trapezoid will be stretched in the order UL, UR, LR, LL
    trapezoid = sys.argv[2]
    pattern = re.compile("^[(), 0-9]+$")
    if not pattern.match(trapezoid):
        raise RuntimeError("Illegal characters in the trapezoid description")
    original = eval(trapezoid)
    original = np.asarray(original)

    raw = skimage.io.imread(base_tif)

    desired = find_edges(raw)

    transform = ProjectiveTransform()
    transform.estimate(desired, original)
    t_image = warp(raw, transform)

    rgb = np.array(t_image * 256, dtype=np.int8)
    im = Image.fromarray(rgb, "RGB")
    im.show()
