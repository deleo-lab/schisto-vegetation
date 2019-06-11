import argparse
import os
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

def parse_args():
    parser = argparse.ArgumentParser(description='Transform a training/test example.')
    parser.add_argument('--base_file', default=None,
                        help='Which file to process')
    parser.add_argument('--trapezoid', default=None,
                        help='Trapezoid to preserve from original file')

    parser.add_argument('--output_suffix', default='_trans',
                        help='Suffix for outputting the transformed file(s)')
    
    args = parser.parse_args()
    return args

def parse_trapezoid(trapezoid):
    # trapezoid will be stretched in the order UL, UR, LR, LL
    pattern = re.compile("^[(), 0-9]+$")
    if not pattern.match(trapezoid):
        raise RuntimeError("Illegal characters in the trapezoid description")
    original = eval(trapezoid)
    original = np.asarray(original)
    return original


if __name__ == '__main__':
    args = parse_args()

    base_file = args.base_file
    raw = skimage.io.imread(base_file)

    original = parse_trapezoid(args.trapezoid)
    desired = find_edges(raw)

    transform = ProjectiveTransform()
    transform.estimate(desired, original)

    base_files = [base_file]

    files = []
    for infile in base_files:
        base, ext = os.path.splitext(infile)
        outfile = base + args.output_suffix + ext
        if os.path.exists(outfile):
            raise RuntimeError("Cowardly refusing to overwrite %s" % outfile)
        files.append((infile, outfile))

    for infile, outfile in files:
        t_image = warp(raw, transform)

        rgb = np.array(t_image * 255, dtype=np.int8)
        im = Image.fromarray(rgb, "RGB")
        im.show()
        im.save(outfile)
        
