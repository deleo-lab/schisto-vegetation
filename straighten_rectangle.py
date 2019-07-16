import argparse
import glob
import math
import os
import re
import sys
import warnings

import numpy as np
from PIL import Image
import skimage.io
from skimage.transform import warp, ProjectiveTransform

def find_edges(raw):
    """
    Return the desired width & height of the result
    """
    shape = raw.shape

    # round up to the nearest 16, as the unet goes through 4 rounds of contraction
    width = math.ceil(shape[1] / 16) * 16
    height = math.ceil(shape[0] / 16) * 16
    return (width, height)

def output_shape(width, height):
    return np.asarray([[0, 0], [width, 0],
                       [width, height], [0, height]])

def parse_args():
    parser = argparse.ArgumentParser(description='Transform a training/test example.')
    parser.add_argument('--base_file', default=None,
                        help='Which file to process')
    parser.add_argument('--shape', default=None,
                        help='Shape to preserve from original file')

    parser.add_argument('--output_suffix', default='_trans',
                        help='Suffix for outputting the transformed file(s)')
    
    parser.add_argument('--transform_all', dest='transform_all',
                        default=False, action='store_true',
                        help='Transform all of 8 channel, cera, etc')

    parser.add_argument('--overwrite', dest='overwrite',
                        default=False, action='store_true',
                        help='Force overwrite of existing files... be careful')

    parser.add_argument('--subdirs', default='8_bands,cera_mask,emergent_mask,land_mask,RGB,water_mask')

    args = parser.parse_args()
    return args

def parse_shape(shape):
    """
    Parse text that looks like this:
      "((64, 0), (511, 0), (511, 511), (60, 511))"
    Returns a numpy array representing this quadrilateral
    """
    # shape will be stretched in the order UL, UR, LR, LL
    pattern = re.compile("^[(), 0-9]+$")
    if not pattern.match(shape):
        raise RuntimeError("Illegal characters in the shape description")
    original = eval(shape)
    original = np.asarray(original)
    return original

def check_mask(mask):
    """
    Verify that the mask is either black or white
    """
    values = {}
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            values[mask[i, j]] = values.get(mask[i, j], 0) + 1
    for k in values.keys():
        if k != 0 and k != 255:
            raise RuntimeError("Mask was not entirely 0 or 255")

def get_sibling_files(base_file, subdirs):
    """
    Look for files with similar names to base_file in parallel dirs
    """
    # we put base_file first so that we can save a read later
    base_file = os.path.abspath(base_file)
    base_files = [base_file]
    parent_dir = os.path.split(os.path.split(base_file)[0])[0]
    base_name = os.path.splitext(os.path.split(base_file)[1])[0]
    print("Processing all files in directory %s with base_name %s" %
          (parent_dir, base_name))
    subdirs = subdirs.split(",")
    for subdir in subdirs:
        path = os.path.join(parent_dir, subdir, base_name)
        images = glob.glob(path + ".*")
        if len(images) > 1:
            raise RuntimeError("Found multiple files: %s" % str(images))
        if len(images) == 0:
            raise RuntimeError("Warning: found no file like %s" % path)
        next_file = os.path.normpath(images[0])
        if next_file != base_file:
            base_files.append(next_file)
    return base_files
        
if __name__ == '__main__':
    """ 
    Takes an image file and a shape as arguments.  Stretches the
    inside of the shape to fill the entire image.  For example, the
    original 35.TIF was slightly cut off on the left, and this script
    can fix that and simultaneously stretch all the masks.

    Example for running this program:
      python -u straighten_rectangle.py --base_file ../training_set/8_bands/35.TIF --shape "((64, 0), (511, 0), (511, 511), (60, 511))" --transform_all

    To run on the extra set:

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_DT_2m.tif --shape "((6, 1), (308, 3), (305, 493), (1, 492))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_FS_2m.tif --shape "((2, 137), (589, 1), (696, 457), (108, 592))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_GK_2m.tif --shape "((85, 1), (367, 45), (295, 509), (2, 493))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_MA_2m.tif --shape "((2, 2), (586, 76), (391, 509), (19, 434))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_ME_2m.tif --shape "((362, 2), (517, 408), (332, 674), (2, 504))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_MT_2m.tif --shape "((2, 302), (312, 3), (554, 182), (260, 501))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_ST_2m.tif --shape "((2, 322), (223, 2), (653, 298), (606, 628))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_ST_2m.tif --shape "((2, 322), (223, 2), (653, 298), (606, 628))" --transform_all --overwrite

      python straighten_rectangle.py --base_file ../extra_set/8_bands/DG_2016_ST_2m.tif --shape "((2, 322), (606, 628), (321, 923), (102, 775))" --transform_all --overwrite --output_suffix _trans_B
    """
    args = parse_args()

    base_file = os.path.normpath(args.base_file)
    raw = {}
    raw = skimage.io.imread(base_file)

    original = parse_shape(args.shape)
    width, height = find_edges(raw)
    desired = output_shape(width, height)

    print("Original shape: %s" % original)
    print("Desired shape: %s" % desired)

    transform = ProjectiveTransform()
    transform.estimate(desired, original)

    if args.transform_all:
        base_files = get_sibling_files(base_file, args.subdirs)
    else:
        base_files = [base_file]

    files = []
    for infile in base_files:
        base, ext = os.path.splitext(infile)
        outfile = base + args.output_suffix + ext
        if os.path.exists(outfile) and not args.overwrite:
            raise RuntimeError("Cowardly refusing to overwrite %s" % outfile)
        files.append((infile, outfile))
        
    for infile, outfile in files:
        print("Transforming %s into %s" % (infile, outfile))
        if infile != base_file:
            raw = skimage.io.imread(infile)
        # order=0 means nearest neighbor rather than interpolation
        if len(raw.shape) == 2:
            # found a mask
            t_image = warp(raw, transform, output_shape=(height, width), order=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")            
                t_image = skimage.img_as_ubyte(t_image)
            check_mask(t_image)
            skimage.io.imsave(outfile, t_image)
        elif len(raw.shape) == 3 and raw.shape[2] == 3:
            # RGB image, presumably converted from satellite
            t_image = warp(raw, transform, output_shape=(height, width), order=0)
            rgb = np.array(t_image * 255, dtype=np.int8)
            im = Image.fromarray(rgb, "RGB")
            im.show()
            im.save(outfile)
        elif len(raw.shape) == 3 and raw.shape[2] == 8:
            # Satellite image
            t_image = warp(raw, transform, output_shape=(height, width), order=0, preserve_range=True)
            t_image = np.asarray(t_image, dtype=raw.dtype)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")            
                skimage.io.imsave(outfile, t_image)
        else:
            raise RuntimeError("Don't know how to handle %s" % infile)
