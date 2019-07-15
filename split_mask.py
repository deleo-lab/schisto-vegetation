"""
Some of the masks we're given are single layer TIF files which need
to be separated into 4 masks.

The default assumption is that the masks are

1 = water
2 = emergent
3 = floating (ceratophyllum)
4 = land
"""

import argparse
import os
import sys

import numpy as np
import skimage.io
default_splits = {
    'water_mask':    1,
    'emergent_mask': 2,
    'cera_mask':     3,
    'land_mask':     4,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Split a 4 value mask file into 4 separate masks.')
    parser.add_argument('--mask_file', default=None,
                        help='Which file to process')
    parser.add_argument('--water_mask', type=int, default=1,
                        help='Mask value to process as water')
    parser.add_argument('--emergent_mask', type=int, default=2,
                        help='Mask value to process as emergent veg')
    parser.add_argument('--cera_mask', type=int, default=3,
                        help='Mask value to process as floating veg')
    parser.add_argument('--land_mask', type=int, default=4,
                        help='Mask value to process as land')
    return parser.parse_args()

def get_splits(args):
    mapped_values = {args.water_mask, args.emergent_mask,
                     args.cera_mask, args.land_mask}
    if len(mapped_values) < 4:
        raise ValueError("Given mask values overlap")
    for value in mapped_values:
        if value < 1 or value > 4:
            raise ValueError("Mask values must be between 1 and 4")
    splits = {
        'water_mask': args.water_mask,
        'emergent_mask': args.emergent_mask,
        'cera_mask': args.cera_mask,
        'land_mask': args.land_mask
    }
    return splits

if __name__ == '__main__':
    """ 
    Converts one of the 1 channel 4 value mask files from the extra
    set into 4 separate mask files, as expected by the model code.

    Since the one channel of mask values has the mask values randomly
    assigned, this script allows you to specify which channel is
    which.  For example. here are command lines for each of the extra files:

    python split_mask.py --mask_file ../extra_set/masks/DG_2016_DT_2m_Class.tif --land_mask 3 --water_mask 2 --emergent_mask 4 --cera_mask 1

    python split_mask.py --mask_file ../extra_set/masks/DG_2016_FS_2m_Class.tif --land_mask 2 --water_mask 3 --emergent_mask 4 --cera_mask 1

    python split_mask.py --mask_file ../extra_set/masks/DG_2016_GK_2m_Class.tif --land_mask 3 --water_mask 2 --emergent_mask 4 --cera_mask 1

    python split_mask.py --mask_file ../extra_set/masks/DG_2016_MA_2m_Class.tif --land_mask 2 --water_mask 4 --emergent_mask 1 --cera_mask 3

    python split_mask.py --mask_file ../extra_set/masks/DG_2016_ME_2m_Class.tif --land_mask 3 --water_mask 1 --emergent_mask 4 --cera_mask 2

    python split_mask.py --mask_file ../extra_set/masks/DG_2016_MT_2m_Class.tif --land_mask 2 --water_mask 3 --emergent_mask 4 --cera_mask 1

    python split_mask.py --mask_file ../extra_set/masks/DG_2016_ST_2m_Class.tif --land_mask 1 --water_mask 3 --emergent_mask 4 --cera_mask 2
    """

    args = parse_args()
    mask_file = args.mask_file
    mask = skimage.io.imread(mask_file)

    # assuming mask_file is something like
    #   extra_set/masks/DG_2016_DT_2m_Class.tif
    # parent_dir is extra_set
    parent_dir = os.path.split(os.path.split(mask_file)[0])[0]
    # base_name is DG_2016_DT_2m_Class
    base_name = os.path.splitext(os.path.split(mask_file)[1])[0]

    print(parent_dir)
    print(base_name)

    splits = get_splits(args)
    
    for mask_name, index in splits.items():
        # out_file for water_mask in the above example will be
        #   extra_set\water_mask\DG_2016_DT_2m_Class.png
        out_file = os.path.join(parent_dir, mask_name, base_name + ".png")
        out_file = os.path.normpath(out_file)
        print("Putting index %d (%s) in %s" % (index, mask_name, out_file))
        
        sub_mask = np.zeros(mask.shape, mask.dtype)
        sub_mask[np.where(mask == index)] = 255

        skimage.io.imsave(out_file, sub_mask)
