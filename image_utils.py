import numpy as np
import rasterio
import skimage.io

# try to map the visible channels to RGB
SAT_TRANSFORM = np.array([[ 0,  0,  0,  0,  0,  1, 0, 0],
                          [ 0,  0,  1,  0,  0,  0, 0, 0],
                          [ 0,  1,  0,  0,  0,  0, 0, 0]]).T

def transform_rgb_image(image):
    """
    Transform an 8 band tif in the 0..2047 range to RGB from 0..1
    """
    rgb = np.tensordot(image, SAT_TRANSFORM, 1) / 2047
    return rgb


def read_8band_tif(image_filename):
    """
    Uses either rasterio or skimage.io to read a .tif file.
    """
    dataset = rasterio.open(image_filename)
    if dataset.count == 8:
        bands = [dataset.read(i) for i in range(1, 9)] # thanks for not indexing by 0
        raw = np.stack(bands, axis=2)
        return raw
    # TODO: images from the original training set get read in as 1
    # channel, 512x8 images by rasterio.  fixing it like this is
    # pretty lazy.  should just transform the images in question.  or
    # maybe this is a bug in rasterio?  i would think 512x512x8 should
    # be read in as 512 channels of 512x8
    if dataset.count == 1 and dataset.read(1).shape[1] == 8:
        return skimage.io.imread(image_filename)
    raise RuntimeError("%s is not an 8 band tif.  Has %d bands.  Unable to process" %
                       (image_filename, dataset.count))
