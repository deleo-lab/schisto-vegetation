# schisto-vegetation
Building deep learning applications to combat Schistosomiasis

The main contribution of this repo is schisto_model.py, which builds
and runs a u-net for recognizing snail habitat in satellite images of
Senegal.  The script should contain sufficient information on how to
run it in the pydoc.

There are a few other scripts in this directory:

convert_image.py turns an 8 channel satellite image into a 3 channel
  RGB image.  It can also display the image.

split_mask.py turns a single mask with 4 values for the outputs into 4
  separate mask files of the format expected by schisto_model.py.

straighten_rectangle.py extracts quadrilaterals from images in cases
  where valid input does not fill the entire rectangle.



Images are read using rasterio, which may require some installation:
https://rasterio.readthedocs.io/en/stable/installation.html
