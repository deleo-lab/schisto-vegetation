import argparse
import glob
import os
import random

from enum import Enum

from PIL import Image
import rasterio
import skimage.io as io
import skimage.transform as trans
import sklearn.metrics
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as backend

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class DataType(Enum):
    HABITAT_SEPARATE = 1
    HABITAT_COMBINED = 2
    VILLAGE = 3

class ModelType(Enum):
    HABITAT = 1
    VILLAGE = 2

class ModelArch(Enum):
    UNET = 1
    PIXEL = 2
    CONV = 3

PATCH_SZ = 256   # should divide by 16
BATCH_SIZE = 8
TRAIN_SZ = 50  # train size
VAL_SZ = 15   # validation size
N_EPOCHS = 500  #150

PATH_TIF = '%s/8_bands/'
WATER_MASK = '%s/water_mask/'
LAND_MASK = '%s/land_mask/'
EMERGENT_MASK = '%s/emergent_mask/'
CERA_MASK = '%s/cera_mask/'

VILLAGE_MASK = '%s/village/'

STARTING_LR = 4e-5

def default_training_dir():
    candidates = ['c:/Users/horat/Documents/hai/schisto/training_set',
                  '/home/john/hai/schisto/training_set']
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

DEFAULT_DIR = default_training_dir()

def random_transform(patch_img, patch_mask):    
    # Apply some random transformations
    random_transformation = np.random.randint(1,8)
    if random_transformation == 1:
        # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:] #[::-1,:,:]
    elif random_transformation == 2:
        # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:] #[:,::-1,:]
    elif random_transformation == 3:
        # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        pass
    return patch_img, patch_mask
    

def get_rand_patch(img, mask, sz):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3
    assert img.shape[0] >= sz
    assert img.shape[1] >= sz
    assert img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)

    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]
    
    #adjusted for masks of size (512,512), no third dimension.

    patch_img, patch_mask = random_transform(patch_img, patch_mask)

    return patch_img, patch_mask


def get_patches(dataset, n_patches, sz):
    """
    Returns X, Y arrays of patches.

    Patches are randomly selected images from the given dataset of
    the requested size.
    Each patch might have a random transformation applied to it.
    """
    x_dict, y_dict = dataset
    x = []
    y = []
    for _ in range(n_patches):
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
    #print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)

def conv_classifier(learning_rate, num_classes, model_name, input_channels=8):
    """
    Another simple baseline which only look at single pixels to classify
    Starts with a 3x3 conv, then does a bunch of 1x1 convs

    Goal is to test how necessary a model like the U-Net actually is

    The best trained model as of 2019-06-13 was 0.952 val accuracy,
    roughly 2% lower than the best U-net.
    """
    inputs = Input((None, None, input_channels))
    conv1 = Conv2D(128, (3, 3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, (1, 1), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.1)(conv2)
    conv3 = Conv2D(64, (1, 1), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(drop2)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(32, (1, 1), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv5 = Conv2D(num_classes, 1, activation = 'softmax')(conv4)

    model = tf.keras.Model(inputs=inputs, outputs=conv5, name=model_name)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


def pixel_classifier(learning_rate, num_classes, model_name, model_input_channels=8):
    """
    A simple baseline which only look at single pixels to classify

    Goal is to test how necessary a model like the U-Net actually is

    The best trained model as of 2019-06-13 was 0.956 val accuracy,
    roughly 2% lower than the best U-net.
    """
    inputs = Input((None, None, input_channels))
    conv1 = Conv2D(64, (1, 1), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(64, (1, 1), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.1)(conv2)
    conv3 = Conv2D(64, (1, 1), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(drop2)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(32, (1, 1), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv5 = Conv2D(num_classes,1, activation = 'softmax')(conv4)

    model = tf.keras.Model(inputs=inputs, outputs=conv5, name=model_name)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
    
def unet(learning_rate, num_classes, model_name, input_channels=8):
    """
    Builds a u-net out of TF Keras layers

    Batch Normalization layers are added in between each pair of Conv2D
    Note that the input size is unconstrained, as the network will
    operate on any size image (past a certain minimum)
    """
    # TODO: https://datascience.stackexchange.com/questions/26942/
    # Update with concat layer that doesn't barf on slightly different sizes
    inputs = Input((None, None, input_channels))
    conv1 = Conv2D(32, (3, 3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    # TODO: it might be better to separate activation so that it goes
    #   conv -> activation -> bn
    # would like to try conv -> bn -> activation
    # sadly TF 2.0 alpha has a bug saving models with a separate
    # activation layer
    # on the bright side, most results out there seem to prefer
    # this order anyway
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.1)(conv5)

    up6 = Conv2D(256, (2, 2), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis=3)
    conv6 = Conv2D(256, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # TODO: it is not clear we want these BN layers - there was
    # slightly better performance with just the down, not the up, but
    # that is with a very small val set
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, (2, 2),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, (2, 2),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(32, (2, 2),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],  axis = 3)
    conv9 = Conv2D(32, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(num_classes,1, activation = 'softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10, name=model_name)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    return model

# try to map the visible channels to RGB
SAT_TRANSFORM = np.array([[ 0,  0,  0,  0,  0,  1, 0, 0],
                          [ 0,  0,  1,  0,  0,  0, 0, 0],
                          [ 0,  1,  0,  0,  0,  0, 0, 0]]).T

GREY_TRANSFORM = np.array([0.299, 0.587, 0.114])

def transform_rgb_image(image):
    rgb = np.tensordot(image, SAT_TRANSFORM, 1) / 2047
    return rgb

# TODO: refactor with convert_image.py
def read_8band_tif(image_filename):
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
        return io.imread(image_filename)
    raise RuntimeError("%s is not an 8 band tif.  Has %d bands.  Unable to process" %
                       (image_filename, dataset.count))

def read_mask(mask_file):
    """
    Reads in a mask file, converts it to 0/1

    Anything > 0 is treated as masked
    """
    base_name, _ = os.path.splitext(mask_file)   # remove the .TIF
    candidates = glob.glob(base_name + ".*")
    if len(candidates) == 0:
        raise RuntimeError("No mask files found matching %s" % mask_file)
    elif len(candidates) > 1:
        raise RuntimeError("Multiple mask files found matching %s" % mask_file)

    mask = io.imread(candidates[0])
    mask[mask > 0] = 1
    return mask

def read_rgb(rgb_file):
    return io.imread(rgb_file)


def split_images(X, Y):
    """
    20% (could be parametrized) are split off into a val set
    Returns (train_set, val_set) where both training sets are
      (X, Y), input channels & masks
    """
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    assert X.keys() == Y.keys()
    files = sorted(X.keys())

    train_len = int(len(files) * 0.8)
    train_files = files[:train_len]
    val_files = files[train_len:]

    for base_name in train_files:
        X_DICT_TRAIN[base_name] = X[base_name]
        Y_DICT_TRAIN[base_name] = Y[base_name]
        print ("Train: %s" % base_name)

    for base_name in val_files:
        X_DICT_VALIDATION[base_name] = X[base_name]
        Y_DICT_VALIDATION[base_name] = Y[base_name]
        print ("Val: %s" % base_name)

    train_set = (X_DICT_TRAIN, Y_DICT_TRAIN)
    val_set = (X_DICT_VALIDATION, Y_DICT_VALIDATION)
    return train_set, val_set

def check_filenames_unique(files):
    """
    Check there are no duplicates if ignoring case

    Simple check to make sure there are no confusing filenames such as
    foo.tif vs foo.TIF.  This can happen on linux (but not windows)
    """
    lowered_set = {}
    for f in files:
        lf = f.lower()
        if lf in lowered_set:
            raise RuntimeError("Confusing filename: %s vs %s" % (f, lowered_set[lf]))
        lowered_set[lf] = f

def read_images(data_type, image_path):
    """
    Reads images from the given path

    If data_type == HABITAT_SEPARATE, four masks are returned, with cera being mask 0
    If data_type == HABITAT_COMBINED, cera and emergent plants are combined
    If data_type == VILLAGE, we just load village data

    Returns (X, Y) where X and Y are dicts from filename to img & mask

    For the snail habitat model, the following organization is expected.
      dataset
        - 8_bands: a directory with 8 band tif files
        - cera_mask: a directory with mask files for floating
        - emergent_mask, land_mask, water_mask: etc etc
        - RGB: a directory with RGB conversions of the 8_bands files
               not necessary except when building heat maps
    Available filenames are taken from the 8_bands dir.
    Mask files are expected to be .png files with 0 for no and 255 for yes
      (this way a human can read them, instead of 0/1 for example)
    Each file should have a parallel name.  So, for example, 
      if there is a .tif in 8_bands named 00.tif, there should be
      corresponding masks 00.png in each of the mask directories.
    """
    tif_path = PATH_TIF % image_path
    print("Loading images from %s" % tif_path)
    tif_path = tif_path + '*.[tT][iI][fF]'
    tif_files = glob.glob(tif_path)
    tif_files = sorted(tif_files)

    check_filenames_unique(tif_files)
    
    X = dict()
    Y = dict()

    for image_filename in tif_files:
        # note: this image is used later when building heat maps
        # of the directory.  would need to re-read the image if
        # it is manipulated at all here.
        img_m = read_8band_tif(image_filename)
        _, base_name = os.path.split(image_filename) # filename -> 05.TIF

        if data_type == DataType.VILLAGE:
            mask_village = read_mask(VILLAGE_MASK % image_path + base_name)
            mask_no_village = 1 - mask_village
            mask = np.stack([mask_no_village, mask_village], axis=2)
        elif data_type in (DataType.HABITAT_COMBINED, DataType.HABITAT_SEPARATE):
            #create 3d mask where each channel is the mask for a specific class
            mask_water = read_mask(WATER_MASK % image_path + base_name)
            mask_land = read_mask(LAND_MASK % image_path + base_name)
            mask_emerg = read_mask(EMERGENT_MASK % image_path + base_name)
            mask_cera = read_mask(CERA_MASK % image_path + base_name)

            if data_type == DataType.HABITAT_COMBINED:
                mask = np.stack([mask_cera + mask_emerg, mask_land, mask_water],
                                axis=2)
            elif data_type == DataType.HABITAT_SEPARATE:
                mask = np.stack([mask_cera, mask_emerg, mask_land, mask_water],
                                axis=2)
        else:
            raise ValueError("Unknown data_type {}".format(data_type))
                
        mask_sum = np.sum(mask, axis=2)
        if np.min(mask_sum) == 0:
            print("Warning: %s has %d unlabeled pixels" %
                  (base_name, len(np.where(mask_sum == 0)[0])))
        if np.max(mask_sum) > 1:
            print("Warning: %s has %d pixels with multiple labels" %
                  (base_name, len(np.where(mask_sum > 1)[0])))

        X[base_name] = img_m
        Y[base_name] = mask

    return (X, Y)

def stack_dataset(dataset):
    """
    Returns the X, Y dicts from read_images as 2 numpy arrays
    """
    x_dict, y_dict = dataset

    if x_dict.keys() != y_dict.keys():
        raise RuntimeError("Got different input and label names")

    X = []
    Y = []
    keys = []
    for k in x_dict.keys():
        x = x_dict[k]
        y = y_dict[k]
        keys.append(k)

        if x.shape[:2] != y.shape[:2]:
            raise RuntimeError("Val set error: %s input has different shape from mask" % k)

        # TODO: make 16 a constant
        # TODO: only do this for unet, not the other model arch
        if x.shape[0] % 16 != 0 or x.shape[1] % 16 != 0:
            new_shape = (x.shape[0] - x.shape[0] % 16,
                         x.shape[1] - x.shape[1] % 16)
            print("Warning: {} has shape {}, and the u-net expects multiples of 16.  Trimming to {}"
                  .format(k, x.shape[:2], new_shape))
            x = x[:new_shape[0], :new_shape[1], :]
            y = y[:new_shape[0], :new_shape[1], :]

        X.append(x)
        Y.append(y)

    new_shape = (min(x.shape[0] for x in X),
                 min(x.shape[1] for x in X))

    if (max(x.shape[0] for x in X) > new_shape[0] or
        max(x.shape[1] for x in X) > new_shape[1]):
        for index, k in enumerate(keys):
            if (x[index].shape[0] > new_shape[0] or
                x[index].shape[1] > new_shape[1]):
                print("Warning: Some of the files in the val set are larger than the others " +
                      "and will need to be trimmed to stack them all together.  " +
                      "To avoid this, use --val_patches, although that means smaller " +
                      "val objects will be used in that case.  One example problem file: %s" % k)
                break


    X = map(lambda x: x[:new_shape[0], :new_shape[1], :], X)
    Y = map(lambda y: y[:new_shape[0], :new_shape[1], :], Y)

    return np.stack(X), np.stack(Y)
    

def train(model, model_filename, train_set, val_set, args):
    # upweight ceratophyllum
    if args.separate_ceratophyllum:
        class_weight = [.8,.3,.1,.3]
    else:
        class_weight = [.8,.1,.3]

    if not args.val_patches:
        # TODO: stack earlier?  we may not actually need
        # to keep the filenames, perhaps
        x_val, y_val = stack_dataset(val_set)

    if args.save_best_only:
        checkpoint_filename = model_filename
    else:
        root, ext = os.path.splitext(model_filename)
        checkpoint_filename = root + ".E{epoch:04d}-VAL{val_loss:05.2f}" + ext
    
    model_checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_loss',
                                       save_best_only=args.save_best_only)
    
    for i in range(0, N_EPOCHS):
        #csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        #tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

        x_train, y_train = get_patches(train_set, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        if args.val_patches:
            x_val, y_val = get_patches(val_set, n_patches=VAL_SZ, sz=PATCH_SZ)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                  epochs=1+i, initial_epoch=i,
                  verbose=2, #shuffle=True,
                  #callbacks=[model_checkpoint, csv_logger, tensorboard],
                  callbacks=[model_checkpoint],
                  validation_data=(x_val, y_val),
                  class_weight=class_weight)
        del x_train, y_train
        print ("Finished epoch %d" % (i+1))

    if not args.save_best_only:
        model.save(model_filename)

def washed_greyscale(image):
    """
    Applies a greyscale transformation to an 8 channel image,
    then remaps it to be between 0.5 and 1.0 in each channel
    """
    rgb_image = transform_rgb_image(image)
    grey = np.tensordot(rgb_image, GREY_TRANSFORM, 1)
    grey = np.expand_dims(grey, axis=2)
    grey = grey - grey.min()
    if grey.max() > 0.0:
        grey = grey / grey.max()
    else:
        grey = grey + 1.0
    grey = grey / 2.0 + 0.5

    return grey

def predict_single_image(model, image):
    """
    Uses the current model to predict a single image
    """
    if len(image.shape) == 3:
        batch = np.expand_dims(image, axis=0)
    else:
        batch = image

    prediction = model.predict(batch)

    if len(image.shape) == 3:
        prediction = np.squeeze(prediction)
    return prediction

def prediction_to_heat_map(prediction, grey):
    """
    Given a prediction and a greyscale image, put the heat map
    of the first channel of the prediction on top of the greyscale.
    """
    cera_prediction = prediction[:, :, 0]
    blue_prediction = np.maximum(0.4 - cera_prediction, 0.0)
    blue_prediction = np.sqrt(blue_prediction)
    red_prediction = np.minimum(cera_prediction / 0.4, 1.0)
    red_prediction = np.sqrt(red_prediction)
    green_prediction = np.maximum(np.minimum((cera_prediction - 0.4) / 0.4, 1.0), 0)

    # multiply pixel by pixel with the grey visualization
    # so the prediction has some texture to it
    heat_map = np.stack([red_prediction, green_prediction,
                         blue_prediction], axis=2)
    heat_map = heat_map * grey

    # map from 0..1 to int8
    heat_map = heat_map * 255
    heat_map = np.array(heat_map, dtype=np.int8)
    return heat_map

def prediction_to_classification(prediction, grey):
    """
    Given a prediction and a greyscale image, put prediction classes
    on top of the heat map.  Only works for up to 4 channels
    """
    classification = np.round(np.squeeze(prediction))

    channels = classification.shape[2]
    if channels == 2:
        red_prediction = classification[:, :, 0]
        green_prediction = classification[:, :, 1]
        blue_prediction = np.zeros_like(green_prediction)
    elif channels == 3 or channels == 4:
        red_prediction = classification[:, :, 0] + classification[:, :, 1]
        green_prediction = classification[:, :, 0] + classification[:, :, 2]
        blue_prediction = classification[:, :, 0]
        if classification.shape[2] == 4:
            blue_prediction = blue_prediction + classification[:, :, 3]
    else:
        raise ValueError("Prediction has unhandled number of channels: %d" % channels)
    classification = np.stack([red_prediction, green_prediction,
                               blue_prediction], axis=2)
    
    # multiply pixel by pixel with the grey visualization
    # so the classification has some texture to it
    classification = classification * grey

    # map from 0..1 to int8
    classification = classification * 255
    classification = np.array(classification, dtype=np.int8)
    return classification



def process_heat_map(model, test_image, display, save_filename=None):
    """Builds a heat map from the given TIF file

    First, the image is converted into a greyscale image where the
    darkest color is 128/255 so that features are somewhat visible.
    Then, colors ranging from blue to red to yellow are superimposed
    based on the model's prediction.  Probability of being cera: 0=blue,
    0.4=red, 0.8 or higher = yellow.

    if save_filename != None, the heat map is saved to that file
    """
    prediction = predict_single_image(model, test_image)

    # grey will be 0.5 to 1, a washed out greyscale version
    # of the original image.  the purpose is to make something that
    # can have blue...red...yellow heat map colors imposed on top
    grey = washed_greyscale(test_image)

    heat_map = prediction_to_heat_map(prediction, grey)
    classification = prediction_to_classification(prediction, grey)
    
    # stack the heat map and the classification side by side
    rgb = np.concatenate([heat_map, classification], axis=1)
    im = Image.fromarray(rgb, "RGB")

    if display:
        im.show()

    if save_filename:
        im.save(save_filename)

    return rgb

    
def process_heat_map_set(model, data_type, in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    X, Y = read_images(data_type, in_dir)
    files = X.keys()
    for base_name in files:
        print("Processing %s" % base_name)
        save_filename = os.path.join(out_dir, os.path.splitext(base_name)[0] + ".bmp")
        test_image = X[base_name]

        heat_map = process_heat_map(model, test_image, display=False)

        # first channel of the mask is the channel we care about.
        # make it three channels, white
        mask = Y[base_name][:,:,0]
        mask = np.stack([mask, mask, mask], axis=2)
        mask = mask * 200
        mask = np.array(mask, dtype=np.int8)
        # add the other masks as R, G, B
        for i in range(1, Y[base_name].shape[2]):
            mask[:, :, i-1] += np.array(Y[base_name][:, :, i] * 255, dtype=np.int8)

        rgb_name = os.path.join(in_dir, 'RGB', base_name)
        rgb_files = glob.glob('%s.*' % rgb_name)
        if len(rgb_files) == 1:
            # also TODO: do this stacking for single heatmaps as well?
            rgb = read_rgb(rgb_files[0])
            rgb = np.array(rgb, dtype=np.int8)
            gold = np.concatenate([rgb, mask], axis=1)
            heat_map = np.concatenate([gold, heat_map], axis=0)
        else:
            if len(rgb_files) > 1:
                print("Warning: more than one file matching %s.*" % rgb_name)
            else:
                print("Warning: no files matching %s.*" % rgb_name)
            heat_map = np.concatenate([mask, heat_map], axis=1)

        im = Image.fromarray(heat_map, "RGB")
        im.save(save_filename)
            
            

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_model', default="unet_3cW_0.h5",
                        help='Filename for saving the model')
    parser.add_argument('--load_model', default=None,
                        help=('Filename for loading a model, either as '
                              'a starting point for training or for testing'))

    parser.add_argument('--model_arch', default=ModelArch.UNET,
                        type=lambda x: ModelArch[x.upper()],
                        help=('Model architecture to use.  unet, 1x1 pixel classifier, or 3x3 conv pixel classifier.  unet/pixel/conv'))

    parser.add_argument('--model_type', default=ModelType.HABITAT,
                        type=lambda x: ModelType[x.upper()],
                        help=('Model type to use: habitat or village'))
    
    parser.add_argument('--separate_ceratophyllum',
                        dest='separate_ceratophyllum',
                        default=True, action='store_true',
                        help='Separate classes for emergent and ceratophyllum')
    parser.add_argument('--no_separate_ceratophyllum',
                        dest='separate_ceratophyllum',
                        action='store_false',
                        help='Combine classes for emergent and ceratophyllum')

    parser.add_argument('--train', dest='train',
                        default=True, action='store_true',
                        help='Train the model (default)')
    parser.add_argument('--no_train', dest='train',
                        action='store_false', help="Don't train the model")

    parser.add_argument('--val_patches', dest='val_patches',
                        default=False, action='store_true',
                        help='Use patches from the val set')
    parser.add_argument('--no_val_patches', dest='val_patches',
                        action='store_false',
                        help="Use the entire block of validation data (default)")

    parser.add_argument('--save_best_only', dest='save_best_only',
                        default=True, action='store_true',
                        help='Only save the best model when training, measured on val set')
    parser.add_argument('--no_save_best_only', dest='save_best_only',
                        action='store_false',
                        help="Save all the models.  Gets quite expensive")

    parser.add_argument('--process_heat_map', default=None,
                        help='A file on which to run the model')
    parser.add_argument('--save_heat_map', default=None,
                        help='Where to save the heat map, if applicable')
    parser.add_argument('--heat_map_dir', default=None,
                        help='Generate heat maps from this directory')
    parser.add_argument('--heat_map_save_dir', default=None,
                        help='A dir to save all the heat maps.  Will use --heat_map_dir if specified, otherwise will use --train_dir')

    parser.add_argument('--train_dir', default=DEFAULT_DIR,
                        help='Where to get the training data')
    
    parser.add_argument('--test_dir', default=None,
                        help='Where to get test data')

    args = parser.parse_args()
    return args

def print_args(args):
    """
    For record keeping purposes, print out the arguments
    """
    args = vars(args)
    keys = sorted(args.keys())
    print('ARGS:')
    for k in keys:
        print('%s: %s' % (k, args[k]))

def image_generator(dataset):
    X, Y = dataset
    assert X.keys() == Y.keys()
    for f in X.keys():
        print("GENERATING %s" % f)
        yield(X[f][np.newaxis], Y[f][np.newaxis])

def evaluate_dataset(model, data_type, test_dir):
    """
    Run the model on all of the files in the given test_dir

    Minor issues: this runs items one at a time instead of batching them
    # TODO: stack files somehow.  A couple issues:
    #   my old laptop can't run 50x512x512x8 sadly
    #   want to handle datasets with images of different sizes
    """
    print("Running test set on %s" % test_dir)
    test_set = read_images(data_type, test_dir)
    print("Number of elements: %d" % len(test_set[0]))
    #results = model.evaluate_generator(image_generator(test_set),
    #                                   steps=len(test_set[0]))
    #print('test loss, test_acc:', results)
    confusion = None
    for X, Y_true in image_generator(test_set):
        Y_pred = predict_single_image(model, X)
        axis = len(Y_true.shape) - 1
        Y_pred = np.argmax(Y_pred, axis=axis)
        Y_true = np.argmax(Y_true, axis=axis)
        if confusion is None:
            confusion = sklearn.metrics.confusion_matrix(Y_true.flatten(), Y_pred.flatten())
        else:
            confusion = confusion + sklearn.metrics.confusion_matrix(Y_true.flatten(), Y_pred.flatten())
    # TODO: pretty print this with labels and everything
    print(confusion)

def choose_heat_map_save_dir(args):
    """
    Choose a place to save heat maps based on the user args.

    Obviously the easiest place to put them is if the user specifies a save dir...
    """
    if args.heat_map_save_dir:
        return args.heat_map_save_dir
    candidate = os.path.join(os.getcwd(), "heat_maps")
    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate
    if not args.heat_map_dir:
        return None
    candidate = args.heat_map_dir + "_heat_maps"
    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate
    return None

def main():
    """Load a model or create a model with random weights.
    Then possibly train it.
    Finally, possibly display a heat map for a given image.

    To train:
      python [script name] --save_model [model name]

    Training images will be loaded from --train_dir

    To continue training an existing model, --load_model [model name]

    To display a heat map, --process_heat_map [8 channel .TIF image]

    To not train, --no_train

    Sample command line to retrain new model:
      python schisto_model.py --save_model softmax.h5

    Sample command line to show a single heat map:
      python schisto_model.py --load_model softmax.h5 --no_train --process_heat_map ../training_set/8_bands/28.TIF
      python schisto_model.py --load_model softmax.h5 --no_train --process_heat_map ../training_set/8_bands/28.TIF --save_heat_map 28_heat_map.bmp

    Sample command line to process a directory data into heat maps,
      with the originals appended to the images.
      If --heat_map_dir is specified, will try to use that directory
      for the images.  Otherwise will use train_dir

      python schisto_model.py --load_model softmax.h5 --no_train --heat_map_save_dir heat_maps

    Sample command to test a model:
      python schisto_model.py --load_model softmax.h5 --no_train --test_dir ../extra_set_trans

      python -u schisto_model.py --save_model village.h5 --train_dir ../village_set --model_type village --val_patches

    """
    args = parse_args()
    print_args(args)

    if args.load_model:
        model = tf.keras.models.load_model(args.load_model)
        output = model.outputs[0]
        num_classes = output.shape[-1]
        print("Loaded model %s.  Number of classes: %d" %
              (args.load_model, num_classes))
        print("Model description: %s" % model.name)
        model_arch, data_type = model.name.split(":")[:2]
        model_arch = ModelArch[model_arch]
        data_type = DataType[data_type]
        if data_type == DataType.HABITAT_SEPARATE:
            # relevant if you retrain a model
            print("Model built to detect snail habitat, using separate ceratophyllum class")
            args.separate_ceratophyllum = True
            args.model_type = ModelType.HABITAT
        elif data_type == DataType.HABITAT_COMBINED:
            print("Model built to detect snail habitat, combining both vegetation classes")
            args.separate_ceratophyllum = False
            args.model_type = ModelType.HABITAT
        elif data_type == DataType.VILLAGE:
            print("Model built to detect villages")
            args.model_type = ModelType.VILLAGE
        else:
            raise ValueError("Unable to determine data type to use: %s" % data_type.name)
    else:
        if args.model_type == ModelType.VILLAGE:
            num_classes = 2
            data_type = DataType.VILLAGE
        elif args.model_type == ModelType.HABITAT and args.separate_ceratophyllum:
            num_classes = 4
            data_type = DataType.HABITAT_SEPARATE
        elif args.model_type == ModelType.HABITAT and not args.separate_ceratophyllum:
            num_classes = 3
            data_type = DataType.HABITAT_COMBINED
        else:
            raise RuntimeError("Unknown model type %s" % args.model_type.name)

        model_name = args.model_arch.name + ":" + data_type.name
        print("Model description: %s" % model_name)
        if args.model_arch == ModelArch.UNET:
            model = unet(STARTING_LR, num_classes, model_name)
        elif args.model_arch == ModelArch.PIXEL:
            model = pixel_classifier(STARTING_LR, num_classes, model_name)
        elif args.model_arch == ModelArch.CONV:
            model = conv_classifier(STARTING_LR, num_classes, model_name)
        else:
            raise RuntimeError("Unknown model architecture %s" % args.model_arch.name)

    if args.train:
        X, Y = read_images(data_type, args.train_dir)
        train_set, val_set = split_images(X, Y)
        train(model, args.save_model, train_set, val_set, args)

    if args.process_heat_map:
        # TODO: would need to save the best model or just reload it
        # if we ran training and wanted to use the best
        print("Running model on %s" % args.process_heat_map)
        process_heat_map(model, read_8band_tif(args.process_heat_map),
                         display=True, save_filename=args.save_heat_map)

    if args.heat_map_dir or args.heat_map_save_dir:
        heat_map_dir = args.heat_map_dir
        if not heat_map_dir:
            heat_map_dir = args.train_dir
        heat_map_save_dir = choose_heat_map_save_dir(args)
        if heat_map_save_dir:
            print("Producing heat maps for all of %s" % heat_map_dir)
            print("Heat maps saved to %s" % heat_map_save_dir)
            process_heat_map_set(model, data_type,
                                 heat_map_dir, heat_map_save_dir)
        else:
            print("Unable to determine where to save heat maps.  Skipping.  Try setting --heat_map_save_dir")

    if args.test_dir:
        evaluate_dataset(model, data_type, args.test_dir)

if __name__ == '__main__':
    main()
