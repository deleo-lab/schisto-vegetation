import argparse
import glob
import os
import random

from PIL import Image
import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as backend

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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

STARTING_LR = 4e-5

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

def conv_classifier(learning_rate, classes, input_channels=8):
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
    conv5 = Conv2D(classes,1, activation = 'softmax')(conv4)

    model = tf.keras.Model(inputs=inputs, outputs=conv5)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
    


def pixel_classifier(learning_rate, classes, input_channels=8):
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
    conv5 = Conv2D(classes,1, activation = 'softmax')(conv4)

    model = tf.keras.Model(inputs=inputs, outputs=conv5)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
    
def unet(learning_rate, classes, input_channels=8):
    """
    Builds a u-net out of TF Keras layers

    Batch Normalization layers are added in between each pair of Conv2D
    Note that the input size is unconstrained, as the network will
    operate on any size image (past a certain minimum)
    """
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
    conv10 = Conv2D(classes,1, activation = 'softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    return model

# try to map the visible channels to RGB
SAT_TRANSFORM = np.array([[ 0,  0,  0, .4, .4, .2, 0, 0],
                          [ 0, .3, .4, .3,  0,  0, 0, 0],
                          [.3, .4, .3,  0,  0,  0, 0, 0]]).T

GREY_TRANSFORM = np.array([0.299, 0.587, 0.114])

def transform_rgb_image(image):
    rgb = np.tensordot(image, SAT_TRANSFORM, 1) / 2048
    return rgb
    
def read_tif(image_filename):
    return io.imread(image_filename)

def read_mask(mask_file):
    return io.imread(mask_file)

def read_rgb(rgb_file):
    return io.imread(rgb_file)

def read_images(num_classes, image_path):
    """
    Reads images from the given path

    20% (could be parametrized) are split off into a val set
    Returns (train_set, val_set) where both training sets are
      (X, Y), input channels & masks
    If num_classes == 4, four masks are returned, with cera being mask 0
    If num_classes == 3, cera and emergent plants are combined
    """
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    tif_files = glob.glob(PATH_TIF % image_path + '*.TIF')
    tif_files = sorted(tif_files)
    
    for image_filename in tif_files:
        # note: this image is used later when building heat maps
        # of the directory.  would need to re-read the image if
        # it is manipulated at all here.
        img_m = read_tif(image_filename)
        _, base_name = os.path.split(image_filename) # filename -> 05.TIF
        base_name, _ = os.path.splitext(base_name)   # remove the .TIF
        
        #create 3d mask where each channel is the mask for a specific class
        mask_water = read_mask(WATER_MASK % image_path + '%s.png'%base_name)
        mask_land = read_mask(LAND_MASK % image_path + '%s.png'%base_name)
        mask_emerg = read_mask(EMERGENT_MASK % image_path +'%s.png'%base_name)
        mask_cera = read_mask(CERA_MASK % image_path + '%s.png'%base_name)

        if num_classes == 3:
            mask = np.stack([mask_cera + mask_emerg, mask_land, mask_water],
                            axis=2)
        elif num_classes == 4:
            mask = np.stack([mask_cera, mask_emerg, mask_land, mask_water],
                            axis=2)
            
        mask = mask/255
        mask_sum = np.sum(mask, axis=2)
        if np.min(mask_sum) == 0:
            print("Warning: %s has %d unlabeled pixels" %
                  (base_name, len(np.where(mask_sum == 0)[0])))
        if np.max(mask_sum) > 1:
            print("Warning: %s has %d pixels with multiple labels" %
                  (base_name, len(np.where(mask_sum > 1)[0])))

        # use 80% of images for train, 20% for validation
        if len(X_DICT_TRAIN) < len(tif_files) * 0.8:
            X_DICT_TRAIN[base_name] = img_m
            Y_DICT_TRAIN[base_name] = mask
            dataset = "train"
        else:
            X_DICT_VALIDATION[base_name] = img_m
            Y_DICT_VALIDATION[base_name] = mask
            dataset = "val"

        print('read %s (%s)' % (image_filename, dataset))

    train_set = (X_DICT_TRAIN, Y_DICT_TRAIN)
    val_set = (X_DICT_VALIDATION, Y_DICT_VALIDATION)
    return train_set, val_set

def stack_dataset(dataset):
    """
    Returns the X, Y dicts from read_images as 2 numpy arrays
    """
    x_dict, y_dict = dataset
    X = []
    Y = []
    for k in x_dict.keys():
        X.append(x_dict[k])
        Y.append(y_dict[k])
    return np.array(X), np.array(Y)
    

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
        print ("Finished epoch %d" % i)

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
    batch = np.expand_dims(image, axis=0)
    prediction = np.squeeze(model.predict(batch))
    return prediction

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

    grey = washed_greyscale(test_image)

    # grey is now from 0.5 to 1, a washed out greyscale version
    # of the original image.  the purpose is to make something that
    # can have blue...red...yellow heat map colors imposed on top

    cera_prediction = prediction[:, :, 0]
    blue_prediction = np.maximum(0.4 - cera_prediction, 0.0)
    blue_prediction = np.sqrt(blue_prediction)
    red_prediction = np.minimum(cera_prediction / 0.4, 1.0)
    red_prediction = np.sqrt(red_prediction)
    green_prediction = np.maximum(np.minimum((cera_prediction - 0.4) / 0.4, 1.0), 0)
    heat_map = np.stack([red_prediction, green_prediction,
                         blue_prediction], axis=2)

    # now we make a similar image, but just of the classification
    # if the prediction for cera is high enough, yellow, otherwise blue
    classification = np.round(np.squeeze(prediction))

    red_prediction = classification[:, :, 0] + classification[:, :, 1]
    green_prediction = classification[:, :, 0] + classification[:, :, 2]
    blue_prediction = classification[:, :, 0]
    if classification.shape[2] > 3:
        blue_prediction = blue_prediction + classification[:, :, 3]
    classification = np.stack([red_prediction, green_prediction,
                               blue_prediction], axis=2)

    vis = np.concatenate([heat_map, classification], axis=1)
    # grey is the greyscale of the original image
    # since the visualization is two images side by side,
    # stack two copies of grey...
    grey = np.concatenate([grey, grey], axis=1)

    # .. then multiply pixel by pixel with the visualization
    # so the prediction has some texture to it
    rgb = vis * grey

    # since values are 0..1, rescale to RGB
    rgb = rgb * 255
    rgb = np.array(rgb, dtype=np.int8)
    im = Image.fromarray(rgb, "RGB")
    
    if display:
        im.show()

    if save_filename:
        im.save(save_filename)

    return rgb

    
def process_heat_map_set(model, dataset, in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    X, Y = dataset
    files = X.keys()
    for base_name in files:
        print("Processing %s" % base_name)
        # base_name should already have the .tif split off
        save_filename = os.path.join(out_dir, base_name + ".bmp")
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

    parser.add_argument('--model_type', default='unet',
                        help=('Model type to use.  unet, 1x1 pixel classifier, or 3x3 pixel classifier.  unet/pixel_classifier/conv_classifier'))

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

    parser.add_argument('--heat_map', default=None,
                        help='A file on which to run the model')
    parser.add_argument('--save_heat_map', default=None,
                        help='Where to save the heat map, if applicable')
    parser.add_argument('--heat_map_dir', default=None,
                        help='A dir to save all the heat maps from train & val')

    parser.add_argument('--training_home', default='c:/Users/horat/Documents/hai/schisto/training_set',
                        help='Where to get the training data')
    
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
        
if __name__ == '__main__':
    """
    Load a model or create a model with random weights.
    Then possibly train it.
    Finally, possibly display a heat map for a given image.

    To train:
      python [script name] --save_model [model name]

    Training images will be loaded from --training_home

    To continue training an existing model, --load_model [model name]

    To display a heat map, --heat_map [8 channel .TIF image]

    To not train, --no_train

    Sample command line to retrain new model:
      python schisto_model.py --save_model foo.h5

    Sample command line to show a single heat map:
      python schisto_model.py --load_model foo.h5 --no_train --heat_map ../training_set/8_bands/28.TIF 

    Sample command line to process all of the training and validation
      data into heat maps, with the originals appended to the images:
      python schisto_model.py --load_model foo.h5 --no_train --heat_map_dir heat_maps
    """
    args = parse_args()
    print_args(args)

    if args.load_model:
        model = tf.keras.models.load_model(args.load_model)
        output = model.outputs[0]
        num_classes = output.shape[-1]
        print("Loaded model %s.  Number of classes: %d" %
              (args.load_model, num_classes))
    else:
        if args.separate_ceratophyllum:
            num_classes = 4
        else:
            num_classes = 3

        if args.model_type == 'unet':
            model = unet(STARTING_LR, num_classes)
        elif args.model_type == 'pixel_classifier':
            model = pixel_classifier(STARTING_LR, num_classes)
        elif args.model_type == 'conv_classifier':
            model = conv_classifier(STARTING_LR, num_classes)
        else:
            raise RuntimeError("Unknown model type %s" % args.model_type)

    if args.train or args.heat_map_dir:
        train_set, val_set = read_images(num_classes, args.training_home)

    if args.train:
        train(model, args.save_model, train_set, val_set, args)

    if args.heat_map:
        # TODO: would need to save the best model or just reload it
        # if we ran training and wanted to use the best
        print("Running model on %s" % args.heat_map)
        process_heat_map(model, read_tif(args.heat_map),
                         display=True, save_filename=args.save_heat_map)

    if args.heat_map_dir:
        print("Producing heat maps for all of %s" % args.training_home)
        process_heat_map_set(model, train_set,
                             args.training_home, args.heat_map_dir)
        process_heat_map_set(model, val_set,
                             args.training_home, args.heat_map_dir)        
