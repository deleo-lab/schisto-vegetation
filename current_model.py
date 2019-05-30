import argparse
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

HOME = "."
path_TIF = '%s/training_set/8_bands_set3/' % HOME
water_mask = '%s/training_set/water_mask_set3/' % HOME
land_mask = '%s/training_set/land_mask_set3/' % HOME
emergent_mask = '%s/training_set/emergent_mask_set3/' % HOME
ceratophyllum_mask = '%s/training_set/cera_mask_set3/' % HOME

num_training_images = 50

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
    x_dict, y_dict = dataset
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    #print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)

def unet(learning_rate, classes, input_channels=8):
    # Can represent variable input dimensions by None
    inputs = Input((None, None, input_channels))
    conv1 = Conv2D(32, (3, 3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    # TODO: it might be better to separate activation so that it goes
    #   conv -> activation -> bn
    # but sadly TF 2.0 alpha has a bug saving models with a separate activation layer
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
    drop4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.1)(conv5)

    up6 = Conv2D(256, (2, 2), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis=3)
    conv6 = Conv2D(256, (3, 3),activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # TODO: it is not clear we want these BN layers - there was
    # slightly better performance with just the up, not the down, but
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
    conv10 = Conv2D(classes,1, activation = 'sigmoid')(conv9)

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
    img_m = io.imread(image_filename)
    return img_m

def read_images(num_classes):
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()
    for i in range (0,num_training_images):
        # TODO: use the directory listing directly rather than
        # expecting hardcoded names
        image_filename = path_TIF + '%d.TIF' % i
        img_m = read_tif(image_filename)
        
        #create 3d mask where each channel is the mask for a specific class
        mask = np.zeros((512,512,num_classes))
        
        mask_water = io.imread(water_mask + '%d.png'%i)
        mask_land = io.imread(land_mask + '%d.png'%i)
        mask_emerg = io.imread(emergent_mask +'%d.png'%i)
        mask_cera = io.imread(ceratophyllum_mask + '%d.png'%i)

        if num_classes == 3:
            mask[:,:,2] = mask_water[:,:]
            mask[:,:,1] = mask_land[:,:]
            mask[:,:,0] = mask_emerg[:,:]
            mask[:,:,0] += mask_cera[:,:]
        elif num_classes == 4:
            mask[:,:,3] = mask_water[:,:]
            mask[:,:,2] = mask_land[:,:]
            mask[:,:,1] = mask_emerg[:,:]
            mask[:,:,0] = mask_cera[:,:]            
            
        mask = mask/255        

        # use 80% of images for train, 20% for validation
        if i < num_training_images * 0.8:
            X_DICT_TRAIN[i] = img_m
            Y_DICT_TRAIN[i] = mask
        else:
            X_DICT_VALIDATION[i] = img_m
            Y_DICT_VALIDATION[i] = mask

        print(image_filename + ' read')

    train_set = (X_DICT_TRAIN, Y_DICT_TRAIN)
    val_set = (X_DICT_VALIDATION, Y_DICT_VALIDATION)
    return train_set, val_set

def stack_dataset(dataset):
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

def display_heat_map(model, filename):
    test_image = read_tif(filename)
    test_batch = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_batch)

    rgb_test_image = transform_rgb_image(test_image)
    grey = np.tensordot(rgb_test_image, GREY_TRANSFORM, 1)
    grey = np.expand_dims(grey, axis=2)
    grey = grey - grey.min()
    if grey.max() > 0.0:
        grey = grey / grey.max()
    else:
        grey = grey + 1.0
    grey = grey / 2.0 + 0.5

    # grey is now from 0.5 to 1... a washed out greyscale version
    # of the original image.  the purpose is to make something that
    # can have blue...red...yellow heat map colors imposed on top

    prediction = np.squeeze(prediction)[:, :, 0]
    blue_prediction = np.maximum(0.4 - prediction, 0.0)
    blue_prediction = np.sqrt(blue_prediction)
    red_prediction = np.minimum(prediction / 0.4, 1.0)
    red_prediction = np.sqrt(red_prediction)
    green_prediction = np.maximum(np.minimum((prediction - 0.4) / 0.4, 1.0), 0)
    heat_map = np.stack([red_prediction, green_prediction,
                         blue_prediction], axis=2)

    rgb = heat_map * grey

    rgb = rgb * 256
    rgb = np.array(rgb, dtype=np.int8)
    im = Image.fromarray(rgb, "RGB")
    im.show()

    
def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_model', default="unet_3cW_0.h5",
                        help='Filename for saving the model')
    parser.add_argument('--load_model', default=None,
                        help=('Filename for loading a model, either as '
                              'a starting point for training or for testing'))

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
                        
    args = parser.parse_args()
    return args

def print_args(args):
    args = vars(args)
    keys = sorted(args.keys())
    print('ARGS:')
    for k in keys:
        print('%s: %s' % (k, args[k]))
        
if __name__ == '__main__':
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

        model = unet(STARTING_LR, num_classes)
        
    if args.train:
        train_set, val_set = read_images(num_classes)
        train(model, args.save_model, train_set, val_set, args)

    if args.heat_map:
        print("Running model on %s" % args.heat_map)
        display_heat_map(model, args.heat_map)
