#!/usr/bin/env python
'''
Author: Olena Boiko

'''
import os
import rasterio
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
import segmentation_models as sm
from keras.optimizers import Adam

def load_patches(data_dir, subset="train", which="images", onehot=False):
    '''
    Load patches from a directory.

    Parameters
    ----------
    data_dir: string
        Path to the training data
    subset: string
        Which subset to load - train or test
    which: string
        Which patches to load - images or labels
    onehot: bool
        Do one-hot encoding, use this only for labels

    Returns
    -------
    array
        Array with patches
    '''
    patches = []

    for file in os.listdir(data_dir + f"{subset}/{which}/"):
        if file.endswith(".tif"):
            filepath = os.path.join(data_dir + f"{subset}/{which}/", file)
            with rasterio.open(filepath) as src:
                single_patch_img = src.read()
                single_patch_img = np.moveaxis(single_patch_img, 0, -1)
                patches.append(single_patch_img)
    patches = np.array(patches)
    if onehot == True:
        patches = to_categorical(patches)
    return patches

def class_counts_to_weights(class_feature_counts, n_classes, sparse=True):
    '''
    Convert class distribution statistics to training weights.

    Parameters
    ----------
    class_feature_counts: dict
        Distribution of feature counts by class
    n_classes: int
        Total number of classes to separate
    sparse: bool
        Specifies if the training patches are fully or sparsely labeled.
        If sparse=True, adds background class 0 with weight 0

    Returns
    -------
    list
        List with training weights
    '''
    class_feature_counts = [class_feature_counts[str(i)] for i in range(1,n_classes)]
    class_weights = [sum(class_feature_counts)/i for i in class_feature_counts]
    if sparse == True:
        class_weights = [0] + class_weights
    return class_weights

def compile_unet(backbone="resnet34", patch_height=256, patch_width=256,
                 patch_channels=4, n_classes=8, activation="softmax", class_weights=None,
                 learning_rate=0.001):
    '''
    Defines and compiles a U-Net model
    Relies on built-in architectures from the Segmentation Models library.

    Parameters
    ----------
    backbone: string
        Name of existing backbone to define U-Net architecture
    patch_height: int
        Height of a patch
    patch_width: int
        Width of a patch
    patch_channels: int
        Count of bands (use 4 for NAIP - red, green, blue, and nir)
    n_classes: int
        Total number of classes to separate
    activation: string
        Name of the activation function
    class_weights: list
        List of class weights to balance out under- and over-represented classes
    learning_rate: float
        Learning rate for the training

    Returns
    -------
    keras model
        An instance of keras model
    '''
    model = sm.Unet(
        backbone,
        encoder_weights=None,
        input_shape=(patch_height, patch_width, patch_channels),
        classes=n_classes,
        activation="softmax",
        decoder_block_type="transpose"
    )
    metrics = [sm.metrics.IOUScore(class_weights=class_weights)]
    loss = sm.losses.CategoricalCELoss(class_weights=class_weights)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
    )
    return model

def load_model_file(path, compile=False):
    '''
    Load a Keras model file from the file system.

    Parameters
    ----------
    path: str
        Path the the Keras model file on the filesystem.
    compile: bool
        Flag indicating whether or not to compile the model.

    Returns
    -------
    keras model
        The instance of keras model that was stored on the file system
    '''
    return load_model(path, compile = compile)
