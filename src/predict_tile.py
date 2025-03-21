#!/usr/bin/env python

'''
Author: Dr. Jeffery Thompson

Original code is from the following source. It comes with MIT License so please mention
the original reference when sharing. Note: clipping not yet implemented

The original code has been adapted to be more memory efficient. Also provides
    for turning off using the D_4 (D4) Dihedral group (mirror option) for predictions
    as well as using a clipping function that side steps the need to average the
    overlapping tiles from the original. Also the original version uses np.mean for
    averaging predictions, where as a geometric mean is appropriate (both overlap
    patches and Dihedral group are correlated, and not independent of one another)

    https://medium.com/kaggle-blog/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey-85395e51e118

# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE

'''
import gc

import numpy as np

from functools import partial
from itertools import product, tee
from scipy import signal
from scipy.stats.mstats import gmean
from tqdm import tqdm

from matplotlib import pyplot as plt

from utilities import rotate_mirror

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False

def _spline_window(window_size, power=2, tukey = False):
    '''
    Create a 2D spline window.

    Parameters
    ----------
    window_size : int
        Size of the patchs
    power : int, optional
        Exponential to use for window, by default 2
    tukey : bool, optional
        Use Tukey's tri-weight function , by default False
        Default is to use Squared spline (power=2) window function
        https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2

    Returns
    -----------
    spline window
    '''
    # use tukey window if specified, else the squared spline from the original code

    if tukey:
        window = signal.windows.tukey(window_size, alpha=0.5)
        window = np.outer(window, window) ** power
    else:
        # this is the original function for _spline_window
        intersection = int(window_size/4)
        window_outer = (abs(2*(signal.windows.triang(window_size))) ** power)/2
        window_outer[intersection:-intersection] = 0

        window_inner = 1 - (abs(2*(signal.windows.triang(window_size) - 1)) ** power)/2
        window_inner[:intersection] = 0
        window_inner[-intersection:] = 0

        window = window_inner + window_outer
        window = window / np.mean(window)

    return window

cached_2d_windows = dict()
def _window_2D(window_size, power=2, **kwargs):
    '''
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.

    Parameters
    ----------
    window_size : int
        Size of patchs
    power : int, optional
        Exponent for the polynomial, by default 2

    kwargs :
        There to pass the boolean for tukey function to the window, if specified

    '''
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)

    if key in cached_2d_windows:
        window = cached_2d_windows[key]
    else:
        if 'tukey' in kwargs.keys():
            window = _spline_window(window_size, power, tukey = True)
        else:
            window = _spline_window(window_size, power)
        #SREENI: Changed from 3, 3, to 1, 1 [below]
        window = np.expand_dims(np.expand_dims(window, 1), 1)
        window = window * window.transpose(1, 0, 2)
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(window[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = window

    return window

def _pad_img(img, window_size, subdivisions, remove_pad=False):
    '''
    Expands the image by zero padding the original input image.
    Basically, adds borders to img for a "valid" border pattern
    according to "window_size" and "subdivisions".

    Parameters
    ----------
    img : numpy array
        Image is an np array of shape (x, y, nb_channels).
    window_size : int
        The patch size used in the modeling process.
    subdivisions : int
        The amount of overlap between patches to use in prediction.
    remove_pad : boolean
        Add padding around the image. Default False.
        If True, then remove padding from image (clip it).
    Returns
    -------
    padded image : numpy array (x + pad, y + pad, nb_channels)
        Returns the expanded image with zeros along the borders
    '''
    if remove_pad:
        aug = int(round(window_size * (1 - 1.0/subdivisions)))
        img_pad = img[
            aug:-aug,
            aug:-aug,
            :
        ]
    else:
        aug = int(round(window_size * (1 - 1.0/subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
        img_pad = np.pad(img, pad_width=more_borders, mode='reflect')
        gc.collect()

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(img_pad)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()

    gc.collect()
    return img_pad

def image_iterator(shape, window_size, subdivisions):
    '''
    Generate iterator of image row, columns indexes for processing the image

    Parameters
    ----------
    shape : tuple
        The shape of the image array in row/height/y, column/width/x order.
        This is the default numpy order for .shape()
    The patch size to use for prediction. Must match the size used in training.
        window_size -> patches_resolution_along_X == patches_resolution_along_Y == window_size
    subdivisions : int
        The amount of overlap to use for smoothing predictions

    Returns
    -------
    iterator
        Iterator with tuples of (row, column) indexes to use in creating and reassembling patches  

    '''

    step = int(window_size/subdivisions)
    # Jeff T. note, this is the spec from the origial
    #   but this is confusing since padx_len is the shape in the y dimension
    #   and pady_len in the x dimension!
    ylen = shape[0]
    xlen = shape[1]

    return  product(
        range(0, ylen-window_size+1, step),
        range(0, xlen-window_size+1, step),
    )

def _overlap_patches(padded_img, window_size, subdivisions, nb_classes) -> np.ndarray:
    '''
    Create tiled, overlapping patches using the window size and subdivisions. Overlap
        is calculated by window_size / subdivisions

    Parameters
    ----------
    padded_img : numpy array
        Zero-padded version of the image to use for prediction
    window_size : int
        The patch size to use for prediction. Must match the size used in training.
        window_size -> patches_resolution_along_X == patches_resolution_along_Y == window_size
    subdivisions : int
        The amount of overlap to use for smoothing predictions
    nb_classes : int
        The number of classes to be predicted.

    Returns
    -------
    numpy array
        Array containing either overlapping patches for prediction or
            reassembled mosaic of predictions 

    '''
    assert len(padded_img.shape) == 3, 'Expected image with shape [x, y, bands]'
    # # Going to use row, column to not be as confusing (maybe)
    # #   start with row 0, then move column-wise; then row 1 then column-wise
    rc_ix = image_iterator(
        shape = (padded_img.shape[0], padded_img.shape[1]),
        window_size = window_size,
        subdivisions = subdivisions
    )
    rc_ix, rc_ix_copy = tee(rc_ix)

    subdivs = [padded_img[rc[0]:rc[0]+window_size, \
                        rc[1]:rc[1]+window_size, :] \
                        for rc in rc_ix]

    return list(rc_ix_copy), np.array(subdivs)

def _patches_to_image(patches, window_size, subdivisions, nb_classes, rc_ix, padded_shape, smooth=True) -> np.ndarray:

    '''
    Reassemble the tile from constituent overlapping patches

    Parameters
    ----------
    patches: np.array
        Array of the overlapping patches to reassemble into the image.
    shape : tuple
        The shape of output image array in row/height/y, column/width/x order.
        This is the default numpy order for .shape()
    window_size : int
        The patch size to use for prediction. Must match the size used in training.
            window_size -> patches_resolution_along_X == patches_resolution_along_Y == window_size
    subdivisions : int
        The amount of overlap to use for smoothing predictions
    nb_classes : int
        The number of classes to be predicted.
    rc_ix : iterator
        Row, Column Index (rc_ix) is the row column index iterator that corresponds to the
            window that is being processed.
    padded_shape: tuple of ints
        The shape of the padded shape
    smooth : boolean
        Determines if the patches should be smoothed when reassembling the tile.
            Default: True

    Returns
    -------
    image : numpy array
        Reconstructed image array from the overlapping patches
    '''
    assert len(patches.shape) == 4, 'Expected array with shape [n patches, window_size, window_size, nb_classes]'

    img = np.zeros(shape=list(padded_shape[:-1]) + [nb_classes])

    rc_ix = iter(rc_ix)
    for patch, rc in zip(patches, rc_ix):

        # need to get the image portion to insert
        #   this can be a fraction of a patch, if the tile
        #   is not a multiple of 256. patches almost always seem
        #   to be 256 x 256 at this stage, probably due to broadcasting
        #   happening in a previous step (likely smoothing)
        #
        #   the difference in dimensions is a problem. but patch can
        #       be reduced to the size of image fraction to insert, as
        #       anything outside those 2d image boundaries is a suspect
        #       value.
        #
        # Hence, compare shapes, and where they do not agree, truncate the
        #   in x, y dims
        img[rc[0]:rc[0]+window_size, rc[1]:rc[1]+window_size] = \
            img[rc[0]:rc[0]+window_size, rc[1]:rc[1]+window_size] + patch

    if smooth:
        return np.divide(img, (subdivisions ** 2))
    else:
        return img

def _smooth_patches(patches, window_size, smooth = True, **kwargs):

    '''
    Smooth the predicted patches.

    Parameters
    ----------
    patches : numpy array
        Array with overlapping patches to smooth.
    window_size : int
        The patch size to use for prediction. Must match the size used in training.
            window_size -> patches_resolution_along_X == patches_resolution_along_Y == window_size
    smooth: bool
        Determines if the patches are smoothed or not. Default: True
    kwargs : additional options
        Place holder for Tukey smoothing option. Not yet implemented

    Returns
    -------
    smoothed patch : numpy array
        Array where prediction probabilities for all patches have been smoothed
    '''
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2, **kwargs)

    if isinstance(patches, list):
        patches = np.stack(patches, axis=0)

    # the original smoothing worked by moving back to the padded image space
    #   first an array of zeros with shape [x, y, nb_classes] was created
    #   next, the algorithm went row by row of that array, updating it
    #       with the patch. when not in the left most edge of the column
    #       it would combine the current patch with the right most side of the
    #       previous patch.
    #
    # Here, smoothing is achieved by working through the array of patches without
    #   converting to the final image space. the logic is to only do the one
    #   function (smoothing) and not convert back to image space
    #
    #   for this to work, we use the patch array, and the row, column indexes
    #       (rc_ix) used to build the array to determine if a patch is on the
    #       left hand side of the image [c index == 0]. in those cases, there
    #       is no previous patch, so we simply have the current patch. by using a
    #       mask of zeros in that case, we simply add the current patch to the mask
    #
    #   the mask is then set to be the right hand side of the current patch
    #       combined with an array of zeros with dims of:
    #            [window_size, window_size / subdivs, nb_classes]
    #       this gives the correct patch dimensions [window, window, nb_classes]
    #       that can be used to track the required portions of the previous
    #       patch for smoothing that can be combined with the next patch when
    #       the loop executes again.
    #
    #   so in effect, we not just smooth in this function, and don't convert
    #       back to image array space. That is to say, we smooth on an array of
    #       dims corresponding to [number_patches, patch x dim, patch y dim, number classes]
    #       rather than having to convert and array with the dims in last line, to
    #       an output array with [padded image y dim, padded image x dim, number classes]
    #
    #   in other words, do on thing per function
    if smooth:
        return np.multiply(patches, WINDOW_SPLINE_2D)
    else:
        return patches

def predict_patches(patches, model, batch_size=32, loop=True):
    '''
    Predict patches using the U-Net model.

    Parameters
    ----------
    patches : numpy array
        Numpy array with patches for prediction. Shape is (n_patches, x, y, n_bands)
    model : keras model
        The trained UNet model to use for prediction,
    batch_size : int
        The batch size to use when predicting tiles. Default: 32
    loop : bool
        The determines if predictions happen using a for loop, or if they
            are predicted using one call. On MSI, prediction using a single
            call did not work, and so prediction  happens on slices of the patches
            array in a loop.

    Returns
    -------
    list
        Predicted segmentation for the patch
    '''
    if loop:
        # to process using slices, crate arrays with start/end indexes
        slice_s = np.arange(0, patches.shape[0], batch_size, dtype='int')
        slice_e = np.arange(batch_size, patches.shape[0], batch_size, dtype='int')

        assert slice_e.shape[0] == slice_s.shape[0] -1, \
            'start slice indexes must be 1 larger than end slice indexes'
        pred_patches = []
        for i in range(slice_s.shape[0]):
            # when on last iteration, only need start index
            if i == slice_s.shape[0] - 1:
                pred_patches.append(model.predict(patches[slice_s[i]:, :, :, :], verbose=0))
            else:
                pred_patches.append(model.predict(patches[slice_s[i]:slice_e[i], :, :, :], verbose=0))

        return np.concatenate(pred_patches, axis=0)

    # NOTE: do not use this method on MSI GPUs -
    #   model.predict as used below, does not run correctly
    #       and produces garbage output. that is why the predictions
    #       are done above in a for loop. :/
    else:
        return model.predict(patches, batch_size = batch_size)

def segment_tile(model, tile, window_size, subdivisions, nb_classes, smooth= True, mirror = True, tukey = False):
    '''
    Applies a smoothing approach to predict segmentation classes using a trained U-Net model in keras.

    Parameters
    ----------
    unet_model : keras model
        The trained semantic model.
    tile : numpy array
        The image tile to generate smoothed predictions for.
            Has shape of (x, y, nb_channels).
    window_size : int
        The patch size used in the trained model
    subdivisions : int
        The overlap between subsequent patches for smoothing.
        Basically, the divisor used with window_size to figure out
        pixel overlap between consecutive patches. Also used for
        zero-padding image during prediction.
    nb_classes : int
        The number of segmentation classes in the trained model.
    smooth : bool
        Determines if smoothing is applied to the tile or not.
    mirror : bool
        Determines whether to applying mirroring to the segmentation process or not.
        Default is to use mirroring during segmentation.

    Returns
    -------
    img_predict : numpy array
        The segmented image, with smoothed predictions.
    '''
    _predict_patches = partial(predict_patches, model=model, batch_size = 256, loop = True)

    img_pad = _pad_img(tile, window_size, subdivisions)
    if mirror:
        print(f'running mirror...')
        pad_mirrors = rotate_mirror(img_pad)

    else:
        pad_mirrors = [img_pad]

    # variable for predictions on the mirrored images
    mirrored_preds = []
    for pm in tqdm(pad_mirrors):
        # for every mirror in  D_4 (D4) Dihedral group
        rc_ix_list, patches = _overlap_patches(
            padded_img = pm,
            window_size = window_size,
            subdivisions = subdivisions,
            nb_classes = nb_classes,
        )

        patches_pred = _predict_patches(patches)
        # the smoothing process currently works by converting from patches
        #   back to padded image
        if smooth:
            patches_pred = _smooth_patches(
                patches=patches_pred,
                window_size= window_size,
            )

        padded_predict = _patches_to_image(
            patches= patches_pred,
            window_size=window_size,
            subdivisions= subdivisions,
            nb_classes = nb_classes,
            rc_ix = rc_ix_list,
            padded_shape = pm.shape,
            smooth = True
        )

        mirrored_preds.append(padded_predict)

    # if mirror was specified:
    #   reverse reverse it;
    #    else it doesn't need reversing
    if mirror:
        print(f'reversing mirror...')
        padded_result = rotate_mirror(mirrored_preds, reverse =True)
        padded_result = np.stack(padded_result, axis = 0)
        # this is a list where image mirrors and rotated images
        #   have been 'undone'. These must be averaged, and because
        #   they are not independent, they should be geometric means
        padded_result =  gmean(padded_result, axis=0)
    else:
        print(f'no mirroring; mirror_preds is type: {type(mirrored_preds)}')
        padded_result = mirrored_preds[0]

    # reverse the padding
    img_predict = _pad_img(
        img = padded_result,
        window_size = window_size,
        subdivisions = subdivisions,
        remove_pad=True
    )
    return img_predict
