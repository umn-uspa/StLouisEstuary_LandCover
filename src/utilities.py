#!/usr/bin/env python
'''
Generic functions and utilities

Author: Olena Boiko and Dr Jeffery Thompson

'''
import numpy as np
from shapely.geometry import Point, Polygon

def flatten(list):
    '''
    Function that converts nested lists into a flat list.
    
    Parameters
    ----------
    list : list
        Nested list to convert to individual elements
    
    Returns
    -------
    flattened list
        Single list individual elements
    '''
    return [li for sub_list in list for li in sub_list]


def rotate(image, reverse=False):
    '''
    Rotate the image to get all 4 90 degree rotations of the image.

    Parameters
    ----------
    image : numpy array
        Input image of shape (x, y, nb_channels).

    reverse : bool
        Flag to determine if the rotation happens in the forward
        direction, or the reverse direction. Reverse is basically
        used to indicate you want to 'undo' the rotation at the
        end of the process.

    Returns
    -------
    rotated
        Returns the 4 rotated/un-rotated images corresponding with
        the 90 degree rotations.
    '''
    rotated = []
    if reverse:
        rotated.append(image[0])
        rotated.append(np.rot90(image[1], axes=(0, 1), k=3))
        rotated.append(np.rot90(image[2], axes=(0, 1), k=2))
        rotated.append(np.rot90(image[3], axes=(0, 1), k=1))
    else:
        rotated.append(np.array(image))
        rotated.append(np.rot90(np.array(image), axes=(0, 1), k=1))
        rotated.append(np.rot90(np.array(image), axes=(0, 1), k=2))
        rotated.append(np.rot90(np.array(image), axes=(0, 1), k=3))
    return rotated

def mirror(image, reverse=False, mirror_index=4):
    '''
    Mirror an image to get each of the 4 mirrors of the image.

    Parameters
    ----------
    image : numpy array
        Input image of shape (x, y, nb_channels).

    reverse : bool
        Flag to indicate whether the mirroring happens or whether
            the results of previous mirroring should be undone.
    mirror_index : int
        The index where the mirrors are found in the image array.
            Used for book keeping basically.

    Returns
    -------
    mirrors
        Returns the 4 mirrored image, including the original.
    '''
    mirrors = []
    if reverse:
        mirrors.append(image[mirror_index][:, ::-1])
        mirrors.append(np.rot90(image[5], axes=(0, 1), k=3)[:, ::-1])
        mirrors.append(np.rot90(image[6], axes=(0, 1), k=2)[:, ::-1])
        mirrors.append(np.rot90(image[7], axes=(0, 1), k=1)[:, ::-1])
    else:
        image = np.array(image)[:, ::-1]
        mirrors.append(np.array(image))
        mirrors.append(np.rot90(np.array(image), axes=(0, 1), k=1))
        mirrors.append(np.rot90(np.array(image), axes=(0, 1), k=2))
        mirrors.append(np.rot90(np.array(image), axes=(0, 1), k=3))
    return mirrors

def rotate_mirror(image, reverse=False):
    '''
    Rotate and mirror an image to get the D_4 (D4) Dihedral group.
    https://en.wikipedia.org/wiki/Dihedral_group. The image is
    rotated and mirrored 8 times, in order to have all the possible
    90 degrees rotations.

    Basically a wrapper function for both rotate and mirror

    Parameters
    ----------
    image : numpy array
        Input image of shape (x, y, nb_channels).

    reverse : bool
        Flag to determine if we are undoing the previous
            rotation and mirroring.

    Returns
    -------
    mirrors
        Returns the 4 mirrored image, including the original.
    '''
    rot_mir =[]
    if reverse:
        rot_mir.append(rotate(image, reverse = True))
        rot_mir.append(mirror(image, reverse = True))
    else:
        rot_mir.append(rotate(image))
        rot_mir.append(mirror(image))

    return flatten(rot_mir)


def extent_to_polygon(left, bottom, right, top, dtype = 'polygon'):
    '''
    Convert bounding spatial extent to polygon.

    Parameters
    ----------
    left : float
        Left edge of bounding box.

    bottom : float
        Bottom edge of bounding box.

    right : float
        Right edge of bounding box.

    top : TYPE
        Top edge of bounding box.

    dtype: str
        Sting for the spatial data type. Currently only 'polygon' is supported.

    Returns
    -------
    polygon : shapely Polygon
        Polygon corresponding with the input coordinates

    '''
    if type(left) is int:
        left = float(left)
        bottom = float(bottom)
        right = float(right)
        top = float(top)

    ul = Point(left, top)
    ur = Point(right, top)
    lr= Point(right, bottom)
    ll = Point(left, bottom)

    if dtype == 'polygon':
        return Polygon([ul, ur, lr, ll, ul])

    else:
        print(f'Option not supported. Return type must be polygon')


def palette()-> dict:
    '''
    Generate the palette for rasterio write_colormap and other uses
    palette reference: https://personal.sron.nl/~pault/#sec:qualitative

    Returns
    -----------
    palette : dict
        Color palette for the output raster; compatible with rasterio write_colormap.
    '''

    _pal = {
        'FORESTED' : {'value' : 1, 'rgb' : (34, 136, 51)},
        'UNVEGETATED UNCONSOLIDATED' :{'value' :  2, 'rgb' : (238, 102, 119)},
        'SCRUB-SHRUB' : {'value' : 3, 'rgb' : (204, 187, 68)},
        'HERBACEOUS' : {'value' : 4, 'rgb' :  (102, 204, 238)} ,
        'HUMAN-MADE STRUCTURES' : {'value' : 5, 'rgb' : (170, 51, 119)},
        'UNVEGETATED ROCKY' : {'value' : 6, 'rgb' : (187, 187, 187)},
        'WATER': {'value' : 7, 'rgb' : (68, 119, 170)}
    }
    return _pal.keys(), {_pal[k]['value'] :_pal[k]['rgb'] for k in _pal.keys()}


def probabilities_to_classes(predictions):
    '''
    Convert the predictions to classes.

    Parameters
    ----------
    predictions : np.array
        The segmentation predictions to convert to classes.

    Returns
    -----------
    argmax : np.array
        The argmax of the prediction probabilities.
    '''

    return np.argmax(predictions, axis=-1)
