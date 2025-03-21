#!/usr/bin/env python
'''
Code for converting polygons to patches that can be used
    by U-Net

Author: Dr. Jeffery Thompson

'''
import json
import os
import rasterio
import timeit

import geopandas as gpd
import numpy as np
import rasterio

from pathlib import Path
from rasterio.windows import Window
from rasterio.features import rasterize as rio_rasterize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from parsers import polygons_to_patches_parser
from utilities import flatten, rotate, mirror, rotate_mirror, extent_to_polygon
from raster_utilities import  raster_metadata

from predict_tile import image_iterator

RANDOM = 42

def update_feature_counts(features, series, set_type):
    '''
    Internal function to initialize/update feature counts when generating patches.
        Features counts are important for determining model weights.

    Parameters:
    -----------
    features : dict
        The dict containing the count of features.
    series : GeoPandas Series
        The series containing the features that are to be counted.
    set_type : str
        Whether the feature is part of 'train' or 'test'

    Returns:
    --------

    features : dict
        Feature counts are updated using the series.
    '''

    valid_types = ['train', 'test']
    assert set_type in valid_types, 'Unexpected set type: one of train, or test expected'

    for s in series:
        if s not in features[set_type].keys():
            features[set_type][s] = 1
        else:
            features[set_type][s] += 1

def create_polygon(affine_coord, window, pixel_size):
    '''
    Create a polygon from components of Rasterio Widow object

    affine_coord : tuple
        The coordinates to the upper left pixel of the image
    window : rasterio Window object
        The window information for the patch to generate a polygon
    pixel_size : float
        The pixel size to use when creating the polygon.

    Returns:
    --------
    extent_to_polygon
        The extent of the polygon is returned (left, bottom, right, top)

    '''

    return extent_to_polygon(
        left = affine_coord[0] + window.col_off * pixel_size,
        bottom= affine_coord[1] - (window.row_off + window.height) * pixel_size,
        right = affine_coord[0] + (window.col_off + window.width) * pixel_size,
        top = affine_coord[1] - window.row_off * pixel_size
    )

def write_patches(path, augmented, file_root, meta):
    '''
    Write out the augmented dataset

    Parameters:
    -----------
    path : str
        Path to store the augmented train-test dataset
    augmented : numpy array
        The augmented dataset to write to the file system.
    file_root : str
        The base name for the output file. Augmented datasets use numbers in them
        to differentiate whether they are mirrored etc.
    meta : rasterio metadata
        Rasterio metadata needed to write the augmented GeoTiffs.

     Returns:
    --------
    None
        Writes the patches as GeoTiff files.
    '''
    Path(path).mkdir(parents=True, exist_ok=True)

    for i in range(len(augmented)):
        imgOut = np.moveaxis(augmented[i], -1, 0)

        with rasterio.open(f"{path}{file_root}-{i}.tif", 'w', **meta) as dst:
            dst.write(imgOut)

def gen_split_indexes(gdf, col, train_pct, test_pct):
    '''
    Splits the train and test data according to provided fractions

    Parameters:
    -----------
    gdf : geopandas GeoDataFrame
        The GeoDataFrame that will be used to generate train-test patches.
    col : str
        The name of the column in the gdf that contains the label values.
    train_pct : float
        The fractional percentage of polygons to use as training data.
    test_pct : float
        The fractional percentage of polygons to use as test data.

    Returns
    -------
    class_weights : dict
        The weighting of the features - needed for U-Net modeling.
    split_dict : dict
        The dictionary of feature types, indicating which index in the
            supplied GeoDataFrame was allocated to the train set, and
            which features were allocated to the test set. Can be
            used to track which features were trained | test in a given
            model run.
    '''

    class_weights = gdf[col].value_counts(sort=False).to_dict()
    split_dict ={}

    for k in class_weights.keys():
        train_ix, test_ix = \
            train_test_split(
                gdf[gdf[col] == k].index.values,
                train_size=train_pct,
                test_size= test_pct,
                random_state = RANDOM
            )

        split_dict[k] ={
            'train' : train_ix,
            'test' : test_ix
        }

    return class_weights, split_dict

def rasterizer(col, geom, fill, shape, transform, dtype):
    '''
    Convert GeoDataFrame column and geometry to raster

    Parameters
    ----------
    col : pandas Series
        Series values to convert to raster
    geom : GeoPandas Geometry
        Geopandas geometry object for Series
    fill : int or float
        The fill/default value for the raster
    shape : tuple
        The (height, width) of the raster being generated
    transform : Affine.transform
        The linear transform for the raster
    dtype : data type
        _description_
        
    Returns
    -------
    raster
        numpy array of corresponding to gridded features
    '''
    return rio_rasterize(
        ((geo, value) for geo, value in zip(geom, col)),
        out_shape = shape,
        fill = fill,
        transform = transform,
        dtype = dtype
    )


def main():
    '''
    Command line script for rasterizing feature patches from the mosaic on the file system

    '''
    parser = polygons_to_patches_parser()
    args = parser.parse_args()
    assert np.isclose(np.sum([args.test_split, args.train_split]), 1.0) == 1.0, 'Split fractions need to add to one.'

    features={}
    counter = 0
    file_len=10

    meta = raster_metadata(args.mosaic_path)
    gdf = gpd.read_file(args.polygons_path).to_crs(meta['crs'])

    # generate rasterio windows, then use those windows to generate polygon geometries
    #   that correspond to the patches for the raster
    # this gets done for the whole raster, once
    windows_rc = image_iterator(
        shape=(meta['height'], meta['width']),
        window_size = args.patch_size,
        subdivisions = args.subdivisions,
        )
    windows = [Window(rc[1], rc[0], args.patch_size, args.patch_size) for rc in windows_rc]

    geoms = [create_polygon(
            affine_coord = (meta['transform'][2],meta['transform'][5]),
            window = win,
            pixel_size=meta['transform'][0]) for win in windows]

    patches_gdf = gpd.GeoDataFrame({
        'window' : windows,
        'geometry' : geoms},
        crs = meta['crs'])

    # now deal with identifying the data splits (e.g. train, validation, test)
    train_pct = args.train_split[0]
    test_pct = args.test_split[0]
    print(f'training percent is: {train_pct} test percent is: {test_pct}')

    class_weights, split_dict = gen_split_indexes(
        gdf = gdf,
        col = 'class',
        train_pct=train_pct,
        test_pct= test_pct
    )

    # for each of the training types (train, valid, test) generate features separately
    #   to do that, get the first key, then use it to get the subkeys from the dict
    for k in split_dict.keys():
        ix = k
        break
    split_type = [k if v is not None else '' for k,v in split_dict[ix].items()]

    # dict for number of polygons per split type
    numPolygons ={}
    nPatches = {st :0 for st in split_type}

    if args.label_path == None:
        label = rasterizer(
                col=  gdf['class'],
                geom= gdf['geometry'],
                fill= 0,
                shape =  (meta['height'], meta['width']),
                transform = meta['transform'],
                dtype='int'
            )
        tmp_meta = meta.copy()
        tmp_meta.update({'count' : 1})
        label_path =".labels_tmp.tif"
        with rasterio.open(label_path, 'w', **tmp_meta) as dst:
            dst.write(label, 1)
    else:
        label_path = args.label_path

    # the label and mosaic file opening is much faster
    # when done once, and not in the loop
    # open the label file and metadata
    lSrc =  rasterio.open(label_path, 'r')
    lMeta = lSrc.meta

    # open the mosaic file
    mSrc = rasterio.open(args.mosaic_path, 'r')
    pMeta = mSrc.meta


    for stype in split_type:
        print(f'processing {stype}...')

        features[stype] = {}

        data_ixs = flatten([split_dict[k][stype].tolist() for k in split_dict.keys()])
        numPolygons[stype] = len(data_ixs)

        # subset the geodata frame for the split type, then find the places where the patch
        #   geodataframe intersects the polygons
        _gdf_sub = gdf[gdf.index.isin(data_ixs)]

        valid_ix = [patches_gdf.intersects(geom, align=True) for geom in _gdf_sub['geometry']]
        valid_ix = np.stack(valid_ix, axis = -1)# valid_ix =  patch_intsct==True
        valid_ix = valid_ix.any(axis=1)
        valid_ix = np.where(valid_ix==True)[0].tolist()

        for valid in tqdm(valid_ix):
            win = patches_gdf.loc[valid]['window']

            # now, get the window data for the labels
            label = lSrc.read(window = win)
            lBounds = lSrc.window_bounds(window = win)
            lTrans = lSrc.window_transform(window = win)
            label = np.moveaxis(label, 0, -1)

            nZeroPix = (label==0).sum()

            if args.patch_size**2 - args.label_thresh >= nZeroPix:


                # first set up the file name
                file_root = f"{counter:0{file_len}d}"

                lMeta.update({
                    'transform':lTrans,
                    'height' : args.patch_size,
                    'width' : args.patch_size,
                })

                # now read the window data from the mosaic
                patch = mSrc.read(window = win)
                patch = np.moveaxis(patch, 0, -1)

                # update patch meta
                pMeta.update({
                    'transform':lTrans,
                    'height' : args.patch_size,
                    'width' : args.patch_size,
                })

                if args.rotate and args.mirror:
                    label_augment = rotate_mirror(label)
                    patch_augmented = rotate_mirror(patch)
                    nPatches[stype] += 8

                elif args.rotate and not args.mirror:
                    label_augment = rotate(label)
                    patch_augmented = rotate(patch)
                    nPatches[stype] += 4

                elif args.mirror and not args.rotate:
                    label_augment = mirror(label)
                    patch_augmented = mirror(patch)
                    nPatches[stype] += 4

                else:
                    label_augment = [label]
                    patch_augmented = [patch]
                    nPatches[stype] += 1

                lPath = args.out_path + f"/{stype}/labels/"

                # now write the label datasets
                write_patches(
                    path = lPath,
                    augmented=label_augment,
                    file_root=file_root,
                    meta = lMeta
                )

                # now write the patches
                write_patches(
                    path = args.out_path + f"/{stype}/images/",
                    augmented= patch_augmented,
                    file_root=file_root,
                    meta = pMeta
                )

                patch_is = _gdf_sub.intersects(patches_gdf.loc[valid]['geometry'])
                patch_features = _gdf_sub[patch_is]
                # for every feature that intersected the patch, update
                #   the feature count
                update_feature_counts(features,patch_features['class'], set_type=stype)
                counter += 1

    splitOut = {}
    for ko in split_dict.keys():
        splitOut[ko] ={}
        for ki in split_dict[ko].keys():
            splitOut[ko][ki] = split_dict[ko][ki].tolist()

    outDict = {'numPatchesUnique' :counter,
               'numTrainPatchesAugmented' :nPatches['train'],
               'numValidTestPatches' : {k: nPatches[k] for k in nPatches.keys() if k != 'train' },
               'numFeaturesPerClass':{},
               'splitIndices' : splitOut,
               'augmentationOptions' : {
                   'rotate' : args.rotate,
                   'mirror' : args.mirror,
                   'overlap' : f'{args.patch_size // args.subdivisions if args.subdivisions > 1 else 0} pixels'
               }
            }
    for k in features.keys():
        outDict['numFeaturesPerClass'][k] = {}
        for k2, v2 in features[k].items():
            outDict['numFeaturesPerClass'][k][k2] = v2

    oFile = Path(args.out_path + 'polygon_to_patches.json')

    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    with open(oFile, 'w') as out:
        out.write(json.dumps(outDict, indent=4))

    if args.label_path == None:
        if os.path.exists(label_path) and label_path == ".labels_tmp.tif":
            os.remove(label_path)
            print(f'cleaning up - temporary label file {label_path} removed')


if __name__ == "__main__":
    n = 1
    run_time = timeit.timeit(stmt='main()',globals=globals(), number=n)
    print(f'Total run-time: {run_time}')

