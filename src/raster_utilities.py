#!/usr/bin/env python
'''
Utilities for working with raster data

Author: Olena Boiko and Dr Jeffery Thompson

'''

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
import rasterio.mask
from rasterio.windows import Window
from rasterstats.io import bounds_window
from rasterio.features import rasterize

from itertools import product

def extract_by_mask (raster_filepath, vector_filepath, nodata=None):
    '''
    Clips the raster inside the study area polygon.

    Parameters
    ----------
    raster_filepath: string
        Path to the raster file
    vector_filepath: string
        Path to the vector file outlining the study area
    nodata: float
        Value to encode missing values to

     Returns
    -------
    cropped: array
        Clipped array
    meta: dict
        Dictionary with raster geo-properties, such as Affine transform, CRS, etc.
    '''
    aoi = gpd.read_file(vector_filepath)
    with rasterio.open(raster_filepath) as src:
        cropped, output_transform = rasterio.mask.mask(
            dataset=src,
            shapes=aoi.to_crs(src.crs)["geometry"],
            crop=True,
            filled=False,
            all_touched=True)
        meta = src.meta.copy()
        meta.update(
            {"height" : cropped.shape[1],
             "width" : cropped.shape[2],
             "transform" : output_transform,
             "nodata" : nodata,
             "driver" : "GTiff",
             "compress" : "lzw",
             "bigtiff" : "yes"
            })
        return cropped, meta

def get_windows(height, width, win_height, win_width, step=None)->list:
    '''
    Use rasterio Window to subset the image for processing.

    Parameters
    ----------
    height: int
        Height of the image being processed.
    width : int
        Width of the image being processed.
    win_height : int
        Height of the subset Window to use.
    win_width : int
        Width of the subset Window to use.
    step : int
        The amount of window overlap. Default None.
        Divides into win_height|win_width.

    Returns
    -----------
    windows : list
        List with all of the rasterio Window subsets for the image.

    '''
    if step is None:
        # use product to calculate row, column dimensions in that order
        # rc_ix is shorthand for row, column index
        #
        # using row, column order as numpy is row, column major
        #   and rest of code assumes going row by row
        rc_ix =  product(
            range(0, (height // win_height) + 1),
            range(0, (width // win_width) +1)
        )
    else:
        exit()

    # iterate thought the rows, and columns to generate the
    #   rasterio Windows. that basically has the format of:
    #       col_offset, row_offset, width, height – Window offsets and dimension.
    #
    # 2024-12-13 rasterio Window API documentation is confusing
    #   going to change these to names parameters rather than guessing order
    #   because the order below is wrong (though API docs suggests it should be
    #   this way). the docs also move between row, column, and column, row
    return [Window(
        col_off = rc[1] * win_width,
        row_off = rc[0] * win_height,
        width = min(win_width, width - rc[1] * win_width),
        height = min(win_height, height - rc[0] * win_height)
    ) for rc in rc_ix]

def load_raster_by_window_corner(raster_filepath, lon, lat, window_size=2000):
    '''
    Reads a smaller subset of large raster, using upper left pixel coordinates.

    Parameters
    ----------
    raster_filepath: string
        Path to the raster file
    lon: float
        Longitude of the upper left corner of the window of interest
    lat: float
        Latitude of the upper left corner of the window of interest
    window_size: int
        Dimension in pixels of a square window

    Returns
    -------
    array
        Array read from the raster
    '''
    with rasterio.open(raster_filepath) as dataset:
        py, px = dataset.index(lon, lat)
        window = rasterio.windows.Window(px, py, window_size, window_size)
        array = dataset.read(window=window)
        array = np.moveaxis(array, 0, -1)
        return array

def mosaic_window(path, window)-> tuple[bool, np.array]:
    '''Get the subset of image to segment
    Parameters
    ----------
    path : pathlib Path
        The path to the image dataset to subset.
    window : rasterio Window
        The window object to use to subset. Corresponds with (col_offset, row_offset, win_width, win_height)

    Returns
    -----------
    valid_data : boolean
        True if tile contained some valid data, False if all nodata were found
    image data : np.array
        Windowed subset of image

    '''
    with rasterio.open(path, 'r') as src:
        img = src.read(window = window)
        valid_data = src.read_masks(window = window)
        print(f'\t image data and mask loaded for the window...')

    if np.any(valid_data>0):
        img = np.moveaxis(img, 0, -1)

        return True, img

    else:
        return False, img

def polygons_to_pixels(gdf, raster_filepath, label_column):
    '''
    Extracts pixel values from a raster using polygons

    Parameters
    ----------
    gdf: GeoDataFrame
        Polygons loaded as geopandas GeoDataFrame
    raster_filepath: string
        Path to the raster file
    label_column: string
        Column in the gdf that contains class values for each polygon

    Returns
    -------
    array
        Array with class labels
    array
        Corresponding array with raster values
    '''
    values = []
    labels = []
    with rasterio.open(raster_filepath, "r") as src:
        # change coordinate system if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
            for (label, geom) in zip(gdf[label_column], gdf["geometry"]):
                # read the raster data matching the geometry bounds of each polygon
                # this is more efficient than reading the entire raster
                window = bounds_window(geom.bounds, src.transform)
                window_affine = src.window_transform(window)
                window_array = src.read(window=window)
                # rasterize the geometry
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=window_array.shape[1:],
                    transform=window_affine,
                    fill=0,
                    dtype="uint8",
                    all_touched=False).astype(bool)
                # for each label pixel (places where the mask is true)
                label_pixels = np.argwhere(mask)
                for (row, col) in label_pixels:
                    data = window_array[:,row,col]
                    values.append(data)
                    labels.append(label)
            src.close()
            return np.array(labels), np.array(values)

def raster_metadata(path):
    ''''
    Return image metadata.

    Parameters
    ----------
    path : str
        Path to the raster data you want metadata for.

    Returns
    -------
    raster metadata
        As per rasterio metadata functions.
    '''
    with rasterio.open(path, 'r') as src:
        meta = src.meta

    return meta

def raster_mosaic (raster_filepath_list, nodata=None, resolution=0.6):
    '''
    Combines individual NAIP tiles into a single mosaic file.

    Parameters
    ----------
    raster_filepath_list: list
        List of filenames of NAIP tiles in order of priority
    nodata: float
        Value to encode missing values to
    resolution: float
        Spatial resolution of the resulting mosaic

    Returns
    -------
    mosaic: array
        Array merged from smaller tiles
    meta: dict
        Dictionary with raster geo-properties, such as Affine transform, CRS, etc.
    '''
    print ("Get metadata from the first raster")
    meta = rasterio.open(raster_filepath_list[0]).meta.copy()
    raster_datasets = []
    for raster_filepath in raster_filepath_list:
        print (raster_filepath)
        raster_dataset = rasterio.open(raster_filepath)
        raster_datasets.append(raster_dataset)
    mosaic, output_transform = merge(raster_datasets, res=(resolution,resolution))
    meta.update(
        {"height" : mosaic.shape[1],
         "width" : mosaic.shape[2],
         "transform" : output_transform,
         "nodata" : nodata,
         "driver" : "GTiff",
         "compress" : "lzw",
         "bigtiff" : "yes"
        })
    return mosaic, meta

def save_raster (array, meta, out_filename):
    '''
    Saves raster to a GeoTiff file.

    Parameters
    ----------
    array: array
        Array to save
    meta: dict
        Dictionary with raster geo-properties, such as Affine transform, CRS, etc.
    out_filename: string
        Path to the output raster file

    Returns
    -------
    None (Saves raster to a file)
    '''
    with rasterio.open(out_filename, "w", **meta) as dst:
        dst.write(array)