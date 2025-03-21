#!/usr/bin/env python

'''
Code for prediction mosaics using U-Net model

Author: Dr. Jeffery Thompson

'''
import rasterio
import timeit
import time

import numpy as np

from pathlib import Path

from parsers import predict_mosaic_parser
from predict_tile import segment_tile
from raster_utilities import raster_metadata, get_windows, mosaic_window
from modeling import load_model_file

def main():
    '''
    Command line script for segmenting image on the file system with smoothing applied in the process
    '''
    parser = predict_mosaic_parser()
    args = parser.parse_args()
    print(f'image_path: {args.image_path}')

    model = load_model_file(args.model_path, compile = False)

    n_classes = args.n_classes
    patch_size = args.patch_size
    subdivisions = args.subdivisions

    win_width = args.tile_width
    win_height = args.tile_height

    print(f'tile dims: win_height: {win_height},  win_width: {win_width}')

    meta = raster_metadata(path = Path(args.image_path))
    outMeta = meta.copy()
    outMeta.update({'dtype': rasterio.int16, 'count' : n_classes, 'nodata': -9999, 'compress':'lzw'})
    print(f'outMeta is: {outMeta}')

    windows = get_windows(
        height= meta['height'],
        width= meta['width'],
        win_height = win_height,
        win_width= win_width
    )

    with rasterio.open(args.out_path, 'w', **outMeta) as dst:
        for win in windows:
            print(f'working on window(row, col):  {win.row_off}, {win.col_off}')
            segment, img = mosaic_window(
                path = args.image_path,
                window = win
            )

            if segment:
                print(f'\t window had valid data. segmenting window...')
                print(f'\t window has dimensions: {img.shape}, type: {type(img)}')

                # this is call to my revised implmentation
                pred_img = segment_tile(
                    model = model,
                    tile = img,
                    window_size = patch_size,
                    subdivisions= subdivisions,
                    nb_classes= n_classes,
                    smooth = True,
                    mirror = True,
                    tukey = False
                )

                # now that we have prediction probs, need to get the classes
                final_probs = pred_img
                nodata_ix = img[:,:,0] == 0
                nodata_ix = np.repeat(nodata_ix[:,:, np.newaxis], args.n_classes, axis=-1) # move up outside of if?
                final_probs = np.multiply(final_probs, 1e4).astype('int16')
                final_probs[nodata_ix] = -9999

                final_probs = np.moveaxis(final_probs, -1, 0)

                dst.write(final_probs, window = win)

                print(f'window: ({win.row_off}, {win.col_off}), with shape (w/h): ({win.width},{win.height}) written to {args.out_path}')

if __name__ == "__main__":
    n = 1
    run_time = timeit.timeit(stmt='main()',globals=globals(), number=n)
    time.sleep(120)
    print(f'Total run-time: {run_time}.')
