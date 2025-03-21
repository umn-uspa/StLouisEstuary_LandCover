#!/usr/bin/env python

'''
Author: Dr. Jeffery Thompson

'''
import rasterio
import timeit

import numpy as np

from pathlib import Path

from parsers import generate_prediction_ensemble_parser
from utilities import palette
from raster_utilities import raster_metadata, get_windows, mosaic_window

def main():
    '''
    Command line script for creating prediction ensemble on the file system.

    '''

    parser = generate_prediction_ensemble_parser()
    args = parser.parse_args()

    print(f'generating input file list from: {args.in_path}...')
        # get files
    files = [p for p in Path(args.in_path).glob("*.tif") if (p.is_file() and p.stat().st_size >0)]
    files.sort()

    print(f' \t grabbing probability metadata from file: {str(files[0])}')
    meta = raster_metadata(path = files[0])
    probMeta = meta.copy()
    probMeta.update({'nodata': -9999, 'compress':'lzw'})

    classMeta = meta.copy()
    classMeta.update({'dtype': 'uint8', 'count' : 1, 'nodata': 255, 'compress':'lzw'})

    maxMeta = probMeta.copy()
    maxMeta.update({'nodata': -9999, 'count' : 1, 'compress':'lzw'})

    aoi_path = Path("/scratch.local/aoi_mask.tiff")

    n_classes = args.n_classes
    win_width = args.tile_width
    win_height = args.tile_height

    windows = get_windows(
        height= meta['height'],
        width= meta['width'],
        win_height = win_height,
        win_width= win_width
    )
    _, _palette = palette()

    prob_path = Path(args.out_path + f"st_louis_naip_unet_ensemble_{len(files)}mods_probabilities.tif")
    class_path = Path(args.out_path + f"st_louis_naip_unet_ensemble_{len(files)}mods_predict.tif")
    max_path = Path(args.out_path + f"st_louis_naip_unet_ensemble_{len(files)}mods_max_probability.tif")

    prob_dst = rasterio.open(prob_path, 'w', sharing=True, **probMeta)
    cl_dst =  rasterio.open(class_path, 'w', sharing=True, **classMeta)
    max_dst = rasterio.open(max_path, 'w', sharing=True, **maxMeta)

    aoi_src = rasterio.open(aoi_path, sharing=True)
    for win in windows:
        print(f'working on window(row, col):  {win.row_off}, {win.col_off}')

        valid_data, aoi = mosaic_window(
            path = aoi_path,
            window = win
        )
        # note: creating a nodata based on aoi mask here
        #   is inefficient. however, if not done here
        #   the mask that is created for a tile of nothing but
        #   no data fails. it's wasted from that standpoint, but
        #   it works. :/
        aoi = np.squeeze(aoi)#.astype('bool')
        nodata_ix = aoi == 0
        nodata_ix = np.repeat(nodata_ix[:,:, np.newaxis], args.n_classes, axis=-1) # move up outside of if?

        tiles = []
        if valid_data:
            print(f'\t tile had valid data. starting data retrieval for tile...')
            print(f'\t window has dimensions: {aoi.shape}, setting AOI mask...')

            for i, file in enumerate(files, start=1):
                print(f"\t loading data from image file: {file}")
                _, img = mosaic_window(
                    path = file,
                    window=win
                )

                # if trying to mask, create masked array
                tiles.append(img.astype('float32'))

            print(f'\t Data from windows loaded. Averaging data...')
            final_probs = np.mean(tiles, axis = 0)
            final_probs = final_probs.astype(probMeta['dtype'])
            final_probs[nodata_ix] = int(probMeta['nodata'])
            final_probs = np.moveaxis(final_probs, -1, 0)

            # write ensemble probs
            prob_dst.write(final_probs, window = win)

            # now for max probs - already has probMeta nodata should just be able to write
            max_dst.write(np.max(final_probs, axis=0), indexes = 1, window = win)
            print(f'window: ({win.row_off}, {win.col_off}) written to {args.out_path}')

            # now for class image. nodata needs changed
            class_img = np.argmax(final_probs, axis=0).astype(classMeta['dtype'])
            class_img[nodata_ix[:,:,0]] = int(classMeta['nodata'])
            cl_dst.write(class_img,indexes = 1, window = win)
            cl_dst.write_colormap(1, _palette)

if __name__ == "__main__":
    n = 1
    run_time = timeit.timeit(stmt='main()',globals=globals(), number=n)
    print(f'Total run-time: {run_time}')
