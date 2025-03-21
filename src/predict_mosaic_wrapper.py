#!/usr/bin/env python
'''
wrapper code for predict_mosaic.py

Author: Dr. Jeffery Thompson
'''
import json
import os
import subprocess
import sys
import timeit

from pathlib import Path
from scipy.stats.mstats import gmean

srcpath = Path(os.getcwd()).joinpath('src')
sys.path.insert(0, str(srcpath))

from parsers import predict_mosaic_parser, remove_argument

def _dict_from_json(file):
    '''
    Open JSON file and return as dict

    Parameters:
    -----------
    file : str
        Path to the JSON file.

    Returns:
    --------
    dict
        The dict derived from the supplied JSON file.

    '''
    with open(file) as src:
        return json.load(src)

def main():
    '''Command line script for segmenting image on the file system

    '''
    parser = predict_mosaic_parser()
    remove_argument(parser=parser, arg="model_path")
    parser.description = "Predict for the entire study area with multiple models."
    parser.add_argument('--json_path', default = None, type = str,  required = True,  help='POSIX path to a JSON file with paths to model files.')

    args = parser.parse_args()

    try:
        models = _dict_from_json(args.json_path)
    except:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            args.json
        )

    for run in models.keys():
        seed = models[run]["seed"]
        model_path = models[run]["model"]

        out_file = f"st_louis_predicted_probs_seed-{seed}.tif"
        out_path = args.out_path + out_file

        if Path(out_path).is_file() and Path(out_path).stat().st_size > 0:
            print(f'predicted probability file: {out_path} exists. skipping... ')
            continue

        print(f'generate full prediction using model:{model_path}')
        result  = subprocess.run([
            "python",
            "predict_mosaic.py",
            "--image_path", args.image_path,
            "--model_path", model_path,
            "--out_path", out_path,
            "--tile_height", f"{args.tile_height}",
            "--tile_width",  f"{args.tile_width}",
            "--n_classes", f"{args.n_classes}",
            "--patch_size", f"{args.patch_size}",
            "--subdivisions", f"{args.subdivisions}"
        ])

        if result.returncode == 0:
            print(f'\t model {model_path} ran successfully.')
            print(result.stdout)
        else:
            print(f"Error running model: {model_path}", result.stderr)

if __name__ == "__main__":
    n = 1
    run_time = timeit.timeit(stmt='main()',globals=globals(), number=n)
    print(f'Total run-time: {run_time}')
