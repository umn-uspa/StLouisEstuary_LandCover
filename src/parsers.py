#!/usr/bin/env python
'''
Create the argparse arguments for all command line functions

Author: Dr. Jeffery Thompson
'''
import argparse

def predict_mosaic_parser():
    '''
    Create the parser for command line arguments.
    Returns
    -------
    parser : arg parse object
        The command line objects required to run the script.
    '''
    parser = argparse.ArgumentParser(description="Predict for the entire study area with a single model.")
    parser.add_argument('--mosaic_path', default = None, type = str,  required = True,  help='POSIX path to mosaic file to segment.')
    parser.add_argument('--model_path', default = None, type = str,  required = True,  help='POSIX path to keras model file.')

    parser.add_argument('--out_path', default = None, type = str, required = True, help='POSIX path to store segmentation results, both classes and their associated probabilities.')
    parser.add_argument('--tile_height', default = None, type = int, required= True, help = 'Tile height to use for processing.' )
    parser.add_argument('--tile_width', default = None, type = int, required= True, help = 'Tile width to use for processing.' )

    parser.add_argument('--n_classes', default = None, type = int, required= True, help = 'Number of classes used in the model.')
    parser.add_argument('--patch_size', default = 256, type = int, required= True, help = 'Patch size used in the modelling process.')
    parser.add_argument('--subdivisions', default = 2, type = int, required= True, help = 'The division factor used to determine how much patches overlap.' \
                        ' E.g. `subdivisions = 2` and `patch_size = 256` will result in subsequent patches overlapping by 128 pixels.')

    return parser

def remove_argument(parser, arg):
    '''
    Function to remove argparse argument.
    Parameters
    ----------
    parser : arg parse object
        The parser to remove an argparse option from
    arg: argparse option
        The option to remove from the parser
    Returns
    -------
    parser
        The argparse parser with the option removed
    '''
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return

def generate_prediction_ensemble_parser():
    '''
    Create the parser for command line arguments needed to generate.
    Returns
    -------
    parser : arg parse object
        The command line objects required to run the script.
    '''
    # for consistency, get the args from predict_mosaic. some are same
    #   but some are unused here
    parser = predict_mosaic_parser()
    unused_args = ['mosaic_path', 'model_path', 'patch_size', 'subdivisions']
    [remove_argument(parser= parser, arg=ua) for ua in unused_args]
    parser.description = "Uses probabilities generated from multiple models and averages them into an ensemble result."
    parser.add_argument('--in_path', default = None, type = str,  required = True,  help='POSIX path to probabilities files from individual models.')
    parser.add_argument('--scale_factor', default = 10000, type = float, required = False,
                        help='Scale factor used to convert probabilities from integers to floats. Default value: 10,000')

    return parser

# processing the command line booleans wasn't working properly.
#  the following is from stackoverflow and fixes the problem
#   https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    '''
    Internal function to properly process Booleans passed in from the command line.
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def polygons_to_patches_parser():
    '''
    Create the parser for command line arguments.
    Returns
    -------
    parser : arg parse object
        The command line objects required to run the script.
    '''
    parser = argparse.ArgumentParser(description="Convert training data from vectors to appropriately sized raster patches for U-Net.")
    parser.add_argument('--mosaic_path', default = None, type = str,  required = True,  help='POSIX path to the raster file image containing input observations (aerial image) for U-Net patches.')
    parser.add_argument('--label_path', default = None, type = str,  required = False,  help='POSIX path to the raster file containing input class labels for U-Net patches.' \
                        ' Default is None. If not specified, a rasterized version of the features will be created on the fly using polygons contained in: --polygons_path')

    parser.add_argument('--polygons_path', default = None, type = str,  required = True,  help=' POSIX path to input polygons to convert to patches. NOTE: Only ESRI Shapefile has been tested.')

    parser.add_argument('--out_path', default = None, type = str, required = True, help='POSIX path to store the output patches.')
    parser.add_argument('--patch_size', default = 256, type = int, required= False, help = 'Patch size used when generating patches. Default is 256 pixels.' )
    parser.add_argument('--subdivisions', default = 2, type = int, required= False, help = 'The division factor used to determine how much patches overlap.'\
                        ' E.g. `subdivisions = 2` and `patch_size = 256` will result in subsequent patches overlapping by 128 pixels.')
    parser.add_argument('--label_thresh', default = 10, type = int, required= False, help = 'Pixels threshold used to determine if a patch is kept. Default is 10 pixels in a 256x256 patch.')
    parser.add_argument('--train_split', default = .9, nargs = 1, type = float,
                        required= False, help = 'Proportions for data set to keep for train,split. Default value: 0.9' )
    parser.add_argument('--test_split', default =  .1, nargs = 1, type = float,
                        required= False, help = 'Proportions for data set to keep for test split. Default value:  0.1' )
    parser.add_argument('--rotate', default = False, type = str2bool, nargs='?', required= False, help = 'Activate patch augmentation using rotation. Default value: False' )
    parser.add_argument('--mirror', default = False , type = str2bool, nargs='?', required= False, help = 'Activate path augmentation using mirroring. Default value: False' )
    return parser
