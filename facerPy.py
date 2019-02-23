#!/usr/bin/env Python

import os
import sys
import time
import logging
import argparse
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

logger = logging.getLogger('facerPy')

def dither(num, thresh = 127):
    derr = np.zeros(num.shape, dtype=int)
    div = 8
    for y in range(num.shape[0]):
        for x in range(num.shape[1]):
            newval = derr[y,x] + num[y,x]
            if newval >= thresh:
                errval = newval - 255
                num[y,x] = 1.
            else:
                errval = newval
                num[y,x] = 0.
            if x + 1 < num.shape[1]:
                derr[y, x + 1] += errval / div
                if x + 2 < num.shape[1]:
                    derr[y, x + 2] += errval / div
            if y + 1 < num.shape[0]:
                derr[y + 1, x - 1] += errval / div
                derr[y + 1, x] += errval / div
                if y + 2< num.shape[0]:
                    derr[y + 2, x] += errval / div
                if x + 1 < num.shape[1]:
                    derr[y + 1, x + 1] += errval / div
    return num[::-1,:] * 255

def ditherImage(input_img, output_path=None, threshold=127, mode='floyd-steinberg'):
    if hasattr(input_img, 'filename'):
        logger.info('Performing image dithering on {}.'.format(os.path.basename(input_img.filename)))
    else:
        logger.info('Performing image dithering.')
    
    dither_img = Image.NONE

    if mode == 'floyd-steinberg':
        dither_img = input_img.convert('1')
    elif mode == 'atkinson':
        m = np.array(input_img)[:,:]
        m2 = dither(m, thresh = threshold)
        dither_img = Image.fromarray(m2[::-1,:])
        dither_img.convert('1')

    if output_path is not None and dither_img is not None:
        logger.info('Saving dithered output to {}.'.format(output_path))
        dither_img.save(output_path)

    return dither_img

def findEdgesInImage(input_img, output_path=None):
    if hasattr(input_img, 'filename'):
        logger.info('Performing image edge filtering on {}.'.format(os.path.basename(input_img.filename)))
    else:
        logger.info('Performing image edge filtering.')

    edge_img = input_img.filter(ImageFilter.FIND_EDGES)

    if output_path is not None:
        logger.info('Saving edge filter output to {}.'.format(output_path))
        edge_img.save(output_path)

    logger.info('Completed edge filtering.')
    return edge_img

def invertImage(input_img, output_path=None):
    if hasattr(input_img, 'filename'):
        logger.info('Performing image inversion on {}.'.format(os.path.basename(input_img.filename)))
    else:
        logger.info('Performing image inversion.')

    inverted_img = ImageOps.invert(input_img)
    if output_path:
        logger.info('Saving inverted image output to {}.'.format(output_path))
        inverted_img.save(output_path)
    return inverted_img

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-file',
        dest='input_file',
        help='file path to the input image.',
    )
    parser.add_argument(
        '-l', '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='critical',
        dest='log_level',
        help='set logging level',
    )
    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help='save processed images',
    )
    parser.add_argument(
        '-e', '--ext',
        default='png',
        help='output filetype',
    )
    parser.add_argument(
        '-dm', '--dither-mode',
        choices=['atkinson','floyd-steinberg'],
        default='floyd-steinberg',
        dest='dither_mode',
        help='select dithering method',
    )
    parser.add_argument(
        '-dt', '--dither-threshold',
        type = int,
        default=127,
        dest='dither_threshold',
        help='black/white threshold',
    )
    parser.add_argument(
        '-dc', '--dither-contrast',
        type = float,
        dest='dither_contrast',
        help = 'boost contrast by specified factor (default = 1)',
    )
    parser.add_argument(
        '-ds', '--dither-sharpness',
        type = float,
        dest='dither_sharpness',
        help = 'boost sharpness by specified factor (default = 1)',
    )
    parser.add_argument(
        '-dr', '--dither-resize',
        type = int,
        dest='dither_resize',
        help = 'resize pre-dither image on longest dimension',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.getLevelName(args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    if not os.path.exists(args.input_file):
        logger.error('Input file path not valid!')
        sys.exit(1)

    if args.save:
        file_name, ext = os.path.splitext(os.path.basename(args.input_file))
        dir_path = os.path.dirname(args.input_file)
        edge_save_path = file_name + '_edges.' + args.ext
        dither_save_path = file_name + '_dithered.' + args.ext
        inverted_save_path = file_name + '_inverted.' + args.ext
    else:
        dither_save_path = None
        edge_save_path = None
        inverted_save_path = None

    start_time = time.time()
    input_img = Image.open(args.input_file)
    edge_img = findEdgesInImage(input_img, edge_save_path)

    pre_dither_img = edge_img.convert('L')
    if args.dither_contrast:
        pre_dither_img = ImageEnhance.Contrast(pre_dither_img).enhance(args.dither_contrast)
    if args.dither_resize:
        pre_dither_img.thumbnail((args.dither_resize,) * 2, 3)
    if args.dither_sharpness:
        pre_dither_img = ImageEnhance.Sharpness(pre_dither_img).enhance(args.dither_sharpness)

    dither_img = ditherImage(pre_dither_img, dither_save_path, args.dither_threshold, args.dither_mode)
    inverted_img = invertImage(dither_img, inverted_save_path)
    end_time = time.time()
    logger.info('Image processing complete! Total Time Taken: {:.2f} seconds.'.format(end_time - start_time))

if __name__ == '__main__':
    main()
