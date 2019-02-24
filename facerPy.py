#!/usr/bin/env Python

import os
import sys
import time
import logging
import argparse
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from scipy.spatial.distance import cdist
import numpy as np
import cv2

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

def ditherImage(input_img, threshold=127, mode='floyd-steinberg'):
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

    return dither_img

def findEdgesInImage(input_img):
    if hasattr(input_img, 'filename'):
        logger.info('Performing image edge filtering on {}.'.format(os.path.basename(input_img.filename)))
    else:
        logger.info('Performing image edge filtering.')

    edge_img = input_img.filter(ImageFilter.FIND_EDGES)

    logger.info('Completed edge filtering.')

    return edge_img

def invertImage(input_img):
    if hasattr(input_img, 'filename'):
        logger.info('Performing image inversion on {}.'.format(os.path.basename(input_img.filename)))
    else:
        logger.info('Performing image inversion.')

    inverted_img = ImageOps.invert(input_img).convert('1')

    return inverted_img

def findFacesInImage(input_img, scaleFactor, minNeighbors, minSize):
    if hasattr(input_img, 'filename'):
        logger.info('Performing face detection on {}.'.format(os.path.basename(input_img.filename)))
    else:
        logger.info('Performing face detection.')

    img = np.array(input_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor, 0, minNeighbors, minSize)

    if len(faces) == 0:
        logger.warning('No faces found!')
        x, y = input_img.size
        return np.array([])

    logger.info(f'Found {len(faces)} face(s)!')

    return faces

def stringyPlotter(input_img, divisor):
    """
        source: https://github.com/wndaiga/StringyPlotter
    """
    if hasattr(input_img, 'filename'):
        logger.info('Performing stringy plotting on {}.'.format(os.path.basename(input_img.filename)))
    else:
        logger.info('Performing stringy plotting.')

    ii = np.array(input_img)
    iii = np.where(ii==1)
    iiii = np.column_stack(reversed(iii))
    iiiii = iiii[np.random.choice(iiii.shape[0],iiii.shape[0]//divisor,replace=False),:]

    the_first = iiiii[0]
    first_mask  = np.ones(iiiii.shape[0], dtype=bool)
    first_mask[[0]] = False
    the_rest = iiiii[first_mask]
    collection = np.array([the_first])

    for x in range(iiiii.shape[0]-1):
        all_distances = cdist(the_rest, [the_first])
        next_distance = np.min(all_distances)
        distance_match = np.where(all_distances == next_distance)[0][0]
        found_next = the_rest[distance_match]
        collection = np.concatenate([collection,np.array([found_next])])
        next_mask  = np.ones(the_rest.shape[0], dtype=bool)
        next_mask[[distance_match]] = False
        next_rest  = the_rest[next_mask]
        next_first = found_next
        the_first = found_next
        the_rest  = next_rest

    svg_template = '<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">{}</svg>'
    path_template = '<path d="{}" fill="none" stroke="black" />"'
    move_template = 'M {} {} '
    line_template = 'L {} {} '

    path_string = move_template.format(*collection[0])
    for x in collection[1:]:
        path_string += line_template.format(*x)

    return svg_template.format(
            input_img.width,
            input_img.height,
            path_template.format(path_string)
        )

def getParser(parser_formatter):
    parser = argparse.ArgumentParser(formatter_class=parser_formatter)
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
        '-ri', '--run-interactive',
        action = 'store_true',
        default = False,
        dest = 'interactive',
        help='run interactively.'
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
    parser.add_argument(
        '-osf', '--opencv-scale-factor',
        type = float,
        default = 1.3,
        dest = 'opencv_scaleFactor',
        help = 'opencv scale factor that specifies how much the image size'+
               'is reduced at each scale',
    )
    parser.add_argument(
        '-omn', '--opencv-min-neighbors',
        type = int,
        default = 6,
        dest = 'opencv_minNeighbors',
        help = 'opencv min-neighbors that specifies the quality of detected faces'+
               ' higher number results in fewer detections of higher quality. 3 - 6'+
               ' is a good start.',
    )
    parser.add_argument(
        '-oms', '--opencv-min-size',
        type = int,
        default = 800,
        dest = 'opencv_minSize',
        help = 'opencv min-size that determines how small your detections can be.'+
               '30 > (30,30) is a good start.',
    )
    parser.add_argument(
        '-op','--opencv-padding',
        type = int,
        default = 0,
        dest = 'opencv_padding',
        help = 'padding applied to facial roi from face detector.'
    )
    parser.add_argument(
        '-sd','--stringy-divisor',
        type = int,
        default = 2,
        dest = 'stringy_divisor',
        help = 'determines black pixel division in the image to obtain a sample size for stringy plotting.'
    )
    parser.add_argument(
        '-si', '--stringy-invert',
        action = 'store_true',
        default = False,
        dest = 'stringy_invert',
        help = 'perform pre-string plotting image inversion.'
    )

    return parser

def newImageObject(input_file, ext, save, string_mod=''):
    if save:
        file_name, _ = os.path.splitext(os.path.basename(input_file))
        dir_path = os.path.dirname(input_file)
        edge_output_path = os.path.join(dir_path, file_name + '_edges{}.'.format('_'+string_mod) + ext)
        dither_output_path = os.path.join(dir_path, file_name + '_dithered{}.'.format('_'+string_mod) + ext)
        inverted_output_path = os.path.join(dir_path, file_name + '_inverted{}.'.format('_'+string_mod) + ext)
        detections_output_path = os.path.join(dir_path, file_name + '_detections{}.'.format('_'+string_mod) + ext)
        stringy_output_path = os.path.join(dir_path, file_name + '_stringy{}.svg'.format('_'+string_mod))
    else:
        dither_output_path = None
        edge_output_path = None
        inverted_output_path = None
        detections_output_path = None
        stringy_output_path = None

    return {
        'input': {
            'img': Image.NONE
        },
        'dither': {
            'output_path': dither_output_path,
            'img': Image.NONE
        },
        'inverted': {
            'output_path': inverted_output_path,
            'img': Image.NONE
        },
        'edge': {
            'output_path': edge_output_path,
            'img': Image.NONE
        },
        'stringy': {
            'output_path': stringy_output_path,
            'img': Image.NONE
        },
    }

def saveImages(imgs):
    if imgs['faces'].get('output_path') and imgs['faces'].get('img'):
        logger.info('Saving faces output to {}.'.format(imgs['faces']['output_path']))
        imgs['faces']['img'].save(imgs['faces']['output_path'])

    for img_objs in imgs['enhanced']:
        for enhancement, img_obj in {k: v for k, v in img_objs.items() if v.get('output_path') and v.get('img')}.items():
            if img_obj['output_path']:
                logger.info('Saving {} output to {}.'.format(enhancement, img_obj['output_path']))
                if enhancement == 'stringy':
                    with open(img_obj['output_path'], 'w') as f:
                        f.write(img_obj['img'])
                else:
                    img_obj['img'].save(img_obj['output_path'])

def interactiveHalt(interactive, msg):
    if interactive:
        continue_process = input('{} (Y/N) > '.format(msg))
        while continue_process not in ['Y', 'N']:
            print('Ooops! Invalid choice selected. Let\'s try this again :)')
            continue_process = input('{} (Y/N) > '.format(msg))

        if continue_process == 'N':
            logging.info('Exiting facerPy on user request.')
            sys.exit(2)

def main():
    parser = getParser(argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error('Input file path not valid!')
        sys.exit(1)

    logging.basicConfig(
        level=logging.getLevelName(args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    start_time = time.time()
    imgs = {
        'original': Image.open(args.input_file),
        'faces': {
            'output_path': os.path.join(
                os.path.dirname(args.input_file),
                os.path.splitext(os.path.basename(args.input_file))[0] + '_faces.' + args.ext
            ),
            'img': Image.NONE
        },
        'enhanced': [],
    }
    faces = findFacesInImage(
        imgs['original'],
        args.opencv_scaleFactor,
        args.opencv_minNeighbors,
        (args.opencv_minSize,)*2
    )

    if len(faces) > 0:
        cv2_img = cv2.cvtColor(np.array(imgs['original']), cv2.COLOR_RGB2BGR)
        for index, (x,y,w,h) in enumerate(faces):
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_obj = newImageObject(args.input_file, args.ext, args.save, '_face{}'.format(index+1))
            img_obj['input']['img'] = Image.fromarray(
                np.array(imgs['original'])[
                    y - args.opencv_padding:y + h + args.opencv_padding,
                    x - args.opencv_padding:x + w  + args.opencv_padding
                ]).convert('L')
            imgs['faces']['img'] = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
            imgs['enhanced'].append(img_obj)
    else:
        interactiveHalt(args.interactive, 'No faces found! Continue?')
        img_obj = newImageObject(args.input_file, args.ext, args.save)
        img_obj['input']['img'] = imgs['original'].convert('L')

        imgs['enhanced'].append(img_obj)

    for index, img_obj in enumerate(imgs['enhanced']):
        if index > 0:
            interactiveHalt(args.interactive, 'Perform processing on next frame?')

        img_obj['edge']['img'] = findEdgesInImage(img_obj['input']['img'])
        pre_dither_img = img_obj['edge']['img']

        if args.dither_contrast:
            pre_dither_img = ImageEnhance.Contrast(pre_dither_img).enhance(args.dither_contrast)
        if args.dither_resize:
            pre_dither_img.thumbnail((args.dither_resize,) * 2, 3)
        if args.dither_sharpness:
            pre_dither_img = ImageEnhance.Sharpness(pre_dither_img).enhance(args.dither_sharpness)

        img_obj['dither']['img'] = ditherImage(
            pre_dither_img,
            args.dither_threshold,
            args.dither_mode,
        )

        img_obj['inverted']['img'] = invertImage(img_obj['dither']['img'])

        if (args.stringy_invert):
            pre_stringy_img = img_obj['inverted']['img']
        else:
            pre_stringy_img = img_obj['dither']['img'].convert('1')

        img_obj['stringy']['img'] = stringyPlotter(pre_stringy_img, args.stringy_divisor)

    saveImages(imgs)

    end_time = time.time()
    logger.info('Image processing complete! Total Time Taken: {:.2f} seconds.'.format(end_time - start_time))

if __name__ == '__main__':
    main()
