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

    inverted_img = ImageOps.invert(input_img).convert('1')
    if output_path:
        logger.info('Saving inverted image output to {}.'.format(output_path))
        inverted_img.save(output_path)
    return inverted_img

def findFacesInImage(input_img, scaleFactor, minNeighbors, minSize, output_path=None):
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

    if output_path:
        logger.info('Saving detections output to {}.'.format(output_path))
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for (x,y,w,h) in faces:
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(output_path, cv2_img)
            # res_cv2_img = cv2.resize(cv2_img, (0, 0), fx=0.15, fy=0.15)
            # cv2.imshow("Face Detections", res_cv2_img)
            # cv2.waitKey(0)

    return faces

def stringyPlotter(input_img, divisor, output_path=None):
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

    final_svg = svg_template.format(
            input_img.width,
            input_img.height,
            path_template.format(path_string)
        )

    if output_path:
        logger.info('Saving stringy output to {}.'.format(output_path))
        with open(output_path, 'w') as f:
            f.write(final_svg)

    return final_svg

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
    parser.add_argument(
        '-os', '--opencv-scale-factor',
        type = float,
        default = 1.3,
        dest = 'opencv_scaleFactor',
        help = 'opencv scale factor that specifies how much the image size'+
               'is reduced at each scale',
    )
    parser.add_argument(
        '-ominn', '--opencv-min-neighbors',
        type = int,
        default = 6,
        dest = 'opencv_minNeighbors',
        help = 'opencv min-neighbors that specifies the quality of detected faces'+
               ' higher number results in fewer detections of higher quality. 3 - 6'+
               ' is a good start.',
    )
    parser.add_argument(
        '-omins', '--opencv-min-size',
        type = int,
        default = 800,
        dest = 'opencv_minSize',
        help = 'opencv min-size that determines how small your detections can be.'+
               '30 > (30,30) is a good start.',
    )
    parser.add_argument(
        '-opad','--opencv-padding',
        type = int,
        default = 0,
        dest = 'opencv_padding',
        help = 'padding applied to facial roi from face detector.'
    )
    parser.add_argument(
        '-sdiv','--stringy-divisor',
        type = int,
        default = 2,
        dest = 'stringy_divisor',
        help = 'determines black pixel division in the image to obtain a sample size for stringy plotting.'
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
        detections_save_path = file_name + '_detections.' + args.ext
        stringy_save_path = file_name + '_stringy.svg'
    else:
        dither_save_path = None
        edge_save_path = None
        inverted_save_path = None
        detections_save_path = None
        stringy_save_path = None

    start_time = time.time()
    input_img = Image.open(args.input_file)
    edge_img = findEdgesInImage(input_img, edge_save_path)
    faces = findFacesInImage(
        input_img,
        args.opencv_scaleFactor,
        args.opencv_minNeighbors,
        (args.opencv_minSize,)*2,
        detections_save_path
    )

    imgs_arr = []
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            face_img = Image.fromarray(
                np.array(edge_img)
                [y - args.opencv_padding:y + h + args.opencv_padding, x - args.opencv_padding:x + w  + args.opencv_padding]
            )
            dith_save_arr = list(dither_save_path)
            inv_save_arr = list(inverted_save_path)
            stringy_save_arr = list(stringy_save_path)
            dith_save_arr.insert(dither_save_path.find('.'),'_face{}'.format(i + 1))
            inv_save_arr.insert(inverted_save_path.find('.'),'_face{}'.format(i + 1))
            stringy_save_arr.insert(stringy_save_path.find('.'),'_face{}'.format(i + 1))

            imgs_arr.append({
                'dither_save_path': ''.join(dith_save_arr),
                'inverted_save_path': ''.join(inv_save_arr),
                'stringy_save_path': ''.join(stringy_save_arr),
                'img': face_img,
            })
    else:
        imgs_arr.append(
            {
                'dither_save_path': dither_save_path,
                'inverted_save_path': inverted_save_path,
                'stringy_save_path': stringy_save_path,
                'img': input_img,
            }
        )

    for img_obj in imgs_arr:
        pre_dither_img = img_obj['img'].convert('L')
        if args.dither_contrast:
            pre_dither_img = ImageEnhance.Contrast(pre_dither_img).enhance(args.dither_contrast)
        if args.dither_resize:
            pre_dither_img.thumbnail((args.dither_resize,) * 2, 3)
        if args.dither_sharpness:
            pre_dither_img = ImageEnhance.Sharpness(pre_dither_img).enhance(args.dither_sharpness)

        dither_img = ditherImage(
            pre_dither_img,
            img_obj['dither_save_path'],
            args.dither_threshold,
            args.dither_mode,
        )
        inverted_img = invertImage(
            dither_img,
            img_obj['inverted_save_path'],
        )
        sringy_img = stringyPlotter(
            dither_img.convert('1'),
            args.stringy_divisor,
            img_obj['stringy_save_path'],
        )

    end_time = time.time()
    logger.info('Image processing complete! Total Time Taken: {:.2f} seconds.'.format(end_time - start_time))

if __name__ == '__main__':
    main()
