# facerPy

## Introduction
facerPy is a Python-based, facial image processing project that
applies varied artistic effects on inputted images.

## Usage
This project was developed in Python3.7.2 with the following dependencies:
- [numpy](https://www.numpy.org/)
- [pillow](https://pillow.readthedocs.io/en/stable/)
- [opencv](https://opencv.org/)
- [scipy](https://www.scipy.org/)
- [cairosvg](https://cairosvg.org)


It's recommended to use an isolated Python environment which can be setup using either
[Pipenv](https://pipenv.readthedocs.io/en/latest/) or [VirtualEnv](https://pypi.org/project/virtualenv/)/[VirtualEnvWrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to install this project's dependencies. Requisite environment setup files are provided for both (`requirements.txt`, `Pipfile[.lock]`).

```
> ./facerPy.py -h
usage: facerPy.py [-h] [-i INPUT_FILE]
                  [-l {debug,info,warning,error,critical}] [-s] [-e EXT] [-ri]
                  [-dm {atkinson,floyd-steinberg}] [-dt DITHER_THRESHOLD]
                  [-dc DITHER_CONTRAST] [-ds DITHER_SHARPNESS]
                  [-dr DITHER_RESIZE] [-osf OPENCV_SCALEFACTOR]
                  [-omn OPENCV_MINNEIGHBORS] [-oms OPENCV_MINSIZE]
                  [-op OPENCV_PADDING] [-sd STRINGY_DIVISOR] [-si]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        file path to the input image. (default: None)
  -l {debug,info,warning,error,critical}, --log-level {debug,info,warning,error,critical}
                        set logging level (default: critical)
  -s, --save            save processed images (default: False)
  -e EXT, --ext EXT     output filetype (default: png)
  -ri, --run-interactive
                        run interactively. (default: False)
  -dm {atkinson,floyd-steinberg}, --dither-mode {atkinson,floyd-steinberg}
                        select dithering method (default: floyd-steinberg)
  -dt DITHER_THRESHOLD, --dither-threshold DITHER_THRESHOLD
                        black/white threshold (default: 127)
  -dc DITHER_CONTRAST, --dither-contrast DITHER_CONTRAST
                        boost contrast by specified factor (default = 1)
                        (default: None)
  -ds DITHER_SHARPNESS, --dither-sharpness DITHER_SHARPNESS
                        boost sharpness by specified factor (default = 1)
                        (default: None)
  -dr DITHER_RESIZE, --dither-resize DITHER_RESIZE
                        resize pre-dither image on longest dimension (default:
                        None)
  -osf OPENCV_SCALEFACTOR, --opencv-scale-factor OPENCV_SCALEFACTOR
                        opencv scale factor that specifies how much the image
                        sizeis reduced at each scale (default: 1.3)
  -omn OPENCV_MINNEIGHBORS, --opencv-min-neighbors OPENCV_MINNEIGHBORS
                        opencv min-neighbors that specifies the quality of
                        detected faces higher number results in fewer
                        detections of higher quality. 3 - 6 is a good start.
                        (default: 6)
  -oms OPENCV_MINSIZE, --opencv-min-size OPENCV_MINSIZE
                        opencv min-size that determines how small your
                        detections can be.30 > (30,30) is a good start.
                        (default: 800)
  -op OPENCV_PADDING, --opencv-padding OPENCV_PADDING
                        padding applied to facial roi from face detector.
                        (default: 0)
  -sd STRINGY_DIVISOR, --stringy-divisor STRINGY_DIVISOR
                        determines black pixel division in the image to obtain
                        a sample size for stringy plotting. (default: 2)
  -si, --stringy-invert
                        perform pre-string plotting image inversion. (default:
                        False)
```

### Example
```
./facerPy.py -i images/sample1.jpg -l info -e png -dm atkinson -dr 3000 -dc 2 -ds
2 -os 1.3 -omn 6 -oms 2000 -op 510 -s -sd 5 -ri
24-Feb-19 23:56:49 - facerPy - INFO - Performing face detection on sample1.jpg.
24-Feb-19 23:56:50 - facerPy - INFO - Found 20 face(s)!
24-Feb-19 23:56:55 - facerPy - INFO - Performing image edge filtering.
24-Feb-19 23:56:55 - facerPy - INFO - Completed edge filtering.
24-Feb-19 23:56:56 - facerPy - INFO - Performing image dithering.
24-Feb-19 23:59:25 - facerPy - INFO - Performing image inversion.
24-Feb-19 23:59:25 - facerPy - INFO - Performing stringy plotting.
Perform processing on next frame? (Y/N) > N
24-Feb-19 23:59:43 - facerPy - INFO - Saving faces output to images/sample1_faces.png.
24-Feb-19 23:59:50 - facerPy - INFO - Saving dither output to images/sample1_dithered_face1.png.
24-Feb-19 23:59:51 - facerPy - INFO - Saving inverted output to images/sample1_inverted_face1.png.
24-Feb-19 23:59:51 - facerPy - INFO - Saving edge output to images/sample1_edges_face1.png.
24-Feb-19 23:59:52 - facerPy - INFO - Saving stringy output to images/sample1_stringy_face1.svg.
24-Feb-19 23:59:52 - facerPy - INFO - Saving stringy png output to images/sample1_stringy_face1.png.
24-Feb-19 23:59:54 - facerPy - INFO - Exiting facerPy on user request.
```
The above snippet takes in an image named `sample1.jpg` and runs the varied effects on it. The results are saved as requested by the save flag `-s`, with logging set to `INFO` and dithering configured to use the `atkinson algorithm`. The output of the above is shown below:

<p align="center">
  <img width="192" height="auto" src="/images/sample1.jpg?raw=true" alt="Original">
  <img width="192" height="auto" src="/images/sample1_faces.png?raw=true" alt="Detected Faces">
  <img width="260" height="auto" src="/images/sample1_edges_face1.png?raw=true" alt="Filtered Edges">
  <img width="260" height="auto" src="/images/sample1_dithered_face1.png?raw=true" alt="Dithered Output">
  <img width="260" height="auto" src="/images/sample1_inverted_face1.png?raw=true" alt="Inverted Output">
  <img width="260" height="auto" src="/images/sample1_stringy_face1.png?raw=true" alt="Stringy Output">
</p>

### Credits
The atkinson dithering algorithm is adapted from the [hyperdither](https://github.com/tgray/hyperdither) project.