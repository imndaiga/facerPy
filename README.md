# facerPy

## Introduction
facerPy is a Python-based, facial image processing project that
applies varied artistic effects on inputted images.

## Usage
This project was developed in Python3.7.2 with the following dependencies:
- [numpy](https://www.numpy.org/)
- [pillow](https://pillow.readthedocs.io/en/stable/)

To run, we recommend using an isolated Python environment which can be setup using either
[Pipenv](https://pipenv.readthedocs.io/en/latest/) or [VirtualEnv](https://pypi.org/project/virtualenv/)/[VirtualEnvWrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). Requisite 
environment setup files are provided for both (`requirements.txt`, `Pipfile[.lock]`).

```
> ./facerPy.py -h              
usage: facerPy.py [-h] [-i INPUT_FILE]
                  [-l {debug,info,warning,error,critical}] [-s]
                  [-dm {atkinson,floyd-steinberg}] [-dt DITHER_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        file path to the input image. (default: None)
  -l {debug,info,warning,error,critical}, --log-level {debug,info,warning,error,critical}
                        set logging level (default: critical)
  -s, --save            save processed images (default: False)
  -dm {atkinson,floyd-steinberg}, --dither-mode {atkinson,floyd-steinberg}
                        select dithering method (default: floyd-steinberg)
  -dt DITHER_THRESHOLD, --dither-threshold DITHER_THRESHOLD
                        black/white threshold (default: 127)
```

### Example
```BASH
./facerPy.py -i headshot.jpg -l info -dm atkinson -s
```
The above snippet takes in an image named `headshot.jpg` and runs the varied effects on it,
saving the results as requested by the save flag `-s`, with the script set to an `INFO` 
logging level and dithering configured to use the `atkinson algorithm`.

### Credits
The atkinson algorthm is adapted from the [hyperdither](https://github.com/tgray/hyperdither) project.