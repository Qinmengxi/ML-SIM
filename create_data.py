import argparse
import numpy as np
from numpy import pi
import os
import glob
import sys
import math
import subprocess
from skimage import io, transform
import MLSIM_datagen.SIMulator_functions

# SIM options to control from command line (only via MLSIM_pipeline.py, not Jupyter)
SIMopt = argparse.Namespace()

# directory of source images used for simulation
SIMopt.sourceimages_path = 'Training_data/DIV2K_subset'
# directory to store training data
SIMopt.root = 'Training_data/SIMdata'
# desired samples for training and validation - e.g. ntrain=2350 and ntest=50
SIMopt.ntrain = 10
SIMopt.ntest = 5
# network input layer channels, e.g. 3x3
SIMopt.nch_in = 9
SIMopt.imageSize = 256
# instances of same source image (randomisation)
SIMopt.nrep = 1
# threads used to generate data
SIMopt.datagen_workers = 4 # only works with MLSIM_pipeline.py
# image extensions to accept, e.g. ['png','jpg','tif']
SIMopt.ext = ['png']

# ------------ Parameters-------------
# phase shifts for each stripe
SIMopt.Nshifts = 3
# number of orientations of stripes
SIMopt.Nangles = 3
# used to adjust PSF/OTF width
SIMopt.scale = 0.9 + 0.1*(np.random.rand()-0.5)
# modulation factor
SIMopt.ModFac = 0.8 + 0.3*(np.random.rand()-0.5)
# orientation offset
SIMopt.alpha = 0.33*pi*(np.random.rand()-0.5)
# orientation error
SIMopt.angleError = 10*pi/180*(np.random.rand()-0.5)
# shuffle the order of orientations
SIMopt.shuffleOrientations = True
# random phase shift errors
SIMopt.phaseError = 0.33*pi*(0.5-np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
# mean illumination intensity
SIMopt.meanInten = np.ones(SIMopt.Nangles)*0.5
# amplitude of illumination intensity above mean
SIMopt.ampInten = np.ones(SIMopt.Nangles)*0.5*SIMopt.ModFac
# illumination freq
SIMopt.k2 = 126 + 30*(np.random.rand()-0.5)
# noise type
SIMopt.usePoissonNoise = False
# noise level (percentage for Gaussian)

SIMopt.NoiseLevel = 8 + 8*(np.random.rand()-0.5)
# 1(to blur using PSF), 0(to blur using OTF)
SIMopt.UsePSF = 0
# include OTF and GT in stack
SIMopt.OTF_and_GT = True
# use a blurred target (according to theoretical optimal reconstruction)
SIMopt.applyOTFtoGT = False

os.makedirs(SIMopt.root, exist_ok=True)

files = []
for ext in SIMopt.ext:
    files.extend(glob.glob(SIMopt.sourceimages_path + "/*." + ext))

if len(files) == 0:
    print('source images not found')
    sys.exit(0)
elif SIMopt.ntrain + SIMopt.ntest > SIMopt.nrep*len(files):
    print('ntrain + opt.ntest is too high given nrep and number of source images')
    sys.exit(0)
elif SIMopt.nch_in > SIMopt.Nangles*SIMopt.Nshifts:
    print('nch_in cannot be greater than Nangles*Nshifts - not enough SIM frames')
    sys.exit(0)

#files = files[:math.ceil( (SIMopt.ntrain + SIMopt.ntest) / SIMopt.nrep )]


# ------------ Main loop --------------
def processImage(SIMopt, file):
    Io = io.imread(file) / 255
    Io = transform.resize(Io, (SIMopt.imageSize, SIMopt.imageSize), anti_aliasing=True)

    if len(Io.shape) > 2 and Io.shape[2] > 1:
        Io = Io.mean(2)  # if not grayscale

    filename = os.path.basename(file).replace('.png', '')

    print('Generating SIM frames for', file)

    for n in range(SIMopt.nrep):
        SIMopt.outputname = '%s/%s_%d.tif' % (SIMopt.root, filename, n)
        I = MLSIM_datagen.SIMulator_functions.Generate_SIM_Image(SIMopt, Io)

for file in files:
    processImage(SIMopt, file)
print('Done generating images,', SIMopt.root)