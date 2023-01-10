import torch
import matplotlib.pyplot as plt
import torchvision
import skimage
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
import scipy.ndimage as ndimage
import torch.nn as nn
import os
from skimage import io,exposure,img_as_ubyte
import glob
import torchvision.transforms as transforms
import argparse
from models import GetModel
import time

def LoadModel(opt):
    print('Loading model')
    print(opt)

    net = GetModel(opt)
    checkpoint = torch.load(opt.weights,map_location=opt.device)

    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k,v in state_dict.items():
        k = 'module.' + k
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    return net


def SIM_reconstruct(model, opt):

    def prepimg(stack,self):

        inputimg = stack[:9]

        if self.nch_in == 6:
            inputimg = inputimg[[0,1,3,4,6,7]]
        elif self.nch_in == 3:
            inputimg = inputimg[[0,4,8]]

        if inputimg.shape[1] > 512 or inputimg.shape[2] > 512:
            print('Over 512x512! Cropping')
            inputimg = inputimg[:,:512,:512]

        inputimg = inputimg.astype('float') / np.max(inputimg) # used to be /255
        widefield = np.mean(inputimg,0)

        if self.norm == 'adapthist':
            for i in range(len(inputimg)):
                inputimg[i] = exposure.equalize_adapthist(inputimg[i],clip_limit=0.001)
            widefield = exposure.equalize_adapthist(widefield,clip_limit=0.001)
            inputimg = torch.from_numpy(inputimg).float()
            widefield = torch.from_numpy(widefield).float()
        else:
            # normalise
            inputimg = torch.from_numpy(inputimg).float()
            widefield = torch.from_numpy(widefield).float()
            widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))

            if self.norm == 'minmax':
                for i in range(len(inputimg)):
                    inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))

        return inputimg,widefield

    os.makedirs('%s' % opt.out,exist_ok=True)
    files = glob.glob('%s/*.tif' % opt.root)

    for iidx,imgfile in enumerate(files):
        starttime = time.time()
        print('[%d/%d] Reconstructing %s' % (iidx+1,len(files),imgfile))
        stack = io.imread(imgfile)

        inputimg, wf = prepimg(stack,opt)
        wf = (255*wf.numpy()).astype('uint8')

        with torch.no_grad():
            sr = model(inputimg.unsqueeze(0).to(opt.device))
            sr = sr.cpu()
            sr = torch.clamp(sr,min=0,max=1)

        sr = sr.squeeze().numpy()
        sr = (255*sr).astype('uint8')
        if opt.norm == 'adapthist':
            sr = exposure.equalize_adapthist(sr,clip_limit=0.01)

        skimage.io.imsave('%s/test_wf_%d.jpg' % (opt.out,iidx), wf)
        skimage.io.imsave('%s/test_sr_%d.jpg' % (opt.out,iidx), sr)
        endtime = time.time()
        print("time comsuming:", endtime - starttime)

opt = argparse.Namespace()

opt.root = 'Test_data/9'
opt.out = 'test_output/9'
opt.task = 'simin_gtout'
opt.norm = 'minmax'
opt.dataset = 'fouriersim'

opt.model = 'rcan'

# data
opt.imageSize = 512
opt.weights = 'DIV2K_randomised_3x3_20200317.pth'

# input/output layer options
opt.scale = 1
opt.nch_in = 9
opt.nch_out = 1

# architecture options 
opt.n_resgroups = 3
opt.n_resblocks = 10
opt.n_feats = 96
opt.reduction = 16
opt.narch = 0

# test options
opt.test = False
opt.cpu = False
opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')


net = LoadModel(opt)
SIM_reconstruct(net,opt)

