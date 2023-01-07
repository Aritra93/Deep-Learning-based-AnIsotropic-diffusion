# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:19:31 2020

@author: HP
"""

import torch
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import ProbDiffusion as pb
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

arguments_strModel = 'bsds500'
arguments_strIn = './images/sample.png' #'cameraman.png' #
arguments_strOut = './out.png'


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        
        
        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load('network-bsds500.pytorch').items() })
        
    # end

    def forward(self, tenInput):
        tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        
        
        return tenScoreOne

        #return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
    # end
# end


netNetwork = None

def estimate(tenInput):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().eval()
    # end

    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    #assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    return netNetwork(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
# end

##########################################################

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = numpy.mean((numpy.array(img1, dtype=numpy.float32) - numpy.array(img2, dtype=numpy.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * numpy.log10(max_value / (numpy.sqrt(mse)))


    #tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
inpImg = cv2.imread('lena512color.tiff')
inpImg = cv2.resize(inpImg, (516,516))

original_img_gray = cv2.cvtColor(inpImg, cv2.COLOR_BGR2GRAY)

sigma = 0.2
inpImg = random_noise(inpImg, var=sigma**2, mode='gaussian')#random_noise(inpImg, mode="speckle")
inpImg = numpy.array(255*inpImg, dtype = 'uint8')

img_gray = cv2.cvtColor(inpImg, cv2.COLOR_BGR2GRAY)
img_gray = img_gray.astype('int')
gray_img = img_gray

inpImg = numpy.stack((img_gray,)*3, axis=-1)


#  #PIL.Image.open('lena512color.tiff')

#### Converting numpy array image to torch float tensor. Estimation of the edges #####

tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(inpImg).transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

tenOutput = estimate(tenInput)

###### End: estimating edges ########



probArray = tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0]
img_numpy = (tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)

revProb = (1-probArray)

for m in range(13):
    out = numpy.zeros((516,516))
    out = pb.diffusedImage(img_gray,revProb) # calling anisotropic diffusion function
    
    img_gray= numpy.zeros((516,516))
    img_gray = out.astype('int')


resultOut = out.astype('uint8')
#plt.imshow(resultOut)


label = 'PSNR: {:.2f} SSIM: {:.2f}'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                          sharex=True, sharey=True)

ax = axes.ravel()

psnr_noisy = calculate_psnr(original_img_gray, gray_img)
psnr_restored = calculate_psnr(original_img_gray, resultOut)

ssim_noisy = ssim(original_img_gray, gray_img)
ssim_restored = ssim(original_img_gray, resultOut)

ax[0].imshow(gray_img)
ax[0].set_xlabel(label.format(psnr_noisy, ssim_noisy))
ax[0].set_title('Noisy image')

ax[1].imshow(resultOut)
ax[1].set_xlabel(label.format(psnr_restored, ssim_restored))
ax[1].set_title('Restored image')

fig.savefig('diffusion_output.png')

plt.tight_layout()
plt.show()


# cv2.imshow("probdiffusion_out",resultOut)
# cv2.imshow("original_input_image",gray_img)
# cv2.waitKey(-1)
cv2.imwrite('probdiffusion_out.png', resultOut)
#cv2.imwrite('probdiffusionarray_out.png', gray_img)
cv2.imwrite('original_input_image.png',gray_img)
# PIL.Image.fromarray((tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
