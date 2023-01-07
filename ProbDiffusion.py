# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:10:27 2020

@author: HP
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import cv2 


resultImg = np.zeros((516,516))

def diffusedImage(img, prob):
    resultImg = np.zeros((516,516))
    (height, width) = img.shape
    
    for i in range(width-2):
        for j in range(height-2):
            subimg = img[i:i+3,j:j+3]
            subprob = prob[i:i+3,j:j+3]
            resultImg[i+1,j+1] = anisotropicDiffusion(subimg,subprob)
    return resultImg

def anisotropicDiffusion(subImg, subProb):
    # middle pixel : subImg[1,1]
    N = subImg[0,1]-subImg[1,1]
    S = subImg[2,1]-subImg[1,1]
    W = subImg[1,0]-subImg[1,1]
    E = subImg[1,2]-subImg[1,1]
    NE = subImg[0,2]-subImg[1,1]
    NW = subImg[0,0]-subImg[1,1]
    SE = subImg[2,2]-subImg[1,1]
    SW = subImg[2,0]-subImg[1,1]
    
    pix = 0.125*(subProb[0,1]*N + subProb[2,1]*S +subProb[1,2]*E + subProb[1,0]*W
                 +0.5*subProb[0,2]*NE + 0.5*subProb[0,0]*NW + 0.5*subProb[2,0]*SW +0.5*subProb[2,2]*SE)
                 
                
    # pix = (subProb[0,1]*subImg[0,1] + subProb[2,1]*subImg[2,1] +
    #        subProb[1,0]*subImg[1,0] + subProb[1,2]*subImg[1,2] + 
    #        subProb[0,2]*subImg[0,2] + subProb[0,0]*subImg[0,0]
    #              + subProb[2,2]*subImg[2,2] + subProb[2,0]*subImg[2,0] + 
    #              subProb[1,1]*subImg[1,1])/9
    # if(subImg[1,1] < pix):
    #     subImg[1,1] = pix
    
    #subImg[1,1] = subImg[1,1] + pix
    
    return (subImg[1,1] + pix)

    

# img_name = 'cameraman.png'
# img = cv2.imread(img_name)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# h,w = img.shape
# img = cv2.resize(img, (516,516))
# prob = np.random.uniform(low=0.0, high=0.1, size=(516,516))
# prob = (1-prob)
# for count in range(10):
#     resultImg = np.zeros((516,516))
#     img = img.astype("int")
#     img = diffusedImage(img, prob)
# # img = data.camera()
# # for i in range(1):
# #     out = cv2.medianBlur(img,11)
# #     img = out
# cv2.imwrite('probdiffusion_out.png', img)
# plt.imshow(resultImg)
