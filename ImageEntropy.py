# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:45:13 2020

@author: HP
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import cv2

def entropy(signal):
    
    lenSig = signal.size
    symset = list(set(signal))
    numsym = len(symset)
    prob = [np.size(signal[signal == i])/(1.0*lenSig) for i in symset]
    ent = np.sum([p*np.log2(1.0/p) for p in prob if p!= 0])
    return ent

image = data.astronaut()
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imageArr = np.array(image)
gray_imgArr = np.array(gray_img)

N = 5
S = gray_imgArr.shape

E = np.array(gray_imgArr)

for row in range(S[0]):
    for col in range(S[1]):
        Lx = np.max([0, col-N])
        Ux = np.max([S[1], col+N])
        Ly = np.max([0, row-N])
        Uy = np.max([S[0], row+N])
        region = gray_imgArr[Ly:Uy,Lx:Ux].flatten()
        E[row, col] = entropy(region)


plt.subplot(1,3,1)
plt.imshow(imageArr)

plt.subplot(1,3,2)
plt.imshow(gray_imgArr, cmap=plt.cm.gray)

plt.subplot(1,3,3)
plt.imshow(E, cmap=plt.cm.jet)
plt.xlabel('Entropy in 10x10 neighbourhood')
plt.colorbar()

plt.show()