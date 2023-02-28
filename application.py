#validation


import os
import random
import math
import shutil
import cv2
import csv
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw

import pywt
import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

import pylab as py
import radialProfile
from scipy import fftpack

import skimage
from skimage import transform
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.filters import try_all_threshold
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)

import sklearn
from sklearn.metrics import mutual_info_score as mi 
from sklearn.metrics import mean_absolute_error as mae
from scipy import stats, signal

from tensorflow.keras import backend as K

from utils import *
from models import *





dspath32='/media/lxia/Expansion/DATA/Patient32/IMAGES/'

pathref='/media/lxia/Expansion/LowDose/lowdoseTest/IMAGES/'
pathsrc32='/media/lxia/Expansion/LDSimulation/DL/PC195/DL_ICASSP/experiment/P32/source/'
pathmatch32='/media/lxia/Expansion/LDSimulation/DL/PC195/DL_ICASSP/experiment/P32/match/'


imnms75=random.sample(os.listdir(dspath75),10)
#imnms75=rmpng(os.listdir(pathsrc75))


ims32=[]
for nm in imnms32:
    imm=readdicom(nm,dspath32,'midastinum')
    plt.imsave(os.path.join(pathsrc32,nm)+'.png',imm,cmap='gray')
    im=readdicom(nm,dspath32,None)
    ims32.append(im)
    

immatched32=imshistmatch(imnms32, dspath32, imnms120, pathref,pathmatch32)


outputsim_rm32="/media/lxia/Expansion/LDSimulation/DL/PC195/DL_ICASSP/experiment/P32/output_match/"
outputdif_rm32="/media/lxia/Expansion/LDSimulation/DL/PC195/DL_ICASSP/experiment/P32/diff_match/"
imgouputs_rm32=model213output(checkpoint,np.array(immatched32))


findscale(imnms80,pathref,imnms32,ims32,imgouputs_rm32,"histogram_correlation_k_p32.csv")


imgoutputs_rm32=[]
for i in range(len(imgouputs_rm32)):
    #print(i)
    nm=imnms32[i]
    plt.imsave(outputdif_rm32+nm+'_2.5.png',2.5*pthres0(imgouputs_rm32[i],250),cmap='gray')
    img=readdicomm(nm,ims32[i]-2.5*pthres0(imgouputs_rm32[i],200),dspath32,'midastinum')+10
    imgoutputs_rm32.append(img)
    plt.imsave(outputsim_rm32+nm+'_2.5.png',img,cmap='gray')

***
histogram matching
***
x1, y1 = ecdf(source.ravel())
x2, y2 = ecdf(template.ravel())
x3, y3 = ecdf(matched.ravel())

plt.figure(figsize=(15,8))

plt.plot(x1, y1 * 100, '-r', lw=5, label='Source')
plt.plot(x2, y2 * 100, '-k', lw=5, label='Reference')
plt.plot(x3, y3 * 100, '--y', lw=5, label='Matched')
#plt.xlim(x1[0], x1[-1])
plt.tick_params(labelsize=15)
plt.xlabel('Pixel value',fontsize=30)
plt.ylabel('Cumulative %',fontsize=30)
plt.legend(loc=5,fontsize=28)
plt.title('Cumulative Histogram Comparison',fontsize=40)
