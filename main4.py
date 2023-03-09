# -*- coding: utf-8 -*-
'''
Created on Wed July 01 20:15:10 2022
low dose simulation
target is variance difference
model 3,4,5: Unet
Adam,epoch=1200
@author: xialumi
'''

from readdata import *
from utils import *
from model_Unet import *

from train_model3 import *
from train_model311 import *
from train_model312 import *
from train_model4 import *
from train_model5 import *
from train_model import *
from train_model6 import *
#from train_model3 import *



import os
import random
import math
import shutil
import cv2
import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


import tensorflow
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError 
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger



imgdiff_train = imgs120_train-imgs80_train
imgdiff_test = imgs120_test-imgs80_test


imgs120fft_train=imgsfft(imgs120_train)[0]
imgs80fft_train=imgsfft(imgs80_train)[0]
imgs120fft_test=imgsfft(imgs120_test)[0]
imgs80fft_test=imgsfft(imgs80_test)[0]


fftdiff_train = imgs120fft_train-imgs80fft_train
fftdiff_test = imgs120fft_test-imgs80fft_test


imgs120fft_train_patch=dim_exp(dmreduce(imgtopatch(imgs120fft_train,64)))
imgs80fft_train_patch=dim_exp(dmreduce(imgtopatch(imgs80fft_train,64)))
imgs120fft_test_patch=dim_exp(dmreduce(imgtopatch(imgs120fft_test,64)))
imgs80fft_test_patch=dim_exp(dmreduce(imgtopatch(imgs80fft_test,64)))



fftdiff_train_patch=dim_exp(dmreduce(imgtopatch(fftdiff_train,64)))
fftdiff_test_patch=dim_exp(dmreduce(imgtopatch(fftdiff_test,64)))


 
'''
#non subsampled transform
imgs120nsct_train=readimgtonsct(imgs120_train)
imgs80nsct_train=readimgtonsct(imgs80_train)
imgs120nsct_test=readimgtonsct(imgs120_test)
imgs80nsct_test=readimgtonsct(imgs80_test)

imgs120nsct_train_patch=dim_exp(dmreduce(nscttopatch(imgs120nsct_train,64)))
imgs80nsct_train_patch=dim_exp(dmreduce(nscttopatch(imgs80nsct_train,64)))
imgs120nsct_test_patch=dim_exp(dmreduce(nscttopatch(imgs120nsct_test,64)))
imgs80nsct_test_patch=dim_exp(dmreduce(nscttopatch(imgs80nsct_test,64)))

'''

''' 
#img wavelet transform patches
imgs120_train_patch=dim_exp(dmreduce(imgwtpatch(imgs120_train,64)))
imgs80_train_patch=dim_exp(dmreduce(imgwtpatch(imgs80_train,64)))
imgs120_test_patch=dim_exp(dmreduce(imgwtpatch(imgs120_test,64)))
imgs80_test_patch=dim_exp(dmreduce(imgwtpatch(imgs80_test,64)))
'''

'''
imgs120_train_patch=dim_exp(dmreduce(imgtopatch(imgs120_train,128)))
imgs80_train_patch=dim_exp(dmreduce(imgtopatch(imgs80_train,128)))
imgs120_test_patch=dim_exp(dmreduce(imgtopatch(imgs120_test,128)))
imgs80_test_patch=dim_exp(dmreduce(imgtopatch(imgs80_test,128)))

inpdiff_train_patch=dim_exp(dmreduce(imgtopatch(inpdiff_train,128)))
inpdiff_test_patch=dim_exp(dmreduce(imgtopatch(inpdiff_test,128)))
'''


res_nums=[3,4,5,6,7,8,9,10]
#lrs=[0.01,0.001,0.0001]
#lrs_nm=['001','301','401']
#lrs=[0.000001,0.0000001,0.00000001]
#lrs_nm=['601','701','801']
lrs=[0.00001,0.000001]
lrs_nm=['501','601']
losses=[MeanAbsoluteError()]
losses_nm=['mae']

#checkpoint0='/mnt/DONNEES/lxia/LDSimulation/DL3/checkpoint/model3_adam401_mae_1492_0.5198.hdf5'
#checkpoint0='/mnt/DONNEES/lxia/LDSimulation/DL3/checkpoint/model3_adam401_mae_2944_0.5163cnd.hdf5'
#train_model31_cnd(checkpoint0,6000,4, (64,64,1), lrs, lrs_nm, losses, losses_nm, imgs120fft_train_patch, fftdiff_train_patch, imgs120fft_test_patch, fftdiff_test_patch)

train_model32m1s(15000,4, (64,64,1), lrs, lrs_nm, losses, losses_nm, imgs120fft_train_patch, fftdiff_train_patch, imgs120fft_test_patch, fftdiff_test_patch)



'''
train_model628(1500,4, (64,64,1), lrs, lrs_nm, losses, losses_nm, imgs120fft_train_patch, fftdiff_train_patch, imgs120fft_test_patch, fftdiff_test_patch)




train_model6(1200,4, (64,64,1), lrs, lrs_nm, losses, losses_nm, imgs120nsct_train_patch, imgs80nsct_train_patch, imgs120nsct_test_patch, imgs80nsct_test_patch)



train_model6(1200,4, (64,64,1), lrs, lrs_nm, losses, losses_nm, imgs120_train_patch, imgs80_train_patch, imgs120_test_patch, imgs80_test_patch)



train_model3(1200,4, (128,128,1), lrs, lrs_nm, losses, losses_nm, imgs120_train_patch, inpdiff_train_patch, imgs120_test_patch, inpdiff_test_patch)



train_model4(1200,4, (128,128,1), lrs, lrs_nm, losses, losses_nm, imgs120_train_patch, inpdiff_train_patch, imgs120_test_patch, inpdiff_test_patch)



train_model5(1200,4, (128,128,1), lrs, lrs_nm, losses, losses_nm, imgs120_train_patch, inpdiff_train_patch, imgs120_test_patch, inpdiff_test_patch)



train_model1(1200,4, (64,64,1), res_nums, lrs,lrs_nm, losses,losses_nm, imgs120_train_patch, imgs80_train_patch, imgs120_test_patch, imgs80_test_patch)
'''





