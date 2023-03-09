#training models

from readdata import *
from utils import *

from model_Unet import *
from model_Resnet import *

from train_model_Unet import *
from train_model_Resnet import *


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


#generate image patcehs

imgdiff_train = imgs120_train-imgs80_train
imgdiff_test = imgs120_test-imgs80_test

imgs120_train_patch=dim_exp(dmreduce(imgtopatch(imgs120_train,64)))
imgs80_train_patch=dim_exp(dmreduce(imgtopatch(imgs80_train,64)))

imgs120_test_patch=dim_exp(dmreduce(imgtopatch(imgs120_test,64)))
imgs80_test_patch=dim_exp(dmreduce(imgtopatch(imgs80_test,64)))

diff_train_patch=dim_exp(dmreduce(imgtopatch(diff_train,64)))
diff_test_patch=dim_exp(dmreduce(imgtopatch(diff_test,64)))


#model training

res_nums=[3,4,5,6,7,8,9,10]
#lrs=[0.01,0.001,0.0001]
#lrs_nm=['001','301','401']
#lrs=[0.000001,0.0000001,0.00000001]
#lrs_nm=['601','701','801']
lrs=[0.00001,0.000001]
lrs_nm=['501','601']
losses=[MeanAbsoluteError()]
losses_nm=['mae']


train_model33(5000,4, (64,64,1), lrs, lrs_nm, losses, losses_nm, imgs120_train_patch, diff_train_patch, imgs120_test_patch, diff_test_patch)

