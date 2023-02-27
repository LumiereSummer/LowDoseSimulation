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

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Reshape, Input, concatenate, add
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import matplotlib.pyplot as plt

'''
resnet
'''
#According to GitHub Repository: https://github.com/calmisential/TensorFlow2.0_ResNet
class BasicBlock(Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn1 = BatchNormalization()
        self.lrelu1 = LeakyReLU(alpha=0.2)
        self.conv2 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding='same')
        self.bn2 = BatchNormalization()
        self.lrelu2 = LeakyReLU(alpha=0.2)
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)
        #x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu1(x)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output
    
    
    

class Basic3Block(Layer):

    def __init__(self, filter_num, stride=1):
        super(Basic3Block, self).__init__()
        self.conv1 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn1 = BatchNormalization()
        self.lrelu1 = LeakyReLU(alpha=0.2)
        self.conv2 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding='same')
        self.bn2 = BatchNormalization()
        self.lrelu2 = LeakyReLU(alpha=0.2)
        self.conv3 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding='same')
        self.bn3 = BatchNormalization()
        self.lrelu3 = LeakyReLU(alpha=0.2)
        
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)
        #x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu1(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.lrelu3(x)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output
    



    
    
class BottleNeck(Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = BatchNormalization()

        self.downsample = Sequential()
        self.downsample.add(Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output





def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for i in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_basic_3block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(Basic3Block(filter_num, stride=stride))

    for i in range(1, blocks):
        res_block.add(Basic3Block(filter_num, stride=1))

    return res_block




def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for i in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block





#model 1

def generator1(res_num,input_shape):

    #encoder
    inp = Input(input_shape)
    
    conv1 = Conv2D(64,3,padding = 'same', activation = 'relu')(inp)
    dp1 = Dropout(0.5)(conv1)
    bn1 = BatchNormalization()(dp1)
    lr1 = LeakyReLU(alpha=0.2)(bn1)
    
    conv2 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr1)
    dp2 = Dropout(0.5)(conv2)
    bn2 = BatchNormalization()(dp2)
    lr2 = LeakyReLU(alpha=0.2)(bn2)
    
    conv3 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr2)
    dp3 = Dropout(0.5)(conv3)
    bn3 = BatchNormalization()(dp3)
    lr3 = LeakyReLU(alpha=0.2)(bn3)
    
    #residual blocks
    rb=make_basic_block_layer(filter_num=256, blocks=res_num)(lr3)
    
    #decoder
    rs1=Resizing(input_shape[0],input_shape[1])(rb)
    dp4=Dropout(0.5)(rs1)
    dconv1=Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(dp4)
    bn4=BatchNormalization()(dconv1)
    lr4=LeakyReLU(alpha=0.2)(bn4)
    
    rs2=Resizing(input_shape[0],input_shape[1])(lr4)
    dp5=Dropout(0.5)(rs2)
    dconv2=Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(dp5)
    bn5=BatchNormalization()(dconv2)
    lr5=LeakyReLU(alpha=0.2)(bn5)

    
    merge1=add([lr1,lr5])
    
    
    rs3=Resizing(input_shape[0],input_shape[1])(merge1)
    dp6=Dropout(0.5)(rs3)
    dconv3=Conv2DTranspose(1, (3,3), strides=(1,1), padding='same')(dp6)
    bn6=BatchNormalization()(dconv3)
    lr6=LeakyReLU(alpha=0.2)(bn6)
    
    outp=add([inp,lr6])
    
    model = Model(inputs = inp, outputs = outp)
    model.summary()
    
    return model
    
        








