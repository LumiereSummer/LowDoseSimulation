#u-net model, with different filter numbers
#the one called "generator33" is used for generating the results in the paper

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Reshape, Input, Concatenate, add, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers.experimental.preprocessing import Resizing



def generator30(input_shape):

    #encoder
    inp = Input(input_shape)
    
    #scale 0
    conv1 = Conv2D(16,3,padding = 'same', activation = 'relu')(inp)
    bn1 = BatchNormalization()(conv1)
    lr1 = LeakyReLU(alpha=0.2)(bn1)
    conv2 = Conv2D(16,3,padding = 'same', activation = 'relu')(lr1)
    bn2 = BatchNormalization()(conv2)
    lr2 = LeakyReLU(alpha=0.2)(bn2)
    conv3 = Conv2D(16,3,padding = 'same', activation = 'relu')(lr2)
    bn3 = BatchNormalization()(conv3)
    lr3 = LeakyReLU(alpha=0.2)(bn3)
    mp1 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr3)
    #dp1 = Dropout(0.5)(conv1)
    
    
    #scale 1
    conv4 = Conv2D(32,3,padding = 'same', activation = 'relu')(mp1)
    bn4 = BatchNormalization()(conv4)
    lr4 = LeakyReLU(alpha=0.2)(bn4)
    conv5 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr4)
    bn5 = BatchNormalization()(conv5)
    lr5 = LeakyReLU(alpha=0.2)(bn5)
    conv6 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr5)
    bn6 = BatchNormalization()(conv6)
    lr6 = LeakyReLU(alpha=0.2)(bn6)
    mp2 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr6)
    #dp2 = Dropout(0.5)(conv2)
    
    
    #scale 2
    conv7 = Conv2D(64,3,padding = 'same', activation = 'relu')(mp2)
    bn7 = BatchNormalization()(conv7)
    lr7 = LeakyReLU(alpha=0.2)(bn7)
    conv8 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr7)
    bn8 = BatchNormalization()(conv8)
    lr8 = LeakyReLU(alpha=0.2)(bn8)
    conv9 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr8)
    bn9 = BatchNormalization()(conv9)
    lr9 = LeakyReLU(alpha=0.2)(bn9)
    mp3 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr9)
    #dp3 = Dropout(0.5)(conv3)
    
    
    #center convolution
    conv10 = Conv2D(64,3,padding = 'same', activation = 'relu')(mp3)
    bn10 = BatchNormalization()(conv10)
    lr10 = LeakyReLU(alpha=0.2)(bn10)
    up1 = UpSampling2D(size=(2,2))(lr10)
    #dp3 = Dropout(0.5)(conv3)
    
    
    #residual blocks
    #rb=make_basic_block_layer(filter_num=128, blocks=res_num)(lr3)
    
    #decoder
    #scale 2
    cct1 = Concatenate()([up1, lr9])
    conv11 = Conv2D(64,3,padding = 'same', activation = 'relu')(cct1)
    bn11 = BatchNormalization()(conv11)
    lr11 = LeakyReLU(alpha=0.2)(bn11)
    conv12 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr11)
    bn12 = BatchNormalization()(conv12)
    lr12 = LeakyReLU(alpha=0.2)(bn12)
    conv13 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr12)
    bn13 = BatchNormalization()(conv13)
    lr13 = LeakyReLU(alpha=0.2)(bn13)
    up2 = UpSampling2D(size=(2,2))(lr13)
    
    #scale 1
    cct2 = Concatenate()([up2, lr6])
    conv14 = Conv2D(32,3,padding = 'same', activation = 'relu')(cct2)
    bn14 = BatchNormalization()(conv14)
    lr14 = LeakyReLU(alpha=0.2)(bn14)
    conv15 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr14)
    bn15 = BatchNormalization()(conv15)
    lr15 = LeakyReLU(alpha=0.2)(bn15)
    conv16 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr15)
    bn16 = BatchNormalization()(conv16)
    lr16 = LeakyReLU(alpha=0.2)(bn16)
    up3 = UpSampling2D(size=(2,2))(lr16)

    #scale 0
    cct3 = Concatenate()([up3, lr3])
    conv17 = Conv2D(16,3,padding = 'same', activation = 'relu')(cct3)
    bn17 = BatchNormalization()(conv17)
    lr17 = LeakyReLU(alpha=0.2)(bn17)
    conv18 = Conv2D(16,3,padding = 'same', activation = 'relu')(lr17)
    bn18 = BatchNormalization()(conv18)
    lr18 = LeakyReLU(alpha=0.2)(bn18)
    conv19 = Conv2D(16,3,padding = 'same', activation = 'relu')(lr18)
    bn19 = BatchNormalization()(conv19)
    lr19 = LeakyReLU(alpha=0.2)(bn19)


    outp = Conv2D(1,3,padding = 'same', activation = 'relu')(lr19)

    
    model = Model(inputs = inp, outputs = outp)
    model.summary()
    
    return model



def generator31(input_shape):

    #encoder
    inp = Input(input_shape)
    
    #scale 0
    conv1 = Conv2D(32,3,padding = 'same', activation = 'relu')(inp)
    bn1 = BatchNormalization()(conv1)
    lr1 = LeakyReLU(alpha=0.2)(bn1)
    conv2 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr1)
    bn2 = BatchNormalization()(conv2)
    lr2 = LeakyReLU(alpha=0.2)(bn2)
    conv3 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr2)
    bn3 = BatchNormalization()(conv3)
    lr3 = LeakyReLU(alpha=0.2)(bn3)
    mp1 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr3)
    #dp1 = Dropout(0.5)(conv1)
    
    
    #scale 1
    conv4 = Conv2D(64,3,padding = 'same', activation = 'relu')(mp1)
    bn4 = BatchNormalization()(conv4)
    lr4 = LeakyReLU(alpha=0.2)(bn4)
    conv5 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr4)
    bn5 = BatchNormalization()(conv5)
    lr5 = LeakyReLU(alpha=0.2)(bn5)
    conv6 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr5)
    bn6 = BatchNormalization()(conv6)
    lr6 = LeakyReLU(alpha=0.2)(bn6)
    mp2 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr6)
    #dp2 = Dropout(0.5)(conv2)
    
    
    #scale 2
    conv7 = Conv2D(128,3,padding = 'same', activation = 'relu')(mp2)
    bn7 = BatchNormalization()(conv7)
    lr7 = LeakyReLU(alpha=0.2)(bn7)
    conv8 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr7)
    bn8 = BatchNormalization()(conv8)
    lr8 = LeakyReLU(alpha=0.2)(bn8)
    conv9 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr8)
    bn9 = BatchNormalization()(conv9)
    lr9 = LeakyReLU(alpha=0.2)(bn9)
    mp3 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr9)
    #dp3 = Dropout(0.5)(conv3)
    
    
    #center convolution
    conv10 = Conv2D(128,3,padding = 'same', activation = 'relu')(mp3)
    bn10 = BatchNormalization()(conv10)
    lr10 = LeakyReLU(alpha=0.2)(bn10)
    up1 = UpSampling2D(size=(2,2))(lr10)


    #decoder
    #scale 2
    cct1 = Concatenate()([up1, lr9])
    conv11 = Conv2D(128,3,padding = 'same', activation = 'relu')(cct1)
    bn11 = BatchNormalization()(conv11)
    lr11 = LeakyReLU(alpha=0.2)(bn11)
    conv12 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr11)
    bn12 = BatchNormalization()(conv12)
    lr12 = LeakyReLU(alpha=0.2)(bn12)
    conv13 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr12)
    bn13 = BatchNormalization()(conv13)
    lr13 = LeakyReLU(alpha=0.2)(bn13)
    up2 = UpSampling2D(size=(2,2))(lr13)
    
    #scale 1
    cct2 = Concatenate()([up2, lr6])
    conv14 = Conv2D(64,3,padding = 'same', activation = 'relu')(cct2)
    bn14 = BatchNormalization()(conv14)
    lr14 = LeakyReLU(alpha=0.2)(bn14)
    conv15 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr14)
    bn15 = BatchNormalization()(conv15)
    lr15 = LeakyReLU(alpha=0.2)(bn15)
    conv16 = Conv2D(64,3,padding = 'same', activation = 'relu')(lr15)
    bn16 = BatchNormalization()(conv16)
    lr16 = LeakyReLU(alpha=0.2)(bn16)
    up3 = UpSampling2D(size=(2,2))(lr16)

    #scale 0
    cct3 = Concatenate()([up3, lr3])
    conv17 = Conv2D(32,3,padding = 'same', activation = 'relu')(cct3)
    bn17 = BatchNormalization()(conv17)
    lr17 = LeakyReLU(alpha=0.2)(bn17)
    conv18 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr17)
    bn18 = BatchNormalization()(conv18)
    lr18 = LeakyReLU(alpha=0.2)(bn18)
    conv19 = Conv2D(32,3,padding = 'same', activation = 'relu')(lr18)
    bn19 = BatchNormalization()(conv19)
    lr19 = LeakyReLU(alpha=0.2)(bn19)


    outp = Conv2D(1,3,padding = 'same', activation = 'relu')(lr19)

    
    model = Model(inputs = inp, outputs = outp)
    model.summary()
    
    return model





def generator32(input_shape):

    #encoder
    inp = Input(input_shape)
    
    #scale 0
    conv1 = Conv2D(128,3,padding = 'same', activation = 'relu')(inp)
    bn1 = BatchNormalization()(conv1)
    lr1 = LeakyReLU(alpha=0.2)(bn1)
    conv2 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr1)
    bn2 = BatchNormalization()(conv2)
    lr2 = LeakyReLU(alpha=0.2)(bn2)
    conv3 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr2)
    bn3 = BatchNormalization()(conv3)
    lr3 = LeakyReLU(alpha=0.2)(bn3)
    mp1 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr3)
    #dp1 = Dropout(0.5)(conv1)
    
    
    #scale 1
    conv4 = Conv2D(256,3,padding = 'same', activation = 'relu')(mp1)
    bn4 = BatchNormalization()(conv4)
    lr4 = LeakyReLU(alpha=0.2)(bn4)
    conv5 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr4)
    bn5 = BatchNormalization()(conv5)
    lr5 = LeakyReLU(alpha=0.2)(bn5)
    conv6 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr5)
    bn6 = BatchNormalization()(conv6)
    lr6 = LeakyReLU(alpha=0.2)(bn6)
    mp2 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr6)
    #dp2 = Dropout(0.5)(conv2)
    
    
    #scale 2
    conv7 = Conv2D(512,3,padding = 'same', activation = 'relu')(mp2)
    bn7 = BatchNormalization()(conv7)
    lr7 = LeakyReLU(alpha=0.2)(bn7)
    conv8 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr7)
    bn8 = BatchNormalization()(conv8)
    lr8 = LeakyReLU(alpha=0.2)(bn8)
    conv9 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr8)
    bn9 = BatchNormalization()(conv9)
    lr9 = LeakyReLU(alpha=0.2)(bn9)
    mp3 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr9)
    #dp3 = Dropout(0.5)(conv3)


    #center convolution
    conv10 = Conv2D(1024,3,padding = 'same', activation = 'relu')(mp3)
    bn10 = BatchNormalization()(conv10)
    lr10 = LeakyReLU(alpha=0.2)(bn10)
    up1 = UpSampling2D(size=(2,2))(lr10)
    #dp3 = Dropout(0.5)(conv3)

    #decoder
    #scale 2
    cct1 = Concatenate()([up1, lr9])
    conv11 = Conv2D(512,3,padding = 'same', activation = 'relu')(cct1)
    bn11 = BatchNormalization()(conv11)
    lr11 = LeakyReLU(alpha=0.2)(bn11)
    conv12 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr11)
    bn12 = BatchNormalization()(conv12)
    lr12 = LeakyReLU(alpha=0.2)(bn12)
    conv13 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr12)
    bn13 = BatchNormalization()(conv13)
    lr13 = LeakyReLU(alpha=0.2)(bn13)
    up2 = UpSampling2D(size=(2,2))(lr13)
    
    #scale 1
    cct2 = Concatenate()([up2, lr6])
    conv14 = Conv2D(256,3,padding = 'same', activation = 'relu')(cct2)
    bn14 = BatchNormalization()(conv14)
    lr14 = LeakyReLU(alpha=0.2)(bn14)
    conv15 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr14)
    bn15 = BatchNormalization()(conv15)
    lr15 = LeakyReLU(alpha=0.2)(bn15)
    conv16 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr15)
    bn16 = BatchNormalization()(conv16)
    lr16 = LeakyReLU(alpha=0.2)(bn16)
    up3 = UpSampling2D(size=(2,2))(lr16)

    #scale 0
    cct3 = Concatenate()([up3, lr3])
    conv17 = Conv2D(128,3,padding = 'same', activation = 'relu')(cct3)
    bn17 = BatchNormalization()(conv17)
    lr17 = LeakyReLU(alpha=0.2)(bn17)
    conv18 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr17)
    bn18 = BatchNormalization()(conv18)
    lr18 = LeakyReLU(alpha=0.2)(bn18)
    conv19 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr18)
    bn19 = BatchNormalization()(conv19)
    lr19 = LeakyReLU(alpha=0.2)(bn19)


    outp = Conv2D(1,3,padding = 'same', activation = 'relu')(lr19)

    
    model = Model(inputs = inp, outputs = outp)
    model.summary()
    
    return model





def generator33(input_shape):

    #encoder
    inp = Input(input_shape)
    
    #scale 0
    conv1 = Conv2D(128,3,padding = 'same', activation = 'relu')(inp)
    bn1 = BatchNormalization()(conv1)
    lr1 = LeakyReLU(alpha=0.2)(bn1)
    conv2 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr1)
    bn2 = BatchNormalization()(conv2)
    lr2 = LeakyReLU(alpha=0.2)(bn2)
    conv3 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr2)
    bn3 = BatchNormalization()(conv3)
    lr3 = LeakyReLU(alpha=0.2)(bn3)
    mp1 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr3)
    #dp1 = Dropout(0.5)(conv1)
    
    
    #scale 1
    conv4 = Conv2D(512,3,padding = 'same', activation = 'relu')(mp1)
    bn4 = BatchNormalization()(conv4)
    lr4 = LeakyReLU(alpha=0.2)(bn4)
    conv5 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr4)
    bn5 = BatchNormalization()(conv5)
    lr5 = LeakyReLU(alpha=0.2)(bn5)
    conv6 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr5)
    bn6 = BatchNormalization()(conv6)
    lr6 = LeakyReLU(alpha=0.2)(bn6)
    mp2 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr6)
    #dp2 = Dropout(0.5)(conv2)
    
    
    #scale 2
    conv7 = Conv2D(1024,3,padding = 'same', activation = 'relu')(mp2)
    bn7 = BatchNormalization()(conv7)
    lr7 = LeakyReLU(alpha=0.2)(bn7)
    conv8 = Conv2D(1024,3,padding = 'same', activation = 'relu')(lr7)
    bn8 = BatchNormalization()(conv8)
    lr8 = LeakyReLU(alpha=0.2)(bn8)
    conv9 = Conv2D(1024,3,padding = 'same', activation = 'relu')(lr8)
    bn9 = BatchNormalization()(conv9)
    lr9 = LeakyReLU(alpha=0.2)(bn9)
    mp3 = MaxPool2D(pool_size=(2, 2),strides=(2, 2))(lr9)
    #dp3 = Dropout(0.5)(conv3)


    #center convolution
    conv10 = Conv2D(2048,3,padding = 'same', activation = 'relu')(mp3)
    bn10 = BatchNormalization()(conv10)
    lr10 = LeakyReLU(alpha=0.2)(bn10)
    up1 = UpSampling2D(size=(2,2))(lr10)
    #dp3 = Dropout(0.5)(conv3)

    #decoder
    #scale 2
    cct1 = Concatenate()([up1, lr9])
    conv11 = Conv2D(1024,3,padding = 'same', activation = 'relu')(cct1)
    bn11 = BatchNormalization()(conv11)
    lr11 = LeakyReLU(alpha=0.2)(bn11)
    conv12 = Conv2D(1024,3,padding = 'same', activation = 'relu')(lr11)
    bn12 = BatchNormalization()(conv12)
    lr12 = LeakyReLU(alpha=0.2)(bn12)
    conv13 = Conv2D(1024,3,padding = 'same', activation = 'relu')(lr12)
    bn13 = BatchNormalization()(conv13)
    lr13 = LeakyReLU(alpha=0.2)(bn13)
    up2 = UpSampling2D(size=(2,2))(lr13)
    
    #scale 1
    cct2 = Concatenate()([up2, lr6])
    conv14 = Conv2D(512,3,padding = 'same', activation = 'relu')(cct2)
    bn14 = BatchNormalization()(conv14)
    lr14 = LeakyReLU(alpha=0.2)(bn14)
    conv15 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr14)
    bn15 = BatchNormalization()(conv15)
    lr15 = LeakyReLU(alpha=0.2)(bn15)
    conv16 = Conv2D(512,3,padding = 'same', activation = 'relu')(lr15)
    bn16 = BatchNormalization()(conv16)
    lr16 = LeakyReLU(alpha=0.2)(bn16)
    up3 = UpSampling2D(size=(2,2))(lr16)

    #scale 0
    cct3 = Concatenate()([up3, lr3])
    conv17 = Conv2D(256,3,padding = 'same', activation = 'relu')(cct3)
    bn17 = BatchNormalization()(conv17)
    lr17 = LeakyReLU(alpha=0.2)(bn17)
    conv18 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr17)
    bn18 = BatchNormalization()(conv18)
    lr18 = LeakyReLU(alpha=0.2)(bn18)
    conv19 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr18)
    bn19 = BatchNormalization()(conv19)
    lr19 = LeakyReLU(alpha=0.2)(bn19)

    outp = Conv2D(1,3,padding = 'same', activation = 'relu')(lr19)

    model = Model(inputs = inp, outputs = outp)
    model.summary()
    
    return model



