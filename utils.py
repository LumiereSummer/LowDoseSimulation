#the functions used


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


from models import *

***
Data Preparation
***

#read dicom images (original)
def readdicom(nm,dspath,window):
    
    ds=dicom.dcmread(os.path.join(dspath,nm))
    #print(ds.SliceLocation)
    if window!=None:
        img=set_window(ds.pixel_array,ds,window)
    else:
        img=ds.pixel_array
    return img


#read dicom images (with modified pixels)
def readdicomm(nm,im,dspath,window):
    
    ds=dicom.dcmread(os.path.join(dspath,nm))
    #print(ds.SliceLocation)
    if window!=None:
        img=set_window(im,ds,window)
    else:
        img=im
    return img

  
#define the window
def window_image(image, ds, window_center, window_width):

    hu = apply_modality_lut(image, ds)
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    window_image = hu.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image

#choice of windows
def set_window(img,ds,window):
    
    window_option={
        'lungs': [-600,1500],
        'midastinum': [50,400],
        'abdomen': [50,250],
        'liver': [30,150],
        'bone': [400,1800]
    }
    
    return window_image(img, ds, window_option[window][0], window_option[window][1])



   
#3D array to 2D
def dmreduce(arr):
    a=[]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            a.append(arr[i][j])
            
    return np.array(a)
    
#3D array to 4D
def dim_exp(imgs):
    imgs_dimep=[]
    for i in range(len(imgs)):
        imgs_dim=[]
        for ele in imgs[i]:
            #print(ele)
            ele_ep=np.expand_dims(ele, axis=1)
            #print(ele_ep.shape)
            imgs_dim.append(ele_ep)
        imgs_dimep.append(imgs_dim)
            
    return np.array(imgs_dimep)

#4D array to 3D
def dim_unexp(img):
    
    imgf=[]
    for i in range(len(img)):
        pix=[]
        for j in range(len(img[i])):
            pix.append(int(img[i][j]))
        imgf.append(pix)
        
    return np.array(imgf)


#pair low dose and high dose images of patient 0
def matchloc(imnms120,imnms80,dspath):
    
    locs120=[]
    imgs120=[]
    nms120=[]
    for i in range(len(imnms120)):
        nm=imnms120[i]
        ds=dicom.dcmread(os.path.join(dspath,nm))
        if ds.SliceLocation>90.0:
            locs120.append(float(ds.SliceLocation))
            imgs120.append(ds.pixel_array)
            nms120.append(nm)
    locs120,imgs120=list(zip(*sorted(zip(locs120,imgs120))))
    locs120,nms120=list(zip(*sorted(zip(locs120,nms120))))
    
    locs80=[]
    imgs80=[]
    nms80=[]
    for i in range(len(imnms80)):
        nm=imnms80[i]
        ds=dicom.dcmread(os.path.join(dspath,nm))
        locs80.append(float(ds.SliceLocation))
        imgs80.append(ds.pixel_array)
        nms80.append(nm)
    locs80,imgs80=list(zip(*sorted(zip(locs80,imgs80))))
    locs80,nms80=list(zip(*sorted(zip(locs80,nms80))))
    
    pairs=[]
    pairnms=[]
    for i in range(len(locs80)):
        for j in range(len(locs120)):
            loc80=locs80[i]
            loc120=locs120[j]
            nm80=nms80[i]
            nm120=nms120[j]
            pair=[]
            pairnm=[]
            if abs(loc80-loc120)<=0.5:
                pair.append(loc120)
                pair.append(loc80)
                pairs.append(pair)
                pairnm.append(nm120)
                pairnm.append(nm80)
                pairnms.append(pairnm)
    
    with open("patient0_pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['120kvp','80kvp'])
        writer.writerows(pairnms)

    return pairs,pairnms



#split images into patches
def imgtopatch(imgs,patch_size):
    
    nms=imgs.shape[0]
    h=imgs.shape[1]
    w=imgs.shape[2]
    w1=[*range(patch_size, int(w/patch_size)*patch_size+1, patch_size)]
    h1=[*range(patch_size, int(h/patch_size)*patch_size+1, patch_size)]
    patch_id=[]
    patch_sp=[]
    c=0
    for i in w1:
        for j in h1:
            patch_id.append(c)
            patch_sp.append((i,j))
            c+=1
    
    #print(patch_id)
    #print(patch_sp)
    patches_nm=len(patch_id)
    
    
    img_patches=[]
    for img in imgs:
        patches=np.zeros((patches_nm,patch_size,patch_size))
        for idx in patch_id:
            i=patch_sp[idx][0]
            j=patch_sp[idx][1]
            patches[idx]=img[i-patch_size:i, j-patch_size:j]
                
        img_patches.append(patches)
        
    return np.array(img_patches)




#assemble patches to images
def patchtoimg(patches,img_size):
    
    nms=patches.shape[0]
    #print(nms)
    patch_nms=patches.shape[1]
    patch_size=patches.shape[2]
    row_nm=int(img_size/patch_size)
    #print(row_nm)
    imgs=[]
    #imgs=np.zeros((nms,img_size,img_size))

    patch_id=np.array_split(range(int(img_size*img_size/patch_size/patch_size)), int(img_size/patch_size))
    for nm in range(nms):
        rows=[]
        for i in range(len(patch_id)):
            row_patches=[]
            for j in patch_id[i]:
                #print(j)
                row_patches.append(list(patches[nm][j]))
                row=np.concatenate(row_patches,axis=1)
            rows.append(row)
        img=np.concatenate(rows,axis=0)
        #plt.figure(figsize=(20,20))
        #plt.imshow(img)
        imgs.append(img)    
  
    return np.array(imgs)
 
   

#remove ".png" from images names
def rmpng(imnms):
  
    for i in range(len(imnms)):
        nm=imnms[i]
        imnms[i]=nm.split('.')[0]
    
    return imnms


  

***
Histogram Matching & Histogram Correlation
***
  
  
#show the global histogram of all images in a dictionary, get these images and their names
def readimhist(path):
    
    imgnms=os.listdir(path)
    imnms=[]
    for nm in imgnms:
        ds=dicom.dcmread(os.path.join(path,nm))
        if 'PixelData' in ds.dir("pixel"):
            if ds.pixel_array.shape==(512,512):
                imnms.append(nm)
    
    imsnms=random.sample(imnms,10)
    imgs=[]
    for nm in imsnms:
        ds=dicom.dcmread(os.path.join(path,nm))
        img=ds.pixel_array
        plt.hist(img, bins=10)
        imgs.append(img)
        
    return imsnms,imgs

 
#show the global histogram of selected images, and get these images
def readimnmhist(path,imsnms):
    
    imgs=[]
    for nm in imsnms:
        ds=dicom.dcmread(os.path.join(path,nm))
        img=ds.pixel_array
        plt.hist(img, bins=10)
        imgs.append(img)
        
    return imgs
    
    
    
#Histogram Matching: 
#Adjust the pixel values of a grayscale image so that its histogram matches that of a target image
def hist_match(source, template):

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

#computing the empirical CDF: Cumulative Histogram
def ecdf(x):
  
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf
 
  
#match images with one reference image
def imsmatch(template,imgs):
    
    imgs_match=[]
    for im in imgs:
        matched = hist_match(im, template)
        imgs_match.append(matched)
    
    return imgs_match

  
#find the corresponding reference image (from many) for selected images 
def findref(pathref,imnmsref,imnms,path):
    
    imnm_refs=[]
    for i in range(len(imnms)):
        print('search reference image of ',imnms[i])
        nm=os.path.join(path,imnms[i])
        im=readdicom(nm,path,'midastinum')
        imnm_ref=matchimnm(im,imnmsref,pathref)[1][0]
        print("reference image for image {} is image {}".format(imnms[i], imnm_ref))
        imnm_refs.append(imnm_ref)
    
    return imnm_refs

#match images with their corresponding reference images
def imshistmatch(imnms, path, imnms_ref, pathref,pathmatch):
    
    matched=[]
    for i in range(len(imnms)):
        nm=os.path.join(path,imnms[i])
        #print(nm)
        im=readdicom(nm,path,None)
        #plt.figure()
        #plt.imshow(im,cmap='gray')
        nm_ref=os.path.join(pathref,imnms_ref[i])
        im_ref=readdicom(nm_ref,pathref,None)
        #plt.figure()
        #plt.imshow(im_ref[101:390,50:460],cmap='gray')
        #im_match=hist_match(im,im_ref[101:390,50:460])
        im_match=hist_match(im,im_ref)
        #plt.figure()
        #plt.imshow(im_match,cmap='gray')
        plt.imsave(os.path.join(pathmatch,imnms[i])+'m.png',im_match,cmap='gray')
        matched.append(im_match)
    
    return matched


#Thresholding: give the threshold pixel vaule to those who exceeds the threshold (the anomaly pixel)
def pthres(img,thres):
    
    w=img.shape[0]
    h=img.shape[1]
    img_thr=np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if img[i][j]>thres:
                img_thr[i][j]=thres
            else:
                img_thr[i][j]=img[i][j]
    
    return img_thr

#Thresholding: give 0 to those who exceeds the threshold pixel value (the anomaly pixel)
def pthres0(img,thres):
    
    w=img.shape[0]
    h=img.shape[1]
    img_thr=np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if img[i][j]>thres:
                img_thr[i][j]=0
            else:
                img_thr[i][j]=img[i][j]
    
    return img_thr


#calculate the histogram correlation of two images
def histcorr(im1,im2):
    
    hist_img1=cv2.calcHist([im1.astype(np.uint8)],[0],None,[10],[-150, 100])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2=cv2.calcHist([im2.astype(np.uint8)],[0],None,[10],[-150, 100])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    histocorr = cv2.compareHist(hist_img2, hist_img1, cv2.HISTCMP_CORREL)
    #print("histogram correlation is :",histocorr)
    
    return histocorr


#find the best scaling factor for simulating the desired dose images
#return to the best scaling factor (kmax) and the best HC (Histogram Correlation) value
def findscale(imnmsref,pathref,imnmsinput,imsinput,outputs,filenm):
    
    ks=[1.0,1.2,1.5,1.8,2.0,2.2,2.5,2.8,3.0]
    
    imrefs=[]
    for i in range(len(imnmsref)):
        imnmref=imnmsref[i]
        imref=readdicom(imnmref,pathref,"midastinum")
        imrefs.append(imref)
    
    imrefs=random.sample(imrefs,10)
    for i in range(len(outputs)):
        hcs=[]
        for k in ks:
            im=readdicomm(nm,imsinput[i]-k*pthres0(outputs[i],250),dspath26,'midastinum')
            hc=[]
            for j in range(len(imrefs)):
                imref=imrefs[j]
                hc.append(histcorr(im,imref))
            hcs.append(hc)
    
    hckmn=[]
    for i in range(len(hcs)):
        hck=[]
        for j in range(len(hcs[i])):
            hck.append(hcs[i][j])
        hckmn.append(np.mean(hck))
    
    hckmns,kss=list(zip(*sorted(zip(hckmn,ks))))
    kmax=kss[-1]
    
    with open(filenm, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['1','1.2','1.5','1.8','2','2.2','2.5','2.8','3.0'])
        writer.writerow(hckmn)
        writer.writerows(hcs)
    
    return hckmns[-1],kmax



***
Model Output
***

  
#apply the trained model on other images : take U Net model as example
#change the checkpoint and model for applying other model
def model213output(checkpoint,imgs):
    
    model0=generator33((64,64,1))
    model0.load_weights(checkpoint)
    
    wtsp=imgtopatch(imgs,64)
    outp=model0.predict(dim_exp(dmreduce(wtsp)))
    otp=[]
    for i in outp:
        otp.append(dim_unexp(i))
    
    opt_arr=np.array(otp)
    opt4d=np.reshape(opt_arr,(int(len(opt_arr)/64),64,64,64))
    outpimg=patchtoimg(opt4d,512)
    
    return outpimg

