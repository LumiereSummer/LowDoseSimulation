#model evaluation

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


******
Intra-image Analyse : Mean pixel value (mean), Standard Deviation (SD), Region Uniformity (RU)
******
def outputanalyse_m(outputpath,imgoutput):
    
    nm=os.listdir(outputpath)

    nm_s=[]
    maxp_s=[]
    minp_s=[]
    meanp_s=[]
    stdp_s=[]
    #vari_s=[]
    unif_s=[]
    
    nm_o=[]
    maxp_o=[]
    minp_o=[]
    meanp_o=[]
    stdp_o=[]
    #vari_o=[]
    unif_o=[]
    
    nm_t=[]
    maxp_t=[]
    minp_t=[]
    meanp_t=[]
    stdp_t=[]
    #vari_t=[]
    unif_t=[]
    
    for i in range(len(nm)):
        
        if nm[i].split('_')[0]=='output':
            nm_o.append(int(nm[i].split('_')[1]))
            output=np.loadtxt(outputpath+nm[i])
            plt.imsave(imgoutput+nm[i].split('.')[0]+'.png',output,cmap='gray')
            maxp_o.append(np.max(output))
            minp_o.append(np.min(output))
            meanp_o.append(np.mean(output))
            stdp_o.append(np.std(output))
            #vari_o.append(variance(output))
            unif_o.append(region_uni(output))
        
        elif nm[i].split('_')[0]=='source':
            nm_s.append(int(nm[i].split('_')[1]))
            source=np.loadtxt(outputpath+nm[i])
            plt.imsave(imgoutput+nm[i].split('.')[0]+'.png',source,cmap='gray')
            maxp_s.append(np.max(source))
            minp_s.append(np.min(source))
            meanp_s.append(np.mean(source))
            stdp_s.append(np.std(source))
            #vari_s.append(variance(source))
            unif_s.append(region_uni(source))
        
        elif nm[i].split('_')[0]=='target':
            nm_t.append(int(nm[i].split('_')[1]))
            target=np.loadtxt(outputpath+nm[i])
            plt.imsave(imgoutput+nm[i].split('.')[0]+'.png',target,cmap='gray')
            maxp_t.append(np.max(target))
            minp_t.append(np.min(target))
            meanp_t.append(np.mean(target))
            stdp_t.append(np.std(target))
            #vari_t.append(variance(target))
            unif_t.append(region_uni(target))
            
    outp_hd=pd.DataFrame(list(zip(nm_s,maxp_s,minp_s,meanp_s,stdp_s,unif_s)),
                         columns=['image index', 'max (high)', 'min (high)', 'mean (high)', 'std (high)', 'region_uni (high)'])
    outp_hd=outp_hd.sort_values(by=['image index'])
    
    outp_sm=pd.DataFrame(list(zip(nm_o,maxp_o,minp_o,meanp_o,stdp_o,unif_o)),
                         columns=['image index', 'max (simu)', 'min (simu)', 'mean (simu)', 'std (simu)', 'region_uni (simu)'])
    outp_sm=outp_sm.sort_values(by=['image index']) 
    
    outp_ld=pd.DataFrame(list(zip(nm_t,maxp_t,minp_t,meanp_t,stdp_t,unif_t)),
                         columns=['image index', 'max (low)', 'min (low)', 'mean (low)', 'std (low)', 'region_uni (low)'])
    outp_ld=outp_ld.sort_values(by=['image index'])
    
    
    imgnms=['high dose images','simulated low dose','low dose images']
    mean_means=[np.mean(meanp_s),np.mean(meanp_o),np.mean(meanp_t)]
    mean_stds=[np.std(meanp_s),np.std(meanp_o),np.std(meanp_t)]
    std_means=[np.mean(stdp_s),np.mean(stdp_o),np.mean(stdp_t)]
    std_stds=[np.std(stdp_s),np.std(stdp_o),np.std(stdp_t)]
    #vari_means=[np.mean(vari_s),np.mean(vari_o),np.mean(vari_t)]
    #vari_stds=[np.std(vari_s),np.std(vari_o),np.std(vari_t)]
    unif_means=[np.mean(unif_s),np.mean(unif_o),np.mean(unif_t)]
    unif_stds=[np.std(unif_s),np.std(unif_o),np.std(unif_t)]
    
    outp_comp=pd.DataFrame(list(zip(imgnms,mean_means,mean_stds,std_means,std_stds,unif_means,unif_stds)),
                           columns=['images','mean of mean','std of mean','mean of std','std of std','mean of region_uni','std of region_uni'])
    
    return outp_hd,outp_sm,outp_ld,outp_comp


styles = [dict(selector="caption",
               props=[("text-align", "middle"),
                      ("font-size", "200%"),
                      ("color", 'black')])]






******
Mutual Information and entropy comparison
******

def mi3(im_o,im_s,im_t):
    
    im_o=im_o.flatten()
    im_s=im_s.flatten()
    im_t=im_t.flatten()
    mi_ot=mi(im_o,im_t)
    mi_st=mi(im_s,im_t)
    mi_os=mi(im_o,im_s)
    ht=mi(im_t,im_t)
    hs=mi(im_s,im_s)
    ho=mi(im_o,im_o)
    #entropy=[ho,hs,ht]
    #mis=[mi_os,mi_ot,mi_st
    mis={'image':['high dose','low dose','simulated low dose'],
         'high dose':[hs,mi_st,mi_os],
         'low dose':[mi_st,ht,mi_ot],
         'simulated low dose':[mi_os,mi_ot,ho]}
    midf = pd.DataFrame(data=mis)
    
    return midf

def mi3s(outputpath):
    
    nm=os.listdir(outputpath)

    nm_s=[]
    nm_o=[]
    nm_t=[]
    
    
    for i in range(len(nm)):
        if nm[i].split('_')[0]=='output':
            nm_o.append(nm[i])
        elif nm[i].split('_')[0]=='source':
            nm_s.append(nm[i])
        elif nm[i].split('_')[0]=='target':
            nm_t.append(nm[i])
   
    nm_pairs=[]
    pair_index=[]
    for i in range(len(nm_o)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o[i].split('_')[1]==nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index.append(int(nm_o[i].split('_')[1]))
                    nm_pairs.append([nm_o[i],nm_s[j],nm_t[k]])
               
    nm_pairs=[x for _, x in sorted(zip(pair_index,nm_pairs))]
    pair_index.sort()
    
    mi_ot=[]
    mi_st=[]
    mi_os=[]
    ht=[]
    hs=[]
    ho=[]
    
    for i in range(len(nm_pairs)):
        #output
        im1=np.loadtxt(outputpath+nm_pairs[i][0])
        #source
        im2=np.loadtxt(outputpath+nm_pairs[i][1])
        #target
        im3=np.loadtxt(outputpath+nm_pairs[i][2])    
        
        im_o=im1.flatten()
        im_s=im2.flatten()
        im_t=im3.flatten()
        mi_ot.append(mi(im_o,im_t))
        mi_st.append(mi(im_s,im_t))
        mi_os.append(mi(im_o,im_s))
        ht.append(mi(im_t,im_t))
        hs.append(mi(im_s,im_s))
        ho.append(mi(im_o,im_o))
    
    
    outp_mi=pd.DataFrame(list(zip(pair_index,hs,ho,ht,mi_os,mi_ot,mi_st)),
                         columns=['image index', 'entropy (high)', 'entropy (simu)', 'entropy (low)', 
                                  'mutual information (simu-high)', 'mutual information (simu-low)','mutual information (low-high)'])
    outp_mi=outp_mi.sort_values(by=['image index'])
    
    
    imgnms=['high dose images','simulated low dose','low dose images','simu-high','simu-low','low-high']
    micps=pd.DataFrame({'images':imgnms,
                       'mean':[np.mean(hs),np.mean(ho),np.mean(ht),np.mean(mi_os),np.mean(mi_ot),np.mean(mi_st)],
                       'std':[np.std(hs),np.std(ho),np.std(ht),np.std(mi_os),np.std(mi_ot),np.std(mi_st)]})
    
    
    return outp_mi,micps




******
Traditional IQA metrics: 
Mean Absolute error (MAE), Root Mean Squared Error (RMSE), 
Structure Similarity Index Measure (SSIM), Peak Signal Noise Ratio (PSNR)
******

def outputanalyse_cp(outputpath):
    
    nm=os.listdir(outputpath)

    nm_s=[]
    nm_o=[]
    nm_t=[]
    
    
    for i in range(len(nm)):
        if nm[i].split('_')[0]=='output':
            nm_o.append(nm[i])
        elif nm[i].split('_')[0]=='source':
            nm_s.append(nm[i])
        elif nm[i].split('_')[0]=='target':
            nm_t.append(nm[i])
   
    nm_pairs=[]
    pair_index=[]
    for i in range(len(nm_o)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o[i].split('_')[1]==nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index.append(int(nm_o[i].split('_')[1]))
                    nm_pairs.append([nm_o[i],nm_s[j],nm_t[k]])
               
    nm_pairs=[x for _, x in sorted(zip(pair_index,nm_pairs))]
    pair_index.sort()
    
    mae_os=[]
    mae_ot=[]
    mae_st=[]
    
    rmse_os=[]
    rmse_ot=[]
    rmse_st=[]
    
    ssim_os=[]
    ssim_ot=[]
    ssim_st=[]
    
    psnr_os=[]
    psnr_ot=[]
    psnr_st=[]
    
    
    for i in range(len(nm_pairs)):
        #output
        im1=np.loadtxt(outputpath+nm_pairs[i][0])
        #source
        im2=np.loadtxt(outputpath+nm_pairs[i][1])
        #target
        im3=np.loadtxt(outputpath+nm_pairs[i][2])
       
        mae_os.append(mae(im1,im2))
        mae_ot.append(mae(im1,im3))
        mae_st.append(mae(im2,im3))
        
        rmse_os.append(rmse(im1,im2))
        rmse_ot.append(rmse(im1,im3))
        rmse_st.append(rmse(im2,im3))
        
        ssim_os.append(ssim(im1,im2))
        ssim_ot.append(ssim(im1,im3))
        ssim_st.append(ssim(im2,im3))
        
        psnr_os.append(cv2.PSNR(im1,im2))
        psnr_ot.append(cv2.PSNR(im1,im3))
        psnr_st.append(cv2.PSNR(im2,im3))
        
        
   
    outp_mae=pd.DataFrame(list(zip(pair_index,mae_os,mae_ot,mae_st)),
                        columns=['image index', 'simu-high', 'simu-low', 'low-high'])
    outp_mae=outp_mae.sort_values(by=['image index'])
    
    outp_rmse=pd.DataFrame(list(zip(pair_index,rmse_os,rmse_ot,rmse_st)),
                        columns=['image index', 'simu-high', 'simu-low', 'low-high'])
    outp_rmse=outp_rmse.sort_values(by=['image index'])
    
    outp_ssim=pd.DataFrame(list(zip(pair_index,ssim_os,ssim_ot,ssim_st)),
                        columns=['image index', 'simu-high', 'simu-low', 'low-high'])
    outp_ssim=outp_ssim.sort_values(by=['image index'])
    
    outp_psnr=pd.DataFrame(list(zip(pair_index,psnr_os,psnr_ot,psnr_st)),
                        columns=['image index', 'simu-high', 'simu-low', 'low-high'])
    outp_psnr=outp_psnr.sort_values(by=['image index'])
    
    
    imgnms=['simu-high','simu-low','low-high']
    mae_means=[np.mean(mae_os),np.mean(mae_ot),np.mean(mae_st)]
    mae_stds=[np.std(mae_os),np.std(mae_ot),np.std(mae_st)]
    rmse_means=[np.mean(rmse_os),np.mean(rmse_ot),np.mean(rmse_st)]
    rmse_stds=[np.std(rmse_os),np.std(rmse_ot),np.std(rmse_st)]
    ssim_means=[np.mean(ssim_os),np.mean(ssim_ot),np.mean(ssim_st)]
    ssim_stds=[np.std(ssim_os),np.std(ssim_ot),np.std(ssim_st)]
    psnr_means=[np.mean(psnr_os),np.mean(psnr_ot),np.mean(psnr_st)]
    psnr_stds=[np.std(psnr_os),np.std(psnr_ot),np.std(psnr_st)]
    
    outp_comp=pd.DataFrame(list(zip(imgnms,mae_means,mae_stds,rmse_means,rmse_stds,ssim_means,ssim_stds,psnr_means,psnr_stds)),
                           columns=['images','mean of mae','std of mae','mean of rmse','std of rmse','mean of ssim','std of ssim','mean of psnr','std of psnr'])
    
    
    
    return outp_mae,outp_rmse,outp_ssim,outp_psnr,outp_comp
    



******
Histogram Correlation
******
def outputanalyse_h(outputpath):
   
    nm=os.listdir(outputpath)
   
    nm_s=[]
    nm_o=[]
    nm_t=[]

   
   
    for i in range(len(nm)):
        if nm[i].split('_')[0]=='output':
            nm_o.append(nm[i])
        elif nm[i].split('_')[0]=='source':
            nm_s.append(nm[i])
        elif nm[i].split('_')[0]=='target':
            nm_t.append(nm[i])
   
    nm_pairs=[]
    pair_index=[]
    for i in range(len(nm_o)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o[i].split('_')[1]==nm_t[k].split('_')[1] and nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index.append(int(nm_o[i].split('_')[1]))
                    nm_pairs.append([nm_o[i],nm_s[j],nm_t[k]])
               
    nm_pairs=[x for _, x in sorted(zip(pair_index,nm_pairs))]
    pair_index.sort()
    #pair_index.sort()
    #print(nm_pairs)
    
    corr_os=[]
    corr_ot=[]
    corr_st=[]
    
   
    for i in range(len(nm_pairs)):
        #output
        im1=np.loadtxt(outputpath+nm_pairs[i][0])
        #source
        im2=np.loadtxt(outputpath+nm_pairs[i][1])
        #target
        im3=np.loadtxt(outputpath+nm_pairs[i][2])
        #print(nm_pairs[i])
        hc=histocompare(im1,im2,im3,'correlation')
        
        corr_os.append(hc[0])
        corr_ot.append(hc[1])
        corr_st.append(hc[2])
   
    outp_corr=pd.DataFrame(list(zip(pair_index,corr_os,corr_ot,corr_st)),
                        columns=['image index', 'simu-high', 'simu-low', 'low-high'])
    outp_corr=outp_corr.sort_values(by=['image index'])
    
    
    imgnms=['simu-high','simu-low','low-high']
    corr_means=[np.mean(corr_os),np.mean(corr_ot),np.mean(corr_st)]
    corr_stds=[np.std(corr_os),np.std(corr_ot),np.std(corr_st)]
    
    outp_comp=pd.DataFrame(list(zip(imgnms,corr_means,corr_stds)),
                           columns=['images','mean of corr','std of corr'])
    
    
    
    return outp_corr,outp_comp




******
Spectrum
******
#spectrum of one image
def spectrum1d(img):
    
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(img)
    
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift( F1 )
    
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2 )**2

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(psd2D)
    
    return psd1D
    
    
#compare specturm of output image and original (high and low dose) images: for one model
def spectrumcompare(outputpath,imgoutput):
    
    nm=os.listdir(outputpath)
   
    nm_s=[]
    nm_o=[]
    nm_t=[]
    
    
    for i in range(len(nm)):
        if nm[i].split('_')[0]=='output':
            nm_o.append(nm[i])
        elif nm[i].split('_')[0]=='source':
            nm_s.append(nm[i])
        elif nm[i].split('_')[0]=='target':
            nm_t.append(nm[i])
   
    nm_pairs=[]
    pair_index=[]
    for i in range(len(nm_o)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o[i].split('_')[1]==nm_t[k].split('_')[1] and nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index.append(int(nm_o[i].split('_')[1]))
                    nm_pairs.append([nm_o[i],nm_s[j],nm_t[k]])
               
    nm_pairs=[x for _, x in sorted(zip(pair_index,nm_pairs))]
    pair_index.sort()
    #print(len(nm_pairs))
    
    for i in range(len(nm_pairs)):
        #output
        im1=np.loadtxt(outputpath+nm_pairs[i][0])
        #source
        im2=np.loadtxt(outputpath+nm_pairs[i][1])
        #target
        im3=np.loadtxt(outputpath+nm_pairs[i][2])    
        
        spect1=spectrum1d(im1)
        spect2=spectrum1d(im2)
        spect3=spectrum1d(im3)
        
        plt.figure(figsize=(15,10))
        plt.semilogy(spect1,label='simulated low dose')
        plt.semilogy(spect2,label='high dose image')
        plt.semilogy(spect3,label='low dose image')
        plt.xlabel('Spatial Frequency',fontsize=15)
        plt.ylabel('Power Spectrum',fontsize=15)
        plt.title('Power Spectrum of images',fontsize=25)
        plt.legend(fontsize=25)
        plt.savefig(imgoutput+'image_'+str(pair_index[i])+'_spectrum.png')
        plt.close()
        
        
        
#compare specturm of output image and original (high and low dose) images: both models
def spectrumcompare2(outputpath1,outputpath2,imgoutput1,imgoutput2):
    
    nm1=os.listdir(outputpath1)
    nm2=os.listdir(outputpath2)
    
    nm_s=[]
    nm_t=[]
    nm_o1=[]
    nm_o2=[]   
    
    for i in range(len(nm1)):
        if nm1[i].split('_')[0]=='output':
            nm_o1.append(nm1[i])
        elif nm1[i].split('_')[0]=='source':
            nm_s.append(nm1[i])
        elif nm1[i].split('_')[0]=='target':
            nm_t.append(nm1[i])
    
    for i in range(len(nm2)):
        if nm2[i].split('_')[0]=='output':
            nm_o2.append(nm2[i])
    
    
    nm_pairs1=[]
    pair_index1=[]
    for i in range(len(nm_o1)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o1[i].split('_')[1]==nm_t[k].split('_')[1] and nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index1.append(int(nm_o1[i].split('_')[1]))
                    nm_pairs1.append([nm_o1[i],nm_s[j],nm_t[k]])
               
    nm_pairs1=[x for _, x in sorted(zip(pair_index1,nm_pairs1))]
    pair_index1.sort()
    
    nm_pairs2=[]
    pair_index2=[]
    for i in range(len(nm_o2)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o2[i].split('_')[1]==nm_t[k].split('_')[1] and nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index2.append(int(nm_o2[i].split('_')[1]))
                    nm_pairs2.append([nm_o2[i],nm_s[j],nm_t[k]])
               
    nm_pairs2=[x for _, x in sorted(zip(pair_index2,nm_pairs2))]
    pair_index2.sort()
    #print(nm_pairs2)
    #print(pair_index2)
    
    for i in range(len(nm_pairs1)):
        #print(i)
        #output1
        im1=np.loadtxt(outputpath1+nm_pairs1[i][0])
        #output2
        im2=np.loadtxt(outputpath2+nm_pairs2[i][0])
        #source
        ims=np.loadtxt(outputpath1+nm_pairs1[i][1])
        #target
        imt=np.loadtxt(outputpath1+nm_pairs1[i][2])    
        
        spect1=spectrum1d(im1)
        spect2=spectrum1d(im2)
        spects=spectrum1d(ims)
        spectt=spectrum1d(imt)
        
        plt.figure(figsize=(15,10))
        plt.semilogy(spect1,label='ResLD',color='blue')
        plt.semilogy(spect2,label='UnetLD',color='red')
        plt.semilogy(spects,label='NDCT',color='orange')
        plt.semilogy(spectt,label='LDCT',color='green')
        plt.xlabel('Spatial Frequency',fontsize=25)
        plt.ylabel('Power Spectrum',fontsize=25)
        plt.title('Power Spectrum of images',fontsize=35)
        plt.legend(fontsize=25)
        #print(pair_index1[i])
        plt.savefig(imgoutput1+'image_'+str(pair_index1[i])+'_spectrum_comparsion.png')
        plt.close()
        
        
        


#compare specturm of output image and original (high and low dose) images: both models, at High Frequency
def spectrumcompare2h(outputpath1,outputpath2,imgoutput1,imgoutput2):
    
    nm1=os.listdir(outputpath1)
    nm2=os.listdir(outputpath2)
    
    nm_s=[]
    nm_t=[]
    nm_o1=[]
    nm_o2=[]   
    
    for i in range(len(nm1)):
        if nm1[i].split('_')[0]=='output':
            nm_o1.append(nm1[i])
        elif nm1[i].split('_')[0]=='source':
            nm_s.append(nm1[i])
        elif nm1[i].split('_')[0]=='target':
            nm_t.append(nm1[i])
    
    for i in range(len(nm2)):
        if nm2[i].split('_')[0]=='output':
            nm_o2.append(nm2[i])
    
    
    nm_pairs1=[]
    pair_index1=[]
    for i in range(len(nm_o1)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o1[i].split('_')[1]==nm_t[k].split('_')[1] and nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index1.append(int(nm_o1[i].split('_')[1]))
                    nm_pairs1.append([nm_o1[i],nm_s[j],nm_t[k]])
               
    nm_pairs1=[x for _, x in sorted(zip(pair_index1,nm_pairs1))]
    pair_index1.sort()
    
    nm_pairs2=[]
    pair_index2=[]
    for i in range(len(nm_o2)):
        for j in range(len(nm_s)):
            for k in range(len(nm_t)):
                if nm_o2[i].split('_')[1]==nm_t[k].split('_')[1] and nm_t[k].split('_')[1]==nm_s[j].split('_')[1]:
                    pair_index2.append(int(nm_o2[i].split('_')[1]))
                    nm_pairs2.append([nm_o2[i],nm_s[j],nm_t[k]])
               
    nm_pairs2=[x for _, x in sorted(zip(pair_index2,nm_pairs2))]
    pair_index2.sort()
    #print(nm_pairs2)
    #print(pair_index2)
    
    for i in range(len(nm_pairs1)):
        #print(i)
        #output1
        im1=np.loadtxt(outputpath1+nm_pairs1[i][0])
        #output2
        im2=np.loadtxt(outputpath2+nm_pairs2[i][0])
        #source
        ims=np.loadtxt(outputpath1+nm_pairs1[i][1])
        #target
        imt=np.loadtxt(outputpath1+nm_pairs1[i][2])    
        
        spect1=spectrum1d(im1)
        spect2=spectrum1d(im2)
        spects=spectrum1d(ims)
        spectt=spectrum1d(imt)
        
        plt.figure(figsize=(15,10))
        plt.semilogy(spect1,label='ResLD',color='blue')
        plt.semilogy(spect2,label='UnetLD',color='red')
        plt.semilogy(spects,label='NDCT',color='orange')
        plt.semilogy(spectt,label='LDCT',color='green')
        plt.xlabel('Spatial Frequency',fontsize=25)
        plt.xlim(left=250)
        plt.ylim(top=5e8)
        plt.ylabel('Power Spectrum',fontsize=25)
        plt.title('Power Spectrum (high frequency) of images',fontsize=35)
        plt.legend(fontsize=25)
        #print(pair_index1[i])
        plt.savefig(imgoutput1+'image_'+str(pair_index1[i])+'_spectrum_comparsion_hf.png')
        plt.close()
        
        
        
******   
main, using the functions above (take resnet output as example)  
******

#Intra-image analyse
outphdr,outpsmr,outpldr,outpcpr=outputanalyse_m(outputpath_r,imgoutput_r)

index = outphdr.index
index.name = "high dose image (resnet)"
outphdr

index = outpcpr.index
index.name = "comparison (resnet)"
outpcpr

#visualisation
plt.figure(figsize=(12,8))
plt.scatter(outphdr['image index'],outphdr['mean (high)'],label='high dose image',color='blue')
plt.scatter(outpldr['image index'],outpldr['mean (low)'],label='low dose image',color='green')
plt.scatter(outpsmr['image index'],outpsmr['mean (simu)'],label='simulated low dose image (resnet)',color='orange')
plt.scatter(outpsmu['image index'],outpsmu['mean (simu)'],label='simulated low dose image (unet)',color='red')
plt.axhline(y=np.nanmean(outphdr['mean (high)']),color='blue')
plt.axhline(y=np.nanmean(outpldr['mean (low)']),color='green')
plt.axhline(y=np.nanmean(outpsmr['mean (simu)']),color='orange')
plt.axhline(y=np.nanmean(outpsmu['mean (simu)']),color='red')
plt.title('mean CT values of whole image',fontsize=20)
plt.xlabel('image index',fontsize=15)
plt.ylabel('CT values',fontsize=15)
plt.legend(fontsize=20)


#Mitual Information and Entropy
outpmir,micpr=mi3s(outputpath_r)
outpmir

index = micpr.index
index.name = "comparison of entropy and mutual information (resnet)"
micpr


#visualisation
plt.figure(figsize=(12,8))
plt.scatter(outpmir['image index'],outpmir['entropy (high)'],label='high dose image',color='blue')
plt.scatter(outpmir['image index'],outpmir['entropy (low)'],label='low dose image',color='green')
plt.scatter(outpmir['image index'],outpmir['entropy (simu)'],label='simulated low dose image (resnet)',color='orange')
plt.scatter(outpmiu['image index'],outpmiu['entropy (simu)'],label='simulated low dose image (unet)',color='red')
plt.axhline(y=np.nanmean(outpmir['entropy (high)']),color='blue')
plt.axhline(y=np.nanmean(outpmir['entropy (low)']),color='green')
plt.axhline(y=np.nanmean(outpmir['entropy (simu)']),color='orange')
plt.axhline(y=np.nanmean(outpmiu['entropy (simu)']),color='red')
plt.title('entropy of whole image',fontsize=20)
plt.xlabel('image index',fontsize=15)
plt.ylabel('CT values',fontsize=15)
plt.legend(fontsize=20)


#Traditional IQA metrics
maesr,rmsesr,ssimsr,psnrsr,outpcppr=outputanalyse_cp(outputpath_r)
index = maesr.index
index.name = "mean absolute error (resnet)"
maesr

index = outpcppr.index
index.name = "Comparison (resnet)"
outpcppr


#visualisation
plt.figure(figsize=(12,8))
plt.scatter(maesr['image index'],maesr['low-high'],label='low-high',color='blue')
plt.scatter(maesr['image index'],maesr['simu-high'],label='simu-high (resnet)',color='green')
plt.scatter(maesu['image index'],maesu['simu-high'],label='simu-high (unet)',color='gray')
plt.scatter(maesr['image index'],maesr['simu-low'],label='simu-low (resnet)',color='orange')
plt.scatter(maesu['image index'],maesu['simu-low'],label='simu-low (unet)',color='red')
plt.axhline(y=np.nanmean(maesr['low-high']),color='blue')
plt.axhline(y=np.nanmean(maesr['simu-high']),color='green')
plt.axhline(y=np.nanmean(maesu['simu-high']),color='gray')
plt.axhline(y=np.nanmean(maesr['simu-low']),color='orange')
plt.axhline(y=np.nanmean(maesu['simu-low']),color='red')
plt.title('mean absolute error',fontsize=20)
plt.xlabel('image index',fontsize=15)
plt.ylabel('CT values',fontsize=15)
plt.legend(fontsize=20)



#Histogram Correlation
corrr,histcpr=outputanalyse_h(outputpath_r)
index = corrr.index
index.name = "Histogram Correlation (resnet)"
corrr

index = histcpr.index
index.name = "Histogram Comparison (resnet)"
histcpr


#visualisation
plt.figure(figsize=(12,8))
plt.scatter(corr['image index'],corr['simu-high'],label='simu-high')
plt.scatter(corr['image index'],corr['simu-low'],label='simu-low')
plt.scatter(corr['image index'],corr['low-high'],label='low-high')
plt.axhline(y=np.nanmean(corr['simu-high']),color='blue')
plt.axhline(y=np.nanmean(corr['simu-low']),color='orange')
plt.axhline(y=np.nanmean(corr['low-high']),color='green')
plt.title('correlation',fontsize=20)
plt.xlabel('image index',fontsize=15)
plt.ylabel('ratio',fontsize=15)
plt.legend(fontsize=20)



#spectrum comparison
spectrumcompare2(outputpath_r,outputpath_u,imgoutput_r,imgoutput_u)










