#parameters
from InputGeneratePatch import *
import pandas as pd


dspath='/mnt/DONNEES/lxia/LDSimulation/LowDoseSimulation/IMAGES/'
csvpath='/mnt/DONNEES/lxia/LDSimulation/LowDoseSimulation/paras_TestLD.csv'

checkpoint_path = '/mnt/DONNEES/lxia/LDSimulation/DL3/checkpoint/'
csvlogger_path = '/mnt/DONNEES/lxia/LDSimulation/DL3/csvlogger/'
performance_path= '/mnt/DONNEES/lxia/LDSimulation/DL3/performance/'

'''
df=pd.read_csv(csvpath,sep=' ')
df120=df.loc[df['kvp']==120.0].sort_values(by=['slice location'])
df120=df120.loc[df120['slice location'] >= 90.0]
df80=df.loc[df['kvp']==80.0].sort_values(by=['slice location'])

#read image

img120train_nm,img80train_nm,img120test_nm,img80test_nm=readimg_nm(df120,df80,0.8)
imgnm_train=pd.DataFrame(list(zip(img120train_nm,img80train_nm)),
                   columns=['high dose image training', 'low dose image training'])
imgnm_test=pd.DataFrame(list(zip(img120test_nm,img80test_nm)),
                   columns=['high dose image testing', 'low dose image testing'])


imgnm_train.to_csv('dataset_train.csv',index=False)
imgnm_test.to_csv('dataset_test.csv',index=False)
'''

imgnms_train=pd.read_csv('/mnt/DONNEES/lxia/LDSimulation/sinogram/dataset_train.csv')
imgnms_test=pd.read_csv('/mnt/DONNEES/lxia/LDSimulation/sinogram/dataset_test.csv')

img120train_nm=list(imgnms_train.iloc[:,0])
img80train_nm=list(imgnms_train.iloc[:,1])

img120test_nm=list(imgnms_test.iloc[:,0])
img80test_nm=list(imgnms_test.iloc[:,1])


imgs120_train,imgs80_train=readimgfromnm(img120train_nm,img80train_nm,dspath,None)
imgs120_test,imgs80_test=readimgfromnm(img120test_nm,img80test_nm,dspath,None)




