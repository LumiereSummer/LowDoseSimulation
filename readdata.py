#read dicom data, form training and testing datasets

from utils import *
import pandas as pd


dspath='/path/to/Patient0/dicomIMAGES/'
csvpath='/path/to/MetaDataPatient0.csv'

checkpoint_path = '/path/to/store/the/checkpoint/'
csvlogger_path = '/path/to/store/the/csvlogger/'
performance_path= '/path/to/store/the/performanceImages/'

'''
#training and testing datasets split
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


imgnms_train=pd.read_csv('/path/to/dataset_train.csv')
imgnms_test=pd.read_csv('/path/to/dataset_test.csv')

img120train_nm=list(imgnms_train.iloc[:,0])
img80train_nm=list(imgnms_train.iloc[:,1])

img120test_nm=list(imgnms_test.iloc[:,0])
img80test_nm=list(imgnms_test.iloc[:,1])


#read images according to csv files
imgs120_train,imgs80_train=readimgfromnm(img120train_nm,img80train_nm,dspath,None)
imgs120_test,imgs80_test=readimgfromnm(img120test_nm,img80test_nm,dspath,None)




