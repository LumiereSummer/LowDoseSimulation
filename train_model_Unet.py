#train Unet models (different versions)

from readdata import *
from model_Unet import *
from utils import *

import matplotlib.pyplot as plt
import numpy as np

import tensorflow
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError 
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger





def train_model30(eps, batchs, input_shape, lrs,lrs_nm,losses,losses_nm, x_train, y_train, x_test, y_test):
    
    for i in range(len(losses)):
        for j in range(len(lrs)):
            print('-----start training model30 , adam '+str(lrs[j])+', '+losses_nm[i]+'-----')
            model=generator30(input_shape)
            model.compile(optimizer=Adam(learning_rate=lrs[j]), loss=losses[i], metrics=['accuracy'])
            csv_logger = CSVLogger(csvlogger_path+'model30'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_training.csv')
            filepath = checkpoint_path+'model30'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_{epoch:02d}_{loss:.4f}.hdf5'             
            checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'auto', save_freq = 'epoch', options = None)

            early = EarlyStopping(monitor='loss', min_delta = 0, patience = 250, verbose = 1, mode = 'auto')

            hist = model.fit(x_train, y_train, batch_size = batchs, epochs = eps, verbose = 1, validation_data=(x_test,y_test), callbacks = [checkpoint,early,csv_logger,timecallback()])

            plt.figure()
            plt.plot(hist.history['accuracy'])
            plt.title('accuracy curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'accuracy curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            plt.figure()
            plt.plot(hist.history['loss'])
            plt.title('loss curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'loss curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            x_test20=x_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'test_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20),cmap='gray')

            x=np.expand_dims(x_test20,axis=0)
            output=model.predict(x)
            print(output.shape)
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,)),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,))-dim_unexp(x_test20),cmap='gray')

            y_test20=y_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(y_test20),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20-y_test20),cmap='gray')

            print('-----finish training model30, adam '+str(lrs[j])+', '+losses_nm[i]+'-----')














def train_model31(eps, batchs, input_shape, lrs,lrs_nm,losses,losses_nm, x_train, y_train, x_test, y_test):
    
    for i in range(len(losses)):
        for j in range(len(lrs)):
            print('-----start training model3 , adam '+str(lrs[j])+', '+losses_nm[i]+'-----')
            model=generator31(input_shape)
            model.compile(optimizer=Adam(learning_rate=lrs[j]), loss=losses[i], metrics=['accuracy'])
            csv_logger = CSVLogger(csvlogger_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_training.csv')
            filepath = checkpoint_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_{epoch:02d}_{loss:.4f}.hdf5'             
            checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'auto', save_freq = 'epoch', options = None)

            early = EarlyStopping(monitor='loss', min_delta = 0, patience = 250, verbose = 1, mode = 'auto')

            hist = model.fit(x_train, y_train, batch_size = batchs, epochs = eps, verbose = 1, validation_data=(x_test,y_test), callbacks = [checkpoint,early,csv_logger,timecallback()])

            plt.figure()
            plt.plot(hist.history['accuracy'])
            plt.title('accuracy curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'accuracy curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            plt.figure()
            plt.plot(hist.history['loss'])
            plt.title('loss curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'loss curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            x_test20=x_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'test_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20),cmap='gray')

            x=np.expand_dims(x_test20,axis=0)
            output=model.predict(x)
            print(output.shape)
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,)),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,))-dim_unexp(x_test20),cmap='gray')

            y_test20=y_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(y_test20),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20-y_test20),cmap='gray')

            print('-----finish training model3, adam '+str(lrs[j])+', '+losses_nm[i]+'-----')




def train_model31_cnd(checkpoint0, eps, batchs, input_shape, lrs,lrs_nm,losses,losses_nm, x_train, y_train, x_test, y_test):
    
    for i in range(len(losses)):
        for j in range(len(lrs)):
            print('-----start training model3 , adam '+str(lrs[j])+', '+losses_nm[i]+'-----')
            model=generator31(input_shape)
            model.load_weights(checkpoint0)
            model.compile(optimizer=Adam(learning_rate=lrs[j]), loss=losses[i], metrics=['accuracy'])
            csv_logger = CSVLogger(csvlogger_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_trainingcnd.csv')
            filepath = checkpoint_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_{epoch:02d}_{loss:.4f}cnd.hdf5'             
            checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'auto', save_freq = 'epoch', options = None)

            early = EarlyStopping(monitor='loss', min_delta = 0, patience = 250, verbose = 1, mode = 'auto')

            hist = model.fit(x_train, y_train, batch_size = batchs, epochs = eps, verbose = 1, validation_data=(x_test,y_test), callbacks = [checkpoint,early,csv_logger,timecallback()])

            plt.figure()
            plt.plot(hist.history['accuracy'])
            plt.title('accuracy curve cnd, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'accuracy curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'cnd.png')

            plt.figure()
            plt.plot(hist.history['loss'])
            plt.title('loss curve cnd, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'loss curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'cnd.png')

            x_test20=x_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'test_adam'+lrs_nm[j]+'_'+losses_nm[i]+'cnd.png',dim_unexp(x_test20),cmap='gray')

            x=np.expand_dims(x_test20,axis=0)
            output=model.predict(x)
            print(output.shape)
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_adam'+lrs_nm[j]+'_'+losses_nm[i]+'cnd.png',np.squeeze(output[0],axis=(2,)),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'cnd.png',np.squeeze(output[0],axis=(2,))-dim_unexp(x_test20),cmap='gray')

            y_test20=y_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_adam'+lrs_nm[j]+'_'+losses_nm[i]+'cnd.png',dim_unexp(y_test20),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'cnd.png',dim_unexp(x_test20-y_test20),cmap='gray')

            print('-----finish training model3, adam '+str(lrs[j])+', '+losses_nm[i]+'-----')






def train_model32(eps, batchs, input_shape, lrs,lrs_nm,losses,losses_nm, x_train, y_train, x_test, y_test):
    
    for i in range(len(losses)):
        for j in range(len(lrs)):
            print('-----start training model3 , adam '+str(lrs[j])+', '+losses_nm[i]+'-----')
            model=generator32(input_shape)
            model.compile(optimizer=Adam(learning_rate=lrs[j]), loss=losses[i], metrics=['accuracy'])
            csv_logger = CSVLogger(csvlogger_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_training.csv')
            filepath = checkpoint_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_{epoch:02d}_{loss:.4f}.hdf5'             
            checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'auto', save_freq = 'epoch', options = None)

            early = EarlyStopping(monitor='loss', min_delta = 0, patience = 250, verbose = 1, mode = 'auto')

            hist = model.fit(x_train, y_train, batch_size = batchs, epochs = eps, verbose = 1, validation_data=(x_test,y_test), callbacks = [checkpoint,early,csv_logger,timecallback()])

            plt.figure()
            plt.plot(hist.history['accuracy'])
            plt.title('accuracy curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'accuracy curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            plt.figure()
            plt.plot(hist.history['loss'])
            plt.title('loss curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'loss curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            x_test20=x_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'test_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20),cmap='gray')

            x=np.expand_dims(x_test20,axis=0)
            output=model.predict(x)
            print(output.shape)
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,)),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,))-dim_unexp(x_test20),cmap='gray')

            y_test20=y_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(y_test20),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20-y_test20),cmap='gray')

            print('-----finish training model3, adam '+str(lrs[j])+', '+losses_nm[i]+'-----')






def train_model33(eps, batchs, input_shape, lrs,lrs_nm,losses,losses_nm, x_train, y_train, x_test, y_test):
    
    for i in range(len(losses)):
        for j in range(len(lrs)):
            print('-----start training model3 , adam '+str(lrs[j])+', '+losses_nm[i]+'-----')
            model=generator33(input_shape)
            model.compile(optimizer=Adam(learning_rate=lrs[j]), loss=losses[i], metrics=['accuracy'])
            csv_logger = CSVLogger(csvlogger_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_training.csv')
            filepath = checkpoint_path+'model3'+'_adam'+lrs_nm[j]+'_'+losses_nm[i]+'_{epoch:02d}_{loss:.4f}.hdf5'             
            checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'auto', save_freq = 'epoch', options = None)

            early = EarlyStopping(monitor='loss', min_delta = 0, patience = 250, verbose = 1, mode = 'auto')

            hist = model.fit(x_train, y_train, batch_size = batchs, epochs = eps, verbose = 1, validation_data=(x_test,y_test), callbacks = [checkpoint,early,csv_logger,timecallback()])

            plt.figure()
            plt.plot(hist.history['accuracy'])
            plt.title('accuracy curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'accuracy curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            plt.figure()
            plt.plot(hist.history['loss'])
            plt.title('loss curve, adam '+str(lrs[j])+', '+losses_nm[i])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(performance_path+'loss curve_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png')

            x_test20=x_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'test_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20),cmap='gray')

            x=np.expand_dims(x_test20,axis=0)
            output=model.predict(x)
            print(output.shape)
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,)),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'output_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',np.squeeze(output[0],axis=(2,))-dim_unexp(x_test20),cmap='gray')

            y_test20=y_test[20]
            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(y_test20),cmap='gray')

            plt.figure(figsize = (20,20))
            plt.imsave(performance_path+'goal_diff_adam'+lrs_nm[j]+'_'+losses_nm[i]+'.png',dim_unexp(x_test20-y_test20),cmap='gray')

            print('-----finish training model3, adam '+str(lrs[j])+', '+losses_nm[i]+'-----')


