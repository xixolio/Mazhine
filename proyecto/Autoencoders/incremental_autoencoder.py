#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 23:00:57 2017
The incremental autoencoder is a brand new idea for pretraining a lstm
whose objetive is to learn a dynamical system (time series). 
@author: ignacio
"""
import numpy as np
import math
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, TimeDistributedDense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

def incremental_lstm_autoencoder(tr_data,lstm_layer,early_stop=False):
    #Se realiza un re-estructuramiento de la data segun (numero_samples,time_steps,features). 
    #Necesariamente el numero de datos originales debe ser divisible por time_steps
    features = tr_data.shape[2]
    time_steps = tr_data.shape[1]
    model = Sequential()
    model.add(LSTM(lstm_layer[0],activation='tanh',inner_activation='sigmoid',stateful=True,
                       batch_input_shape=(1,time_steps,features),return_sequences=True))
    
    model.add(TimeDistributedDense(features*time_steps,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adadelta')
    weights=model.get_weights()
    
#   data = np.zeros((tr_data.shape[0],time_steps,features))
#    data[:,1:,:] = tr_data
    tr_output = np.zeros((tr_data.shape[0],time_steps,time_steps*features))
    
    for i in range(time_steps):
        tr_output[:,i,0:(i+1)*features] = tr_data[:,0:(i+1),:].reshape(tr_data.shape[0],(i+1)*features)
    
    
    if early_stop==True:
        count = 0
        validation=24*5
        tolerance_count = 0
        last_best = 100000
        for i in range(1):
            count+=1
            model.fit(tr_data[:-validation,:,:],tr_output[:-validation,:],
                      nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            error=0
            for j in range(validation):
                predicted=model.predict(tr_data[-validation+j,:,:].reshape(1,time_steps,features))
                if False:
                    error += np.sum(np.abs(tr_output[-validation+j,:,:]-predicted)**2)
                else:
                    
                    predicted_speed=np.sqrt(predicted[0,0]**2+predicted[0,1]**2)
                    real_speed = np.sqrt(tr_output[-validation+j,0]**2 + tr_output[-validation+j,1]**2)
                    error += np.sum(np.abs(predicted_speed - real_speed)**2)
                #error += np.sum(np.abs(tr_output[-validation+j,:,:]-predicted))
            if last_best>error:
                last_best = error
                tolerance_count = 0
            else:
                tolerance_count+=1
            if tolerance_count > 10:
                break
            model.reset_states()
            print error
        
        count -= tolerance_count
        
        model.set_weights(weights)
        model.reset_states()
        
        for i in range(count):
            model.fit(tr_data,tr_output,nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            model.reset_states()
        
    else:
        for i in range(30):
            model.reset_states()
            hist =model.fit(tr_data,tr_output,nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
#            print hist.history['loss'][0]
            if hist.history['loss'][0] <= 0.02:
                break
            


    return model
    

