#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:33:27 2016

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

def train_lstm(tr_data,tr_output,lstm_layer,time_steps,from_model=False,model=False,ae=False,formato=False,early_stop=False):
    #Se realiza un re-estructuramiento de la data segun (numero_samples,time_steps,features). 
    #Necesariamente el numero de datos originales debe ser divisible por time_steps
    if formato:
        features = tr_data.shape[2]
        x = tr_data.shape[0]
        y = tr_data.shape[1]
        output_dim = tr_output.shape[1]
        tr_output = np.reshape(tr_output[-x*y:,:],(x,y,output_dim))
    else:
        n_datos = tr_data.shape[0]
        features = tr_data.shape[1]
        n_samples = n_datos/time_steps
        output_dim = tr_output.shape[1]
        
        tr_data = np.reshape(tr_data[n_datos%time_steps:,:],(n_samples,time_steps,features))
        tr_output = np.reshape(tr_output[n_datos%time_steps:,:],(n_samples,time_steps,output_dim))
    
    #Se crea el modelo segun las especificaciones entregadas en lstm_layer (numero de caps,neuronas por capa)
    if from_model:
        model.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.reset_states()
    else:
        model = Sequential()
        model.add(LSTM(lstm_layer[0], activation='tanh',inner_activation='sigmoid',stateful=True,
                       batch_input_shape=(1,time_steps,features),return_sequences=True,dropout_W=0.3,dropout_U=0.3,W_regularizer=l2(l=0.01)))
        for i in range(1,len(lstm_layer)):
            model.add(LSTM(lstm_layer[i], activation='tanh',inner_activation='sigmoid',stateful=True,return_sequences=True,dropout_W=0.03,dropout_U=0.03,W_regularizer=l2(l=0.01)))
        #model.add(LSTM(lstm_layer[-1], activation='tanh',inner_activation='sigmoid',stateful=True))
    model.add(TimeDistributedDense(output_dim=output_dim,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adadelta')
    weights=model.get_weights()
    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    
    if early_stop==True:
        count = 0
        validation=np.ceil(24.*5/time_steps).astype(int)
        tolerance_count = 0
        last_best = 100000
        for i in range(100):
            count+=1
            model.fit(tr_data[:-validation,:,:],tr_output[:-validation,:,:],nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            error=0
            for j in range(validation):
                predicted=model.predict(tr_data[-validation+j,:,:].reshape(1,time_steps,features))
                if ae:
                    error += np.sum(np.abs(tr_output[-validation+j,:,:]-predicted)**2)
                else:
                    
                    predicted_speed=np.sqrt(predicted[0,:,0]**2+predicted[0,:,1]**2)
                    real_speed = np.sqrt(tr_output[-validation+j,:,0]**2 + tr_output[-validation+j,:,1]**2)
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
        
        for i in range(count):
            model.fit(tr_data,tr_output,nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            model.reset_states()
        
    else:
        for i in range(100):
            model.reset_states()
            hist =model.fit(tr_data,tr_output,nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            print hist.history['loss'][0]
            if hist.history['loss'][0] <= 0.02:
                break
            

        
#    while(True):
#        model.fit(tr_data[:-24*v,:,:],tr_output[:-24*v,:],nb_epoch=1,batch_size=1,shuffle=False)
#        mse = 0
#        for j in range(v):
#            base_data = tr_data[-24*(v-j),:,:]
#            for i in range(24):
#                predicted = model.predict(base_data.reshape(1,1,-1),batch_size=1)
#                mse += np.abs(np.sqrt(predicted[0,0].reshape(-1)**2+predicted[0,1].reshape(-1)**2) - np.sqrt(tr_output[-24*v+i,1].reshape(-1)**2 + tr_output[-24*v+i,0].reshape(-1)**2))
#                base_data = np.concatenate((base_data.reshape(1,-1)[:,4:],predicted.reshape(1,-1)),axis=1)
#                base_data = np.reshape(base_data,(1,1,base_data.shape[1]))
#            
#        if(mse>prev_mse2 and prev_mse>prev_mse2):
#            break
#        model.reset_states()
#        print mse
#        prev_mse2 = prev_mse
#        prev_mse = mse
#        
#    model.fit(tr_data[-24:,:,:],tr_output[-24:,:],nb_epoch=1,batch_size=1,shuffle=False)
    

    return model
    
