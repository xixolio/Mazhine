#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 03:03:00 2017

@author: ignacio
"""
import numpy as np
import math
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, TimeDistributedDense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

def train_lstm2(tr_data,tr_output,lstm_layer,time_steps,from_model=False,model=False):
    #Se realiza un re-estructuramiento de la data segun (numero_samples,time_steps,features). 
    #Necesariamente el numero de datos originales debe ser divisible por time_steps
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
    else:
        model = Sequential()
        model.add(LSTM(lstm_layer[0], activation='tanh',inner_activation='sigmoid',stateful=True,
                       batch_input_shape=(1,time_steps,features),return_sequences=True))
        for i in range(1,len(lstm_layer)):
            model.add(LSTM(lstm_layer[i], activation='tanh',inner_activation='sigmoid',stateful=True,return_sequences=True))
        #model.add(LSTM(lstm_layer[-1], activation='tanh',inner_activation='sigmoid',stateful=True))
    model.add(TimeDistributedDense(output_dim=output_dim,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adadelta')
    weights=model.get_weights()
    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    
    count = 0
    validation=np.ceil(24./time_steps).astype(int)
    tolerance_count = 0
    last_best = 100000
    n = tr_data.shape[0]
    sets = 10
    train_portion = int(n*4./(5.*sets))
    validation_portion = int(n*1./(5.*sets))
    offset= train_portion + validation_portion
    for i in range(100):
        count+=1
        error=0
        for j in range(sets):
            predicted=model.predict(tr_data[j*offset:j*offset+validation_portion,:,:].reshape(validation_portion,time_steps,features),batch_size=1)
            predicted_speed=np.sqrt(predicted[:,:,0]**2+predicted[:,:,1]**2)
            real_speed = np.sqrt(tr_output[j*offset:j*offset+validation_portion,:,0]**2 + tr_output[j*offset:j*offset+validation_portion,:,1]**2)
            error += np.sum(np.abs(predicted_speed - real_speed))
            if j==(sets-1):
                model.fit(tr_data[j*offset+validation_portion:,:,:],tr_output[j*offset+validation_portion:,:,:],nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            else:
                model.fit(tr_data[j*offset+validation_portion:(j+1)*offset,:,:],tr_output[j*offset+validation_portion:(j+1)*offset,:,:],nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
        if last_best>error:
            last_best = error
            tolerance_count = 0
        else:
            tolerance_count+=1
        if tolerance_count > 2:
            break
        model.reset_states()
        #print error
        
    
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