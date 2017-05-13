#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 00:18:22 2016

@author: ignacio
"""
from keras.layers import Input, Dense, Dropout
from keras.models import Model,Sequential
from keras.callbacks import EarlyStopping
import numpy as np
def train_autoencoder(data,out,encoding_dims,activation,out_act):
#    input_data = Input(shape=(data.shape[1],))
    drop_rate = 0.1    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model = Sequential()
    #model.add(Dropout(drop_rate,input_shape=(data.shape[1],)))
    model.add(Dense(encoding_dims[0],activation=activation,input_shape=(data.shape[1],)))
    for i in range(1,len(encoding_dims)):
        model.add(Dense(encoding_dims[i],activation=activation))
        
    for i in range(len(encoding_dims)-1):
        model.add(Dense(encoding_dims[-i-1],activation=activation))
    
    model.add(Dense(data.shape[1],activation=out_act))
    model.compile(optimizer='adadelta',loss='mean_squared_error')
    division = data.shape[0]/20
    for i in range(20):
        if(i<19):
            model.fit(data[i*division:(i+1)*division,:],data[i*division:(i+1)*division,:],nb_epoch=100,shuffle=True,
                    validation_split=24*1./data[i*division:(i+1)*division,:].shape[0],batch_size=1,
                    callbacks=[early_stopping],verbose=True)
        else:
            model.fit(data[i*division:,:],data[i*division:,:],nb_epoch=100,shuffle=True,
                    validation_split=24*1./data[i*division:,:].shape[0],batch_size=10,
                    callbacks=[early_stopping],verbose=True)
    encoder = Sequential.from_config(model.get_config())
    encoder.set_weights(model.get_weights())
    for i in range(len(encoding_dims)):
        encoder.layers.pop()
    encoder.outputs = [encoder.layers[-1].output]
    encoder.layers[-1].outbound_nodes = []
    encoder.compile(optimizer='adadelta',loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    predictor = Sequential.from_config(encoder.get_config())
    predictor.set_weights(encoder.get_weights())
    predictor.add(Dense(4,activation='linear'))
    predictor.compile(optimizer='adadelta',loss='mean_squared_error')
    predictor.fit(data,out,nb_epoch=1000,shuffle=True,
                    validation_split=24*10./data.shape[0],batch_size=1,
                    callbacks=[early_stopping],verbose=True)
  #  predictor.outputs = [encoder.layers[-1].output]
   # encoder.layers[-1].outbound_nodes = []
    prediccions = np.zeros((data.shape[0],4))
    for i in range(data.shape[0]):
       prediccions[i,:] = predictor.predict(data[i,:].reshape(1,-1)).reshape(1,-1)
    return model,encoder,predictor,prediccions