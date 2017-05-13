#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 21:46:54 2016

@author: ignacio
"""
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def test_performance(tr_data,tr_output,ts_data,ts_output,layers,epochs,activation):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(tr_data.shape[1],)))
    model.add(Dense(layers[0],activation=activation))
    for i in range(len(layers)-1):
        model.add(Dense(layers[i],activation=activation))
    model.add(Dense(output_dim=4,activation='linear'))
    model.compile(optimizer='sgd',loss='mean_absolute_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(tr_data,tr_output,validation_split=24.*5/tr_data.shape[0],shuffle=False,callbacks=[early_stopping],nb_epoch=epochs,batch_size=1)
    mse = 0;
    mses = np.zeros((24,1))
    xy_mses =np.zeros((24,1))
    base_data = ts_data
    predicted = np.zeros((24,4))
    for i in range(24):
        mses[i,0]= model.evaluate(base_data.reshape(1,-1), ts_output[i,:].reshape(1,-1), batch_size=1)
        mse+=mses[i,0]
        predicted[i,:] = model.predict(base_data.reshape(1,-1),batch_size=1)
        xy_mses[i,:] = np.sum(np.abs(predicted[i,0:2]-ts_output[i,0:2]))/2.
        base_data = np.concatenate((base_data.reshape(1,-1)[:,4:],predicted[i,:].reshape(1,-1)),axis=1)
        
    mse = mse/24.
    return mse,mses,xy_mses,predicted