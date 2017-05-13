#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:30:05 2016

@author: ignacio
"""
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def create_model_from_ae(prev_model,data,output,test_data,test_output,activation_func,epochs,batch):

    model.add(Dense(output.shape[1],activation='linear'))
    model.compile(optimizer='sgd',loss='mean_absolute_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(data,output,validation_split=5.*24/data.shape[0],shuffle=False,callbacks=[early_stopping],nb_epoch=1000,batch_size=1)
    
    acumulated_loss = 0;
    base_data = test_data;
    print test_output.shape
    xy_loss =0
    for i in range(24):
        acumulated_loss += model.evaluate(base_data.reshape(1,-1), test_output[i,:].reshape(1,-1), batch_size=1)
        predicted = model.predict(base_data.reshape(1,-1),batch_size=1)
        xy_loss= np.mean(np.abs(predicted[0:2]-test_output[i,0:2].reshape(-1,1)))
        base_data = np.concatenate((base_data.reshape(1,-1)[:,4:],predicted.reshape(1,-1)),axis=1)
        
    return acumulated_loss,model
    