#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:31:20 2016

@author: ignacio
"""
import numpy as np
def test_performance_ae(model,ts_data,ts_output):

    mse = np.zeros((24,1))
    base_data = ts_data
    predicted = np.zeros((24,4))
    for i in range(24):
        #%mses[i,0]= model.evaluate(base_data.reshape(1,-1), ts_output[i,:].reshape(1,-1), batch_size=1,verbose=False)
        #%mse+=mses[i,0]
        predicted[i,:] = model.predict(base_data.reshape(1,-1),batch_size=1)
        mse[i,0] = np.abs(np.sqrt(predicted[i,0]**2+predicted[i,1]**2) - np.sqrt(ts_output[i,1]**2+ts_output[i,0]**2))
        base_data = np.concatenate((base_data.reshape(1,-1)[:,4:],predicted[i,:].reshape(1,-1)),axis=1)
        
    return mse,predicted
    
