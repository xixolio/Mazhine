#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 04:35:40 2016

@author: ignacio
"""

import numpy as np
def test_performance_ae2(model,model2,ts_data,ts_output,max_norm,min_norm,diff_ts):

    mse = np.zeros((24,1))
    base_data = ts_data
    base_data2= diff_ts
    predicted = np.zeros((24,4))
    predicted2 = np.zeros((24,4))
    predicted1 = np.zeros((24,4))
    for i in range(24):
        #%mses[i,0]= model.evaluate(base_data.reshape(1,-1), ts_output[i,:].reshape(1,-1), batch_size=1,verbose=False)
        #%mse+=mses[i,0]
        print i
        predicted1[i,:] = model.predict(base_data.reshape(1,-1),batch_size=1)
        predicted2[i,:] = model2.predict(base_data2.reshape(1,-1),batch_size=1)
        base_data2 = np.concatenate((base_data2.reshape(1,-1)[:,4:],predicted2[i,:].reshape(1,-1)),axis=1)
        predicted2[i,:] = predicted2[i,:].reshape(1,-1)*(max_norm-min_norm)+min_norm
        predicted[i,:] = predicted1[i,:]+predicted2[i,:]
        mse[i,0] = np.abs(np.sqrt(predicted[i,0]**2+predicted[i,1]**2) - np.sqrt(ts_output[i,1]**2+ts_output[i,0]**2))
        base_data = np.concatenate((base_data.reshape(1,-1)[:,4:],predicted[i,:].reshape(1,-1)),axis=1)
        
    return mse,predicted