#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:15:19 2016
This function takes de whole data and transforms it into vectors with a given 
lag size. Moreover, its also transformed into time_steps, where lag is the 
number of time_steps to later on traina  LSTM with it (sorry for bad englisherino).
It also returns the vectors output, which correspond to t timesteps ahead.


@author: ignacio
"""
import numpy as np
import math
from normalize import normalize
 
def data_processing(whole_data,lag,time_steps,folds,training_size):
    datasets_window = int((np.shape(whole_data)[0]-training_size+24)/(folds-1))
    sets=[]
    outs=[]
    for i in range(folds):
        data = whole_data[i*datasets_window:i*datasets_window+training_size+24]
        angles = data[:,1]*2*math.pi/360.
        x_vector = np.cos(angles)*data[:,0]
        y_vector = np.sin(angles)*data[:,0]
        data = np.concatenate((x_vector.reshape(-1,1),y_vector.reshape(-1,1),data[:,2:]),axis=1)
        norm_data,max_values,min_values = normalize(data[0:-24,:])
        n = norm_data.shape[0]
        m = n-lag+1
        vectors = np.zeros((m,4*lag))
        for j in range(m):
            vectors[j,:] = norm_data[j:j+lag,:].reshape(1,-1)
        m2 = m-time_steps+1
        ts_vectors = np.zeros((m2-1,time_steps,4*lag))
        out = np.zeros((m2-1,4))
        for j in range(m2-1):
            ts_vectors[j,:,:] = vectors[j:j+time_steps,:].reshape(1,time_steps,4*lag)
            out[j,:] = norm_data[j+lag+time_steps-1,:]
        sets.append(ts_vectors) 
        outs.append(out)
        
    return max_values,min_values,sets,outs
