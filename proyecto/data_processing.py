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
 
def data_processing(whole_data,lag,time_steps,folds,training_size):
    datasets_window = int((np.shape(whole_data)[0]-training_size+24)/(folds-1))
    sets=[]
    for i in range(folds):
        data = whole_data[i*datasets_window:i*datasets_window+training_size+24]
        angles = data[:,1]*2*math.pi/360.
        x_vector = np.cos(angles)*data[:,0]
        y_vector = np.sin(angles)*data[:,0]
        data = np.concatenate((x_vector.reshape(-1,1),y_vector.reshape(-1,1),data[:,2:]),axis=1)
        norm_data,max_values,min_values = normalize(data[0:-24,:])
        n = norm_data.shape[0]
        m = n-lag+1
        vectors = np.zeros((n))
        
    return max_values,min_values,time_step_tr_sets,time_step_ts_sets
    
    
#x_vector = cos(angles').*norm_data(1,:);
#y_vector = sin(angles').*norm_data(1,:);
#data = [x_vector; y_vector];
#final_data = zeros(2*lag,size(data,2)-lag+1);
#
#for i=1:size(data,2)-lag+1
#    for j=1:lag
#        final_data(2*j-1:2*j,i) = norm_data(:,i+j-1);
#    end
#end
#
#end