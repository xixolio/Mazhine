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
 
def data_processing(whole_data,lag,folds,training_size,output_lag):
    datasets_window = int((np.shape(whole_data)[0]-training_size+24)/(folds-1))
    training_datasets = []
    testing_datasets = []
    testing_output_datasets = []
    training_output_datasets = []
    time_step_tr_sets = []
    time_step_tr_out_sets = []
    time_step_ts_sets = []
    time_step_tr_out_sets = []
    for i in range(folds):
        data = whole_data[i*datasets_window:i*datasets_window+training_size+24]
        angles = data[:,1]*2*math.pi/360.
        x_vector = np.cos(angles)*data[:,0]
        y_vector = np.sin(angles)*data[:,0]
        data = np.concatenate((x_vector.reshape(-1,1),y_vector.reshape(-1,1),data[:,2:]),axis=1)
        norm_data,max_values,min_values = normalize(data[0:-24,:])
        size_data = norm_data.shape[0]
        final_data = np.zeros((size_data-(lag+output_lag-1),4*lag))
        time_step_data = np.zeros((size_data-(lag+output_lag-1),lag,4))
        output_data = np.zeros((size_data-(lag+output_lag-1),output_lag*4))
        for i in range(final_data.shape[0]):
            for j in range(lag):
                final_data[i,4*j:4*(j+1)] = norm_data[i+j,:]
                time_step_data[i,j,:] = norm_data[i+j,:]
            
            for j in range(output_lag):
                output_data[i,4*j:4*(j+1)] = norm_data[i+j+lag,:]
                           
#        output_data = norm_data[lag:,:]
        training_datasets.append(final_data)
        training_output_datasets.append(output_data)
        testing_datasets.append(norm_data[-lag:].reshape(1,4*lag))
        testing_output_datasets.append((data[-24:,:]-min_values)/(max_values-min_values))
        
        time_step_tr_sets.append(time_step_data)
        time_step_ts_sets.append(norm_data[-lag:].reshape(1,lag,4))
        
    return training_datasets,training_output_datasets,testing_datasets,testing_output_datasets,max_values,min_values,time_step_tr_sets,time_step_ts_sets
    
    
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