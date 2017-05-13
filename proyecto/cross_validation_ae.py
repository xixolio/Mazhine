#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:44:05 2016

@author: ignacio
"""
import numpy as np


def cross_validation_ae(data,output,layers):
    k=5
    training_folds = []
    training_outputs =[]
    testing_folds = []
    testing_outputs = []
    size = data.shape[0] - k*24
    space = 24
    random_tries = 1
    current_layers = []
    temporal_layers = []
    temporal_models = []
    current_model = []
    for i in range(k):
        training_folds.append(data[i*space:i*space+size,:])
        training_outputs.append(output[i*space:i*space+size,:])
        testing_folds.append(data[i*space+size,:])
        testing_outputs.append(output[i*space+size:i*space+size+24,:])
     
    prev_mse = 100000
    count = 0
    for i in range(k):
        temporal_current_model = []
        for i in range(random_tries):
            temporal_current_model.append([])
        current_model.append(temporal_current_model)
        
    while(True):
        print count
        count+=1
        mse = np.zeros((len(layers),1))
        for j in range(len(layers)):
            temporal_layers = list(current_layers)
        #    temporal_layers = current_layers
            if len(current_layers) != 0:
                if current_layers[-1] < layers[j]:
                    mse[j,0] = 100000
                    temporal_layers.append(layers[j])
                    continue
            else:
                temporal_layers = [layers[j]]

            for i in range(k): 

                for ran in range(random_tries):
                    autoencoder,encoder,predictor = train_autoencoder(training_folds[i],training_outputs[i],temporal_layers,'relu')
                    
                    dummy,dummy2,dummy3,single_mse,dummy4 = test_performance_ae(predictor,testing_folds[i],testing_outputs[i])
                    mse[j,0] += single_mse

        layer_index= np.argmin(mse)
        if prev_mse <= np.amin(mse):
            return current_layers,count
        current_layers.append(layers[layer_index])
        prev_mse = np.amin(mse)