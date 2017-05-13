#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:10:00 2016

@author: ignacio
"""
import numpy as np

def cross_validation(data,output,layers):
    k=3
    training_folds = []
    training_outputs =[]
    testing_folds = []
    testing_outputs = []
    size = data.shape[0] - k*24
    space = 24
    random_tries = 3
    current_layers = []
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
        count+=1
        mse = np.zeros((len(layers),1))
        temporal_models = []
        for j in range(len(layers)):
            #temp_layers = list(current_layers)
            if len(current_layers) != 0:
                if current_layers[-1] < layers[j]:
                    mse[j,0] = 100000
                    temporal_models.append(current_model[0])
                    continue
            temporal_models_fold =[]
            for i in range(k): 
                temporal_models_rand =[]
                for ran in range(random_tries):
                    temp_mse,temporal_model = create_model([current_model[i][ran]],layers[j],training_folds[i],training_outputs[i],testing_folds[i],testing_outputs[i],'relu',1000,500)
                    temporal_models_rand.append(temporal_model)
                    mse[j,0] += temp_mse
                temporal_models_fold.append(temporal_models_rand)
            temporal_models.append(temporal_models_fold)
            
        layer_index= np.argmin(mse)
        if prev_mse <= np.amin(mse):
            return current_layers,count
        current_layers.append(layers[layer_index])
        current_model = temporal_models[layer_index]
        prev_mse = np.amin(mse)