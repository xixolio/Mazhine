#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:30:44 2017

@author: ignacio
"""

import os
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense
import sys
sys.path.insert(0,'/user/i/iaraya/files/Mazhine/proyecto2/Modelos')
from lstm_autoencoder import LSTM_Autoencoder
from data_processing import data_processing



if __name__ == "__main__":

    lag = int(sys.argv[1])
    time_steps = int(sys.argv[2])
    layer = int(sys.argv[3])
    real_data = np.loadtxt(open('/user/i/iaraya/files/Mazhine/proyecto2/serie1.b08C2.csv',"rb"),delimiter=",",skiprows=1,usecols=range(1,5))
    maxv,minv,sets,outs = data_processing(real_data,lag,time_steps,10,5*30*24)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    error = 0
    for j in range(10):
        tr_set = sets[j][:-24,:,:]
        ts_set = sets[j][-24:,:,:]
        tr_out = outs[j][:-24,:]
        ts_out = outs[j][-24:,:]
        autoencoder,encoder,decoder,model = LSTM_Autoencoder(tr_set,[layer])
        autoencoder.fit(tr_set,tr_set,validation_split=0.2,callbacks=[early_stopping],epochs=1000,verbose=0)
        error += np.mean((ts_set[:,-1,:] - autoencoder.predict(ts_set[:,-1,:]))**2)
        
    f=open("lstm_autoencoder_recons.txt","a")
    f.write(str(lag)+" "+str(time_steps)+" "+str(layer)+" ")
    f.write(str(np.mean(error))+"\n")
    f.close()