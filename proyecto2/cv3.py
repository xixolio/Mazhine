#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:30:44 2017

@author: ignacio
"""

import os
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Model,Sequential
from keras.layers import Dense
import sys
sys.path.insert(0,'/user/i/iaraya/Mazhine/proyecto2/Modelos')
from lstm_autoencoder2 import LSTM_Autoencoder2
from data_processing import data_processing



if __name__ == "__main__":

    lag = int(sys.argv[1])
    time_steps = int(sys.argv[2])
    layer = int(sys.argv[3])
    player = int(sys.argv[4])
    real_data = np.loadtxt(open('/user/i/iaraya/Mazhine/proyecto2/serie1.b08C2.csv',"rb"),delimiter=",",skiprows=1,usecols=range(1,5))
    maxv,minv,sets,outs = data_processing(real_data,lag,time_steps,10,5*30*24)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    error = 0
    for j in range(10):
        tr_set = sets[j][:-24,:,:]
        ts_set = sets[j][-24:,:,:]
        tr_out = outs[j][:-24,:]
        ts_out = outs[j][-24:,:]
        autoencoder,encoder,decoder,model = LSTM_Autoencoder2(tr_set,[layer])
        autoencoder.fit(tr_set,tr_set[:,-1,:],validation_split=0.2,callbacks=[early_stopping],epochs=1000,verbose=0)
        activation=encoder.predict(tr_set)
        model = Sequential()
        model.add(Dense(player,layer))
        

    f=open("lstm_ae_recons_dr01.txt","a")
    f.write(str(lag)+" "+str(time_steps)+" "+str(layer)+" ")
    f.write(str(np.mean(error))+"\n")
    f.close()