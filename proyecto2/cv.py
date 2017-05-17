#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:54:11 2017

@author: ignacio
"""
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
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
    val_errors=np.zeros((10,1))
    for j in range(10):
        tr_set = sets[j][:-24,:,:]
        ts_set = sets[j][-24:,:,:]
        tr_out = outs[j][:-24,:]
        ts_out = outs[j][-24:,:]
        autoencoder,encoder,decoder,model = LSTM_Autoencoder(tr_set,[layer])
        autoencoder.fit(tr_set,tr_set,validation_split=0.2,callbacks=[early_stopping],epochs=200,verbose=0)

        model.set_weights(encoder.get_weights())
        x = model.layers[-1].output
        x = Dense(4)(x)
        model2 = Model(inputs=model.layers[0].output,outputs=x)
        model2.compile(loss='mse',optimizer='adadelta')
        counter = 0
        min_error=100000
        tolerance=0
        for i in range(200):
            model2.fit(tr_set[:-24*5,:,:],tr_out[:-24*5,:],batch_size=1,shuffle=False,verbose=0)
            error=model2.evaluate(tr_set[-24*5:,:,:],tr_out[-24*5:,:],batch_size=1,verbose=0)
            model2.reset_states()
            if error>min_error:
                tolerance+=1
            else:
                min_error=error
                tolerance=0
            if tolerance==5:
                break
            counter+=1
        
        for i in range(counter-tolerance):
            model2.reset_states()
            model2.fit(tr_set,tr_out,batch_size=1,shuffle=False,verbose=0)
            
        #we get validation error
        vector = ts_set[0,:,:]
        vector = vector.reshape(1,time_steps,4*lag)
        error_vector = np.zeros((24,1))
        for i in range(24):
            p=model2.predict(vector)
            error_vector[i,0] = np.sqrt((p[0,0]-ts_out[i,0])**2+(p[0,1]-ts_out[i,1])**2)
            vector[0,0:-1,:] = vector[0,1:,:]
            vector[0,-1,:-4]=vector[0,-1,4:]
            vector[0,-1,-4:] = p.reshape(1,1,4)
        val_errors[j,0]=np.mean(error_vector)
    f=open("validation_lstm_autoencoder.txt","a")
    f.write(str(lag)+" "+str(time_steps)+" "+str(layer)+" ")
    f.write(str(np.mean(val_errors))+"\n")
    f.close()
                
        
        