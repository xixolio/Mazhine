#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:11:54 2017
The porpuse of this code is to train a lstm using the return_sequences=False
scheme that professor Carlos Valle pointed out on our meeting. This scheme 
consists of using time steps to process individual vectors and at the last time 
step make a prediction,
@author: ignacio
"""


import numpy as np
import math
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, TimeDistributedDense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

def train_lstm_return_false(tr_data,tr_output,lstm_layer,early_stop=False,from_model=False,model2=[]):
    #Se realiza un re-estructuramiento de la data segun (numero_samples,time_steps,features). 
    #Necesariamente el numero de datos originales debe ser divisible por time_steps
    features = tr_data.shape[2]
    time_steps = tr_data.shape[1]
    model = Sequential()
    model.add(LSTM(lstm_layer[0],activation='tanh',inner_activation='sigmoid',stateful=True,
                       batch_input_shape=(1,time_steps,features),return_sequences=False))
    
    if from_model:
        model2.layers.pop()
        model2.outputs = [model2.layers[-1].output]
        model2.layers[-1].outbound_nodes = []
        model2.compile(optimizer='adadelta',loss='mean_squared_error')
        model.compile(loss='mean_squared_error',optimizer='adadelta')      
        model.set_weights(model2.get_weights())
        
    model.add(Dense(tr_output.shape[1],activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adadelta')
    
    weights=model.get_weights()
    if early_stop==True:
        count = 0
        validation=24*5
        tolerance_count = 0
        last_best = 100000
        for i in range(1):
            count+=1
            model.fit(tr_data[:-validation,:,:],tr_output[:-validation,:],
                      nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            error=0
            for j in range(validation):
                predicted=model.predict(tr_data[-validation+j,:,:].reshape(1,time_steps,features))
                if False:
                    error += np.sum(np.abs(tr_output[-validation+j,:,:]-predicted)**2)
                else:
                    
                    predicted_speed=np.sqrt(predicted[0,0]**2+predicted[0,1]**2)
                    real_speed = np.sqrt(tr_output[-validation+j,0]**2 + tr_output[-validation+j,1]**2)
                    error += np.sum(np.abs(predicted_speed - real_speed)**2)
                #error += np.sum(np.abs(tr_output[-validation+j,:,:]-predicted))
            if last_best>error:
                last_best = error
                tolerance_count = 0
            else:
                tolerance_count+=1
            if tolerance_count > 10:
                break
            model.reset_states()
            print error
        
        count -= tolerance_count
        
        model.set_weights(weights)
        model.reset_states()
        
        for i in range(count):
            model.fit(tr_data,tr_output,nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            model.reset_states()
        
    else:
        for i in range(30):
            print i
            model.reset_states()
            hist =model.fit(tr_data,tr_output,nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
 #$           print hist.history['loss'][0]
 #           if hist.history['loss'][0] <= 0.02:
 #               break
            

        
#    while(True):
#        model.fit(tr_data[:-24*v,:,:],tr_output[:-24*v,:],nb_epoch=1,batch_size=1,shuffle=False)
#        mse = 0
#        for j in range(v):
#            base_data = tr_data[-24*(v-j),:,:]
#            for i in range(24):
#                predicted = model.predict(base_data.reshape(1,1,-1),batch_size=1)
#                mse += np.abs(np.sqrt(predicted[0,0].reshape(-1)**2+predicted[0,1].reshape(-1)**2) - np.sqrt(tr_output[-24*v+i,1].reshape(-1)**2 + tr_output[-24*v+i,0].reshape(-1)**2))
#                base_data = np.concatenate((base_data.reshape(1,-1)[:,4:],predicted.reshape(1,-1)),axis=1)
#                base_data = np.reshape(base_data,(1,1,base_data.shape[1]))
#            
#        if(mse>prev_mse2 and prev_mse>prev_mse2):
#            break
#        model.reset_states()
#        print mse
#        prev_mse2 = prev_mse
#        prev_mse = mse
#        
#    model.fit(tr_data[-24:,:,:],tr_output[-24:,:],nb_epoch=1,batch_size=1,shuffle=False)
    

    return model
    
