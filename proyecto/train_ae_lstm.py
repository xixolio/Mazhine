#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 03:54:34 2016

@author: ignacio
"""
from keras.layers import TimeDistributedDense

def train_ae_lstm(ae,tr_data,tr_output,ae_layers,lstm_layer):
    d = tr_data.shape[1]
    n = tr_data.shape[0]
    model = Sequential()
 #   model.add(Dropout(0.1,input_shape=(tr_data.shape[1],)))
    model.add(TimeDistributedDense(ae_layers[0],activation='relu',batch_input_shape=(1,1,tr_data.shape[1])))
    for i in range(len(ae_layers)-1):
        model.add(TimeDistributedDense(ae_layers[i+1],activation='relu'))
    model.add(TimeDistributedDense(4,activation='sigmoid'))
    model.set_weights(ae.get_weights())
    model.layers.pop()
#    model.outputs = [ae.layers[-1].output]
#    model.layers[-1].outbound_nodes = []
    model.add(LSTM(lstm_layer[i], activation='tanh',inner_activation='sigmoid',stateful=True,return_sequences=True))
    for i in range(1,len(lstm_layer)):
        model.add(LSTM(lstm_layer[i], activation='tanh',inner_activation='sigmoid',stateful=True,))
    model.add(Dense(output_dim=4,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adadelta')
    prev_mse = 100000
    prev_mse2 = 100000
    tr_data = np.reshape(tr_data,(tr_data.shape[0],1,tr_data.shape[1]))

    while(True):
        model.fit(tr_data[:-24,:,:],tr_output[:-24,:],nb_epoch=1,batch_size=1,shuffle=False)
        mse = 0
        base_data = tr_data[-24,:,:]
        for i in range(24):
            predicted = model.predict(base_data.reshape(1,1,-1),batch_size=1)
            mse += np.abs(np.sqrt(predicted[0,0].reshape(-1)**2+predicted[0,1].reshape(-1)**2) - np.sqrt(tr_output[-24+i,1].reshape(-1)**2 + tr_output[-24+i,0].reshape(-1)**2))
            base_data = np.concatenate((base_data.reshape(1,-1)[:,4:],predicted.reshape(1,-1)),axis=1)
            base_data = np.reshape(base_data,(1,1,base_data.shape[1]))
            
        if(mse>prev_mse2 and prev_mse>prev_mse2):
            break
        model.reset_states()
        print mse
        prev_mse2 = prev_mse
        prev_mse = mse
        
    model.fit(tr_data[-24:,:,:],tr_output[-24:,:],nb_epoch=1,batch_size=1,shuffle=False)
    return model