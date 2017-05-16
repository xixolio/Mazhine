#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:17:02 2017
lstm_autoencoders is going to provide 2 ways to train an autoencoder. The main
objetive for this is to test which one is capable of better reconstructing
the inicial input so that we later on use that one to pretrain a lstm on the
task of wind prediction

The first type of autoencoder is going to have just one final output, the input
of the last vector fed to the lstm, meaning that only the last hidden activation
will be a lower dimensional representation of the input with all the meaningful
features needed to reconstruct it.

The second type is going to have multiple outputs, one for each time-step. Each
output then equals the input at that time-step, meaning that each hidden state
will be forced to be a hidden representation of its input.
@author: ignacio
"""
from keras.models import Model
from keras.layers import Dense,LSTM,Input



def LSTM_Autoencoder(tr_data,layers):
    features = tr_data.shape[2]
    time_steps = tr_data.shape[1]
    l = len(layers)
    
    inputs = Input(shape=(time_steps,features))
    dummy=inputs
    for i in range(len(layers)-1):
        dummy = LSTM(layers[i],
                     return_sequences=True)(dummy)
    encoded = LSTM(layers[-1],
                   return_sequences=True)(dummy)
    
    dummy = encoded
    for i in range(len(layers)-1):
        dummy = LSTM(layers[i-2],
                     return_sequences=True)(dummy)
    decoded = Dense(features,activation='linear')(dummy)
    
    autoencoder = Model(inputs=inputs, outputs = decoded)
    
    encoder = Model(inputs=inputs, outputs = encoded)
    
    decoder_inputs = Input(shape=(time_steps,layers[-1]))
    dummy = decoder_inputs
    for i in range(len(layers)):
        dummy = autoencoder.layers[-l+i](dummy)
    decoder = Model(inputs=decoder_inputs, outputs=dummy)
    
    autoencoder.compile(optimizer='adadelta',loss='mse')
    
    #we get the model for the final prediction
    inputs = Input(batch_shape=(1,time_steps,features))
    dummy=inputs
    for i in range(len(layers)-1):
        dummy = LSTM(layers[i],
                     return_sequences=True,stateful=True)(dummy)
    encoded = LSTM(layers[-1],
                   return_sequences=False,stateful=True)(dummy)
    model = Model(inputs=inputs, outputs = encoded)
    model.compile(optimizer='adadelta',loss='mse')
 
    return autoencoder,encoder,decoder,model
    
    

    

        