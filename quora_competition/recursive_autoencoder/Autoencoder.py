#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:14:46 2017

@author: ignacio
"""

from keras.models import Sequential,Model
from keras.layers import Dense,Input

def Autoencoder(input_dim,layers):
    l = len(layers)
    inputs = Input(shape=(input_dim,))
    dummy=inputs
    for i in range(len(layers)-1):
        dummy = Dense(layers[i])(dummy)
    encoded = Dense(layers[-1])(dummy)
    
    dummy = encoded
    for i in range(len(layers)-1):
        dummy = Dense(layers[i-2])(dummy)
    decoded = Dense(input_dim)(dummy)
    
    autoencoder = Model(inputs=inputs, outputs = decoded)
    
    encoder = Model(inputs=inputs, outputs = encoded)
    
    decoder_inputs = Input(shape=(layers[-1],))
    dummy = decoder_inputs
    for i in range(len(layers)):
        dummy = autoencoder.layers[-l+i](dummy)
    decoder = Model(inputs=decoder_inputs, outputs=dummy)
    
    autoencoder.compile(optimizer='adadelta',loss='mse')
    
    return encoder,decoder,autoencoder