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

def lstm_autoencoders(tr_data,lstm_layer,early_stop=False):
    features = tr_data.shape[2]
    time_steps = tr_data.shape[1]
    model = Sequential()
    model.add(LSTM(lstm_layer[0],activation='tanh',inner_activation='sigmoid',stateful=True,
                       batch_input_shape=(1,time_steps,features),return_sequences=False))
    
    model.add(TimeDistributedDense(features*time_steps,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adadelta')
    weights=model.get_weights()
    decoder_weights.append(model.layers[-1].get_weights())
    
    for i in range(1,len(encoding_dims)):
        decoder_weights.append(model.layers[-1].get_weights())
        remove_layer(model)
        activations = model.predict(data)
        model.layers[i-1].trainable = False
        model.add(Dense(encoding_dims[i],activation=activation))
        model.add(Dense(encoding_dims[i-1],activation='sigmoid'))
        model.compile(optimizer='adadelta', loss='mean_squared_error')
        model.fit(data,activations,nb_epoch=100,shuffle=True,verbose=False,
              validation_split=24*5./n_samples,batch_size=32,callbacks=[early_stopping])
    
    decoder_weights.append(model.layers[-1].get_weights())
    remove_layer(model)
    
def remove_layer(model):
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.compile(optimizer='adadelta',loss='mean_squared_error')
        