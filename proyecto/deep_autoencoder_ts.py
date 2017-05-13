#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 00:18:22 2016

@author: ignacio

This function trains a deep autoencoder doing a layer-wise pre-training as each 
new layer is added to the model. Denoising autoencoder is used to pre-train 
each layer. Moreover, we use TimeDistributedDense layers since the final
objective is to remove the decoding layers and add to the model a LSTM layer
for time-series learning. Alternatively, the model can be fine-tuned after 
removing the decoding layers by providing output data

Inputs:
data: the training data
encoding_dims: a list of size N with the size of each layer, being the 0 element
the size for the first autoencoder and the N-1 the one for the last autoencoder.
activation: activation function for the TimeDistributedDense layers.
out: (alternative) output data used to fine-tune the model
out_activation: (alternative) output activation function when fine-tunning the 
model
fine_tune: (alternative) set to True if we want to fine-tune the model using 
the actual data outputs

Outputs:
model: final model without any decoding layer. This is the encoder
decoder: the decoder layer

"""
from keras.layers import Input, Dense, Dropout, TimeDistributed
from keras.models import Model,Sequential
from keras.callbacks import EarlyStopping
import numpy as np
def train_deep_autoencoder(data,encoding_dims,activation,
                           out=[],out_activation='sigmoid',fine_tune=False):
#    input_data = Input(shape=(data.shape[1],))
#    drop_rate = 0.1   
    """ training data is shaped as (n_batches,time_steps,features). Zeros are
    added at the beggining if time_steps doesnt perfectly divide the number
    of samples """
    n_samples = data.shape[0]
    features = data.shape[1]
#    missing =n_samples%time_steps
#    n_batches = (missing+n_samples)/time_steps
#    datashape = np.zeros((n_samples+missing,features))
#    datashape[missing:,:] = data
#    input_data = np.reshape(datashape,(n_batches,time_steps,features))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    decoder_weights = []
    "model is created and trained layer by layer"
    model = Sequential()
    #model.add(Dropout(drop_rate,input_shape=(data.shape[1],)))
#    model.add(TimeDistributed(Dense(encoding_dims[0],activation=activation,
#                                    batch_input_shape=(1,time_steps,features))))
    model.add(Dense(encoding_dims[0],activation=activation,input_shape=(features,)))
    model.add(Dense(features,activation='sigmoid'))
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    model.fit(data,data,nb_epoch=100,shuffle=True,verbose=False,
              validation_split=24*5./n_samples,batch_size=32,callbacks=[early_stopping])
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
    
    """ layers are set to be trainable again. This may not be necessary if
    we dont want to fine-tune the deep model or if we want the layers freezed
    after the lstm is inserted"""
    for i in range(len(encoding_dims)-1):
        model.layers[i].trainable = True
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    
    if fine_tune:
        model.add(Dense(out.shape[1],activation=out_activation))
        model.compile(optimizer='adadelta',loss='mean_squared_error')
        model.fit(data,out,nb_epoch=100,shuffle=True, verbose=False,
              validation_split=24*5./n_samples,batch_size=32,callbacks=[early_stopping])
        remove_layer(model)
    
    """ decoder construction"""
    
    decoder = Sequential()
    
    for i in range(1,len(encoding_dims)):
        decoder.add(Dense(encoding_dims[-(i+1)],input_dim=encoding_dims[-i],activation='sigmoid'))
        decoder.layers[i-1].set_weights(decoder_weights[-i])
        
    decoder.add(Dense(features,activation='sigmoid'))
    decoder.layers[-1].set_weights(decoder_weights[0])
    model.compile(optimizer='adadelta',loss='mean_squared_error')
        
    return model,decoder
    
        
#    for i in range(len(encoding_dims)-1):
#        model.add(Dense(encoding_dims[-i-1],activation=activation))
    
#    model.add(Dense(data.shape[1],activation=out_act))
#    model.compile(optimizer='adadelta',loss='mean_squared_error')
#    division = data.shape[0]/20
#    for i in range(20):
#        if(i<19):
#            model.fit(data[i*division:(i+1)*division,:],data[i*division:(i+1)*division,:],nb_epoch=100,shuffle=True,
#                    validation_split=24*1./data[i*division:(i+1)*division,:].shape[0],batch_size=1,
#                    callbacks=[early_stopping],verbose=True)
#        else:
#            model.fit(data[i*division:,:],data[i*division:,:],nb_epoch=100,shuffle=True,
#                    validation_split=24*1./data[i*division:,:].shape[0],batch_size=10,
#                    callbacks=[early_stopping],verbose=True)
#    encoder = Sequential.from_config(model.get_config())
#    encoder.set_weights(model.get_weights())
#    for i in range(len(encoding_dims)):
#        encoder.layers.pop()
#    encoder.outputs = [encoder.layers[-1].output]
#    encoder.layers[-1].outbound_nodes = []
#    encoder.compile(optimizer='adadelta',loss='mean_squared_error')
#    
#    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#
#    predictor = Sequential.from_config(encoder.get_config())
#    predictor.set_weights(encoder.get_weights())
#    predictor.add(Dense(4,activation='linear'))
#    predictor.compile(optimizer='adadelta',loss='mean_squared_error')
#    predictor.fit(data,out,nb_epoch=1000,shuffle=True,
#                    validation_split=24*10./data.shape[0],batch_size=1,
#                    callbacks=[early_stopping],verbose=True)
#  #  predictor.outputs = [encoder.layers[-1].output]
#   # encoder.layers[-1].outbound_nodes = []
#    prediccions = np.zeros((data.shape[0],4))
#    for i in range(data.shape[0]):
#       prediccions[i,:] = predictor.predict(data[i,:].reshape(1,-1)).reshape(1,-1)
#    return model,encoder,predictor,prediccions

def remove_layer(model):
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.compile(optimizer='adadelta',loss='mean_squared_error')