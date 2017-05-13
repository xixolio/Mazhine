#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 06:19:50 2016

@author: ignacio
"""

def train_fann(data,out,encoding_dims):
    drop_rate = 0.1    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model = Sequential()
    model.add(Dropout(drop_rate,input_shape=(data.shape[1],)))
    
    for i in range(len(encoding_dims)):
        model.add(Dense(encoding_dims[i],activation='relu'))
        
    model.add(Dense(4,activation='sigmoid'))
    model.compile(optimizer='adadelta',loss='mean_squared_error')
    model.fit(data,out,nb_epoch=1000,shuffle=True,
                    validation_split=24*10./data.shape[0],batch_size=32,
                    callbacks=[early_stopping],verbose=True)
    
    
    return model