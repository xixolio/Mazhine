#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:05:08 2017
In this version we arrange the dataset in a different manner in order to
test if by changing the vectors positions we could better train a lstm. The 
idea behind this aproach is that in keras lstm, bptt is restricted to the amount
of timesteps we set for the model, thus if for instance, we set the number of
timesteps to 3, there are going to be a set of specific triplets of vectors
with which the model will be trained. Vector 1 to 3, vector 4 to 6, and so on,
but since we are working with a timeseries, it is to be expected that vector 2
to 4 are also related, as to is vector 3 to 5.  The aim of this code is, therefore,
to allow for such combinations of vectors to be used as well during training.
@author: ignacio
"""

import numpy as np
import math
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, TimeDistributedDense,TimeDistributed
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

def train_lstm3(tr_data,tr_output,lstm_layer,time_steps,from_model=False,
               pre_model=[],ae=False,formato=False,early_stop=False,epochs=10,
               loss_threshold=0.02,activation=''):
    """Se realiza un re-estructuramiento de la data segun (numero_samples,time_steps,features). 
    Necesariamente el numero de datos originales debe ser divisible por time_steps"""
    if formato:
        """formato means that the training data already comes in the required format"""
        features = tr_data.shape[2]
        x = tr_data.shape[0]
        y = tr_data.shape[1]
        output_dim = tr_output.shape[1]
        tr_output = np.reshape(tr_output[-x*y:,:],(x,y,output_dim))
    else:
        n_datos = tr_data.shape[0]
        features = tr_data.shape[1]
        output_dim = tr_output.shape[1]
        missing = time_steps-n_datos%time_steps
#        tr_data = np.concatenate((np.zeros((missing,features)),tr_data),axis=0)
#        tr_output = np.concatenate((np.zeros((missing,output_dim)),tr_output),axis=0)
        
        n_samples = n_datos/time_steps
        awesome_datasets = []
        awesome_outputs = []
#       awesome_data = np.zeros((time_steps,n_samples,time_steps,features))
#        awesome_output = np.zeros((time_steps,n_samples,time_steps,output_dim))
        
#        for i in range(missing):
#            
##            temp_data = np.reshape(tr_data[n_datos%time_steps:,:],(n_samples,time_steps,features))
##            temp_output = np.reshape(tr_output[n_datos%time_steps:,:],(n_samples,time_steps,output_dim))
#            temp_data = tr_data[i+1:-(time_steps-i-1),:]
#            temp_output = tr_output[i+1:-(time_steps-i-1),:]
#            awesome_data[i,:-1,:,:] = np.reshape(temp_data,(n_samples-1,time_steps,features))
#            awesome_output[i,:-1,:,:] = np.reshape(temp_output,(n_samples-1,time_steps,output_dim))
#        
#        for i in range(time_steps-missing):
#            temp_data = tr_data[i+1:-(time_steps-missing-i-1),:]
#            temp_output = tr_output[i+1:-(time_steps-missing-i-1),:] 
#            
#        for i in range(missing+1):
#            temp_data = np.concatenate((np.concatenate((np.zeros((missing-i,features)),tr_data),axis=0),
#                                       np.zeros((i,features))),axis=0)
#            temp_output = np.concatenate((np.concatenate((np.zeros((missing-i,output_dim)),tr_output),axis=0),
#                                       np.zeros((i,features))),axis=0)
         
        n_samples1 = (n_datos + missing)/time_steps
        n_samples2 = n_samples1 + 1
        """Datasets arranged in different manners are built. Dataset 0 is
        always the one with the original positions. It should be considered 
        prunning those datasets ending with a bunch of zeros since it may
        teach the model a non-proper vector sequence"""
        for i in range(missing+1):
            data_shape = np.zeros((n_datos + missing,features))
            output_shape = np.zeros((n_datos + missing,output_dim))
            if i==0:
                data_shape[missing:,:] = tr_data
                output_shape[missing:,:] = tr_output
            else:
                data_shape[missing-i:-i,:] = tr_data
                output_shape[missing-i:-i,:] = tr_output

            awesome_datasets.append(np.reshape(data_shape,(n_samples1,time_steps,features)))
            awesome_outputs.append(np.reshape(output_shape,(n_samples1,time_steps,output_dim)))
            
            
                
                
        for i in range(time_steps-(missing+1)):
            data_shape = np.zeros((n_datos+missing+time_steps,features))
            output_shape = np.zeros((n_datos + missing + time_steps,output_dim))
            data_shape[time_steps-i-1:-(missing+i+1)] = tr_data
            output_shape[time_steps-i-1:-(missing+i+1)] = tr_output
            awesome_datasets.append(np.reshape(data_shape,(n_samples2,time_steps,features)))
            awesome_outputs.append(np.reshape(output_shape,(n_samples2,time_steps,output_dim)))

        """tentativo"""
        for i in range(1,time_steps):
            awesome_datasets[i]= awesome_datasets[i][:-1,:,:]
            awesome_outputs[i]= awesome_outputs[i][:-1,:,:]
    
    #Se crea el modelo segun las especificaciones entregadas en lstm_layer (numero de caps,neuronas por capa)
    model = Sequential()
    
    if from_model:
        """from_model means we are using an already pretrained model. Such case
        arises when we use autoencoders. Since the model uses Dense layers,
        its needed to replicate the model architecture using TimeDistributed
        layers and copy the weights"""
        model.add(TimeDistributed(Dense(pre_model.layers[0].output_dim,
                                    activation=activation),batch_input_shape=(1,time_steps,features)))
        model.layers[0].set_weights(pre_model.layers[0].get_weights())
        for i in range(1,len(pre_model.layers)):
            model.add(TimeDistributed(Dense(pre_model.layers[i].output_dim,
                                            activation=activation)))
            model.layers[i].set_weights(pre_model.layers[i].get_weights())
    
        model.add(LSTM(lstm_layer[0], activation='tanh',inner_activation='sigmoid',
                       stateful=True,return_sequences=True,dropout_W=0,
                       dropout_U=0, W_regularizer=l2(l=0.01)))
    else:
        """model is created using the given specifications"""
        model.add(LSTM(lstm_layer[0], activation='tanh',inner_activation='sigmoid',
                       stateful=True,batch_input_shape=(1,time_steps,features),
                       return_sequences=True,dropout_W=0,dropout_U=0,
                       W_regularizer=l2(l=0.01)))
    
    for i in range(1,len(lstm_layer)):
        model.add(LSTM(lstm_layer[i], activation='tanh',inner_activation='sigmoid',
                       stateful=True,return_sequences=True,dropout_W=0.03,
                       dropout_U=0.03,W_regularizer=l2(l=0.01)))
    #model.add(LSTM(lstm_layer[-1], activation='tanh',inner_activation='sigmoid',stateful=True))
    model.add(TimeDistributedDense(output_dim=output_dim,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adadelta')
    weights=model.get_weights()
    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    """ not for use in this setting since there havent decided yet how to pick
    the validation sets in order for them to be equal for each awesome set 
    (maybe average them through all the sets but that would be costly)"""
    if early_stop==True:
        count = 0
        validation=np.ceil(24.*5/time_steps).astype(int)
        tolerance_count = 0
        last_best = 100000
        for i in range(100):
            count+=1
            ds = i%time_steps
            model.fit(awesome_datasets[ds][:-validation,:,:],
                      awesome_outputs[ds][:-validation,:,:],nb_epoch=1,
                    batch_size=1,shuffle=False,verbose=False)
 #           model.fit(tr_data[:-validation,:,:],tr_output[:-validation,:,:],nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
            error=0
            for j in range(validation):
                predicted=model.predict(awesome_datasets[ds][-validation+j,:,:].reshape(1,time_steps,features))
                if ae:
                    error += np.sum(np.abs(tr_output[-validation+j,:,:]-predicted)**2)
                else:
                    predicted_speed=np.sqrt(predicted[0,:,0]**2+predicted[0,:,1]**2)
                    real_speed = np.sqrt(awesome_outputs[ds][-validation+j,:,0]**2 + awesome_outputs[ds][-validation+j,:,1]**2)
                    error += np.sum(np.abs(predicted_speed - real_speed)**2)
                #error += np.sum(np.abs(tr_output[-validation+j,:,:]-predicted))
            if last_best>error:
                last_best = error
                tolerance_count = 0
            else:
                tolerance_count+=1
            if tolerance_count > time_steps+1:
                break
            model.reset_states()
            print error
        
        count -= tolerance_count
        model.set_weights(weights)
        model.reset_states()
        for i in range(count):
            model.fit(awesome_datasets[ds],
                      awesome_outputs[ds],nb_epoch=1,
                    batch_size=1,shuffle=False,verbose=False)
            model.reset_states()
        
    else:
        for i in range(epochs):
            for j in range(time_steps):
                model.reset_states()
                hist =model.fit(awesome_datasets[j],awesome_outputs[j],nb_epoch=1,batch_size=1,shuffle=False,verbose=False)
                print hist.history['loss'][0]
                if hist.history['loss'][0] <= 0.02:
                    break
            

        
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