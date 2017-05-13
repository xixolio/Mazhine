#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 21:38:55 2016
 Testing of the lstm using the return_sequences=False scheme.
@author: ignacio
"""

import numpy as np
def test_performance_lstm_return_false(model,tr_data,ts_data,ts_output):


    predicted = np.zeros((24,4))
    mae=np.zeros((24,1))
    mape = np.zeros((24,1))
    mse = np.zeros((24,1))
    base_data = ts_data

    
    model.reset_states()
    model.predict(tr_data,batch_size=1)
    
    
    for j in range(24):
        
        #Se realiza la predicci
#        print model.predict(base_data,batch_size=1).shape
        predicted[j,:] = model.predict(base_data,batch_size=1)
#        print "ts1"
#        print ts_data
        base_data[0,:-1,:] = base_data[0,1:,:]
        base_data[0,-1,:] = predicted[j,:]
#        print "ts2"
#        print ts_data
#        wait = raw_input("PRESS ENTER TO CONTINUE.")
        #Metricas de performance
        predicted_speed = np.sqrt(predicted[j,0]**2 + predicted[j,1]**2)
        real_speed = np.sqrt(ts_output[j,0]**2 + ts_output[j,1]**2)
        e = real_speed - predicted_speed
        mae[j,0] = np.abs(e)
        mape[j,0] = np.abs(e/real_speed)
        mse[j,0] = e**2
        
    
    return mae,mape,mse,predicted