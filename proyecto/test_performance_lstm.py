#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 21:38:55 2016

@author: ignacio
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:31:20 2016

@author: ignacio
"""
import numpy as np
def test_performance_lstm(model,tr_data,ts_data,ts_output,time_steps):
    n_datos = tr_data.shape[0]
    n_samples = n_datos / time_steps
    features = tr_data.shape[1]

    tr_data = np.reshape(tr_data[n_datos%time_steps:,:],(n_samples,time_steps,features))
#    ts_data = np.reshape(ts_data,(ts_data.shape[0],1,ts_data.shape[1]))
    base_data = ts_data
    predicted = np.zeros((24,4))
    mae=np.zeros((24,1))
    mape = np.zeros((24,1))
    mse = np.zeros((24,1))
    base_data = np.zeros((math.ceil(24./time_steps),time_steps,features))
    current_batch = 0
    
        
    for j in range(24):
        
        #Se realiza la prediccion
        if j%time_steps==0 and j!=0:
            tr_data = np.concatenate((tr_data,base_data[current_batch,:,:].reshape(1,time_steps,features)),axis=0)
            current_batch+=1
        i = j%time_steps
        model.reset_states()
        
        #Se predice hasta el ultimo sample de entrenamiento
        model.predict(tr_data,batch_size=1)  
#        print "tr"
#        print tr_data[-2:,:,:]
        base_data[current_batch,i,:] = ts_data
#        print "base"
#        print base_data[current_batch,:,:]
#        wait = raw_input("PRESS ENTER TO CONTINUE.")
        predicted[j,:] = (model.predict(base_data[current_batch,:,:].reshape(1,time_steps,features),batch_size=1)[0,i,:]).reshape(1,-1)
#        print "ts1"
#        print ts_data
        ts_data = np.concatenate((ts_data.reshape(1,-1)[:,4:],predicted[j,:].reshape(1,-1)),axis=1)
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