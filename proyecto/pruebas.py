#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:10:36 2016

@author: ignacio
"""
season= autoencoders[0].predict(tr_sets[0])
residual = tr_sets[0] - season
residual = residual - np.amin(residual)
s=season.reshape(season.shape[0],season.shape[1]/4.,4)
r= residual.reshape(residual.shape[0],residual.shape[1]/4.,4) 
#%%
residual=np.abs(residual)
mse =np.zeros((10,24))
predictions = []
encoder =[]
for i in range(1):
    print i
    model= train_autoencoder(residual,residual,[24*4,24*4],'relu','sigmoid')
    #algo,pred =test_performance_ae(predictor,ts_sets[i],out_ts[i])
    #predictions.append(predictor)
    #autoencoders.append(model)
    #predictions.append(pred)
    #mse2[i,:] = algo.reshape(1,-1)
#%%
pred_residual = model.predict(residual)
pr =pred_residual.reshape(pred_residual.shape[0],pred_residual.shape[1]/4.,4)