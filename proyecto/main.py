import math
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
import matplotlib.pyplot as plt


real_data = np.loadtxt(open('serie1.b08C2.csv',"rb"),delimiter=",",skiprows=1,usecols=range(1,5))

#%%
months = 5
days = 30
lag=6

##selected_layers,count=cross_validation(tr_sets[0],out_tr[0],[24*3,24*2,24,12])
##%%
#ds=6
#autoencoder,encoder = train_autoencoder(tr_sets[ds],out_tr[ds],[220,170,120,100],'relu')
#
#%%
lags = [12,24,36]
output_lag = [2,3,4]
timesteps = [24,36]
blocks = [10,15,20]
runs = 5



predictions = []


for lag in lags:
    tr_sets,out_tr,ts_sets,out_ts,max_values,min_values,TS_tr_sets,TS_ts_sets = data_processing(real_data,lag,10,5*30*24,2)
    for timestep in timesteps:
        for block in blocks:
            mse =np.zeros((runs,10,24))
            mape =np.zeros((runs,10,24))
            mae = np.zeros((runs,10,24))
            for j in range(runs):
                for i in range(10):
                    print str(lag)+' '+str(timestep)+' '+str(block)+' '+str(j)+' '+str(i)
                    
                    model1,decoder = train_deep_autoencoder(tr_sets[i],[3*4*lag/4,4*lag/2],'relu',
                           out_tr[i],out_activation='sigmoid',fine_tune=True)
                    
                    model = train_lstm3(tr_sets[i],out_tr[i],[block],timestep,
                                        early_stop=True,epochs=20,loss_threshold=0
                                        ,from_model=True,pre_model=model1,
                                        activation='relu')
                    
                    tmae,tmape,tmse,prediction =test_performance_lstm(model,
                                    tr_sets[i],ts_sets[i],out_ts[i],timestep)
                    mae[j,i,:]=tmae.reshape(1,-1)
                    
                    mape[j,i,:]=tmape.reshape(1,-1)
                    print mae[j,i,:]
                    mse[j,i,:]=tmse.reshape(1,-1)
            runs_mae=np.mean(mae,axis=0)
            runs_mape=np.mean(mape,axis=0)
            runs_rmse=np.sqrt(np.mean(mse,axis=1))
            
            media_mae =np.mean(runs_mae,axis=0)
            media_mape = np.mean(runs_mape,axis=0)
            media_rmse = np.mean(runs_rmse,axis=0)
            
            std_mae = np.std(runs_mae,axis=0)
            std_mape = np.std(runs_mape,axis=0)
            std_rmse = np.std(runs_rmse,axis=0)
            f = open('Resultados_ad_deep_ae_dummy.txt','a')
            f.write('{0:2d} {1:3d} {1:4d} '.format(lag, timestep,blocks))
            for i in range(24):
                f.write(' '+str(media_mae[i])+' '+str(std_mae[i])+' '+str(media_mape[i])+' '+str(std_mape[i])+' '+str(media_rmse[i])+' '+str(std_rmse[i]))
            f.write('\n')
            f.close()

f.close()
#%%
""" Incremental autoencoder test """
lags = [12,24,36]
output_lag = [2,3,4]
timesteps = [24,36]
blocks = [10,15,20]
runs = 5



predictions = []


for lag in lags:
    tr_sets,out_tr,ts_sets,out_ts,max_values,min_values,TS_tr_sets,TS_ts_sets = data_processing(real_data,lag,10,5*30*24,1)
    for timestep in timesteps:
        for block in blocks:
            mse =np.zeros((runs,10,24))
            mape =np.zeros((runs,10,24))
            mae = np.zeros((runs,10,24))
            for j in range(runs):
                for i in range(1,10):
                    print str(lag)+' '+str(timestep)+' '+str(block)+' '+str(j)+' '+str(i)
                    

                    model = incremental_lstm_autoencoder(TS_tr_sets[i],[block],early_stop=False)
                    
                    model2 = train_lstm_return_false(TS_tr_sets[i],out_tr[i],[block],
                                        early_stop=False,from_model=True,model2=model)
                    
                    tmae,tmape,tmse,prediction =test_performance_lstm_return_false(model2,
                                    TS_tr_sets[i],TS_ts_sets[i],out_ts[i])
                    mae[j,i,:]=tmae.reshape(1,-1)
                    
                    mape[j,i,:]=tmape.reshape(1,-1)
                    print mae[j,i,:]
                    mse[j,i,:]=tmse.reshape(1,-1)
            runs_mae=np.mean(mae,axis=0)
            runs_mape=np.mean(mape,axis=0)
            runs_rmse=np.sqrt(np.mean(mse,axis=1))
            
            media_mae =np.mean(runs_mae,axis=0)
            media_mape = np.mean(runs_mape,axis=0)
            media_rmse = np.mean(runs_rmse,axis=0)
            
            std_mae = np.std(runs_mae,axis=0)
            std_mape = np.std(runs_mape,axis=0)
            std_rmse = np.std(runs_rmse,axis=0)
            f = open('Resultados_incremental_ae.txt','a')
            f.write('{0:2d} {1:3d} {1:4d} '.format(lag, timestep,blocks))
            for i in range(24):
                f.write(' '+str(media_mae[i])+' '+str(std_mae[i])+' '+str(media_mape[i])+' '+str(std_mape[i])+' '+str(media_rmse[i])+' '+str(std_rmse[i]))
            f.write('\n')
            f.close()

f.close()
            
#%%

lags = [12,24,36]
timesteps = [24,36]
blocks = [10,15,20]
runs = 5



predictions = []


for lag in lags:
    tr_sets,out_tr,ts_sets,out_ts,max_values,min_values,TS_tr_sets,TS_ts_sets = data_processing(real_data,lag,10,5*30*24)
    for block in blocks:
            mse =np.zeros((runs,10,24))
            mape =np.zeros((runs,10,24))
            mae = np.zeros((runs,10,24))
            for j in range(runs):
                for i in range(10):
                    print str(lag)+' '+str(timestep)+' '+str(block)+' '+str(j)+' '+str(i)
                                    
                    model = train_lstm_return_false(TS_tr_sets[i],out_tr[i],[block],
                                        early_stop=True)
                    
                    tmae,tmape,tmse,prediction =test_performance_lstm_return_false(model,
                                    TS_tr_sets[i],TS_ts_sets[i],out_ts[i])
                    mae[j,i,:]=tmae.reshape(1,-1)
                    
                    mape[j,i,:]=tmape.reshape(1,-1)
                    print mae[j,i,:]
                    mse[j,i,:]=tmse.reshape(1,-1)
            runs_mae=np.mean(mae,axis=0)
            runs_mape=np.mean(mape,axis=0)
            runs_rmse=np.sqrt(np.mean(mse,axis=1))
            
            media_mae =np.mean(runs_mae,axis=0)
            media_mape = np.mean(runs_mape,axis=0)
            media_rmse = np.mean(runs_rmse,axis=0)
            
            std_mae = np.std(runs_mae,axis=0)
            std_mape = np.std(runs_mape,axis=0)
            std_rmse = np.std(runs_rmse,axis=0)
            f = open('Resultados_return_false.txt','a')
            f.write('{0:2d} {1:3d} {1:4d} '.format(lag, timestep,blocks))
            for i in range(24):
                f.write(' '+str(media_mae[i])+' '+str(std_mae[i])+' '+str(media_mape[i])+' '+str(std_mape[i])+' '+str(media_rmse[i])+' '+str(std_rmse[i]))
            f.write('\n')
            f.close()

f.close()
            
#%%
lags = [12,24, 36]
#lags=[36]
timesteps = [12,24,36]
#timesteps = [36]
blocks = [10, 15, 20]
#blocks=[20]
runs = 5


predictions = []

for lag in lags:
    tr_sets,out_tr,ts_sets,out_ts,max_values,min_values = data_processing(real_data,lag,10,5*30*24)
    for timestep in timesteps:
        for block in blocks:
            mse =np.zeros((runs,10,24))
            mape =np.zeros((runs,10,24))
            mae = np.zeros((runs,10,24))
            for j in range(runs):
                for i in range(10):
                    print str(lag)+' '+str(timestep)+' '+str(block)+' '+str(j)+' '+str(i)
    #                corrupted_set = tr_sets[i] + np.random.normal(loc=0.0, scale=0.1, size=tr_sets[i].shape) 
                    model = train_lstm(tr_sets[i],tr_sets[i],[block],timestep,ae=True,early_stop=True)
                    model = train_lstm(tr_sets[i],out_tr[i],[block],timestep,from_model=True,model=model,early_stop=True,ae=True)
                    for i in range(0):
                        corrupted_set = tr_sets[i] + np.random.normal(loc=0.0, scale=0.25, size=tr_sets[i].shape) 
                        model = train_lstm(corrupted_set,tr_sets[i],[lag*2],timestep,ae=True,from_model=True,model=model)
                        model = train_lstm(tr_sets[i],out_tr[i],[block],timestep,from_model=True,model=model)
                    tmae,tmape,tmse,prediction =test_performance_lstm(model,tr_sets[i],ts_sets[i],out_ts[i],timestep)
                    mae[j,i,:]=tmae.reshape(1,-1)
                    mape[j,i,:]=tmape.reshape(1,-1)
                    print mape[j,i,:]
                    mse[j,i,:]=tmse.reshape(1,-1)
            runs_mae=np.mean(mae,axis=0)
            runs_mape=np.mean(mape,axis=0)
            runs_rmse=np.sqrt(np.mean(mse,axis=1))
            
            media_mae =np.mean(runs_mae,axis=0)
            media_mape = np.mean(runs_mape,axis=0)
            media_rmse = np.mean(runs_rmse,axis=0)
            
            std_mae = np.std(runs_mae,axis=0)
            std_mape = np.std(runs_mape,axis=0)
            std_rmse = np.std(runs_rmse,axis=0)
            
            f = open('Resultados_autoencoder_dropout.txt','a')
            f.write('{0:2d} {1:3d} {2:4d} '.format(lag, timestep,block))
            for i in range(24):
                f.write(' '+str(media_mae[i])+' '+str(std_mae[i])+' '+str(media_mape[i])+' '+str(std_mape[i])+' '+str(media_rmse[i])+' '+str(std_rmse[i]))
            f.write('\n')
            f.close()

f.close()

#%%
#%%
lags = [12,24, 36]
#lags=[36]
timesteps = [12,24,36]
#timesteps = [36]
blocks = [10, 15, 20]
#blocks=[20]
runs = 5


predictions = []

f = open('Resultados_autoencoder2.txt','a')
for lag in lags:
    tr_sets,out_tr,ts_sets,out_ts,max_values,min_values = data_processing(real_data,lag,10,5*30*24)
    for timestep in timesteps:
        for block in blocks:
            mse =np.zeros((runs,10,24))
            mape =np.zeros((runs,10,24))
            mae = np.zeros((runs,10,24))
            for j in range(runs):
                for i in range(10):
                    print str(lag)+' '+str(timestep)+' '+str(block)+' '+str(j)+' '+str(i)
                    corrupted_set = tr_sets[i] + np.random.normal(loc=0.0, scale=0.1, size=tr_sets[i].shape) 
                    model2 = train_lstm(corrupted_set,tr_sets[i],[lag*2],timestep,ae=True)
                    model2.pop()
                    model2.outputs = [model2.layers[-1].output]
                    model2.layers[-1].outbound_nodes = []
                    #reshape del training set
                    tr_data=tr_sets[i]
                    n_datos = tr_data.shape[0]
                    features = tr_data.shape[1]
                    n_samples = n_datos/timestep
                    
                    tr_data = np.reshape(tr_data[n_datos%timestep:,:],(n_samples,timestep,features))
                    model2.reset_states()
                    encoded_input = model2.predict(tr_data,batch_size=1)
                    model2 = train_lstm(encoded_input,out_tr[i],[block],timestep,formato=True,ae=True)
                    for i in range(0):
                        model2.pop()
                        model2.outputs = [model2.layers[-1].output]
                        model2.layers[-1].outbound_nodes = []
                        corrupted_set = tr_sets[i] + np.random.normal(loc=0.0, scale=0.25, size=tr_sets[i].shape) 
                        model2 = train_lstm(corrupted_set,tr_sets[i],[lag*2],timestep,ae=True)
                        model2.pop()
                        model2.outputs = [model2.layers[-1].output]
                        model2.layers[-1].outbound_nodes = []
                        #reshape del training set
                        tr_data=tr_sets[i]
                        n_datos = tr_data.shape[0]
                        features = tr_data.shape[1]
                        n_samples = n_datos/timestep
                        
                        tr_data = np.reshape(tr_data[n_datos%timestep:,:],(n_samples,timestep,features))
                        model2.reset_states()
                        encoded_input = model2.predict(tr_data,batch_size=1)
                        model = train_lstm(encoded_input,out_tr[i],[block],timestep,formato=True)
                    tmae,tmape,tmse,prediction =test_performance_lstm(model2,tr_sets[i],ts_sets[i],out_ts[i],timestep)
                    mae[j,i,:]=tmae.reshape(1,-1)
                    mape[j,i,:]=tmape.reshape(1,-1)
                    print mape[j,i,:]
                    mse[j,i,:]=tmse.reshape(1,-1)
            runs_mae=np.mean(mae,axis=0)
            runs_mape=np.mean(mape,axis=0)
            runs_rmse=np.sqrt(np.mean(mse,axis=1))
            
            media_mae =np.mean(runs_mae,axis=0)
            media_mape = np.mean(runs_mape,axis=0)
            media_rmse = np.mean(runs_rmse,axis=0)
            
            std_mae = np.std(runs_mae,axis=0)
            std_mape = np.std(runs_mape,axis=0)
            std_rmse = np.std(runs_rmse,axis=0)
            
            f = open('Resultados_reduccion_dropout.txt','a')
            f.write('{0:2d} {1:3d} {2:4d}'.format(lag, timestep, block))
            for i in range(24):
                f.write(' '+str(media_mae[i])+' '+str(std_mae[i])+' '+str(media_mape[i])+' '+str(std_mape[i])+' '+str(media_rmse[i])+' '+str(std_rmse[i]))
            f.write('\n')
            f.close()

f.close()

#%%
f = open('Resultados_dropout.txt','r')
resultados =[]
resultados_autoencoder=[]
#f2=open('Resultados_autoencoder.txt','r')
for line in f:
    resultados.append(line)
    
#for line in f2:
#    resultados.append(line)
#%%
r=[]
r_ae=[]
for line in resultados:
    r.append(line.split(' ')[7:])

#for line in resultados_autoencoder:
#    r_ae.append(line.split(' ')[6:])
#%%
media = []
for i in range(24):
    media.append(np.mean(np.asarray(r[i]).astype(int)))
