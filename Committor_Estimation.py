#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:40:10 2022

@author: valerian

"""
import Datasets
import numpy as np
import Machine_Learning
import Feedforward_NN_to_publish, AMC
# import matplotlib.pyplot as plt
from NewGen_Reservoir import NewGen_ResNet
import Model_Cimatoribus, Model_wind_driven
from Logarithm_score import LogaScore, diff_score
# from Dynamical_Galerkin_Approximation import DynGalApp

C = Model_Cimatoribus.CimatoribusModel()
idx, noise = 60, 0.4
p_c = {"idx":idx, "noise":noise}

sigma = 0.45
beta = 0.198
WD = Model_wind_driven.WindDrivenModel(1, sigma=sigma, beta=beta)
p_wd = {"sigma":sigma, "beta":beta}

Cima = {'model':C, 'params':p_c}
Wind = {'model':WD, 'params':p_wd}

models = {'C':Cima, 'WD':Wind}

Nt = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]

nb_dataset = 1

to_use = "WD"

par = models[to_use]['params']
handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                "/Users/valerian/Desktop/Data/")
ref_comm = handler.open_data("Committor")
score = LogaScore(ref_comm)

test = handler.open_data("Trajectory", test=True)
true_comm = handler.open_data("Committor", test=True)
lab_test = handler.open_data("Labels", test=True)

true_scores = score.get_score(test,true_comm,lab_test)
m, e = np.mean(true_scores), np.percentile(true_scores, [5,95])
print(m, e[0], e[1])

#%%

### AMC/KNN

save = False

to_use = 'WD'
if to_use == "C":
    # var = ["St", "Sts", "Sn", "Ss", "D", 'Sd']
    var = ["Sdiff", "Sn", "Ss"]
    # var = ["Sdiff", "D"]
    K = 25
elif to_use == "WD":
    var = ["A1", "A2", "A3", "A4"]
    # var = ["A1", "A2", "A3"]
    # var = ["A1", "A2", "A4"]
    # var = ["A1+A3", "A2", "A4"]
    K = 15

norm = True

par = models[to_use]['params']
handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                "/Users/valerian/Desktop/Data/")
ref_comm = handler.open_data("Committor")
score = LogaScore(ref_comm)

test = handler.open_data("Trajectory", test=True)
true_comm = handler.open_data("Committor", test=True)
lab_test = handler.open_data("Labels", test=True)

sub = 1
    
# name = "_1_"+str(sub)+"_subsampling_K_"+str(K)+"_norm_"+str(norm)+"_"+"_".join(var)+"_"+"_".join([str(e).replace(".",",") for i in range(len(par)) for e in list(par.items())[i]])
name = "_K_"+str(K)+"_norm_"+str(norm)+"_"+"_".join(var)+"_"+"_".join([str(e).replace(".",",") for i in range(len(par)) for e in list(par.items())[i]])
dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name+"/AMC_KNN/"

list_K = [10, 15, 20, 25, 30, 50, 75, 100, 150, 250]

# for K in list_K:
# print(K)
# name = "_K_"+str(K)+"_norm_"+str(norm)+"_"+"_".join(var)+"_"+"_".join([str(e).replace(".",",") for i in range(len(par)) for e in list(par.items())[i]])
#Initialize method
amc = AMC.AnaloguesMethod(models[to_use]['model'], K)

# Compute committor with AMC only
# possible_analogues = np.arange(np.max(test.shape)-1)
# comm_amc = amc.process_AMC(test, possible_analogues, var, norm=norm)
# idx = []
# for i in range(100):
#     if np.all((comm_amc[i]>=0)&(comm_amc[i]<=1)):
#         idx.append(i)
# print(len(idx))

# score_amc = score.get_score(test[idx], comm_amc[idx], lab_test[idx])
# print(np.mean(score_amc), np.percentile(score_amc, [5,95]))

# diff_amc = diff_score(comm_amc[idx], true_comm[idx])
# print(np.mean(diff_amc), np.percentile(diff_amc, [5,95]))

# score_amc = np.repeat(score_amc[np.newaxis], len(Nt), axis=0)
# diff_amc = np.repeat(diff_amc[np.newaxis], len(Nt), axis=0)
# all_times_amc = np.repeat(amc.all_times[idx][np.newaxis], len(Nt), axis=0)
# print(np.mean(all_times_amc), np.percentile(all_times_amc, [5,95]))

# if save:
#     res_amc = {"logscore": np.round(score_amc,decimals=3), 
#                 "diffscore": np.round(diff_amc,decimals=3),
#                 "time_test": np.round(all_times_amc,decimals=3),
#                 "time_train": np.full((len(Nt),len(idx)), np.nan)}
#     np.save(dirsave+"AMC"+name, res_amc)
     
#     comm_amc[comm_amc>1], comm_amc[comm_amc<0] = 1, 0
#     score_amc = score.get_score(test, comm_amc, lab_test)
#     print(np.mean(score_amc), np.percentile(score_amc, [5,95]))
    
#     diff_amc = diff_score(comm_amc, true_comm)
#     print(np.mean(diff_amc), np.percentile(diff_amc, [5,95]))
    
#     score_amc = np.repeat(score_amc[np.newaxis], len(Nt), axis=0)
#     diff_amc = np.repeat(diff_amc[np.newaxis], len(Nt), axis=0)
#     all_times_amc = np.repeat(amc.all_times[np.newaxis], len(Nt), axis=0)
#     print(np.mean(all_times_amc), np.percentile(all_times_amc, [5,95]))

    # res_amc = {"logscore": np.round(score_amc,decimals=3), 
    #             "diffscore": np.round(diff_amc,decimals=3),
    #             "time_test": np.round(all_times_amc,decimals=3),
    #             "time_train": np.full((len(Nt),100), np.nan)}
    # np.save(dirsave+"AMC"+name+"_total", res_amc)


# Initialize files to save
score_knn = np.zeros((len(Nt),100))
diff_knn = np.zeros((len(Nt),100))
all_times_knn_train = np.zeros((len(Nt),100))
all_times_knn_test = np.zeros((len(Nt),100))

# Estimation loop with KNN
for i, nt in enumerate(Nt):
    print(nt)
    #Get data
    handler.Nt = nt
    train = handler.open_data("Trajectory")[::sub]
    possible_analogues = np.arange(np.max(train.shape)-1)
    
    #Compute committor
    comm_knn = amc.process_KNN(train, test, possible_analogues, var, norm=norm)
    
    #Compute score
    score_knn[i] = score.get_score(test,comm_knn,lab_test)
    diff_knn[i] = diff_score(comm_knn, true_comm)
    print(np.mean(score_knn[i]), np.std(score_knn[i]))
    print(np.mean(diff_knn[i]), np.std(diff_knn[i]))
    
    #Save committor, time, score
    all_times_knn_train[i] = np.repeat(amc.time_amc, 100)
    all_times_knn_test[i] = amc.time_knn
    print(amc.time_amc)
    print(np.mean(all_times_knn_test[i]), np.std(all_times_knn_test[i]))

    if save:
        res_knn = {"logscore": np.round(score_knn,decimals=3),
                    "diffscore": np.round(diff_knn,decimals=3),
                    "time_train": np.round(all_times_knn_train,decimals=3),
                    "time_test": np.round(all_times_knn_test,decimals=3)}
        np.save(dirsave+"KNN"+name, res_knn)

#%%

### FEEDFORWARD NEURAL NETWORK

save = True

to_use = 'WD'
if to_use == "C":
    var = ["Sdiff", "D"]
elif to_use == "WD":
    var = ["A1", 'A2', "A3", "A4"]

network = "Dense"
layers = [64,128,256]
LR = 1e-4

sub = 1

dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name+"/ML/"
par = models[to_use]['params']
# other = "1_"+str(sub)+"_subsampling_"
other = "500_only"
name = network+"_"+"_".join([str(l) for l in layers])+"_LR_"+str(LR).replace(".",",")+"_"+other+"_".join(var)\
    +"_"+"_".join([str(e).replace(".",",") for i in range(len(par)) for e in list(par.items())[i]])

handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                "/Users/valerian/Desktop/Data/")
ref_comm = handler.open_data("Committor")
score = LogaScore(ref_comm)

test = handler.open_data("Trajectory", test=True)
true_comm = handler.open_data("Committor", test=True)
lab_test = handler.open_data("Labels", test=True).astype(int)
true_scores = score.get_score(test, true_comm, lab_test)

#Initialize method
nb_epochs, nb_runs = 30, 20
dir_ml = "/Users/valerian/Desktop/Data/save_best_state_ml/"
# nn = Feedforward_NN_to_publish.NeuralNetworks(nb_epochs, nb_runs, dir_ml, LR=LR)
nn = Machine_Learning.NeuralNetworks(nb_epochs, nb_runs, dir_ml, LR=LR)

test = models[to_use]['model'].select_var(test, var)

score_ml = np.zeros((len(Nt),nb_runs,100))
diff_ml = np.zeros((len(Nt),nb_runs,100))
all_times_train = np.zeros((len(Nt),nb_runs)) 
all_times_test = np.zeros((len(Nt),nb_runs,100))

# Estimation loop
for i, nt in enumerate(Nt[-1:]):
    print(nt)
    #Get data
    handler.Nt = nt
    train, lab_train = handler.open_data("Trajectory")[::sub], handler.open_data("Labels")[::sub]
    print(train.shape)
    train = models[to_use]['model'].select_var(train, var)

    #Compute committor
    comm = nn.process(train, lab_train, test, network, (len(var),layers))
    
    #Compute score
    for k in range(nb_runs):
        score_ml[i,k] = score.get_score(test,comm[k],lab_test)
        diff_ml[i,k] = diff_score(comm[k],true_comm)
    print(np.mean(score_ml[i]), np.std(score_ml[i]))
    print(np.mean(diff_ml[i]), np.std(diff_ml[i]))
    
    # print(np.mean(diff_ml[i],axis=1))
    # k_max = np.argmax(np.mean(diff_ml[i],axis=1))
    # for j in range(10):
    #     plt.plot(true_comm[j],label="Truth")
    #     plt.plot(comm[k_max,j],label="Prediction")
    #     plt.ylabel("Committor", fontsize=13)
    #     plt.xlabel("Time (arbitrary unit)", fontsize=13)
    #     plt.legend()
    #     plt.title("Feedforward Network", fontsize=13)
    #     plt.show()
    
    #Save committor, time, score
    all_times_train[i] = nn.all_times_train
    all_times_test[i] = nn.all_times_test
    
    print(np.mean(nn.all_times_train), np.std(nn.all_times_train))
    print(np.mean(nn.all_times_test), np.std(nn.all_times_test))
    
    if save:
        res_ml = {"logscore": np.round(score_ml,decimals=3),
                  "diffscore": np.round(diff_ml,decimals=3),
                  "time_train": np.round(all_times_train,decimals=3),
                  "time_test": np.round(all_times_test,decimals=3)}
        np.save(dirsave+name, res_ml)
    
#%%

### DYNAMICAL GALERKIN APPROXIMATION

from Dynamical_Galerkin_Approximation import DynGalApp

save = False

to_use = 'C'
if to_use == 'C':
    # var = ["St", "Sts", "Sn", "Ss", "D", "Sd"]
    var = ["Sdiff", "Sn", "Ss"]
    # var = ["Sdiff", "D"]
    nt = 3
    nb_modes = 10
    # d, k_eps0, k_eps = 0.45, 15, -15
    d, k_eps0, k_eps = 2.6, 20, -9
    # d, k_eps0, k_eps = None, None, None
else:
    var = ["A1", "A2", "A3", "A4"]
    # var = ["A1+A3", "A2", "A4"]
    nt = 3
    nb_modes = 200
    # d, k_eps0, k_eps = 0.75, 5, -5
    d, k_eps0, k_eps = None, None, None

process = "train_in_test"
# process = "full_train"

name = "Train_in_test_"+str(nt)+"_transi_"+str(nb_modes)+"_modes_d_"+\
    str(d).replace('.',',')+"_k_eps0_"+str(k_eps0)+"_k_eps_"+str(k_eps)+"_"+"_".join(var)
# name = str(nb_modes)+"_modes"+"_"+"_".join(var)
# name = str(nb_modes)+"_modes_d_"+str(d).replace('.',',')+"_k_eps0_"+str(k_eps0)+"_k_eps_"+str(k_eps)+"_"+"_".join(var)

dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name+"/DGA/"

par = models[to_use]['params']
handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                "/Users/valerian/Desktop/Data/")
ref_comm = handler.open_data("Committor")
score = LogaScore(ref_comm)

test = handler.open_data("Trajectory", test=True)
true_comm = handler.open_data("Committor", test=True)
lab_test = handler.open_data("Labels", test=True)

for nb_modes in [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]:
    
    dga = DynGalApp(M=nb_modes, d=d, k_eps0=k_eps0, k_eps=k_eps)
    
    if process == "full_train":
        ref_comm = handler.open_data("Committor")
        score = LogaScore(ref_comm)
        
        score_dga, diff_dga = np.zeros((len(Nt),test.shape[0])), np.zeros((len(Nt),test.shape[0]))
        time_train_dga, time_test_dga = np.zeros((len(Nt),test.shape[0])), np.zeros((len(Nt),test.shape[0]))
        
        for i, nt in enumerate(Nt):
            print(nt)
            handler.Nt = nt
            train = handler.open_data("Trajectory")
            
            comm_dga = dga.process_DGA(models[to_use]['model'], test, train_traj=train, process=process, var=var)
        
            score_dga[i] = score.get_score(test, comm_dga, lab_test)
            diff_dga[i] = diff_score(comm_dga, true_comm)
            time_train_dga[i] = np.repeat(dga.time_train, test.shape[0])
            time_test_dga[i] = dga.times_test
            
            print(np.mean(score_dga[i]), np.percentile(score_dga[i],[5,95]))
            print(np.mean(diff_dga[i]), np.percentile(diff_dga[i],[5,95]))
            print(dga.time_train)
            print(np.mean(time_test_dga), np.percentile(time_test_dga,[5,95]))
            print()
                
            if save:
                res_dga= {"logscore": np.round(score_dga,decimals=3),
                          "diffscore": np.round(diff_dga,decimals=3),
                          "time_train": np.round(time_train_dga,decimals=3),
                          "time_test": np.round(time_test_dga,decimals=3)}
                np.save(dirsave+name, res_dga)
    
    else:
        comm_dga = dga.process_DGA(models[to_use]['model'], test, label=lab_test, N_transi=nt, process=process, var=var)
        
        score_dga = score.get_score(test, comm_dga, lab_test)
        diff_dga = diff_score(comm_dga, true_comm)
        
        score_dga = np.repeat(score_dga[np.newaxis], len(Nt), axis=0)
        diff_dga = np.repeat(diff_dga[np.newaxis], len(Nt), axis=0)
        all_times_test_dga = np.repeat(dga.times_test[np.newaxis], len(Nt), axis=0)
        
        print(np.mean(score_dga), np.percentile(score_dga, [5,95]))
        print(np.mean(diff_dga), np.percentile(diff_dga, [5,95]))
        print(np.mean(dga.times_test), np.percentile(dga.times_test, [5,95]))
        if process == "train_in_test":
            all_times_train_dga = np.repeat(dga.times_train[np.newaxis], len(Nt), axis=0)
            print(np.mean(dga.times_train), np.percentile(dga.times_train, [5,95]))
        else:
            all_times_train_dga = np.repeat(np.nan, len(Nt))
        
        if save:
            res_dga = {"logscore": np.round(score_dga,decimals=3), 
                        "diffscore": np.round(diff_dga,decimals=3),
                        "time_test": np.round(all_times_test_dga,decimals=3),
                        "time_train": np.round(all_times_train_dga,decimals=3)}
            np.save(dirsave+name, res_dga)       
    
#%%

### NEW-GENERATION RESERVOIR

save = False

to_use = 'WD'
if to_use == 'C':
    var = ["Sdiff", "Sn", "Ss", "D"]
    config = {'k':2, 's':1, 'p':4, 'alpha':1e-9}
else:
    var = ["A1", "A2", "A3", "A4"]
    config = {'k':2, 's':1, 'p':6, 'alpha':1e-6}

name = "New_generation_"+"_".join(var)+"_"+"_".join([str(e).replace(".",",") for i in range(len(config)) for e in list(config.items())[i]])

par = models[to_use]['params']
dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name+"/Reservoir/"
handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                "/Users/valerian/Desktop/Data/")
ref_comm = handler.open_data("Committor")
score = LogaScore(ref_comm)

test = handler.open_data("Trajectory", test=True)
true_comm = handler.open_data("Committor", test=True)
lab_test = handler.open_data("Labels", test=True)

test = models[to_use]['model'].select_var(test, var)

score_res, diff_res = np.zeros((len(Nt),100)), np.zeros((len(Nt),test.shape[0]))
time_train_res, time_test_res = np.zeros(len(Nt)), np.zeros((len(Nt),test.shape[0]))

for i, nt in enumerate(Nt[-1:]):
    print(nt)
    handler.Nt = nt
    train, train_comm = handler.open_data("Trajectory"), handler.open_data("Committor")
    train = models[to_use]['model'].select_var(train, var)
    
    esn = NewGen_ResNet(config['k'], config['s'], config['p'], config['alpha'])
    pred_test = esn.process_infer(train, train_comm, test)
    pred_test[pred_test>1], pred_test[pred_test<0] = 1, 0
    
    score_res[i] = score.get_score(test, pred_test, lab_test)
    diff_res[i] = diff_score(pred_test, true_comm)  
    time_train_res[i] = esn.time_train
    time_test_res[i] = esn.times_test
    
    # for j in range(10):
    #     plt.plot(true_comm[j],label="Truth")
    #     plt.plot(pred_test[j], label="NGRN estimate")
    #     plt.ylabel("Committor", fontsize=13)
    #     plt.xlabel("Time (arbitrary unit)", fontsize=13)
    #     plt.legend()
    #     plt.title("New Generation Reservoir Computing", fontsize=13)
    #     plt.show()
    
    m_log, e_log = np.mean(score_res[i]), np.percentile(score_res[i], [5,95])
    m_diff, e_diff = np.mean(diff_res[i]), np.percentile(diff_res[i], [5,95])
    m_test, e_test = np.mean(time_test_res[i]), np.percentile(time_test_res[i], [5,95])
    print(m_log, e_log[0], e_log[1])
    print(m_diff, e_diff[0], e_diff[1])
    print(time_train_res[i])
    print(m_test, e_test[0], e_test[1])
    print()
   
if save:
    res_rl= {"logscore": np.round(score_res,decimals=3),
              "diffscore": np.round(diff_res,decimals=3),
              "time_train": np.round(time_train_res,decimals=3),
              "time_test": np.round(time_test_res,decimals=3)}
    np.save(dirsave+name, res_rl)

#%%

### RESERVOIR COMPUTING

# to_use = 'WD'
# if to_use == 'C':
#     # var = ["St", "Sts", "Sn", "Ss", "D", "Sd"]
#     var = ["Sdiff", "Sn", "Ss", "D"]
#     # var = ["Sdiff", "D"]
#     config = {'N':200, 'proba':0.15, 'sigma':1, 'alpha':0.7, 'beta':1e-9, 'spr':0.6,
#               'bias':0.5}
# else:
#     var = ["A1", "A2", "A3", "A4"]
#     config = {'N':200, 'proba':0.15, 'sigma':1, 'alpha':0.8, 'beta':1e-9, 'spr':0.8,
#               'bias':0.5}

# nb_runs = 10

# Min, Mout = len(var), 1
# name = "Mout_"+str(Mout)+"_"+"_".join(var)+"_"+\
#        "_".join([str(e).replace(".",",") for i in range(len(config)) for e in list(config.items())[i]])

# par = models[to_use]['params']
# dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name+"/Reservoir/"
# handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
#                                 "/Users/valerian/Desktop/Data/")
# ref_comm = handler.open_data("Committor")
# score = LogaScore(ref_comm)

# test = handler.open_data("Trajectory", test=True)
# true_comm = handler.open_data("Committor", test=True)
# lab_test = handler.open_data("Labels", test=True)

# test = models[to_use]['model'].select_var(test, var)

# score_res, diff_res = np.zeros((len(Nt),nb_runs,100)), np.zeros((len(Nt),nb_runs,test.shape[0]))
# time_train_res, time_test_res = np.zeros((len(Nt),nb_runs)), np.zeros((len(Nt),nb_runs,test.shape[0]))

# Nt = [100]

# for i, nt in enumerate(Nt):
#     print(nt)
#     handler.Nt = nt
#     train, train_comm = handler.open_data("Trajectory"), handler.open_data("Committor")
#     train = models[to_use]['model'].select_var(train, var)
    
#     esn = ResNet(config['N'], Min, Mout, config['sigma'], config['alpha'],
#                  config['beta'], config['proba'], config['spr'], bias=config['bias'])
#     pred_test = esn.process_reservoir(nb_runs, train, train_comm, test)
    
#     for k in range(nb_runs):
#         score_res[i,k] = score.get_score(test, pred_test[k], lab_test)
#         diff_res[i,k] = diff_score(pred_test[k], true_comm)  
#     time_train_res[i] = esn.times_train
#     time_test_res[i] = np.repeat(esn.times_test, test.shape[0]).reshape(nb_runs,-1)
    
#     print(np.mean(diff_res[i],axis=1))
#     k_max = np.argmax(np.mean(diff_res[i],axis=1))
#     print(k_max)
    # for j in range(10):
    #     plt.plot(true_comm[j],label="Truth")
    #     plt.plot(pred_test[k_max,j], label="Prediction")
    #     plt.ylabel("Committor", fontsize=13)
    #     plt.xlabel("Time (arbitrary unit)", fontsize=13)
    #     plt.legend()
    #     plt.title("Reservoir Computing", fontsize=13)
    #     plt.show()
    
    # print(np.mean(score_res[i]), np.std(score_res[i])) 
    # print(np.mean(diff_res[i]), np.std(diff_res[i]))
    # print(np.mean(time_train_res[i]), np.std(time_train_res[i]))
    # print(np.mean(time_test_res[i]), np.std(time_test_res[i]))
   
# res_rl= {"logscore": np.round(score_res,decimals=3),
#           "diffscore": np.round(diff_res,decimals=3),
#           "time_train": np.round(time_train_res,decimals=3),
#           "time_test": np.round(time_test_res,decimals=3)}
# np.save(dirsave+name, res_rl)

