#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:42:46 2022

Main code to estimate the committor using one of the methods described in the article. 
You can choose either the AMOC or double-gyre model
You can choose of one the 5 methods: AMC, KNN, FFN, DGA or RC

@author: Valerian Jacques-Dumas
"""

import Datasets
import numpy as np
import Machine_Learning, AMC
import matplotlib.pyplot as plt
from NewGen_Reservoir import NewGen_ResNet
import Model_Cimatoribus, Model_wind_driven
from Logarithm_score import LogaScore, diff_score
from Dynamical_Galerkin_Approximation import DynGalApp


# Define both models and needed info to load datasets
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

# Choose the model to use and the method to use 

to_use = "WD"
assert to_use in ["AMOC", "WD"]

method = "Reservoir"
assert method in ["AMC", "KNN", "Feedforward NN", "DGA", "Reservoir"]

#Whether to save the results or not
save = False
dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name

# We load the data corresponding to the chosen model
par = models[to_use]['params']
handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                "/Users/valerian/Desktop/Data/")
# Load the true committor of the largest dataset to define the climatology of the logarithm score
ref_comm = handler.open_data("Committor")
score = LogaScore(ref_comm)

# Load test dataset
test = handler.open_data("Trajectory", test=True)
true_comm = handler.open_data("Committor", test=True)
lab_test = handler.open_data("Labels", test=True)

# For reference and comparison, the logarithm score of the true committor
# We need its average, 5th and 95th percentile
true_scores = score.get_score(test,true_comm,lab_test)
m, e = np.mean(true_scores), np.percentile(true_scores, [5,95])

# Process of each method

if method == "AMC" or method == "KNN":
    
    var = ["Sdiff", "Sn", "Ss"] if to_use=="C" else ["A1", "A2", "A3", "A4"]
    K = 25 if to_use=="C" else 10
    norm = False if to_use=="C" else True

    name = "K_"+str(K)+"_"+"_".join(var)+"_"+"_".join([str(e).replace(".",",") for i in range(len(par)) for e in list(par.items())[i]])
    dirsave += "/AMC_KNN/"

    #Initialize method
    amc = AMC.AnaloguesMethod(models[to_use]['model'], K)
    
    if method == "AMC":
        # Compute committor with AMC only
        possible_analogues = np.arange(np.max(test.shape)-1)
        comm_amc = amc.process_AMC(test, possible_analogues, var, norm=norm)
        
        # Only retain the estimated committors having values in [0,1]
        idx = []
        for i in range(100):
            if np.all((comm_amc[i]>=0)&(comm_amc[i]<=1)):
                idx.append(i)
        print(len(idx)) #Number of retained trajectories
        
        # Logarithm score
        score_amc = score.get_score(test[idx], comm_amc[idx], lab_test[idx])
        print(np.mean(score_amc), np.std(score_amc))
        
        # Difference score
        diff_amc = diff_score(comm_amc[idx], true_comm[idx])
        print(np.mean(diff_amc), np.std(diff_amc))
        
        # Reshape the scores to make plotting easier (see Plot_results.py)
        score_amc = np.repeat(score_amc[np.newaxis], len(Nt), axis=0)
        diff_amc = np.repeat(diff_amc[np.newaxis], len(Nt), axis=0)
        all_times_amc = np.repeat(amc.all_times[idx][np.newaxis], len(Nt), axis=0)
        print(np.mean(all_times_amc), np.std(all_times_amc))
        
        if save:
            res_amc = {"logscore": np.round(score_amc,decimals=3), 
                        "diffscore": np.round(diff_amc,decimals=3),
                        "time_test": np.round(all_times_amc,decimals=3),
                        "time_train": np.full((len(Nt),len(idx)), np.nan)}
            np.save(dirsave+"AMC"+name, res_amc)

    else:
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
            train = handler.open_data("Trajectory")
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



elif method == "Feedforward NN":
    
    var = ["Sdiff", "D"] if to_use=="C" else ["A1", 'A2', "A3", "A4"]

    layers = [64,128,256]
    LR = 1e-4

    dirsave += "/ML/"
    
    name = "_".join([str(l) for l in layers])+"_LR_"+str(LR).replace(".",",")+"_"+"_".join(var)\
        +"_"+"_".join([str(e).replace(".",",") for i in range(len(par)) for e in list(par.items())[i]])

    #Initialize method
    nb_epochs, nb_runs = 30, 20
    # The directory where to save the best state of the NN after each epoch
    dir_ml = "/Users/valerian/Desktop/Data/save_best_state_ml/"
    nn = Machine_Learning.NeuralNetworks(nb_epochs, nb_runs, dir_ml, LR=LR)

    test = models[to_use]['model'].select_var(test, var)

    score_ml = np.zeros((len(Nt),nb_runs,100))
    diff_ml = np.zeros((len(Nt),nb_runs,100))
    all_times_train = np.zeros((len(Nt),nb_runs)) 
    all_times_test = np.zeros((len(Nt),nb_runs,100))

    # Estimation loop
    for i, nt in enumerate(Nt):
        print(nt)
        #Get data
        handler.Nt = nt
        train, lab_train = handler.open_data("Trajectory"), handler.open_data("Labels")
        # If applicable, reduce set of variables of train data
        train = models[to_use]['model'].select_var(train, var)

        #Compute committor
        comm = nn.process(train, lab_train, test, (len(var),layers))
        
        #Compute score
        for k in range(nb_runs):
            score_ml[i,k] = score.get_score(test,comm[k],lab_test)
            diff_ml[i,k] = diff_score(comm[k],true_comm)
        print(np.mean(score_ml[i]), np.std(score_ml[i]))
        print(np.mean(diff_ml[i]), np.std(diff_ml[i]))
        
        print(np.mean(diff_ml[i],axis=1))
        k_max = np.argmax(np.mean(diff_ml[i],axis=1))
        for j in range(10):
            plt.plot(true_comm[j],label="Truth")
            plt.plot(comm[k_max,j],label="Prediction")
            plt.ylabel("Committor", fontsize=13)
            plt.xlabel("Time (arbitrary unit)", fontsize=13)
            plt.legend()
            plt.title("Feedforward Network", fontsize=13)
            plt.show()
        
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



elif method == "DGA":
    
    dirsave += "/DGA/"
    
    var = ["Sdiff", "Sn", "Ss"] if to_use=="C" else ["A1", "A2", "A3", "A4"]
    
    process = "train_in_test"
    assert process in ["full_train", "no_train", "train_in_test"]
    give_params = True
    
    d, k_eps0, k_eps = None, None, None
    if give_params:
        if to_use == 'C':
            d, k_eps0, k_eps = 0.45, 15, -15
        else:
            d, k_eps0, k_eps = 0.75, 5, -5

    nb_modes = 10

    dga = DynGalApp(M=nb_modes, d=d, k_eps0=k_eps0, k_eps=k_eps)

    if process == "full_train":
        
        name = "Full_train_"+str(nb_modes)+"_modes_d_"+str(d).replace('.',',')+\
            "_k_eps0_"+str(k_eps0)+"_k_eps_"+str(k_eps)+"_"+"_".join(var)
        
        score_dga, diff_dga = np.zeros((len(Nt),test.shape[0])), np.zeros((len(Nt),test.shape[0]))
        time_train_dga, time_test_dga = np.zeros((len(Nt),test.shape[0])), np.zeros((len(Nt),test.shape[0]))
        
        for i, nt in enumerate(Nt[4:]):
            print(nt)
            handler.Nt = nt
            train = handler.open_data("Trajectory")
            
            comm_dga = dga.process_DGA(models[to_use]['model'], test, train_traj=train, process=process, var=var)
        
            score_dga[i] = score.get_score(test, comm_dga, lab_test)
            diff_dga[i] = diff_score(comm_dga, true_comm)
            time_train_dga[i] = np.repeat(dga.time_train, test.shape[0])
            time_test_dga[i] = dga.times_test
            
            print(np.mean(score_dga[i]), np.std(score_dga[i]))
            print(np.mean(diff_dga[i]), np.std(diff_dga[i]))
            print(dga.time_train)
            print(np.mean(time_test_dga), np.std(time_test_dga))
            print()
                
            if save:
                res_dga= {"logscore": np.round(score_dga,decimals=3),
                          "diffscore": np.round(diff_dga,decimals=3),
                          "time_train": np.round(time_train_dga,decimals=3),
                          "time_test": np.round(time_test_dga,decimals=3)}
                np.save(dirsave+name, res_dga)

    else:      
        
        nb_transi = 3
        if process == "train_in_test":
            name = "Train_in_test_"+str(nb_transi)+"_transi_"+str(nb_modes)+"_modes_d_"+\
                str(d).replace('.',',')+"_k_eps0_"+str(k_eps0)+"_k_eps_"+str(k_eps)+"_"+"_".join(var)
        else:
            name = "No_train_"+str(nb_modes)+"_modes_d_"+str(d).replace('.',',')+\
                "_k_eps0_"+str(k_eps0)+"_k_eps_"+str(k_eps)+"_"+"_".join(var)
        
        comm_dga = dga.process_DGA(models[to_use]['model'], test, label=lab_test, N_transi=nt, process=process, var=var)
        
        score_dga = score.get_score(test, comm_dga, lab_test)
        diff_dga = diff_score(comm_dga, true_comm)
        
        score_dga = np.repeat(score_dga[np.newaxis], len(Nt), axis=0)
        diff_dga = np.repeat(diff_dga[np.newaxis], len(Nt), axis=0)
        all_times_test_dga = np.repeat(dga.times_test[np.newaxis], len(Nt), axis=0)
        
        print(np.mean(score_dga), np.std(score_dga))
        print(np.mean(diff_dga), np.std(diff_dga))
        print(np.mean(dga.times_test), np.std(dga.times_test))
        
        if process == "train_in_test":
            all_times_train_dga = np.repeat(dga.times_train[np.newaxis], len(Nt), axis=0)
            print(np.mean(dga.times_train), np.std(dga.times_train))
        else:
            all_times_train_dga = np.repeat(np.nan, len(Nt))
        
        if save:
            res_dga = {"logscore": np.round(score_dga,decimals=3), 
                        "diffscore": np.round(diff_dga,decimals=3),
                        "time_test": np.round(all_times_test_dga,decimals=3),
                        "time_train": np.round(all_times_train_dga,decimals=3)}
            np.save(dirsave+name, res_dga)       



elif method == "Reservoir":

    var = ["Sdiff", "Sn", "Ss", "D"] if to_use=="C" else ["A1", "A2", "A3", "A4"]   
    if to_use == 'C':
        config = {'k':3, 's':1, 'p':4, 'alpha':1e-9}
    else:
        config = {'k':2, 's':1, 'p':6, 'alpha':1e-6}

    dirsave += "/Reservoir/"
    name = "New_generation_"+"_".join(var)+"_"+"_".join([str(e).replace(".",",") for i in range(len(config)) for e in list(config.items())[i]])

    test = models[to_use]['model'].select_var(test, var)

    score_res, diff_res = np.zeros((len(Nt),100)), np.zeros((len(Nt),test.shape[0]))
    time_train_res, time_test_res = np.zeros(len(Nt)), np.zeros((len(Nt),test.shape[0]))

    for i, nt in enumerate(Nt):
        print(nt)
        handler.Nt = nt
        train, train_comm = handler.open_data("Trajectory"), handler.open_data("Committor")
        train = models[to_use]['model'].select_var(train, var)
        
        esn = NewGen_ResNet(config['k'], config['s'], config['p'], config['alpha'])
        pred_test = esn.process(train, train_comm, test)
        pred_test[pred_test>1], pred_test[pred_test<0] = 1, 0
        
        score_res[i] = score.get_score(test, pred_test, lab_test)
        diff_res[i] = diff_score(pred_test, true_comm)  
        time_train_res[i] = esn.time_train
        time_test_res[i] = esn.times_test
        
        for j in range(10):
            plt.plot(true_comm[j],label="Truth")
            plt.plot(pred_test[j], label="NGRN estimate")
            plt.ylabel("Committor", fontsize=13)
            plt.xlabel("Time (arbitrary unit)", fontsize=13)
            plt.legend()
            plt.title("New Generation Reservoir Computing", fontsize=13)
            plt.show()
        
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

