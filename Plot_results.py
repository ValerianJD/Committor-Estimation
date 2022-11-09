#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:25:49 2022

Plot and compare performance and computation time of each method,
against size of the training set

@author: Valerian Jacques-Dumas
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import Model_Cimatoribus
import Model_wind_driven

# Define each model 

C = Model_Cimatoribus.CimatoribusModel()
idx, noise = 60, 0.4
p_c = {"idx": idx, "noise": noise}

sigma = 0.45
beta = 0.198
WD = Model_wind_driven.WindDrivenModel(1, sigma=sigma, beta=beta)
p_wd = {"sigma": sigma, "beta": beta}

Cima = {'model': C, 'params': p_c}
Wind = {'model': WD, 'params': p_wd}

models = {'C': Cima, 'WD': Wind}

Nt = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]

# Whether to save the plots 
save = True
savename = "no_DGA_no_RC"

# What model to use
to_use = "C"
par = models[to_use]['params']
dirsave = "/Users/Valerian/Desktop/Results/"+models[to_use]['model'].name+"/"

# What curves to plot
# We can plot several of each but you have to manually indicate colors...
nb_conf_amc, nb_conf_ml, nb_conf_dga, nb_conf_res = 1, 1, 0, 0

def check_dico(dic, nb):
    """
    For convenience.
    Useful if we want to plot several curves of the same method.
    Instead of manually copying certain parameters, just provide them once and 
    this method copies them.

    Parameters
    ----------
    dic : dict
        The dictionary containing the informations about the plot
    nb : int
        The number of times to copy redundant information

    Returns
    -------
    dic : dict
        Updated version of dic
        
    """
    for k, v in dic.items():
        if type(v) != np.ndarray and type(v) != list:
            dic[k] = np.repeat(v, nb)
        elif len(v) != nb or (type(v) == np.ndarray and len(v.shape) == 1):
            dic[k] = np.repeat(v[np.newaxis], nb, axis=0)
    return dic

# For each method, enter the details of the results to plot

# AMC-KNN CONFIG
conf_amc = {'norm': False,
            'color': "b",
            'label': 'AMC',
            'other': ""}
conf_amc['K'] = 25 if to_use =='C' else 15
if to_use == "C":
    conf_amc['var'] = np.array(["Sdiff", "Sn", "Ss"])
elif to_use == "WD":
    conf_amc['var'] = np.array(["A1", "A2", "A3", "A4"])
    
conf_knn = conf_amc.copy()
conf_knn['color'] = "g"
conf_knn['label'] = "KNN"

# FEEDFORWARD NN CONFIG
conf_ml = {'LR': 1e-4,
           'layers': np.array([64, 128, 256]),
           'model': "Dense",
           'label': "FFN",
           'color': "r",
           'other': ""}
conf_ml['var'] = np.array(["Sdiff", "D"]) if to_use=="C" else np.array(["A1", "A2", "A3", "A4"])

# DGA CONFIG
conf_dga = {'color': "y",
            'other': "Train_in_test_3_transi_",
            'label': "DGA"}
conf_dga['process'] = "Train_in_test"
assert conf_dga['process'] in ["Full_train", "No_train", "Train_in_test"]
if conf_dga['process'] == "Train_in_test":
    nb_transi = 3
    conf_dga['process'] += "_"+str(nb_transi)+"_transi"

if to_use == "C":
    conf_dga['var'] = np.array(["Sdiff", "Sn", "Ss"])
    conf_dga['d'], conf_dga['k_e0'], conf_dga['k_eps'] = 0.45, 15, -15
    conf_dga['nb_modes'] = 10
elif to_use == "WD":
    conf_dga['var'] = np.array(["A1", "A2", "A3", "A4"])
    conf_dga['d'], conf_dga['k_e0'], conf_dga['k_eps'] = 0.75, 5, -5
    conf_dga['nb_modes'] = 200
    
# RESERVOIR CONFIG
conf_res = {'color': "m",
        'label':"RC"}

if to_use == "C":
    conf_res['var'] = np.array(["Sdiff", "Sn", "Ss", "D"])
    conf_res['params'] = {'k': 2, 's': 1, 'p': 4, 'alpha': 1e-9}
elif to_use == "WD":
    conf_res['var'] = np.array(["A1", "A2", "A3", "A4"])
    conf_res['params'] = {'k': 2, 's': 1, 'p': 6, 'alpha': 1e-6}

# We automatically check every method and find the right files

all_configs, res, all_names = [], [], []

if nb_conf_amc > 0:
    all_names_amc, all_names_knn = [], []
    conf_amc = check_dico(conf_amc, nb_conf_amc)
    conf_knn = check_dico(conf_knn, nb_conf_amc)
    all_configs.append(conf_amc)
    all_configs.append(conf_knn)
    for i in range(nb_conf_amc):
        n = "_K_"+str(conf_amc['K'][i])+"_norm_"+str(conf_amc['norm'][i])+"_" +\
            "_".join(conf_amc['var'][i])+"_" +\
            "_".join([str(e).replace(".", ",") for i in range(len(par)) for e in list(par.items())[i]])
        all_names_amc.append("AMC_KNN/AMC"+n)
        all_names_knn.append("AMC_KNN/KNN"+conf_knn['other'][i]+n)
    all_names += all_names_amc+all_names_knn

if nb_conf_ml > 0:
    conf = check_dico(conf_ml, nb_conf_ml)
    all_configs.append(conf)
    for i in range(nb_conf_ml):
        n = conf['model'][i]+"_"+"_".join([str(l) for l in conf['layers'][i]]) +"_LR_"+str(conf['LR'][i]).replace(".", ",")+\
            "_"+conf['other'][i]+"_".join(conf['var'][i]) +"_"+"_".join([str(e).replace(".", ",") 
                                                                         for i in range(len(par)) for e in list(par.items())[i]])
        all_names.append("ML/"+n)
    
if nb_conf_dga > 0:
    conf = check_dico(conf_dga, nb_conf_dga)
    all_configs.append(conf)
    for i in range(nb_conf_dga):
        n = conf['process'][i]+"_"+str(conf['nb_modes'][i])+"_modes_d_" +\
            str(conf['d'][i]).replace('.', ',')+"_k_eps0_"+str(conf['k_e0'][i]) +\
            "_k_eps_"+str(conf['k_eps'][i])+"_"+"_".join(conf['var'][i])
        all_names.append("DGA/"+n)
    
if nb_conf_res > 0:
    conf = check_dico(conf_res, nb_conf_res)
    all_configs.append(conf)
    for i in range(nb_conf_res):
        n = "New_generation_"+"_".join(conf['var'][i])+"_" +\
            "_".join([str(e).replace(".", ",") for j in range(
                len(conf['params'][i])) for e in list(conf['params'][i].items())[j]])
        all_names.append("Reservoir/"+n)

# Load all results arrays
for n in all_names:
    res.append(np.load(dirsave+n+".npy", allow_pickle=True).item())

colors = np.concatenate([c['color'] for c in all_configs])
labels = np.concatenate([c['label'] for c in all_configs])

# Plotting itself
logscore, a1 = plt.subplots()
diffscore, a2 = plt.subplots()
trainplot, a3 = plt.subplots()
testplot, a4 = plt.subplots()
axs = [a1, a2, a3, a4]
titles = ["Logarithm score", "Difference score", "Training time", "Testing time"]

# Transparence of the shaded errorbars
alpha = 0.15

# You can increase start or lower end to plot a close-up
start, end = 0, len(Nt)

# The true logarithm scores for reference
m_true = 0.756 if to_use == "C" else 0.562
e_true = np.array([0.628, 0.898]) if to_use == "C" else np.array([0.338, 0.767])
m_true, e_true = np.repeat(m_true, len(Nt)), np.repeat(e_true[:, np.newaxis], len(Nt), axis=1)
axs[0].plot(Nt[start:end], m_true[start:end], color="k", label="Truth")
axs[0].fill_between(Nt[start:end], e_true[0, start:end], e_true[1, start:end], color="k", alpha=alpha)
axs[1].plot(Nt[start:end], np.repeat(1, end-start), color="k", label="Truth")

for i in range(len(res)):
    print(labels[i])
    log, diff = res[i]['logscore'], res[i]['diffscore']
    if log.ndim == 3:
        log, diff = np.mean(log, axis=1), np.mean(diff, axis=1)
    m_log, e_log = np.mean(log, axis=1), np.percentile(log, [5, 95], axis=1)
    m_diff, e_diff = np.mean(diff, axis=1), np.percentile(diff, [5, 95], axis=1)

    m_log[m_log == 0], m_diff[m_diff == 0] = np.nan, np.nan
    e_log[e_log == 0], e_diff[e_diff == 0] = np.nan, np.nan
    
    axs[0].plot(Nt[start:end], m_log[start:end], color=colors[i], label=labels[i])
    axs[0].fill_between(Nt[start:end], e_log[0, start:end], e_log[1, start:end], color=colors[i], alpha=alpha)
    axs[1].plot(Nt[start:end], m_diff[start:end], color=colors[i], label=labels[i])
    axs[1].fill_between(Nt[start:end], e_diff[0, start:end], e_diff[1, start:end], color=colors[i], alpha=alpha)

    time_train, time_test = res[i]['time_train'], res[i]['time_test']
    if time_test.ndim == 3:
        time_test = np.mean(time_test, axis=1)
    if time_train.ndim > 1:
        m_train, e_train = np.mean(time_train, axis=1), np.percentile(time_train, [5, 95], axis=1)
    else:
        m_train, e_train = time_train, np.zeros((2, len(Nt)))
    m_test, e_test = np.mean(time_test, axis=1), np.percentile(time_test, [5, 95], axis=1)

    m_train[m_train == 0], m_test[m_test == 0] = np.nan, np.nan
    e_train[e_train == 0], e_test[e_test == 0] = np.nan, np.nan

    axs[2].plot(Nt[start:end], m_train[start:end], color=colors[i], label=labels[i])
    axs[2].fill_between(Nt[start:end], e_train[0, start:end], e_train[1, start:end], color=colors[i], alpha=alpha)
    axs[3].plot(Nt[start:end], m_test[start:end], color=colors[i])
    axs[3].fill_between(Nt[start:end], e_test[0, start:end], e_test[1, start:end], color=colors[i], alpha=alpha)

for i, a in enumerate(axs):
    a.set_xscale('log')
    a.set_xticks(Nt[start:end])
    a.grid(True)
    a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.legend()
    a.set_xlabel(r"Number of transitions $N_T$ in the training set", fontsize=13)
    a.set_ylabel(titles[i], fontsize=13)
axs[1].legend(loc="lower right")
axs[2].legend(loc="upper left")
axs[2].set_yscale('log')
axs[3].set_yscale('log')

if save:
    logscore.savefig(dirsave+"Logarithm_"+savename, dpi=500)
    diffscore.savefig(dirsave+"Difference_"+savename, dpi=500)
    trainplot.savefig(dirsave+"Train_time_"+savename, dpi=500)
    testplot.savefig(dirsave+"Test_time_"+savename, dpi=500)
plt.show()