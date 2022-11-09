#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:00:26 2022

@author: valerian
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import Model_Cimatoribus
import Model_wind_driven

C = Model_Cimatoribus.CimatoribusModel()
idx, noise = 60, 0.4
p_c = {"idx": idx, "noise": noise}

sigma = 0.4
beta = 0.04
WD = Model_wind_driven.WindDrivenModel(1, sigma=sigma, beta=beta)
p_wd = {"sigma": sigma, "beta": beta}

Cima = {'model': C, 'params': p_c}
Wind = {'model': WD, 'params': p_wd}

models = {'C': Cima, 'WD': Wind}

Nt = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]


def check_dico(dic, nb):
    for k, v in dic.items():
        if type(v) != np.ndarray and type(v) != list:
            dic[k] = np.repeat(v, nb)
        elif len(v) != nb or (type(v) == np.ndarray and len(v.shape) == 1):
            dic[k] = np.repeat(v[np.newaxis], nb, axis=0)
    return dic

#%%

save = True
to_use = "C"
nb_conf_amc, nb_conf_ml, nb_conf_dga, nb_conf_res = 1, 1, 1, 1
savename = "all_methods_newgen_resnet"

par = models[to_use]['params']
dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name+"/"

all_configs, res, all_names = [], [], []

# AMC-KNN CONFIG
conf_amc = {'norm': False,
            'color': "b",
            'other': "_New"}
conf_amc['K'] = 25 if to_use == 'C' else 10
conf_knn = conf_amc.copy()
# conf_knn['color'] = ["b", "g", "r", "y", "indigo", "m", "c"] #, "c", "tab:olive", "tab:brown", "tab:cyan", "tab:pink"]
conf_knn['color'] = "g"
conf_knn['other'] = "_New"
# conf_knn['other'] = "_1_2_subsampling"
# conf_knn["other"] = ["", "_1_2_subsampling", "_1_3_subsampling", "_1_4_subsampling", "_1_5_subsampling", "_1_6_subsampling", "_1_10_subsampling"]
if to_use == "C":
    conf_amc['var'], conf_knn['var'] = np.array(["Sdiff", "Sn", "Ss"]), np.array(["Sdiff", "Sn", "Ss"])
    # conf_amc['label'], conf_knn['label'] = "AMC, $S_{diff}, S_n, S_s$", "KNN, $S_{diff}, S_n, S_s$"
    conf_amc['label'], conf_knn['label'] = "AMC", "KNN"
elif to_use == "WD":
    conf_amc['var'], conf_knn['var'] = np.array(
        ["A1", "A2", "A3", "A4"]), np.array(["A1", "A2", "A3", "A4"])
    conf_amc['label'], conf_knn['label'] = "AMC", "KNN",

if nb_conf_amc > 0:
    all_names_amc, all_names_knn = [], []
    conf_amc = check_dico(conf_amc, nb_conf_amc)
    conf_knn = check_dico(conf_knn, nb_conf_amc)
    all_configs.append(conf_amc)
    all_configs.append(conf_knn)
    for i in range(nb_conf_amc):
        n = "_K_"+str(conf_amc['K'][i])+"_norm_"+str(conf_amc['norm'][i])+"_" +\
            "_".join(conf_amc['var'][i])+"_" +\
            "_".join([str(e).replace(".", ",") for i in range(len(par))
                     for e in list(par.items())[i]])
        all_names_amc.append("AMC_KNN/AMC"+n)
        all_names_knn.append("AMC_KNN/KNN"+conf_knn['other'][i]+n)
    all_names += all_names_amc+all_names_knn

# ML CONFIG
conf = {'LR': 1e-4,
        'layers': np.array([64, 128, 256]),
        'net': "Dense",
        'other': "",
        'color': "r"}
# 'other':["", "1_2_subsampling_", "1_4_subsampling_", "1_6_subsampling_"],
# 'color':["r", "b", "g", "k"]}
if to_use == "C":
    # conf['var'], conf['label'] = np.array(["Sdiff", "D"]), "Feedforward NN, $S_{diff}, D$"
    conf['var'], conf['label'] = np.array(["Sdiff", "D"]), "FFN"
elif to_use == "WD":
    conf['var'], conf['label'] = np.array(
        ["A1", "A2", "A3", "A4"]), "FFN"
if nb_conf_ml > 0:
    conf = check_dico(conf, nb_conf_ml)
    all_configs.append(conf)
    for i in range(nb_conf_ml):
        n = conf['net'][i]+"_"+"_".join([str(l) for l in conf['layers'][i]]) +\
            "_LR_"+str(conf['LR'][i]).replace(".", ",")+"_"+conf['other'][i]+"_".join(conf['var'][i]) +\
            "_"+"_".join([str(e).replace(".", ",")
                         for i in range(len(par)) for e in list(par.items())[i]])
        all_names.append("ML/"+n)

# DGA CONFIG
conf = {'color': "y",
        'other': "Train_in_test_3_transi_",
        'excep': ""}
if to_use == "C":
    # conf['var'], conf['label'] = np.array(["Sdiff", "Sn", "Ss"]), "DGA, $S_{diff}, S_n, S_s$"
    conf['var'], conf['label'] = np.array(["Sdiff", "Sn", "Ss"]), "DGA"
    conf['d'], conf['k_e0'], conf['k_eps'] = 0.45, 15, -15
    conf['nb_modes'] = 10
elif to_use == "WD":
    conf['var'], conf['label'] = np.array(["A1", "A2", "A3", "A4"]), "DGA"
    conf['d'], conf['k_e0'], conf['k_eps'] = 0.8, 7, -7
    conf['nb_modes'] = 200
if nb_conf_dga > 0:
    conf = check_dico(conf, nb_conf_dga)
    all_configs.append(conf)
    for i in range(nb_conf_dga):
        if len(conf['excep'][i]) > 0:
            n = conf['excep'][i]+"_"+"_".join(conf['var'][i])
        else:
            n = conf['other'][i]+str(conf['nb_modes'][i])+"_modes_d_" +\
                str(conf['d'][i]).replace('.', ',')+"_k_eps0_"+str(conf['k_e0'][i]) +\
                "_k_eps_"+str(conf['k_eps'][i])+"_"+"_".join(conf['var'][i])
        all_names.append("DGA/"+n)

# RESERVOIR CONFIG
# conf = {'Mout':1,
#         'color':"m"}
# if to_use == "C":
#     conf['var'], conf['label'] = np.array(["Sdiff", "Sn", "Ss", "D"]), "Reservoir, $S_{diff}, S_n, S_s, D$"
#     conf['params'] = {'N':200, 'proba':0.15, 'sigma':1, 'alpha':0.7, 'beta':1e-9,
#                       'spr':0.6,'bias':0.5}
# elif to_use == "WD":
#     conf['var'], conf['label'] = np.array(["A1", "A2", "A3", "A4"]), "Reservoir"
#     conf['params'] = {'N':200, 'proba':0.15, 'sigma':1, 'alpha':0.8, 'beta':1e-9, 'spr':0.8,
#                       'bias':0.5}
# if nb_conf_res>0:
#     conf = check_dico(conf,nb_conf_res)
#     all_configs.append(conf)
#     for i in range(nb_conf_res):
#         n = "Mout_"+str(conf['Mout'][i])+"_"+"_".join(conf['var'][i])+"_"+\
#             "_".join([str(e).replace(".",",") for j in range(len(conf['params'][i])) for e in list(conf['params'][i].items())[j]])
#         all_names.append("Reservoir/"+n)

# NEWGEN RESERVOIR CONFIG
conf = {'color': "m"}
if to_use == "C":
    # conf['var'], conf['label'] = np.array(["Sdiff", "Sn", "Ss", "D"]), "Reservoir, $S_{diff}, S_n, S_s, D$"
    conf['var'], conf['label'] = np.array(
        ["Sdiff", "Sn", "Ss", "D"]), "RC"
    conf['params'] = {'k': 2, 's': 1, 'p': 4, 'alpha': 1e-9}
elif to_use == "WD":
    conf['var'], conf['label'] = np.array(
        ["A1", "A2", "A3", "A4"]), "RC"
    conf['params'] = {'k': 2, 's': 3, 'p': 6, 'alpha': 1e-6}
if nb_conf_res > 0:
    conf = check_dico(conf, nb_conf_res)
    all_configs.append(conf)
    for i in range(nb_conf_res):
        n = "New_generation_"+"_".join(conf['var'][i])+"_" +\
            "_".join([str(e).replace(".", ",") for j in range(
                len(conf['params'][i])) for e in list(conf['params'][i].items())[j]])
        all_names.append("Reservoir/"+n)

for n in all_names:
    res.append(np.load(dirsave+n+".npy", allow_pickle=True).item())

colors = np.concatenate([c['color'] for c in all_configs])
labels = np.concatenate([c['label'] for c in all_configs])

logscore, a1 = plt.subplots()
diffscore, a2 = plt.subplots()
trainplot, a3 = plt.subplots()
testplot, a4 = plt.subplots()
axs = [a1, a2, a3, a4]
titles = ["Logarithm score", "Difference score",
          "Training time", "Testing time"]

alpha = 0.15
start, end = 0, len(Nt)

m_true = 0.756 if to_use == "C" else 0.493
e_true = np.array([0.628, 0.898]) if to_use == "C" else np.array(
    [0.285, 0.726])
m_true, e_true = np.repeat(m_true, len(Nt)), np.repeat(
    e_true[:, np.newaxis], len(Nt), axis=1)
axs[0].plot(Nt[start:end], m_true[start:end], color="k", label="Truth")
axs[0].fill_between(Nt[start:end], e_true[0, start:end],
                    e_true[1, start:end], color="k", alpha=alpha)
axs[1].plot(Nt[start:end], np.repeat(1, end-start), color="k", label="Truth")

for i in range(len(res)):
    print(labels[i]+"\n")

    log, diff = res[i]['logscore'], res[i]['diffscore']
    if log.ndim == 3:
        log, diff = np.mean(log, axis=1), np.mean(diff, axis=1)
    m_log, e_log = np.mean(log, axis=1), np.percentile(log, [5, 95], axis=1)
    # m_log, e_log = np.mean(log, axis=-1), np.std(log, axis=-1)
    # m_diff, e_diff = np.mean(diff, axis=-1), np.std(diff, axis=-1)
    m_diff, e_diff = np.mean(diff, axis=1), np.percentile(
        diff, [5, 95], axis=1)

    m_log[m_log == 0], m_diff[m_diff == 0] = np.nan, np.nan
    e_log[e_log == 0], e_diff[e_diff == 0] = np.nan, np.nan

    # print(m_log)
    # print(e_log)
    # print()
    # print(m_diff)
    # print(e_diff)
    # print()

    axs[0].plot(Nt[start:end], m_log[start:end],
                color=colors[i], label=labels[i])
    axs[0].fill_between(Nt[start:end], e_log[0, start:end],
                        e_log[1, start:end], color=colors[i], alpha=alpha)
    axs[1].plot(Nt[start:end], m_diff[start:end],
                color=colors[i], label=labels[i])
    axs[1].fill_between(Nt[start:end], e_diff[0, start:end],
                        e_diff[1, start:end], color=colors[i], alpha=alpha)

    time_train, time_test = res[i]['time_train'], res[i]['time_test']
    if time_test.ndim == 3:
        time_test = np.mean(time_test, axis=1)
    if time_train.ndim > 1:
        m_train, e_train = np.mean(time_train, axis=1), np.percentile(
            time_train, [5, 95], axis=1)
    else:
        m_train, e_train = time_train, np.zeros((2, len(Nt)))
    m_test, e_test = np.mean(time_test, axis=1), np.percentile(
        time_test, [5, 95], axis=1)

    m_train[m_train == 0], m_test[m_test == 0] = np.nan, np.nan
    e_train[e_train == 0], e_test[e_test == 0] = np.nan, np.nan

    # print(m_train)
    # print(e_train)
    # print()
    # print(m_test)
    # print(e_test)
    # print()

    axs[2].plot(Nt[start:end], m_train[start:end],
                color=colors[i], label=labels[i])
    axs[2].fill_between(Nt[start:end], e_train[0, start:end],
                        e_train[1, start:end], color=colors[i], alpha=alpha)
    axs[3].plot(Nt[start:end], m_test[start:end],
                color=colors[i], label=labels[i])
    axs[3].fill_between(Nt[start:end], e_test[0, start:end],
                        e_test[1, start:end], color=colors[i], alpha=alpha)

for i, a in enumerate(axs):
    a.set_xscale('log')
    a.set_xticks(Nt[start:end])
    a.grid(True)
    a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.legend(loc='lower right')
    a.set_xlabel(
        r"Number of transitions $N_T$ in the training set", fontsize=13)
    a.set_ylabel(titles[i], fontsize=13)
axs[2].set_yscale('log')
axs[3].set_yscale('log')

if save:
    logscore.savefig(dirsave+"Logarithm_"+savename, dpi=500)
    diffscore.savefig(dirsave+"Difference_"+savename, dpi=500)
    trainplot.savefig(dirsave+"Train_time_"+savename, dpi=500)
    testplot.savefig(dirsave+"Test_time_"+savename, dpi=500)
plt.show()
