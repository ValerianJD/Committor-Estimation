#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:39:57 2022

@author: valerian
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

## SEVERAL DATASETS

base_dir =  #Results directory

to_use = "Cimatoribus"
assert to_use in ["Cimatoribus", "Wind_driven"]
dirsave = base_dir+to_use+"/"

# C = Model_Cimatoribus.CimatoribusModel()
idx, noise = 60, 0.4
params = {"idx": idx, "noise": noise}

# sigma = 0.45
# beta = 0.198
# WD = Model_wind_driven.WindDrivenModel(1, sigma=sigma, beta=beta)
# p_wd = {"sigma": sigma, "beta": beta}

# Cima = {'model': C, 'params': p_c}
# Wind = {'model': WD, 'params': p_wd}

dirsave = base_dir+to_use+"/"
end_filename = "_".join([str(e).replace(".", ",") for i in range(len(params)) for e in list(params.items())[i]])

# method = "Reservoir"
# assert method in ["KNN", "FeedForward_Network", "DGA", "Reservoir"]

methods = ["KNN", "FeedForward_Network", "DGA", "Reservoir"]
labels = ["AMC", "FFNN", "DGA", "RC"]
colors = ["b", "tab:orange", "g", "m"]

Nt = [10, 20 ,30 ,50 ,75, 100, 150, 200, 300, 400, 500]
alpha = 0.15

titles = ["Normalized logarithm score", "Difference score", "Training time (s)", "Testing time (s)"]

fig, ax = plt.subplots(2,2, figsize=(15,10))

if to_use == "Cimatoribus":
    names = [ #List of names of the files to plot ]
    ax[0,0].plot(Nt, np.repeat(0.756,11), color="k", label="Truth", lw=2)
    ax[0,0].fill_between(Nt, 0.628, 0.898, color="k", alpha=alpha)
else:
    names = [ #List of the names of the files to plot]
    ax[0,0].plot(Nt, np.repeat(0.562,11), color="k", label="Truth", lw=2)
    ax[0,0].fill_between(Nt, 0.338, 0.767, color="k", alpha=alpha)

ax[0,1].plot(Nt, np.ones(11), color="k", label="Truth", lw=2)

for i, n in enumerate(names):
    loga, diff = np.zeros((11,100)), np.zeros((11,100))
    train, test = np.zeros((11,100)), np.zeros((11,100))
    if to_use == "Cimatoribus":    
        data = np.load(dirsave+n, allow_pickle=True).item()
    else:
        data = np.load(dirsave+n, allow_pickle=True).item()
    if data["time_train"].shape[-1] < 100:
        data["time_train"] = np.repeat(data["time_train"][...,np.newaxis], 100, axis=-1)
    loga, diff = data["logscore"], data["diffscore"]
    train, test = data["time_train"], data["time_test"]
    if "ML" in n or "FeedForward_Network" in n:
        loga, diff = np.mean(loga, axis=1), np.mean(diff, axis=1)
        test = np.mean(test, axis=1)
 
    m, e = np.mean(loga, axis=1), np.percentile(loga, [5,95], axis=1)
    m[m==0], e[e==0] = np.nan, np.nan
    ax[0,0].plot(Nt, m, color=colors[i], label=labels[i])
    ax[0,0].fill_between(Nt, e[0], e[1], color=colors[i], alpha=alpha)

    m, e = np.mean(diff, axis=1), np.percentile(diff, [5,95], axis=1)
    m[m==0], e[e==0] = np.nan, np.nan
    ax[0,1].plot(Nt, m, color=colors[i], label=labels[i])
    ax[0,1].fill_between(Nt, e[0], e[1], color=colors[i], alpha=alpha)

    if "ML" in n or "FeedForward_Network" in n:
        m, e = np.mean(train, axis=(1,2)), np.percentile(train, [5,95], axis=(1,2))
    else:
        m, e = np.mean(train, axis=1), np.percentile(train, [5,95], axis=1)
    m[m==0], e[e==0] = np.nan, np.nan
    ax[1,0].plot(Nt, m, color=colors[i], label=labels[i])
    ax[1,0].fill_between(Nt, e[0], e[1], color=colors[i], alpha=alpha)

    m, e = np.mean(test, axis=1), np.percentile(test, [5,95], axis=1)
    m[m==0], e[e==0] = np.nan, np.nan
    ax[1,1].plot(Nt, m, color=colors[i], label=labels[i])
    ax[1,1].fill_between(Nt, e[0], e[1], color=colors[i], alpha=alpha)
    
for i, a in enumerate(ax.reshape(1,4)[0]):
    a.set_xscale('log')
    a.set_xticks(Nt, fontsize=14)
    a.grid(True)
    a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if i==0 or i==1:
        a.legend(loc="lower right")
    else:
        a.legend()
    a.set_xlabel(r"Number of transitions $N_T$ in the training set", fontsize=14)
    a.set_ylabel(titles[i], fontsize=14)
ax[1,0].set_yscale('log')
ax[1,1].set_yscale('log')
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()
