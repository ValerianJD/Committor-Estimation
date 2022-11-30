#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:39:57 2022

@author: valerian
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

to_use = "Cimatoribus"
assert to_use in ["Cimatoribus", "Wind_driven"]

method = "KNN"
assert method in ["KNN", "Feedforward NN", "DGA", "Reservoir"]

dirsave = "/Users/Valerian/Desktop/Results_final/"+to_use+"/"
filename = "KNN_K_25_norm_False_Sdiff_Sn_Ss_idx_60_noise_0,4"

Nt = [10, 20 ,30 ,50 ,75, 100, 150, 200, 300, 400, 500]
alpha = 0.15

colors = ["k", "b", "g", "r", "y", "m", "c", "tab:brown", "tab:orange", "tab:pink"]

loga, diff = np.zeros((10,11,100)), np.zeros((10,11,100))
train, test = np.zeros((10,11,100)), np.zeros((10,11,100))

for i in range(1,11):
    data = np.load(dirsave+method+"/"+filename+"_"+str(i)+".npy", allow_pickle=True).item()
    loga[i-1], diff[i-1] = data["logscore"], data["diffscore"]
    train[i-1], test[i-1] = data["time_train"], data["time_test"]
    
m_loga, e_loga = np.mean(loga, axis=2), np.percentile(loga, [5,95], axis=2)
m_diff, e_diff = np.mean(diff, axis=2), np.percentile(diff, [5,95], axis=2)
m_train, e_train = np.mean(train, axis=2), np.percentile(train, [5,95], axis=2)
m_test, e_test = np.mean(test, axis=2), np.percentile(test, [5,95], axis=2)

m_loga[m_loga==0], e_loga[e_loga==0] = np.nan, np.nan
m_diff[m_diff==0], e_diff[e_diff==0] = np.nan, np.nan
m_train[m_train==0], e_train[e_train==0] = np.nan, np.nan
m_test[m_test==0], e_test[e_test==0] = np.nan, np.nan

for i in range(10):
    plt.plot(Nt, m_loga[i], c=colors[i])
    plt.fill_between(Nt, e_loga[0,i], e_loga[1,i], color=colors[i], alpha=alpha)
plt.xscale('log')
plt.xticks(ticks=Nt, labels=Nt)
plt.grid(True)
plt.show()


for i in range(10):
    plt.plot(Nt, m_diff[i], c=colors[i])
    plt.fill_between(Nt, e_diff[0,i], e_diff[1,i], color=colors[i], alpha=alpha)
plt.xscale('log')
plt.xticks(ticks=Nt, labels=Nt)
plt.grid(True)
plt.show()

for i in range(10):
    plt.plot(Nt, m_train[i], c=colors[i])
    plt.fill_between(Nt, e_train[0,i], e_train[1,i], color=colors[i], alpha=alpha)
plt.xscale('log')
plt.yscale('log')
plt.xticks(ticks=Nt, labels=Nt)
plt.grid(True)
plt.show()

for i in range(10):
    plt.plot(Nt, m_test[i], c=colors[i])
    plt.fill_between(Nt, e_test[0,i], e_test[1,i], color=colors[i], alpha=alpha)
plt.xscale('log')
plt.yscale("log")
plt.xticks(ticks=Nt, labels=Nt)
plt.grid(True)
plt.show()

    