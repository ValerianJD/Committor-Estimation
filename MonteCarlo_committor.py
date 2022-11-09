#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:49:14 2022

@author: valerian
"""

from time import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# from Model_wind_driven import WindDrivenModel
from Model_Cimatoribus import CimatoribusModel


class MonteCarlo():
    
    def __init__(self, model, model_params, N):
        self.model = model
        self.params = model_params
        self.N = N
        self.pool = mp.Pool(6)
    
    def committor_estim(self,traj):
        L = traj.shape[0]
        committor = np.zeros(L)
        on, off = self.model.is_on(traj), self.model.is_off(traj)
        committor[on], committor[off] = 0, 1
        others = self.model.is_transi(traj)
        committor[others] = self.pool.map(self.process, traj[others])
        return committor
    
    def process(self, state):
        return self.model.MC_comp_comm(self.N, state, *self.params)
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
           
if __name__ == "__main__":

#     m = WindDrivenModel(1)
    
#     sigma = 0.4
#     beta = 0.04
#     L = 5000
#     rng = np.random.default_rng()
#     a2 = rng.uniform(-2.5, 2.5, 1)
#     a4 = rng.uniform(-2.5, 2.5, 1)
#     a1 = rng.uniform(-2.5, 2.5, 1)
#     a3 = rng.uniform(-2.5, 2.5, 1) - a1
#     ic = np.vstack((a1,a2,a3,a4)).T
#     traj = m.trajectory(1, L, ic[0], sigma, beta)
    
#     fp = m.steady
#     up, down = fp[str(sigma)][0], fp[str(sigma)][1]
    
#     plt.plot(traj[:,0]+traj[:,2], traj[:,1])
#     plt.show()
    
#     plt.plot(np.linalg.norm(traj-up,axis=1))
#     plt.axhline(y=0.5, c="k", ls="--", lw=3)
#     plt.show()
    
#     p = (sigma, beta)
#     est = MonteCarlo(m, p, 100)
    
#     comm = est.committor_estim(traj)
    
#     plt.plot(comm)
#     plt.show()

    C = CimatoribusModel(seed=23)
    n_t, n_y = 10, 500
    idx, noise = 60, 0.4
    t = C.trajectory(n_t, n_y, idx, noise)
    # t_dim = C.re_dim(t)
    # plt.plot(t_dim[:,2]-t_dim[:,1], t_dim[:,4])
    # plt.show()
    
    p = (idx, noise)
    est = MonteCarlo(C, p, 100)
    comm = np.zeros((10,5000))
    for i in range(10):
        t0 = time()
        comm[i] = est.committor_estim(t[i])
        t1 = time()
        print(t1-t0)
    
    # plt.plot(comm)
    # plt.show()
    
#%%

    comm_julia = np.load("/Users/Valerian/Desktop/comm_julia.npy")
    print(comm_julia.shape, comm.shape)
    for i in range(10):
        plt.plot(comm[i])
        plt.plot(comm_julia[:,i],alpha=0.5)
        plt.show()



