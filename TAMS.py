#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:39:49 2021

@author: Valerian Jacques-Dumas
"""

import numpy as np
import AMC as Analogues
# import Committor_DMD as DMD
import time
# import multiprocessing as mp



class TAMS():
    def __init__(self, N, k_max, Tmax, model, committor_method, seed=None):
        self.model = model
        self.N = N
        self.k_max = k_max,
        self.Tmax = Tmax
        self.committor_method = committor_method
        self.rng = np.random.default_rng(seed=seed)
    
    def pretraining(self, params, training_set, target):
        self.params = params

    def run(self, model, ic):
        k, w = 1, 1
        
        # Create N trajectories
        traj = self.model.trajectory(self.N, self.Tmax, *self.params.values(), init_state=ic)
        Nt, d = traj.shape[1:] 
        
        while k <= k_max:
            #Evaluate the score function along each
            score = self.committor_method.process(traj)
            
            #Compute the max of each score
            Q = np.around(np.max(score, axis=1),2)
            
            #Find the worst trajectories
            L, Q_min = np.argmin(Q), np.min(Q)
            if Q_min >= 1:
                break
            ind_Q = np.argmax(score[L],axis=1)
            
            # Update w
            w *= (1-len(L)/self.N)
            
            # Create new trajectories
            traj = np.delete(traj, L, axis=0)
            print(traj.shape)
            new_ind = self.rng.uniform(0,traj.shape[0],size=len(L))
            for i in range(len(L)):
                new_traj = traj[new_ind[i],:ind_Q[i]]
                print(new_traj.shape)
                new_T = (Nt-ind_Q[i])*model.dt
                new_traj = np.concatenate((new_traj,self.model.trajectory(1, new_T, *params.values(), init_state=new_traj[-1])[0]),axis=1)
                print(new_traj.shape)
                traj = np.concatenate((traj,new_traj[np.newaxis]),axis=0)
                
            # Update k
            k += 1
        
        return np.count_nonzero(Q>=1)*w/self.N