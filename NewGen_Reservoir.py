#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:58:02 2022

@author: valerian
"""

#%%

import numpy as np
from time import time
from scipy.special import binom
from itertools import combinations_with_replacement

class NewGen_ResNet():
    def __init__(self, k, s, p, alpha):
        """
        Parameters
        ----------
        k : int
            The number of timesteps the RC should consider
        s : int
            The stride between these timesteps
        p : int
            The degree of the highest-order monomial in the nonlinear vector
        alpha : float
            The ridge-regression parameter
        """
        self.k, self.s, self.p, self.alpha = k, s, p, alpha
    
    def infer(self, traj, target=None):
        """
        The inference task of the RC. 
        If target is prescribed, the function is in "training" mode. If target is None,
        the function is in "testing" mode.

        Parameters
        ----------
        traj : Either the training of test trajectory

        target : Another timeseries, that of the variable we want ot predict. 
        It shape must be (dimension, timesteps)

        Returns
        -------
        Simply computes the output layer (train mode) or returns the estimated variable (test mode)
        
        """
        d, time = traj.shape
        self.dlin = d*self.k  #Size of the lineart part of the feature vector
        # dtot is the total size of the feature vector
        # the binomial factor describes the size of the nonlinear features vector
        # 1 corresponds to the additional constant
        self.dtot = int(binom(self.dlin+self.p-1, self.p)) + self.dlin + 1
        # List of the indices to multiply in order to create the nonlinear features vector 
        self.combi = np.array(list(combinations_with_replacement(np.arange(self.dlin), self.p)))
        
        # The linear features vector
        x = np.zeros((self.dlin, time))
        for i in range(self.k):
            x[d*i:d*(i+1), i*self.s:] = traj[:,:time-i*self.s]
        
        # Computation of the total features vector
        out = np.ones((self.dtot, time))
        out[1:self.dlin+1] = x 
        # Computation of the nonlinear features vector
        out[1+self.dlin:] = np.product(x[self.combi],axis=1)
        
        if target is None:
            # In test mode, return the estimated variable
            return self.W_out @ out
        # Compute the output layer
        self.W_out = target @ out.T @ np.linalg.pinv(out @ out.T + self.alpha*np.identity(self.dtot))
    
    def training(self, train, target, path):
        """
        Training for the inference task.

        Parameters
        ----------
        train : train trajectory

        target : target timeseries to predict. must be of shape (dimension, timesteps)

        path : where to save the training of the Reservoir Network

        Returns
        -------
        Saves the trained feature vector at indicated path. 

        """
        self.infer(train, target=target)
        self.path = path
        np.save(self.path, self.W_out)

    def testing(self, traj, path=None):
        """
        Testing for the inference task.

        Parameters
        ----------
        traj : timeseries on which to compute the committor. Must be of shape (nb_trajectories, dimension, timesteps)

        path : if None, we re-use the same path as in training

        Returns
        -------
        The estimated committor on every trajectory  

        """
        path = self.path if path==None else path
        self.W_out = np.load(path)
        committor = np.zeros((traj.shape[0],traj.shape[2]))
        for i in range(traj.shape[0]):
            committor[i] = self.infer(traj[i])
        return committor
    
# %%
