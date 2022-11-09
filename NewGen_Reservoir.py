#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:58:02 2022

@author: valerian
"""

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
        if d>time:
            traj = traj.T
            d, time = time, d

        # Only compute dlin, dtot and combi in train mode
        if target is not None:
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
        
    def process_infer(self, train, target, test):
        """
        The total inference process, encompassing training and testing phase

        Parameters
        ----------
        train : train trajectory

        target : target timeseries to predict. must be of shape (dimension, timesteps)
            
        test : test trajectories

        Returns
        -------
        pred : The estimated variables along the test trajectories

        """
        self.times_test = np.zeros(test.shape[0])
        pred = np.zeros((test.shape[:-1]))
        
        # Training phase
        t0 = time()
        self.infer(train, target=target)
        t1 = time()
        self.time_train = t1-t0
        
        # Testing phase
        for i in range(test.shape[0]):   
            t0 = time()
            pred[i] = self.infer(test[i])
            t1 = time()
            self.times_test[i] = t1-t0
            
        return pred

    