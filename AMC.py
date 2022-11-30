#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:35:40 2021

@author: Valerian Jacques-Dumas
"""

import numpy as np
from tqdm import tqdm
from time import time
from scipy.spatial import KDTree
from scipy.sparse.linalg import eigs

class AnaloguesMethod():
    def __init__(self, model, K, norm=False):
        """
        Parameters
        ----------
        model : The dynamical system from which we get data. It should be a class
                with the right format, it must contain the methods 
                    "is_on" -> marks the on-states
                    "is_off" -> marks the off-states
                    "select_var" -> selects a reduced number of variables
        K : Number of analogues, integer
        norm: Whether the trajectories whould be normalized by their variance
        
        """
        
        self.model = model
        self.K = K
        self.norm = norm
        
    def process_AMC(self, test, possible_analogues, var, norm=False):
        """
        Complete AMC process

        Parameters
        ----------
        test : The test trajectories, shape = (number of trajectories, timesteps, dimension)
        possible_analogues : Indices of the timesteps that can be analogues 
                             For instance, the last time of the trajectories cannot be analogues
                             If trajectories are stacked, the last timestep of each cannot be analogue
        var : Reduced number of variables to lwer the trajectories' dimension
        norm : Should the trajectories be normalized by their variance (default=False)

        Returns
        -------
        comm_knn : Committor of each test trajectory

        """
        
        N = test.shape[0]
        self.all_times = np.zeros(N)
        comm = np.zeros(test.shape[:-1])
        
        # Data pre-processing: reduce number of variables (if applicable), find the
        #on and off-states
        samples = self.model.select_var(test,var)
        self.nb_samples = test.shape[1]
        on, off = self.model.is_on(test), self.model.is_off(test)
        
        self.poss_analogues = possible_analogues
        self.norm = norm
        
        for i in tqdm(range(N)):
            self.samples, self.on, self.off = samples[i], on[i], off[i]
            t0 = time()
            comm[i] = self.AMC()
            t1 = time()
            self.all_times[i] = t1-t0
        
        return comm
    
    def process_KNN(self, train, test, possible_analogues, var, norm=False):
        """
        Complete KNN process

        Parameters
        ----------
        train : The train trajectory, shape = (timesteps, dimension)
        test : The test trajectories, shape = (number of trajectories, timesteps, dimension)
        possible_analogues : Indices of the timesteps that can be analogues 
                             For instance, the last time of the trajectories cannot be analogues
                             If trajectories are stacked, the last timestep of each cannot be analogue
        var : Reduced number of variables to lwer the trajectories' dimension
        norm : Should the trajectories be normalized by their variance (default=False)

        Returns
        -------
        comm_knn : Committor of each test trajectory

        """
        
        N = test.shape[0]
        self.time_amc, self.time_knn = 0, np.zeros(N)
        self.norm = norm
        
        # Training phase with the application of AMC
        # Reduce the trajectory dimension (if applicable), find the on and off-states
        self.samples = self.model.select_var(train, var)
        self.nb_samples = train.shape[0]
        self.on, self.off = self.model.is_on(train), self.model.is_off(train)
        self.poss_analogues = possible_analogues
        t0_amc = time()
        _ = self.AMC()
        t1_amc = time()
        self.time_amc = t1_amc-t0_amc
        
        # Testing phase, application of KNN
        samples = self.model.select_var(test, var)
        self.nb_samples = test.shape[1]
        comm_knn = np.zeros(test.shape[:-1])
        for i in tqdm(range(N)):
            self.samples = samples[i]
            t0_knn = time()
            comm_knn[i] = self.KNN()
            t1_knn = time()
            self.time_knn[i] = t1_knn - t0_knn
        
        return comm_knn
        
    def AMC(self):
        
        # The index of states that cannot be analogues
        ind_not_analogues = list(set(list(np.arange(self.nb_samples))) - set(self.poss_analogues))
        
        # In both cases, build the K-d tree to search for the analogues
        if self.norm:
            var_samples = np.var(self.samples,axis=0)
            norm_samples = self.samples.copy() / var_samples
            norm_analogues = self.samples[self.poss_analogues] / var_samples
            tree = KDTree(norm_analogues)
            _, raw_analogues = tree.query(norm_samples, k=self.K+1, workers=6)
        else:
            tree = KDTree(self.samples[self.poss_analogues])
            _, raw_analogues = tree.query(self.samples, k=self.K+1, workers=6)
            
        # Build the list of analogues, paying attention to states that cannot be analogues
        analogues = np.zeros((self.nb_samples,self.K),dtype=int)
        analogues[self.poss_analogues] = raw_analogues[self.poss_analogues,1:] #Remove sample itself from list of its analogues
        analogues[ind_not_analogues] = raw_analogues[ind_not_analogues,:-1] #Last sample is not in list of possible analogues
    
        #Definition of the attractors with the conditions for on and off-states
        idx_states_A, idx_states_B = np.nonzero(self.on)[0], np.nonzero(self.off)[0]
    
        #Define all states belonging either to A, B or none
        all_others = np.nonzero(~(self.on|self.off))[0]
        nb_others = all_others.shape[0]
        
        #Construction of the transition matrix from the Markov chain
        transition = np.zeros((nb_others,self.nb_samples))
        analogues = analogues[all_others]+1
        idx_trans = (np.repeat(np.arange(nb_others),self.K),analogues.flatten())
        transition[idx_trans] = 1/self.K

        #Define the new transition matrix with the global states A and B 
        G_tilde = np.zeros((nb_others+2, nb_others+2))
        G_tilde[:nb_others,:nb_others] = transition[:,all_others].copy()
        G_tilde[:nb_others,-2] = np.sum(transition[:,idx_states_A],axis=1)
        G_tilde[:nb_others,-1] = np.sum(transition[:,idx_states_B],axis=1)
        G_tilde[-2,-2], G_tilde[-1,-1] = 1, 1
    
        #Computing the leading eigenvectors of G_tilde (sparse)
        _, vectors = eigs(G_tilde, k=2, which='LM')
        v1, v2 = np.real(vectors[:,0]), np.real(vectors[:,1])
            
        #We now that the committor is a linear combination of both eigenvectors
        #The imposed conditions are: q[states_A] = 0 and q[states_B] = 1
        alpha = v2[-2]/(v1[-1]*v2[-2]-v2[-1]*v1[-2])
        beta = v1[-2]/(v1[-2]*v2[-1] - v1[-1]*v2[-2])    
        
        committor = np.zeros(self.nb_samples)
        committor[idx_states_B] = 1
        committor[all_others] = (alpha*v1 + beta*v2)[:-2] #last 2 values are always 0 and 1 for A and B
    
        # If we are to re-use the committor with KNN, we only save the states 
        #where it has been correctly estimated, with values in [0,1]
        idx = np.nonzero((committor>=0)&(committor<=1))
        self.committor = committor[idx]
        self.states = self.samples[idx]

        return committor
    
    def KNN(self):
        
        # Look for the K closest neighbours
        # Easier than for AMC because the state itself cannot be in the list of its
        #closest neighbours
        if self.norm:
            var_states = np.var(self.states,axis=0)
            norm_states = self.states.copy() / var_states
            tree = KDTree(norm_states)
            _, neighbours = tree.query(self.samples/var_states, k=self.K, workers=6)
        else:
            tree = KDTree(self.states)
            _, neighbours = tree.query(self.samples, k=self.K, workers=6)
            
        # Fast computation of the averaged committor on the neighbours of each point
        return np.mean(self.committor[neighbours], axis=1)