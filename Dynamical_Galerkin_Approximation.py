#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 18:20:58 2022

@author: valerian
"""

print("salut")

import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class DynGalApp():
    def __init__(self, M=100, k0=7, d=0.45, k_eps0=15, k_eps=-15):
        self.M = M    #nb_modes
        self.k0 = k0  #nb of neighbours in the bandwidth vector
        
        # Definition of the hyperparameters
        self.d = d if d is not None else None
        self.eps0 = 2**k_eps0 if k_eps0 is not None else None
        self.epsilon = 2**k_eps if k_eps is not None else None
        
        # True if d, eps_0 and eps are prescribed
        # Otherwise, indicates that we have to estimate them
        self.none = True if (self.d is None or self.eps0 is None or self.epsilon is None) else False
    
    def process_DGA(self, model, traj, process="train_in_test", label=None, N_transi=10,
                    N_points=None, train_traj=None, var=None):
        """
        The complete process of applying DGA to a set of trajectories. We return 
        the committor estimate on each of them.

        Parameters
        ----------
        model : The dynamical system from which we get data. It should be a class
                with the right format, it must contain the methods 
                    "is_transi" -> marks the transition states
                    "is_off" -> marks the off-states
                    "select_var" -> selects a reduced number of variables

        traj : The trajectories on which to estimate the committor
               shape = (number of trajectories (if applicable), timesteps, dimension)
               
        process : The DGA train/test process to apply, choose among three:
                      "train_in_test" (default) -> only provide test trajectories,
            training is done on its first N_transi transitions or first N_points points,
            testing is done on the rest. label must be provided
                      "no_train" -> no pre-training of the modes is done, from-scratch-computation
            on EVERY test trajectory
                      "full_train" -> train_traj must be provided as a training set. 
            Training will be performed on it before the modes are applied to traj
            
        label : Labels of the traj, more precisely:
                    label[i,j] = 1 if timestep j of traj[i] is an off-state or leads to an off-state
                    label[i,j] = 0 if timestep j of traj[i] is an on-state or leads to an on-state
        N_transi : The number of transitions on which to do the training in the 
                    case of "train_in_test"
        N_points : The number of points on which to do the training in the 
                    case of "train_in_test". If both N_transi and N_points are provided,
                    N_points will be preferred
        train_traj : In the case of "full_train", the train trajectory. 
                     shape = (timesteps, dimension)
        var : If provided, the reduced set of model variables on which to perform DGA

        Returns
        -------
        comm : The estimated committor for every trajectory in traj
               shape = (number of trajectories in traj, timesteps)

        """
        
        self.process = process
        
        # Pre-processing of the trajectories: find off-states, transition states,
        #and (if applicable) reduce the dimension
        traj = traj if traj.ndim==3 else traj[np.newaxis]
        N = traj.shape[0]
        r = model.is_off(traj).astype(int)
        t = model.is_transi(traj).astype(int)
        if var is not None:
            traj = model.select_var(traj, var)
        if train_traj is not None:
            t_train = model.is_transi(train_traj).astype(int)
            if var is not None:
                train_traj = model.select_var(train_traj, var)
        
        comm = np.zeros(traj.shape[:-1])
        self.times_test = np.zeros(N)
        
        if process == "train_in_test":
            self.times_train = np.zeros(N)

            # Separation between train and test parts of the trajectories, based
            #on N_points or N_transi
            if N_points is None:
                all_transi = np.nonzero(label[:,:-1]-label[:,1:])
                idx = np.unique(all_transi[0], return_index=True)[1]+N_transi-1
                stop = all_transi[1][idx]
            else:
                stop = np.repeat(N_points,N)
            
            for i in tqdm(range(N)):
                train, test = traj[i,:stop[i]], traj[i, stop[i]:]
                
                # Training with fast computation of the distance matrix
                self.train = True
                t0 = time()
                mat_traj = np.repeat(train[np.newaxis],traj.shape[1],axis=0) - traj[i][:,np.newaxis]
                self.distances = np.linalg.norm(mat_traj, axis=2)**2
                comm[i,:stop[i]] = self.DGA(train, r=r[i,:stop[i]], idx_transi=t[i,:stop[i]])
                t1 = time()
                self.times_train[i] = t1-t0
                
                # Testing phase
                self.train = False
                t0 = time()
                comm[i,stop[i]:] = self.DGA(test, r=r[i,stop[i]:], idx_transi=t[i,stop[i]:])
                t1 = time()
                self.times_test[i] = t1-t0
                
        elif process == "no_train":
            
            # Testing phase. self.train=True for convenience, we still need to 
            #compute useful matrices and possibly estimate hyperparameters as during
            #a training phase
            self.train = True
            for i in tqdm(range(N)):
                t0 = time()
                comm[i] = self.DGA(traj[i], r=r[i], idx_transi=t[i])
                t1 = time()
                self.times_test[i] = t1-t0
                
        elif process == "full_train":
            
            # Training phase
            self.train = True
            t0 = time()
            self.DGA(train_traj, idx_transi=t_train)
            t1 = time()
            self.time_train = t1-t0
            
            # Testing phase
            self.train = False
            for i in tqdm(range(N)):
                t0 = time()
                comm[i] = self.DGA(traj[i], r=r[i], idx_transi=t[i])
                t1 = time()
                self.times_test[i] = t1-t0
        
        #Estimates of the committor may take values outside of [0,1]
        comm[comm<0], comm[comm>1] = 0, 1
        return comm
    
    def bandwidth(self, l):
        """
        Computation of the bandwidth vector and matrix

        Parameters
        ----------
        l : trajectory of shape = (timesteps, dimension)

        Returns
        -------
        self.mat_bw the beandwidth matrix

        """
        delay = 1 if np.all(len(l)==len(self.traj)) else 0
        tree = KDTree(l)    
        dist, _ = tree.query(self.traj, k=self.k0+delay, workers=6)
        self.all_bw = np.sum(dist**2, axis=1) / self.k0
        if self.train:
            self.mat_bw = 2*self.all_bw[:,np.newaxis]*self.all_bw[np.newaxis]
            self.all_bw_pool, self.mat_bw_pool = np.copy(self.all_bw), np.copy(self.mat_bw)
        else:
            self.mat_bw = 2*self.all_bw[:,np.newaxis]*self.all_bw_pool[np.newaxis]

    def density(self, D):
        """
        Computation of the density vector and density matrix.

        Parameters
        ----------
        D : Distance matrix 
            shape = (len(train_traj), len(trai_traj)) if self.train == 1
            shape = (len(test_traj), len(train_traj)) if self.train == 0

        Returns
        -------
        self.mat_dens the density matrix 

        """
        K0 = np.exp( - D / (self.eps0*self.mat_bw))
        self.all_dens = (2*np.pi*self.eps0)**(-self.d/2) * np.sum(K0, axis=1) / (len(self.pool)*self.all_bw**self.d)
        if self.train:
            self.mat_dens = (self.all_dens[:,np.newaxis]*self.all_dens[np.newaxis])**(-1/self.d)
            self.all_dens_pool, self.mat_dens_pool = np.copy(self.all_dens), np.copy(self.mat_dens)
        else:
            self.mat_dens = (self.all_dens[:,np.newaxis]*self.all_dens_pool[np.newaxis])**(-1/self.d)
    
    def kernel(self, D):
        self.K = np.exp( - D / (self.epsilon*self.mat_dens))
    
    def sum_k(self, m):
        """
        Sum of the exponential of the matrix m

        """
        return np.sum(np.exp(m))
    
    def comp_k(self, mat, D):
        """
        Find the index of the maximmum of a function of mat
        epsilon_0, d and epsilon are computed from it
        D is always the distance matrix (see help(prepare))

        Returns
        -------
        d
        k such that epsilon_0 = 2**k or epsilon = 2**k

        """
        
        guess = -40
        frac_sum = self.sum_k(mat/2.**(guess+1)) / self.sum_k(mat/2.**guess)
        while np.isclose(frac_sum,1):
            guess += 10
            frac_sum = self.sum_k(mat/2.**(guess+1)) / self.sum_k(mat/2.**guess)
        guess += 10
        sum_plus, sum_max = self.sum_k(mat/2.**(guess+1)), self.sum_k(mat/2.**(guess+2))
        frac_sum, new_frac = sum_plus / self.sum_k(mat/2.**guess), sum_max / sum_plus
        while frac_sum < new_frac:
            guess += 1
            frac_sum, new_frac = new_frac, self.sum_k(mat/2.**(guess+2)) / sum_max
            sum_max *= new_frac

        return 2*np.log(frac_sum)/np.log(2), guess
    
    def prepare(self, D):
        """
        Estimation of the hyperparameters

        Parameters
        ----------
        D : Distance matrix 
            shape = (len(train_traj), len(trai_traj)) if self.train == 1
            shape = (len(test_traj), len(train_traj)) if self.train == 0

        Returns
        -------
        Defines hyperparameters and useful matrices as internal variables

        """
        
        # Bandwidh vector and matrix
        self.bandwidth(self.traj)
        
        # Estimate of epsilon_0 and d
        self.d, g = self.comp_k( - D / self.mat_bw, D)
        self.eps0 = 2.**g
        
        # Density vector and matrix
        self.density(D)
        
        # Estimate of epsilon
        _, g = self.comp_k( - D / self.mat_dens, D)
        self.epsilon = 2.**g
        
        print(self.d, self.eps0, self.epsilon)
    
    def kernel_direct(self, l, D):
        """
        Summarized computation of the kernel in the case of the process "train_in_test"
        where the distance matrix can be more easily obtained.

        Parameters
        ----------
        l : Trajectory on which the kernel is computed (train if self.train==1, 
                                                        test otherwise)
        D : Distance matrix
            shape = (len(train_traj), len(trai_traj)) if self.train == 1
            shape = (len(test_traj), len(train_traj)) if self.train == 0

        Returns
        -------
        self.K = kernel useful for computing the transition matrix

        """
        
        n = l.shape[0]
        
        # Bandwidth vector, KDTree returns distances of k closest neighbours
        # self.train = 1 or 0 -> if we look for the closest neighbours in the test set itself,
        #we want the neighbours [1,k] (the closest neghbour is always oneself if present)
        tree = KDTree(l)
        dist, _ = tree.query(self.traj, k=self.k0+self.train, workers=6)
        all_bw = np.sum(dist[:,self.train:]**2, axis=1) / self.k0
        
        # Bandwidth matrix
        if self.train:
            mat_bw = all_bw[:,np.newaxis]@all_bw[np.newaxis]
            self.all_bw_pool = np.copy(all_bw)
        else:
            mat_bw = all_bw[:,np.newaxis]@self.all_bw_pool[np.newaxis]
        
        # K0 vector and matrix for the density matrix
        K0 = np.exp( - D / (2*self.eps0*mat_bw))
        vec_K0 = np.sum(K0, axis=1)**(-1/self.d)
        if self.train:
            mult_K0 = vec_K0[:,np.newaxis]@vec_K0[np.newaxis]
            self.vec_K0_pool = np.copy(vec_K0)
        else:
            mult_K0 = vec_K0[:,np.newaxis]@self.vec_K0_pool[np.newaxis]
        
        # Density matrix and final kernel we want
        mat_dens = 2*np.pi*self.eps0*(n**(2/self.d))*mat_bw*mult_K0
        self.K = np.exp( - D / (self.epsilon*mat_dens))
        
    def DGA(self, traj, r=None, idx_transi=None): 
        """
        Actual DGA process: computation of the modes and committor estimation

        Parameters
        ----------
        traj : The trajectory on which to estimate the committor 
                shape = (timesteps, dimension)
        r : Indicator of the off-states of the trajectory (r[i]=1 if traj[i] is off
                                                           r[i]=0 otherwise)
        idx_transi : Indices of the transition states
                    
        Returns
        -------
        estim_comm : the committor estimate

        """
        
        self.traj = traj
        
        if self.process == "train_in_test":
            # In this case, we already computed the distance matrix with a faster process
            if self.train:
                self.pool = traj #Keeps track of the test trajectory, we will need it
                dist_mat = self.distances[:len(traj)]
            else:
                dist_mat = self.distances[self.distances.shape[1]:]
        
        else:
            # Computation of the distance matrix
            if self.train:
                mat_traj = np.repeat(self.traj[np.newaxis],traj.shape[0],axis=0)
                self.pool = traj
            else:
                mat_traj = np.repeat(self.pool[np.newaxis],traj.shape[0],axis=0)
            mat_traj -= traj[:,np.newaxis]
            dist_mat = np.linalg.norm(mat_traj, axis=2)**2
                
        if self.none:
            # The hyperparameters are not prescribed
            # We need to estimate them so as to compute the kernel for the transition matrix
            
            if self.train:
                #Computation of the hyperparameters
                # At the same time, we make the bandwidth and density matrices for the train set
                self.prepare(dist_mat)
            else:
                # In test traj, we use the same hyperparams
                # We still need to re-compute bandwith and density matrices
                self.bandwidth(self.pool)
                self.density(dist_mat)
                
            self.kernel(dist_mat)
        
        else:
            # Hyperparameters are prescribed
            # A single function contains the kernel-computation process
            self.kernel_direct(self.pool, dist_mat)
        
        if self.train:
            # Construction of the transition matrix 
            # sub_P is the submatrix defined on the transitions
            # The modes are the eigenvectors of sub_P and 0 elsewhere
            P_dmap = (self.K / np.sum(self.K,axis=1)).T
            idx_transi = np.nonzero(idx_transi)[0]
            sub_P = P_dmap[idx_transi][:,idx_transi]
            
            # Compute the eigenvectors
            vals, vecs = np.linalg.eig(sub_P)
            # We keep the M leading real eigenvalues
            idx_real = np.isreal(vals)
            vals, vecs = vals[idx_real], vecs[:,idx_real]
            vals = np.abs(vals)
            # Eigenvalues are not ordered, we need to order them (descending order)
            if len(vals)>self.M:
                idx_modes = np.argsort(vals)[:(len(vals)-1-self.M):-1]
            else:
                idx_modes = np.argsort(vals)[::-1]
                
            self.modes = np.zeros(( np.min([self.M,len(vals)]), len(self.traj) ))
            self.vals = vals[idx_modes]
            self.modes[:,idx_transi] = np.real(vecs[:,idx_modes].T)     #Shape (M, len(transi))
            
            if self.process != "full_train":
                # If process==train_in_test or process==no_train, we want the 
                #committor on the training set
                
                ### COMPUTE PRODUCTS
                # Left-hand term of the equation
                L = np.dot(self.modes[:,:-1], (self.modes[:,1:]-self.modes[:,:-1]).T)
                # Right-hand term
                r_vec = np.dot(self.modes[:,:-1], (r[1:]-r[:-1]))
            
                ### DEDUCE COMMITTOR
                try:
                    # In some rare cases, this matrix operation may be ill-defined
                    a = np.linalg.solve(L, -r_vec) 
                    estim_comm = r + a @ self.modes
                except np.linalg.LinAlgError:
                    estim_comm = np.zeros(len(self.traj))
                return estim_comm

        else:
            # Extension of the modes            
            modes_extend = np.zeros((self.modes.shape[0], len(self.traj)))
            idx_transi = np.nonzero(idx_transi)[0]
            numer = np.dot(self.K[idx_transi], self.modes.T).T
            denom = np.dot(self.vals[:,np.newaxis], np.sum(self.K[idx_transi],axis=1)[np.newaxis])
            modes_extend[:,idx_transi] = numer / denom
    
            ### COMPUTE PRODUCTS
            L = np.dot(modes_extend[:,:-1], (modes_extend[:,1:]-modes_extend[:,:-1]).T)
            r_vec = np.dot(modes_extend[:,:-1], (r[1:]-r[:-1]))
    
            ### DEDUCE COMMITTOR
            try:
                a = np.linalg.solve(L, -r_vec) 
                estim_comm = r + a @ modes_extend
            except np.linalg.LinAlgError:
                estim_comm = np.zeros(len(self.traj))
            return estim_comm