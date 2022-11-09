#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:51:40 2022

@author: valerian
"""

import numpy as np

class LogaScore():
    def __init__(self, ref_comm):
        """
        Parameters
        ----------
        ref_comm : shape = (timesteps)
            The chosen reference committor which will act as a climatology

        """
        self.ref_comm = ref_comm
        avg_tot = np.mean(self.ref_comm[(self.ref_comm>0) & (self.ref_comm<1)])
        self.climato = -avg_tot*np.log(avg_tot)
    
    def get_score(self, traj, comm, labels):
        """
        It is advised to give a batch of trajectories at once, it is more convenient

        Parameters
        ----------
        traj : shape = (Number of trajs, timesteps, dimension)
            The trajectories where the committor is evaluated
        comm : shape = (Number of trajs, timesteps)
            The estimated committors
        labels : shape = (Number of trajs, timesteps)
            List of 1 and 0, 1 if the current state leads to an off-state (or is an off-state),
            0 otherwise. 

        Returns
        -------
        list of shape (N_traj)
            The logarithm score for each input trajectory
        
        """
        
        if comm.ndim == 1:  # If we have a single trajectory
            traj, comm, labels = traj[np.newaxis], comm[np.newaxis], labels[np.newaxis]
        N = comm.shape[0]  # Number of trajectories

        all_score = np.zeros(N)
        for i in range(N):
            
            # Check whether the trajectory visits the on and off-states
            if len(np.unique(labels[i]))==1:
                # If not, return a value so that the final score is 0 (convention)
                all_score[i] = -self.climato
                continue
            
            #At the end of the trjs, some points may not be labelled, their label is -1, we must update the time series
            ind_labelled = np.nonzero(labels[i]>=0)
            t, l, c = traj[i][ind_labelled], labels[i][ind_labelled], comm[i][ind_labelled]
            #The logarithm score can only be evaluated outside out the on and off-states
            idx_correct = np.nonzero((c>0) & (c<1))[0]
            l = l[idx_correct]
            c = c[idx_correct]
            all_score[i] = np.sum(l*np.log(c)+(1-l)*np.log(1-c)) / t.shape[0]
        
        return 1 + all_score/self.climato
    
def diff_score(est, true):
    """
    Computation of the difference score. It is normalized so that 1 is the perfect
    match and 0 is the opposite. 

    Parameters
    ----------
    est : list of shape (N, timesteps)
        N estimated committors
    true : list of shape (N, timesteps)
        The N corresponding true committors

    Returns
    -------
    The list of the N difference scores 

    """
    max_dist = np.linalg.norm(np.where(true>0.5,true,1-true),axis=-1)
    return 1 - np.linalg.norm(est-true,axis=-1)/max_dist
