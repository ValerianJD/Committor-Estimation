#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:44:59 2022

@author: valerian
"""

import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

class MonteCarlo():
    """
    Class handling the Monte-Carlo estimation of the committor function
    
    """
    def __init__(self, model, model_params, N):
        """
        model is the dynamical systems model
        model_params are the required parameters
        N is the number of Monte-Carlo runs to perform

        """
        self.model = model
        self.params = model_params
        self.N = N
        self.pool = mp.Pool(6) #The number of workers for the parallel computation
    
    def committor_estim(self,traj):
        L = traj.shape[0]
        committor = np.zeros(L) #The on-states already have a committor of 0
        off = self.model.is_off(traj) #The of-states have a committor of 1
        committor[off] = 1
        others = self.model.is_transi(traj) #We actually compute the committor on all the other states
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


class DataHandler():
    def __init__(self, model, params:dict, Nt:int, nb_dataset, dirsave):
        """
        Parameters
        ----------
        model : Dynamical System Class
            Required variables: name, capped
            Required methods: is_on, is_off, trajectory, check_traj_fine
        params : dict
            The parameters of the model to compute the trajectories
        Nt : int
            The maximum number of transitions the new dataset should contain
        nb_dataset : int
            We can make several datasets for the same parameters. It just helps
            distinguishing them
        dirsave : str
            The directory wher to save the dataset

        """    
        self.model = model
        self.params = params
        self.Nt = Nt
        self.nb_dataset = nb_dataset
        
        self.dirsave = dirsave
        if self.dirsave[-1] != "/":
            self.dirsave += "/"
        
    def name(self):
        """
        We create the name of the dataset, containing the model's name and the parameters

        """
        self.filename = self.model.name+"_"
        for k,p in self.params.items():
            self.filename += k+"_"+str(p).replace(".", ",")+"_"
        self.filename += str(self.nb_dataset)
        
    def make_train_set(self, tmax, ic, nb_mc_runs):
        """
        Compute a training set containing Nt transitions
        We compute it by concatenating trajectories of time duration tmax

        Parameters
        ----------
        tmax : int (or float)
            The time length of each sub-trajectory
        ic : list of float
            The initial condition of the first sub-trajectory. It is often close to the steady on-state
            If the length of the trajectories in the model is capped, then ic is 
            the initial condition of every sub-trajectory
        nb_mc_runs : int
            The number of runs used in the Monte-Carlo estimation of the committor 
            once the training trajectory has been computed.

        """
        # transi is the indices of the transition states and labels=1 if the state leads to an off-state
        all_traj, all_labels, all_transi = [], [], []
        count, all_count = 0, [] # Counts the number of transitions, all_count lists all values of count
        
        # We keep computing the trajectory (by bits of size tmax) as long as
        #we haven't reached the number of transitions Nt
        while count < self.Nt:
            traj = self.model.trajectory(1, tmax, *self.params.values(), init_state=ic)[0]
            N_steps = traj.shape[0]
            transi, labels = self.compute_transi_labels(traj)
            add_count = len(transi)
            
            if not self.model.capped: #Means there is no max traj length
                # Update of the initial condition: it will be the last computed state
                len_ic = 1 if (ic is None or ic.ndim==1) else ic.shape[0]
                ic = traj[-len_ic:]
            
            # If we have too many transitionsm we cut the last trajectory
            if count+add_count > self.Nt:
                # transi[self.Nt-count-1][-1] is the last element of the Nt_th transition
                # We add +2 so that the last state of the traj is either an on or off-state
                #to be able to label this last transition
                traj = traj[:transi[self.Nt-count-1][-1]+2]
                labels = labels[:transi[self.Nt-count-1][-1]+2]
                transi = transi[:self.Nt-count]
                add_count = self.Nt-count  #len(transi)
            
            all_count.append(add_count)
            count += add_count

            if self.model.capped:
                all_traj.append(traj)
                all_labels.append(labels)
                # Update the indices of transitions so that they do not restart at 0
                #at every new trajectory
                for i, t in enumerate(transi):
                    transi[i] = [e + len(all_count)*N_steps for e in t]
                all_transi.append(transi)
            else:
                # From the 2nd trajectory computed, the init state is the last state of the former traj
                # We mustn't repeat that state in the data
                all_traj += list(traj[int(len(all_traj)>0):])
            
        if self.model.capped:
            all_traj = np.concatenate(all_traj)
            all_labels = np.concatenate(all_labels)
            all_transi = np.concatenate(all_transi, axis=0, dtype=object)
        else:
            all_traj = np.array(all_traj)
            all_transi, all_labels = self.compute_transi_labels(all_traj)
            all_traj = all_traj[:all_transi[self.Nt-1][-1]+2]
            all_labels = all_labels[:all_transi[self.Nt-1][-1]+2]
            all_transi = all_transi[:self.Nt]
          
        print(all_traj.shape)
        data = {"Trajectory":all_traj, "Transitions":all_transi, "Labels":all_labels}
        self.name() #Update filename if necessary
        np.save(self.dirsave+self.filename, data)
        
        # Compute the Monte-Carlo estimation of the committor function
        # m = MonteCarlo(self.model, list(self.params.values()), nb_mc_runs)
        # ref_comm = m.committor_estim(all_traj)
        # data["Committor"] = ref_comm
        # np.save(self.dirsave+self.filename, data)
        
    def compute_transi_labels(self, traj):
        """
        Automatic computation of the transitions and labels in any trajectory of a model
        Since a state is labelled according to what follows in the trajectory, here
        we look at the trajectory in reverse order

        Returns
        -------
        A list of transitions: a list of lists of indices. Each sublist is a transition
        and contains the indices of all the states forming that transition.
        
        The labels of the trajectory, with the following convention:
            labels[i] = 1 if the state at the index i is an off-state or will lead 
          to an off-state before an on-state
            labels[i] = 0 if the state i is an on-state or leads to an on-state
            labels[i] = -1 for the last states at the end of the trajectory that do not 
          lead to an on or off-state and that cannot be labelled

        """
        N_steps = traj.shape[0]
        on, off = self.model.is_on(traj), self.model.is_off(traj)
        
        # Indicator array and label array initialized this way for convenience
        # In the indicator, on-states=1 and off-state=0
        indic = np.zeros(N_steps,dtype=int) + (on+2*off)*np.ones(N_steps,dtype=int)
        labels = -np.ones(N_steps,dtype=int)
        
        idx_transi = []
        mem_stop = []
        for i in range(N_steps-2,-1,-1):
            if indic[i]==0 and indic[i+1]>0:   #state i is a transition and i+1 is not
                mem_stop = [i, indic[i+1]]
            #We check if, during a transition, i is either on or off and i+1 is not
            # It means we reached the beginning of the transition (we look at the traj in reverse order)
            elif len(mem_stop)>0 and indic[i]>0 and indic[i+1]==0:
                labels[i+1:mem_stop[0]+1] += mem_stop[1]
                # A transition is from on to off or off to on
                # So if indic[i]==mem_stop[1], it is not a transition
                if indic[i] != mem_stop[1]:
                    idx_transi.append(np.arange(i+1,mem_stop[0]+1))
                mem_stop = []
        #In case the trajectory start out of the on or off-zone
        if len(mem_stop)>0:
            labels[:mem_stop[0]+1] = mem_stop[1]
        return idx_transi[::-1], labels+indic  

    def make_test_set(self, N_traj, tmax, ic, nb_mc_runs):
        """
        Computation of N_traj test trajectories.
        Unlike the train trajectories, they have a fixed length tmax, number of transitions doesn't matter
        
        Parameters
        ----------
        N_traj : int
            Number of trajectories to generate.
        tmax : int (or float)
            Total time duration of each test trajectory.
        ic : list of floats
            The initial condition of each trajectory.
        nb_mc_runs : int
            The number of Monte-Carlo runs in the committor estimate.
        start : int
            Defaults at 0. The test set is saved every time a new trajectory is computed. 
            So if it takes too long, you can stop the computation and restart it later by 
            setting 'start' to where it stopped. 

        """
        self.name()
        
        # We check if the computation had not been started already
        if not "Test_set_"+self.filename+".npy" in os.listdir(self.dirsave):
            traj = self.model.trajectory(N_traj, tmax, *self.params.values(), init_state=ic)
            lab, comm = np.zeros(traj.shape[:2]), -np.ones(traj.shape[:2])
            # "start" Defaults at 0. The test set is saved every time a new trajectory is computed. 
            # So if it takes too long, you can stop the computation and restart it later
            # The correct value of "start" will be automatically determined (see else statement)
            start = 0
            for i in range(N_traj):
                # We check if the computed traj meet potential criteria defined in the model
                #(i.e. make sure the traj do not reach a certain zone of the phase space)
                while not self.model.check_traj_fine(traj[i]):
                    traj[i] = self.model.trajectory(1, tmax, *self.params.values(), init_state=ic)
                _, lab[i] = self.compute_transi_labels(traj[i])
            data = {"Trajectory": traj, "Labels":lab}
            np.save(self.dirsave+"Test_set_"+self.filename, data)
        
        else:
            data = np.load(self.dirsave+"Test_set_"+self.filename+".npy", allow_pickle=True).item()
            traj = data["Trajectory"]
            if 'Committor' in data.keys():
                comm = data['Committor']
            else:
                comm = -np.ones(traj.shape[:2])
            # comm = data['Committor'] if start>0 else 
            # Where to restart the computation of the committor: first time that the committor contains -1 (initialization)
            start = np.nonzero(np.any(comm==-1,axis=1))[0][0] 
            
        # Monte-Carlo committor estimation
        # m = MonteCarlo(self.model, list(self.params.values()), nb_mc_runs)
        # for i in tqdm(range(start,N_traj)):
        #     plt.plot(traj[i,:,0]+traj[i,:,2],traj[i,:,1])
        #     plt.scatter(self.model.up[0]+self.model.up[2], self.model.up[1],c="g",s=30)
        #     plt.scatter(self.model.down[0]+self.model.down[2], self.model.down[1],c="r",s=30)
        #     plt.show()
        #     comm[i] = m.committor_estim(traj[i])
        #     plt.plot(comm[i])
        #     plt.show()
        #     data["Committor"] = comm
        #     np.save(self.dirsave+"Test_set_"+self.filename, data)
        # m.pool.close()
        
    def open_data(self, name, test=False):
        """
        Load data, either train or test.

        Parameters
        ----------
        test : False to get the training set
        
        name : "Trajectory", "Committor", "Labels"
               If test=False, the list of transitions "Transitions" is also available
            DESCRIPTION.

        """
        self.name()
        dico = np.load(self.dirsave+test*"Test_set_"+self.filename+".npy", allow_pickle=True).item()
        if test:
            return dico[name]
        last = dico["Transitions"][self.Nt-1][-1]+2
        return dico[name][:last]  
    
    
    
if __name__ == "__main__":   
    from Model_wind_driven import WindDrivenModel
    from Model_Cimatoribus import CimatoribusModel
    
    model = "WD"
    train, test = False, True
    if model == "C":
        m = CimatoribusModel()
        p = {"idx":60, "noise":0.15}
    elif model == "WD":
        sigma, beta = 0.45, 0.198
        m = WindDrivenModel(1,sigma=sigma,beta=beta)
        p = {"sigma":sigma, "beta":beta}
    
    Nt = 500
    data = DataHandler(m, p, Nt, 1, "/Users/valerian/Desktop/Data/")
    if train:
        data.make_train_set(5000, None, 1000)
    if test:
        data.make_test_set(100, 5000, None, 1000)