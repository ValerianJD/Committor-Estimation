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

    # def train_predict(self, train):
    #     d, traintime = train.shape[0], train.shape[1]-1
    #     self.dlin = d*self.k
    #     self.dtot = int(binom(self.dlin+self.p-1, self.p)) + self.dlin + 1
    #     self.combi = np.array(list(combinations_with_replacement(np.arange(self.dlin), self.p)))
        
    #     x = np.zeros((self.dlin, traintime)) #linear
    #     for i in range(self.k):
    #         x[d*i:d*(i+1), i:] = train[:,:traintime-i*self.s]

    #     out_train = np.ones((self.dtot, traintime)) #full features
    #     out_train[1:self.dlin+1] = x
    #     out_train[1+self.dlin:] = np.product(x.T[:,self.combi], axis=-1).T  
        
    #     self.W_out = (train[:,1:]-train[:,:-1]) @ out_train.T @ np.linalg.pinv(out_train @ out_train.T + self.alpha*np.identity(self.dtot))
        
    # def test_predict(self, test):
    #     d, testtime = test.shape[0], test.shape[1]-self.s*self.k+1
        
    #     out_test = np.ones(self.dtot)           # full feature vector
    #     x_test = np.zeros((self.dlin,testtime)) # linear part
        
    #     for i in range(self.k):
    #         x_test[d*i:d*(i+1), 0] = test[:,self.s*self.k-i*self.s-1]

    #     for j in range(testtime-1):
    #         out_test[1:self.dlin+1] = x_test[:,j] # shift by one for constant
    #         out_test[self.dlin+1:] = np.product(x_test[self.combi,j], axis=1)  
    #         x_test[d:self.dlin,j+1] = x_test[0:(self.dlin-d),j]
    #         x_test[0:d,j+1] = x_test[0:d,j] + self.W_out @ out_test #prediction

    #     return x_test
    
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
        
#%%

if __name__ == "__main__":

    ### LORENZ SYSTEM TESTCASE
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    sigma = 10
    beta = 8 / 3
    rho = 28
    tf = 80 #Ending time
    h = 0.025 #Step size for RK4
    t_eval = np.linspace(0,tf,int(tf//h)+1)
    
    # r = [17.67715816276679, 12.931379185960404, 43.91404334248268]
    r = [1,1,0]
    
    def lorenz(t, y):
        dy0 = sigma * (y[1] - y[0])
        dy1 = y[0] * (rho - y[2]) - y[1]
        dy2 = y[0] * y[1] - beta * y[2]
        return [dy0, dy1, dy2]
    
    lorenz_soln = solve_ivp(lorenz, (0, tf), r , t_eval=t_eval, method='RK45')
    traj = lorenz_soln.y
    
    for i in range(3):
        plt.plot(traj[i])
        plt.show()
    
    alpha = 2.5e-6
    k, s, p = 2, 1, 2
    
    esn = NewGen_ResNet(k, s, p, alpha)
    
    warmup = 400
    traintime, testtime = 800, 400
    esn.train_predict(traj[:,warmup:warmup+traintime+1])
    x = esn.test_predict(traj[:,-testtime-s*k:])
    for i in range(3):
        plt.plot(x[i])
        plt.plot(traj[i,-testtime-1:])
        plt.show()
    
    warmup = 500
    traintime, testtime = 300, 1000
    k, s, p = 4, 5, 2
    esn.k, esn.s, esn.p = k, s, p
    t0 = time()
    esn.infer(traj[::2, warmup:warmup+traintime], target=traj[1, warmup:warmup+traintime])
    t1 = time()
    z = esn.infer(traj[::2, -testtime-s*k:])
    t2 = time()
    print(t1-t0)
    print(t2-t1)
    plt.plot(traj[1,-testtime:])
    plt.plot(z[s*k:])
    plt.show()
    
#%%
    
### COMMITTOR PROBLEM
    
    import Datasets
    import Model_Cimatoribus, Model_wind_driven
    from Logarithm_score import LogaScore, diff_score
    
    C = Model_Cimatoribus.CimatoribusModel()
    idx, noise = 60, 0.4
    p_c = {"idx":idx, "noise":noise}
    
    sigma, beta = 0.4, 0.04
    WD = Model_wind_driven.WindDrivenModel(1, sigma=sigma, beta=beta)
    p_wd = {"sigma":sigma, "beta":beta}
    
    Cima = {'model':C, 'params':p_c}
    Wind = {'model':WD, 'params':p_wd}
    
    models = {'C':Cima, 'WD':Wind}
    
    Nt = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
    
    nb_dataset = 1
    
    to_use = 'C'
    if to_use == 'C':
        var = ["St", "Sts", "Sn", "Ss", "D", "Sd"]
        # var = ["St", "Sts", "Sn", "Ss", "D"]
        # var = ["Sdiff", "Sn", "Ss", "D"]
        # var = ["Sdiff", "D"]
        config = {'k':2, 's':1, 'p':4, 'alpha':1e-9}
    else:
        var = ["A1", "A2", "A3", "A4"]
        config = {'k':3, 's':2, 'p':4, 'alpha':1e-6}
    
    # name = "Mout_"+str(Mout)+"_"+"_".join(var)+"_"+\
    #        "_".join([str(e).replace(".",",") for i in range(len(config)) for e in list(config.items())[i]])
    
    par = models[to_use]['params']
    dirsave = "/Users/valerian/Desktop/Results/"+models[to_use]['model'].name+"/Reservoir/"
    handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                    "/Users/valerian/Desktop/Data/")
    ref_comm = handler.open_data("Committor")
    score = LogaScore(ref_comm)
    
    test = handler.open_data("Trajectory", test=True)
    true_comm = handler.open_data("Committor", test=True)
    lab_test = handler.open_data("Labels", test=True)
    
    test = models[to_use]['model'].select_var(test, var)
    
    score_res, diff_res = np.zeros((len(Nt),100)), np.zeros((len(Nt),test.shape[0]))
    # time_train_res, time_test_res = np.zeros(len(Nt)), np.zeros((len(Nt),test.shape[0]))
    
    for i, nt in enumerate(Nt):
        print(nt)
        handler.Nt = nt
        train, train_comm = handler.open_data("Trajectory"), handler.open_data("Committor")
        train = models[to_use]['model'].select_var(train, var)
        
        esn = NewGen_ResNet(config['k'], config['s'], config['p'], config['alpha'])
        
        t0 = time()
        esn.infer(train, target=train_comm)
        t1 = time()
        print(t1-t0)
        
        t0 = time()
        comm_est = esn.test_infer(test)
        comm_est[comm_est<0], comm_est[comm_est>1] = 0, 1
        t1 = time()
        print(t1-t0)
        score_res[i] = score.get_score(test, comm_est, lab_test)
        diff_res[i] = diff_score(comm_est, true_comm)  
        m_log, e_log = np.mean(score_res[i]), np.percentile(score_res[i], [5,95])
        m_diff, e_diff = np.mean(diff_res[i]), np.percentile(diff_res[i], [5,95])
        print(m_log, e_log[0], e_log[1])
        print(m_diff, e_diff[0], e_diff[1])
        print()
    
        comm_est = np.zeros(true_comm.shape)
        times = []
        for j in range(100):
            t0  = time()
            est = esn.infer(test[j])
            est[est<0], est[est>1] = 0, 1
            t1 = time()
            times.append(t1-t0)
            comm_est[j] = est
            # plt.plot(true_comm[j])
            # plt.plot(comm_est[j])
            # plt.show()
        print(np.sum(times))
        # print(np.mean(times), np.std(times))
        
        score_res[i] = score.get_score(test, comm_est, lab_test)
        diff_res[i] = diff_score(comm_est, true_comm)  
        m_log, e_log = np.mean(score_res[i]), np.percentile(score_res[i], [5,95])
        m_diff, e_diff = np.mean(diff_res[i]), np.percentile(diff_res[i], [5,95])
        print(m_log, e_log[0], e_log[1])
        print(m_diff, e_diff[0], e_diff[1])
        print()
        # time_train_res[i] = esn.times_train
        # time_test_res[i] = np.repeat(esn.times_test, test.shape[0]).reshape(nb_runs,-1)
