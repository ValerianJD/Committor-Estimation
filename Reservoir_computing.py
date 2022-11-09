#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:04:50 2022

@author: valerian
"""

import numpy as np
from time import time
from tqdm import tqdm
import networkx as nx
from scipy.linalg import eigvals
from networkx.generators.random_graphs import fast_gnp_random_graph

class ResNet():
    def __init__(self, N, Min, Mout, sigma, alpha, beta, proba, spr, bias=0):
        self.N = N
        self.Min, self.Mout = Min, Mout
        self.sigma, self.alpha, self.beta = sigma, alpha, beta
        self.bias = bias
        self.proba, self.spr = proba, spr
        self.rng = np.random.default_rng()
    
    def create_reservoir(self):
        self.Win = np.zeros((self.N_runs,self.N,self.Min))
        connect = self.rng.choice(self.Min, size=(self.N_runs,self.N), replace=True)
        coeffs = self.rng.uniform(-self.sigma, self.sigma, size=(self.N_runs,self.N))
        connect = (np.repeat(np.arange(self.N_runs),self.N),np.tile(np.arange(self.N),self.N_runs),connect.flatten())
        self.Win[connect] = coeffs.flatten()
        
        self.W = np.zeros((self.N_runs,self.N,self.N)).astype(float)
        for j in range(self.N_runs):
            reservoir = fast_gnp_random_graph(self.N, self.proba, directed=True)
            self.W[j] = nx.adjacency_matrix(reservoir).toarray()
        self.W *= self.rng.uniform(-1,1,size=self.W.shape)
        radius = np.array([max(abs(eigvals(self.W[k]))) for k in range(self.N_runs)])
        self.W *= np.repeat(self.spr/radius, self.N**2).reshape(self.W.shape)
    
    def train(self, data, labels, k):
        T = data.shape[0]
        Win, W = self.Win[k], self.W[k]
        self.states = np.zeros((self.N,T+1))
        in_data = Win @ (data.T) + self.bias
        for i in range(1,T+1):
            self.states[:,i] = (1-self.alpha)*self.states[:,i-1] +\
                               self.alpha*np.tanh(in_data[:,i-1] +\
                                                  W @ self.states[:,i-1])
        self.states = self.states[:,1:]
        
        self.Wout = (labels.T @ self.states.T) @ np.linalg.inv(self.states @ self.states.T + self.beta*np.eye(self.N))
    
    def test(self, test, k, warmup=10):
        Win, W = self.Win[k], self.W[k]
        warm_state = np.zeros((self.N,test.shape[0]))
        self.test_states = np.zeros((self.N,test.shape[1]+1,test.shape[0]))
        test = np.transpose(test, (1,2,0))
        
        for i in range(1,warmup):
            warm_state = (1-self.alpha)*warm_state +\
                          self.alpha*np.tanh(Win @ test[i] + W @ warm_state + self.bias)
        
        self.test_states[:,0] = warm_state
        for i in range(1,test.shape[0]+1):
            self.test_states[:,i] = (1-self.alpha)*self.test_states[:,i-1] +\
                                    self.alpha*np.tanh(Win @ test[i-1] +\
                                                        W @ self.test_states[:,i-1] +\
                                                        self.bias)
        self.test_states = np.transpose(self.test_states[:,1:], (2,0,1))
        return self.Wout @ self.test_states    
    
    def process_reservoir(self, N_runs, train_traj, lab, test_traj):
        self.N_runs = N_runs
        self.times_train, self.times_test = np.zeros(N_runs), np.zeros(N_runs)
        pred = np.zeros((N_runs,)+test_traj.shape[:-1])
        
        self.create_reservoir()
        
        for n in tqdm(range(N_runs)):
            t0 = time()
            self.train(train_traj, lab, n)
            t1 = time()
            self.times_train[n] = t1-t0
            
            t0 = time()
            pred[n] = self.test(test_traj, n)
            t1 = time()
            self.times_test[n] = t1-t0
            
        pred[pred>1], pred[pred<0] = 1, 0
        self.times_test /= test_traj.shape[0]
        return pred

if __name__ == "__main__":

    import Datasets
    import matplotlib.pyplot as plt
    import Model_Cimatoribus, Model_wind_driven
    from Logarithm_score import LogaScore, diff_score

    C = Model_Cimatoribus.CimatoribusModel()
    idx, noise = 60, 0.15
    p_c = {"idx":idx, "noise":noise}

    sigma = 0.4
    beta = 0.04
    WD = Model_wind_driven.WindDrivenModel(1, sigma=sigma, beta=beta)
    p_wd = {"sigma":sigma, "beta":beta}

    Cima = {'model':C, 'params':p_c}
    Wind = {'model':WD, 'params':p_wd}

    models = {'C':Cima, 'WD':Wind}

    Nt = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
    nb_dataset = 1    

    to_use = 'C'
    if to_use == "C":
        # var = ["St", "Sts", "Sn", "Ss", "D", "Sd"]
        var = ["Sdiff", "Sn", "Ss", "D"]
        # var = ["Sdiff", "D"]
    else:
        # var = ["A1", "A2", "A3", "A4"]
        var = ["A3", "A2", "A1"]
    
    par = models[to_use]['params']
    
    handler =  Datasets.DataHandler(models[to_use]['model'], par, np.max(Nt), nb_dataset,
                                    "/Users/valerian/Desktop/Data/")
    # ref_comm = handler.open_data("Committor")
    # score = LogaScore(ref_comm)
    
    if to_use == "C":
        ### CIMATORIBUS PARAMS
        N = 200 
        proba = 0.15
        sigma = 1
        alpha = 0.7
        beta = 1e-9
        spr = 0.6
        sigma_b = 0.5
    elif to_use == "WD":
        ### WIND-DRIVEN PARAMS
        N = 100
        proba = 0.15
        sigma = 1
        alpha = 0.8
        beta = 1e-9
        spr = 0.8
        sigma_b = 0.3
    
    Min, Mout = len(var), 1
    
    test = handler.open_data("Trajectory", test=True)
    true_comm = handler.open_data("Committor", test=True)
    lab_test = handler.open_data("Labels", test=True)
    
    test = models[to_use]['model'].select_var(test, var)
    
    Nt = [200]
    
    train, train_comm = test[0], true_comm[0]
    score = LogaScore(train_comm)
    
    plt.plot(train_comm)
    plt.show()

    score_rl = np.zeros((10,3))
    diff_rl = np.zeros((10,3))

    esn = ResNet(N, Min, Mout, sigma, alpha, beta, proba, spr, bias=sigma_b)
    pred_test = esn.process_reservoir(10, train, train_comm, test[1:4])
    print(np.mean(esn.times_train), np.std(esn.times_train))
    print(np.mean(esn.times_test), np.std(esn.times_test))
    print()
    
    for k in range(10):
        score_rl[k] = score.get_score(test, pred_test[k], lab_test[1:4])
        diff_rl[k] = diff_score(pred_test[k], true_comm[1:4])            
    best = np.argmax(np.mean(diff_rl,axis=-1))
    print(np.mean(score_rl), np.std(score_rl))
    print(np.mean(diff_rl), np.std(diff_rl))
    print(np.mean(score_rl[best]), np.std(score_rl[best]))
    print(np.mean(diff_rl[best]), np.std(diff_rl[best]))
    
    # for nt in Nt:
    #     handler.Nt = nt
    #     train, train_comm = handler.open_data("Trajectory"), handler.open_data("Committor")
    #     train = models[to_use]['model'].select_var(train, var) 
    #     score_rl, diff_rl = np.zeros((10,100)), np.zeros((10,100))

    #     esn = ResNet(N, Min, Mout, sigma, alpha, beta, proba, spr, bias=sigma_b)
    #     pred_test = esn.process_reservoir(10, train, train_comm, test)
    #     print(np.mean(esn.times_train), np.std(esn.times_train))
    #     print(np.mean(esn.times_test), np.std(esn.times_test))
    #     print()
        
    #     for k in range(10):
    #         # score_rl[k] = score.get_score(test, pred_test[k], lab_test)
    #         diff_rl[k] = diff_score(pred_test[k], true_comm)            
    #     best = np.argmax(np.mean(diff_rl,axis=-1))
    #     print(np.mean(score_rl), np.std(score_rl))
    #     print(np.mean(diff_rl), np.std(diff_rl))
    #     print(np.mean(score_rl[best]), np.std(score_rl[best]))
    #     print(np.mean(diff_rl[best]), np.std(diff_rl[best]))

#%%



# def train_progressive(self, data, labels):
#     T = data.shape[0]
#     self.states = np.zeros((self.N,T+1))
#     in_data = self.Win @ (data.T) + self.bias
#     for i in range(1,T+1):
#         self.states[:,i] = (1-self.alpha)*self.states[:,i-1] +\
#             self.alpha*np.tanh(in_data[:,i-1] + self.W @ self.states[:,i-1] + (self.W_ofb @ self.Wout)@self.states[:,i-1])
#         self.Wout = (labels[:i].T @ self.states[:,1:i+1].T) @ np.linalg.inv(self.states[:,1:i+1] @ self.states[:,1:i+1].T + self.beta*np.eye(self.N))
#         if self.Mout == 1:
#             self.Wout = self.Wout[np.newaxis]
#     self.states = self.states[:,1:]

#     return self.Wout @ self.states

# class ResNet_predict_next():
#     def __init__(self, N, Min, Mout, sigma, alpha, beta, proba, spr, bias=0):
#         self.N = N
#         self.Min, self.Mout = Min, Mout
#         self.sigma, self.alpha, self.beta = sigma, alpha, beta
#         self.bias = bias
        
#         self.create_reservoir(proba, spr)
        
#     def create_reservoir(self, proba, spr):
#         rng = np.random.default_rng()
#         self.Win = np.zeros((self.N,self.Min))
#         connect = rng.choice(self.Min, self.N, replace=True)
#         coeffs = rng.uniform(-self.sigma, self.sigma, size=self.N)
#         for i, elt in enumerate(connect):
#             self.Win[i,elt] = coeffs[i]
            
#         reservoir = erdos_renyi_graph(self.N, proba, directed=True)
#         with warnings.catch_warnings():
#             # ignore all caught warnings
#             warnings.filterwarnings("ignore")
#             self.W = nx.adjacency_matrix(reservoir).toarray()
#         self.W = np.multiply(self.W, rng.uniform(-1,1,size=self.N**2).reshape(self.N,self.N))
#         radius = max(abs(eigvals(self.W)))
#         self.W *= spr/radius
        
#     def train(self, data, labels):
#         T = data.shape[0]
#         self.states = np.zeros((self.N,T))
#         for i in range(1,T):
#             self.states[:,i] = (1-self.alpha)*self.states[:,i-1] +\
#                 self.alpha*np.tanh(self.Win @ data[i-1] + self.W @ self.states[:,i-1] + self.bias)
                
#         self.Wout = (labels.T @ self.states.T) @ np.linalg.inv(self.states @ self.states.T + self.beta*np.eye(self.N))
        
#         if self.Min == self.Mout:
#             self.W_final = self.Win @ self.Wout + self.W
            
#         return self.Wout @ self.states
        
#     def train2(self, data, labels):
#         T = data.shape[0]
#         self.states = np.zeros((self.N,T))
#         for i in range(1,T):
#             self.states[:,i] = (1-self.alpha)*self.states[:,i-1] +\
#                 self.alpha*np.tanh(self.Win @ data[i-1] + self.W @ self.states[:,i-1] + self.bias)
#         self.states = self.states[:,1:]
        
#         self.Wout = (labels.T @ self.states.T) @ np.linalg.inv(self.states @ self.states.T + self.beta*np.eye(self.N))
        
#         if self.Min == self.Mout:
#             self.W_final = self.Win @ self.Wout + self.W
            
#         return self.Wout @ self.states

#     def test(self,test,warmup=100):
#         warm_state = np.zeros((self.N,warmup))
#         for i in range(1,warmup):
#             warm_state[:,i] = (1-self.alpha)*warm_state[:,i-1] +\
#                           self.alpha*np.tanh(self.Win @ test[i-1] + self.W @ warm_state[:,i-1] + self.bias)
              
#         # for i in range(self.Mout):
#         #     plt.plot((self.Wout @ warm_state)[i,1:])
#         #     plt.plot(test[1:warmup,i])
#         #     plt.show()

#         self.test_states = np.zeros((self.N,test.shape[0]-warmup+1))
#         self.test_states[:,0] = warm_state[:,-1]
#         for i in range(1,test.shape[0]-warmup+1):
#             self.test_states[:,i] = (1-self.alpha)*self.test_states[:,i-1] +\
#                                     self.alpha*np.tanh(self.W_final @ self.test_states[:,i-1] + self.bias)
        
#         return self.Wout @ self.test_states
    
#     def test2(self, test, warmup=100):
#         warm_state = np.zeros((self.N,warmup))
#         for i in range(1,warmup):
#             warm_state[:,i] = (1-self.alpha)*warm_state[:,i-1] +\
#                           self.alpha*np.tanh(self.Win @ test[i] + self.W @ warm_state[:,i-1] + self.bias)
        
#         # for i in range(self.Mout):
#         #     plt.plot((self.Wout @ warm_state)[i,1:])
#         #     plt.plot(test[2:warmup+1,i])
#         #     plt.show()
            
#         self.test_states = np.zeros((self.N,test.shape[0]-warmup+1))
#         self.test_states[:,0] = warm_state[:,-1]
#         for i in range(1,test.shape[0]-warmup+1):
#             self.test_states[:,i] = (1-self.alpha)*self.test_states[:,i-1] +\
#                                     self.alpha*np.tanh(self.W_final @ self.test_states[:,i-1] + self.bias)
        
#         return self.Wout @ self.test_states


# sigma = 10.0 #Variable for dx/dt
# rho = 28.0 #Variable for dy/dt
# beta = 8/3 #Variable for dz/dt
# t = 0 #Starting time
# tf = 100 #Ending time
# h = 0.01 #Step size for RK4

# def derivative(r,t):
#     x = r[0]
#     y = r[1]
#     z = r[2]
#     return np.array([sigma * (y - x), x * (rho - z) - y, (x * y) - (beta * z)])

# time = np.array([]) #Empty time array to fill for the x-axis
# x = np.array([]) #Empty array for x values
# y = np.array([]) #Empty array for y values
# z = np.array([]) #Empty array for z values
# r = np.array([1.0, 1.0, 1.0]) #Initial conditions array

# while (t <= tf ):        #Appending values to graph
#     time = np.append(time, t)
#     z = np.append(z, r[2])
#     y = np.append(y, r[1])
#     x = np.append(x, r[0])        #RK4 Step method
#     k1 = h*derivative(r,t)
#     k2 = h*derivative(r+k1/2,t+h/2)
#     k3 = h*derivative(r+k2/2,t+h/2)
#     k4 = h*derivative(r+k3,t+h)
#     r += (k1+2*k2+2*k3+k4)/6        #Updating time value with step size
#     t = t + h

# traj = np.vstack((x,y,z))[:,:-1].T
# train = traj[:5000]
# test = traj[6000:]

# # Lorenz system config
# N = 500 #Nb of states in the network
# proba = 0.3
# sigma = 0.1
# alpha = 0.95
# beta = 1e-5
# spr = 1.2
# sigma_b = 0.1

# warmup = 200

# # train, test = models[to_use]['model'].re_dim(train), models[to_use]['model'].re_dim(test)

# print(train.shape, test.shape)

# Min, Mout = train.shape[1], train.shape[1]
# esn = ResNet_predict_next(N, Min, Mout, sigma, alpha, beta, proba, spr, bias=sigma_b)
# pred = esn.train(train, train)

# pred_test = esn.test(test, warmup=warmup)
# for i in range(Mout):
#     plt.plot(test[warmup:,i])
#     plt.plot(pred_test[i])
#     plt.show()

# pred = esn.train2(train, train[1:])
# pred_test = esn.test2(test, warmup=warmup)
# for i in range(Mout):
#     plt.plot(test[warmup:,i])
#     plt.plot(pred_test[i])
#     plt.show()

#%%

# def test3(self, test):
#     self.test_states = np.zeros((self.N,test.shape[0]+1))
#     for i in range(1,test.shape[0]+1):
#         self.test_states[:,i] = (1-self.alpha)*self.test_states[:,i-1] +\
#                                 self.alpha*np.tanh(self.Win @ test[i-1] + self.W @ self.test_states[:,i-1] + self.bias)
#     self.test_states = self.test_states[:,1:]
#     return self.Wout @ self.test_states

# import torch
# from torch import nn
# Activation functions to test
# sigmoid = lambda z: 1.0/(1.0 + np.exp(-z))
# relu = lambda z: np.where(z>0, z, 0)
# tanh = lambda z: np.tanh(z)

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_uniform_(m.weight)
#         m.bias.data.fill_(0.01)
        
# class ESN_Readout(nn.Module):
#     def __init__(self, N, nb_neurons=64):
#         super(ESN_Readout, self).__init__()
#         # self.n = nb_neurons
#         self.n = N // 2
#         self.network = []
        
#         self.input = nn.Sequential(
#             nn.Linear(N, 4*self.n),
#             nn.ReLU(),
#             nn.BatchNorm1d(4*self.n))
#         self.network.append(self.input)
#         self.hidden = nn.Sequential(
#             nn.Linear(4*self.n, self.n),
#             nn.ReLU(),
#             nn.BatchNorm1d(self.n))
#         self.network.append(self.hidden)
#         self.output = nn.Linear(self.n, 2)
#         self.network.append(self.output)
        
#         for l in self.network:
#             l.apply(init_weights)
        
#     def forward(self, x):
#         for l in self.network:
#             x = l(x)
#         return x

# params = np.load("/Users/valerian/Desktop/Codes/Castellana_params.npy", allow_pickle=True).item()
# dir_data = "/Users/valerian/Desktop/Dataset_AMC/train_test_sets_from_500/"
# dir_ml = "/Users/valerian/Desktop/Dataset_AMC/results_ML/"
# idx, noise_str = 60, "0,4"
# nt = 200

# fname = "idx_"+str(idx)+"_noise_"+noise_str+"_Nt_"+str(nt)+"_1"
# name_traj = "Trajectory_" + fname + ".npy"
# name_ref = "Monte-Carlo_committor_" + fname + ".npy"
# traj = np.load(dir_data+name_traj)
# ref_comm = np.load(dir_data+name_ref)
# logscore = LogaScore(ref_comm, traj, params)

# test_set = np.load(dir_data+"Test_set_idx_"+str(idx)+"_noise_"+noise_str+".npy")
# truth_set = np.load(dir_data+"Truth_test_set_idx_"+str(idx)+"_noise_"+noise_str+".npy")

# subtrajs = []
# comm_train = []
# i = 0
# while (i+1)*5000 < traj.shape[0]:
#     subtrajs.append(traj[i*5000:(i+1)*5000])
#     comm_train.append(ref_comm[i*5000:(i+1)*5000])
#     i += 1
# subtrajs.append(traj[i*5000:])
# comm_train.append(ref_comm[i*5000:])

# # var = ["St", "Sts", "Sn", "Ss", "D"]
# var = ["Sts", "Sn", "Ss"]

# labels_train = []
# traj_train, traj_train_full = [], []
# for k,t in enumerate(subtrajs):
#     t_dim = utils.re_dim(t, params)
#     l = utils.comp_labels(t_dim, params)
#     labs = l[l>=0]
#     temp_l = np.zeros((2,labs.shape[0]))
#     for i in range(labs.shape[0]):
#         temp_l[labs[i],i] = 1
#     labels_train.append(temp_l)
#     comm_train[k] = comm_train[k][l>=0][:,np.newaxis]
#     traj_train_full.append(t[l>=0])
#     traj_train.append(select_var(var, t_dim[l>=0]))
    
# traj_conc = np.concatenate(traj_train,axis=0)
# comm_conc = np.concatenate(comm_train,axis=0)
# # labels_conc = np.concatenate(labels_train,axis=1)

# data_comm_stack = np.concatenate([traj_conc,comm_conc],axis=1)

# test_dim = utils.re_dim(test_set, params)
# traj_test_full, traj_test = [], []
# comm_test = []
# test_comm_stack = []
# labels_test = []
# for k,t in enumerate(test_set):
#     t_dim = utils.re_dim(t, params)
#     l = utils.comp_labels(t_dim, params)
#     labs = l[l>=0]
#     temp_l = np.zeros((2,labs.shape[0]))
#     for i in range(labs.shape[0]):
#         temp_l[labs[i],i] = 1
#     labels_test.append(temp_l)
#     comm_test.append(truth_set[k,(l>=0)][:,np.newaxis])
#     traj_test_full.append(t[l>=0])
#     traj_test.append(select_var(var, t_dim[l>=0]))
#     # test_comm_stack.append(np.concatenate([traj_test[-1],comm_test[-1][:,np.newaxis]],axis=1))
#     # red_t = select_var(var, all_test[i])
#     # all_red_t.append(torch.Tensor(red_t))
    
# labels_ml = np.concatenate(labels_test, axis=1)

# time_train = np.zeros((nb_runs,len(traj_train)))
# time_test = np.zeros((nb_runs,len(traj_test)))

# current_dir = os.getcwd()+"/"
# params = np.load(current_dir+"Castellana_params.npy", allow_pickle=True).item()
# name = current_dir+'equilibria/'
# all_eq_files = np.sort(os.listdir(name))[1:]

#Useful for computing the noise amplitude, normalized by volume and time for each variable
# noise_amp_base = 1e6*params["S0"]/(params["Vn"]*params["S_dim"])*params["t_dim"]*np.array([0,0,-1,params["Vn"]/params["Vs"],0])

#Loading the forcing amplitude
# init_state_nondim, target_state_nondim, unstable = utils.open_equilibria(name,all_eq_files,idx,params)

# trend_train = np.zeros((train.shape[0]-w_size,train.shape[1]))
# trend_test = np.zeros((train.shape[0]-w_size,train.shape[1]))
# for i in range(train.shape[0]-w_size):
#     trend_train[i] = np.median(train[i:i+w_size],axis=0)
#     trend_test[i] = np.median(data_test[i:i+w_size],axis=0)
    
# data = comm_train[0]
# data_test = comm_test[0]

# print(data.shape, data_test.shape)

# Min, Mout = data.shape[1], data.shape[1]
# print(Min, Mout)
# esn = ResNet(N, Min, Mout, sigma, alpha, beta, 0.2, spr, bias=sigma_b)
# # esn.train(data)
# # esn.compute_Wout(data)
# esn.train(trend_train)
# esn.compute_Wout(trend_train)
# pred = esn.get_training()

# print(pred.shape)

# plt.plot(data[:200])
# plt.plot(trend_train[:200])
# plt.plot(pred[:,:200])
# plt.xlabel("Time", fontsize=12)
# plt.ylabel("Committor", fontsize=12)
# plt.show()
# # plt.plot(data[:500])
# plt.plot(trend_train[:2000])
# plt.plot(pred[:,:2000])
# plt.xlabel("Time", fontsize=12)
# plt.ylabel("Committor", fontsize=12)
# plt.show()

# warmup = 400
# pred_test = esn.test(trend_test, warmup=warmup)

# for i in range(data.shape[1]):
#     plt.plot(trend_test[warmup:warmup+200,i])
#     plt.plot(pred_test[i,:200])
#     plt.xlabel("Time", fontsize=12)
#     plt.ylabel("Committor", fontsize=12)
#     plt.show()

#%%

    # Min, Mout = len(var), 1
    # esn = ResNet(N, Min, Mout, sigma, alpha, beta, 0.1, spr, bias=sigma_b)
    # esn.train(data)
    # esn.compute_Wout(comm_train[0])
    # pred = esn.get_training()[0]
    # pred[pred<0] = 0
    # pred[pred>1] = 1
    
    # plt.plot(comm_train[0][:1000])
    # plt.plot(pred[:1000])
    # plt.show()
    
    # pred_test = esn.test_no_pred(data_test)[0]
    # pred_test[pred_test<0] = 0
    # pred_test[pred_test>1] = 1
    # plt.plot(comm_test[0][:1000])
    # plt.plot(pred_test[:1000])
    # plt.show()
    
    
    # T = traj_conc.shape[0]
    # test_ml = np.concatenate(x_test,axis=1)

    # X_train, X_test = [], []
    # y_train, y_test = [], []
    
    # for i in range(len(traj_train)):
    #     for j in range(len(traj_train[i])):
    #         X_train.append(np.array(x_train[i][:,j]))
    #         y_train.append(np.array(labels_train[i][:,j]))
    
    # for  i in range(len(traj_test)):
    #     for j in range(len(traj_test[i])):
    #         X_test.append(np.array(x_test[i][:,j]))
    #         y_test.append(np.array(labels_test[i][:,j]))
    
    # # for i in range(T):
    # #     X_train.append(np.array(x_co[:,i]))
    # #     y_train.append(np.array(labels_conc[:,i]))
    # # for i in range(test_ml.shape[1]):
    # #     X_test.append(np.array(test_ml[:,i]))
    # #     y_test.append(np.array(labels_ml[:,i])) 

    # X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    # y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # net = DenseLayers(np.arange(N))
    # LR = 5.5e-4
    # nb_epochs = 20
    # optimizer = torch.optim.Adam(net.parameters(), amsgrad=True, lr=LR)
    # scheduler = ReduceLROnPlateau(optimizer, patience=2)
    # loss_fn = nn.CrossEntropyLoss()
    # for _ in range(nb_epochs):
    #     train(train_dataloader, net, loss_fn, optimizer)
    #     val_loss = test(test_dataloader, net, loss_fn)
    #     scheduler.step(val_loss)

    # net.eval()
    # with torch.no_grad():
    #     for i in range(len(traj_test)):
    #         pred = net(torch.Tensor(x_test[i].T)).numpy()
    #         mc_comm = softmax(pred,axis=1)[:,1]
    #         # plt.plot(truth_set[i][:1000])
    #         # plt.plot(mc_comm[:1000])
    #         # plt.show()
    #         score_ml[i,k] = logscore.get_score(traj_test_full[i], mc_comm)
    
    # # net.eval()
    # # with torch.no_grad():
    # #     pred = net(torch.Tensor(x_conc[:,:1000].T)).numpy()
    # #     mc_comm = softmax(pred,axis=1)[:,1]
    # #     plt.plot(ref_comm[:1000])
    # #     plt.plot(mc_comm[:1000])
    # #     plt.show()
    # #     score_ml[i,k] = logscore.get_score(traj_test[i], mc_comm)
    
    # print(np.mean(score_ml[:,k]), np.median(score_ml[:,k]), np.std(score_ml[:,k]))
    # print()

# print(np.mean(np.mean(score_test,axis=1)))
# print(np.median(np.mean(score_test,axis=1)))
# print(np.std(np.mean(score_test,axis=1)))
# print()
# print(np.mean(np.mean(score_test_full,axis=1)))
# print(np.median(np.mean(score_test_full,axis=1)))
# print(np.std(np.mean(score_test_full,axis=1)))
# print()
# print(np.mean(np.mean(score_ml,axis=1)))
# print(np.median(np.mean(score_ml,axis=1)))
# print(np.std(np.mean(score_ml,axis=1)))
# print()
    
# var_str = "_".join(var)
# np.save(dir_ml+"Time_train_Reservoir_computing_from_traj", time_train)
# np.save(dir_ml+"Time_test_Reservoir_computing_from_traj", time_test)
# np.save(dir_ml+"Test_scores_partial_traj_Reservoir_computing_linear_readout_var_"+var_str+"_N_"+str(N)+"_alpha_0,8_beta_1e-9_sigma_0,3", score_test)
