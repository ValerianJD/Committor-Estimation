#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:00:52 2022

@author: valerian
"""

from time import time
import numpy as np
from tqdm import tqdm
import copy

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import Architectures
from scipy.special import softmax

class NeuralNetworks():
    def __init__(self, nb_epochs, nb_runs, dir_ml, batch_size=32, LR=1e-4, loss_fn=nn.CrossEntropyLoss(), seed=None):
        self.rng = np.random.default_rng(seed)
        self.dir_ml = dir_ml
        
        self.loss_fn = loss_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.LR = LR
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.nb_runs = nb_runs
        
        self.conv_delay = 0
    
    def test_data(self, traj):
        if self.conv_delay > 1:
            all_test = np.zeros((traj.shape[0],traj.shape[1]-self.conv_delay,traj.shape[2],self.conv_delay))
            for i,t in enumerate(traj):
                for j in range(traj.shape[1]-self.conv_delay):
                    all_test[i,j] = t[j:j+self.conv_delay].T
        else:
            all_test = traj.copy()
        self.all_test = torch.Tensor(all_test)
    
    def k_split(self, traj, labels):
        traj_idx = np.arange(traj.shape[0])
        self.rng.shuffle(traj_idx)
        split_data = np.array_split(traj_idx, self.nb_runs)
        self.train_data, self.valid_data = [], []
        for i in range(self.nb_runs):
            valid_t, valid_l = traj[split_data[i]], labels[split_data[i]]
            train_t, train_l = np.delete(traj,split_data[i],axis=0), np.delete(labels,split_data[i],axis=0)
            t, v = self.build_datasets(train_t, valid_t, train_l, valid_l, stack=self.conv_delay)
            self.train_data.append(t)
            self.valid_data.append(v)
    
    def build_datasets(self, train_traj, valid_traj, train_labels, valid_labels, stack=0):
        shape, shape_valid = train_traj.shape[0], valid_traj.shape[0]
        target = np.zeros((shape-stack,2))
        target_valid = np.zeros((shape_valid-stack,2))
        assert (stack==0 or stack>1), "Wrong stack value"

        x_train, x_valid = [], []
        y_train, y_valid = [], []
        for i in range(shape-stack):
            if stack>1:
                x_train.append(np.array(train_traj[i:i+stack]).T)
            else:
                x_train.append(np.array(train_traj[i]))
            target[i,train_labels[i+stack]] = 1
            y_train.append(np.array(target[i]))
            
        for i in range(shape_valid-stack):
            if stack>1:
                x_valid.append(np.array(valid_traj[i:i+stack]).T)
            else:
                x_valid.append(np.array(valid_traj[i]))
            target_valid[i,valid_labels[i+stack]] = 1
            y_valid.append(np.array(target_valid[i])) 

        if len(x_train)%self.batch_size == 1:
            x_train, y_train = x_train[:-1], y_train[:-1]
            
        x_train, x_valid = torch.Tensor(np.array(x_train)), torch.Tensor(np.array(x_valid))
        y_train, y_valid = torch.Tensor(np.array(y_train)), torch.Tensor(np.array(y_valid))

        train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size)
        
        return train_dataloader, valid_dataloader
    
    def build_only_train_set(self, train_traj, train_labels, stack=0):
        shape = train_traj.shape[0]
        target = np.zeros((shape-stack,2))
        assert (stack==0 or stack>1), "Wrong stack value"

        x_train, y_train = [], []
        for i in range(shape-stack):
            if stack>1:
                x_train.append(np.array(train_traj[i:i+stack]).T)
            else:
                x_train.append(np.array(train_traj[i]))
            target[i,train_labels[i+stack]] = 1
            y_train.append(np.array(target[i]))

        if len(x_train)%self.batch_size == 1:
            x_train, y_train = x_train[:-1], y_train[:-1]
            
        x_train, y_train = torch.Tensor(np.array(x_train)), torch.Tensor(np.array(y_train))
        train_dataset = TensorDataset(x_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        return train_dataloader
    
    def process(self, train_traj, train_labels, test_traj, type_nn, params, cpname="best"):
        self.all_times_train, self.all_times_test = np.zeros(self.nb_runs), np.zeros((self.nb_runs,test_traj.shape[0]))
        self.cpn = cpname
        
        self.k_split(train_traj, train_labels)
        self.test_data(test_traj)
        
        self.loss = np.zeros((self.nb_runs, self.nb_epochs))
        comm = np.zeros((self.nb_runs,self.all_test.shape[0],self.all_test.shape[1]))
        
        assert type_nn in ["Dense", "RNN", "LSTM", "Convnet1d"], "Wrong NN type"
        for i in tqdm(range(self.nb_runs)):
            self.network = getattr(Architectures, type_nn)(*params)
            # self.optimizer = torch.optim.Adam(self.network.parameters(), amsgrad=True, lr=self.LR)
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.LR, momentum=0.9)
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)
            
            self.loss[i], train_time = self.learning(self.train_data[i], self.valid_data[i])
            comm[i], test_time = self.testing()
            self.all_times_train[i] = train_time
            self.all_times_test[i] = test_time
                
        return comm

    # def progressive_learning(self, test_traj, test_labels, type_nn, params):
    #     assert type_nn in ["DenseLayers", "RNN", "LSTM", "Convnet1d"], "Wrong NN type"
    #     self.network = getattr(Architectures, type_nn)(*params)
    #     self.optimizer = torch.optim.Adam(self.network.parameters(), amsgrad=True, lr=self.LR)
        
    #     self.test_data(test_traj)
    #     all_comm = np.zeros(self.all_test.shape[:2])
    #     for i in tqdm(range(len(self.all_test))):
    #         #Test phase
    #         with torch.no_grad():
    #             pred = self.network(torch.Tensor(self.all_test[i])).numpy()
    #             all_comm[i] = softmax(pred,axis=1)[:,1]
    #         #Improvement phase
    #         t = self.build_only_train_set(test_traj[:(i+1)].reshape((i+1)*test_traj.shape[1],-1),
    #                                       test_labels[:(i+1)].reshape((i+1)*test_traj.shape[1],-1),stack=self.conv_delay)
    #         for k in range(self.nb_epochs):
    #             self.train(t)
    
    #     return all_comm

    def train(self, train_set):
        # losses = []
        self.network.train()
        for batch, (X, y) in enumerate(train_set):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.network(X)
            loss = self.loss_fn(pred, y)
            # losses.append(loss.detach().numpy())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self, valid_set):
        num_batches = len(valid_set)
        self.network.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in valid_set:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.network(X)
                test_loss += self.loss_fn(pred, y).item()
        test_loss /= num_batches
        return test_loss
    
    def learning(self, t, v):
        full_loss = np.zeros(self.nb_epochs)
        min_loss = np.inf
        train_t0 = time()
        for k in range(self.nb_epochs):
            self.train(t)
            val_loss = self.test(v)
            full_loss[k] = val_loss
            if k == 0 or val_loss<min_loss:
                min_loss = val_loss
                torch.save(self.network.state_dict(), self.dir_ml+self.cpn+".pt")
            self.scheduler.step(val_loss)
        train_t1 = time()
        return full_loss, train_t1-train_t0

    def testing(self):
        all_comm = np.zeros(self.all_test.shape[:2])
        time_test = np.zeros((len(self.all_test)))
        checkpoint = copy.deepcopy(self.network)
        checkpoint.load_state_dict(torch.load(self.dir_ml+self.cpn+".pt"))
        checkpoint.eval()
        with torch.no_grad():
            for i in range(len(self.all_test)):
                t0 = time()
                pred = checkpoint(torch.Tensor(self.all_test[i])).numpy()
                all_comm[i] = softmax(pred,axis=1)[:,1] 
                t1 = time()
                time_test[i] = t1-t0
        return all_comm, time_test