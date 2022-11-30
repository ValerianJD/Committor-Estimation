#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 3 15:00:52 2022

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
from scipy.special import softmax

class Dense(nn.Module):
    def __init__(self, in_neurons, out_feat):
        """
        Construction of the NN

        Parameters
        ----------
        in_neurons : Size of the input data, i.e. number of variables in the trajectories
        out_feat : list of int
            Number of neurons on every layer

        """
        super(Dense, self).__init__()
        n_layers = len(out_feat)
        in_feat = in_neurons
        layers = []
        for i in range(n_layers):
            # Each layer is made of : a layer of linear neurons, the ReLU activation function, batch normalization
            layers.append(nn.Linear(in_feat, out_feat[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_feat[i]))
            in_feat = out_feat[i]
        # Last layer returns 2 numbers, trying to predict the one-hot encoded labels
        layers.append(nn.Linear(out_feat[-1],2))
        
        # Initialization of the weights
        for l in layers:
            l.apply(self.init_weights) 
        self.network = nn.Sequential(*layers)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, x):
        return self.network(x)

class NeuralNetworks():
    def __init__(self, nb_epochs, nb_runs, dir_ml, batch_size=32, LR=1e-4, loss_fn=nn.CrossEntropyLoss(), seed=None):
        self.rng = np.random.default_rng(seed)
        self.dir_ml = dir_ml  #The directory where to save the best trained state of the NN after each epoch
        self.loss_fn = loss_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # nb_runs is the parameter of the k-fold splitting
        self.LR, self.batch_size, self.nb_epochs, self.nb_runs = LR, batch_size, nb_epochs, nb_runs
    
    def build_datasets(self, train_traj, valid_traj, train_labels, valid_labels):
        """
        Builds the train and validation datasets.
        We make pyTorch Dataloader, that nicely wrap samples, labels and separates them in batches

        All inputs must be of shape (timesteps, dimension) 
            (where dimension = 1 for the labels)
            
        """
        # We build the arrays containing the one-hot encoded labels
        # y[i] = [1,0] if the state at index i leads to an on-state (label 0)
        # y[i] = [0,1] if the state at index i leads to an off-state (label 1)
        l_train, l_valid = train_labels.shape[0], valid_labels.shape[0]
        y_train, y_valid = np.zeros((l_train,2)), np.zeros((l_valid,2))
        y_train[np.arange(l_train),train_labels] = 1
        y_valid[np.arange(l_valid),valid_labels] = 1
            
        # x_train and x_valid simply contain the samples, i.e. the traj
        x_train, x_valid = torch.Tensor(train_traj), torch.Tensor(valid_traj)
        y_train, y_valid = torch.Tensor(y_train), torch.Tensor(y_valid)
        # If the last batch only contains a single sample, we will have a shape error 
        # So we remove the last sample if necessary for convenience
        if len(train_traj) % self.batch_size == 1:
            x_train, y_train = x_train[:-1], y_train[:-1]

        train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size)
        
        return train_dataloader, valid_dataloader
    
    def k_split(self, traj, labels):
        """
        The k-fold function. 
        The original train set is split in nb_runs sub_arrays
        The NN process is performed nb_runs times.
        Every time, the validation set is a different sub-array and the actual train set is the nb_runs-1 remaining sub-arrays 

        traj and labels are of shape (timesteps, dimension)
        
        We create two lists: train_data contains the nb_runs train sets
                             valid_data contains the nb_runs balidation sets

        """
        traj_idx = np.arange(traj.shape[0])
        self.rng.shuffle(traj_idx)
        split_data = np.array_split(traj_idx, self.nb_runs)
        self.train_data, self.valid_data = [], []
        for i in range(self.nb_runs):
            valid_t, valid_l = traj[split_data[i]], labels[split_data[i]]
            train_t, train_l = np.delete(traj,split_data[i],axis=0), np.delete(labels,split_data[i],axis=0)
            t, v = self.build_datasets(train_t, valid_t, train_l, valid_l)
            self.train_data.append(t)
            self.valid_data.append(v)
    
    def train(self, train_set):
        self.network.train()  # Allow gradient tracking, mandatory for training mode
        for batch, (X, y) in enumerate(train_set):
            X, y = X.to(self.device), y.to(self.device)
            # Make sure gradients are set to 0
            self.optimizer.zero_grad()
            # Make a prediction
            pred = self.network(X)
            # ompute the loss function
            loss = self.loss_fn(pred, y)
            # Compute the gradients of the loss function
            loss.backward()
            # Adjust the learning weights
            self.optimizer.step()

    def test(self, valid_set):
        num_batches = len(valid_set)
        # Set the NN to testing mode : the gradients cannot change anymore
        self.network.eval()
        test_loss = 0
        with torch.no_grad(): #Ensure gradient calculation is disabled
            for X, y in valid_set:
                X, y = X.to(self.device), y.to(self.device)
                # Make a prediction
                pred = self.network(X)
                # Update the test loss
                test_loss += self.loss_fn(pred, y).item()
        # Average the test_loss over all batches 
        return test_loss / num_batches
    
    def learning(self, t, v):
        """
        Wrapper function to loop though all epochs for training. 
        At each epoch, we call the function train

        Parameters
        ----------
        t : train dataloader
        v : validation dataloader

        Returns
        -------
        full_loss : list of the validation loss function at the end of each epoch 
        t1 - t0 is the total training time

        """
        full_loss = np.zeros(self.nb_epochs)
        min_loss = np.inf
        train_t0 = time()
        for k in range(self.nb_epochs):
            self.train(t) # Perform training
            val_loss = self.test(v) #Compute the loss function on the validation set and store it
            full_loss[k] = val_loss
            # Check whether the new validation loss is lower than the former best
            # If yes, save the current state of the NN, it is the best until now
            if k == 0 or val_loss<min_loss:
                min_loss = val_loss
                torch.save(self.network.state_dict(), self.dir_ml+self.cpn+".pt")
            self.scheduler.step(val_loss) # Check whether that new loss triggers the next step of the scheduler (see function process)
        train_t1 = time()
        return full_loss, train_t1-train_t0

    def testing(self):
        """
        Wrapper function estimating the committor for each test trajectory.

        Returns
        -------
        all_comm : array containing the whole committor for each test trajectory
        time_test : array containing the testing time for each test trajectory

        """
        all_comm = np.zeros(self.all_test.shape[:2])
        time_test = np.zeros((len(self.all_test)))
        # Load the best state of the NN
        checkpoint = copy.deepcopy(self.network)
        checkpoint.load_state_dict(torch.load(self.dir_ml+self.cpn+".pt"))
        checkpoint.eval()  #Make sure we are in test mode
        with torch.no_grad():  #Make sure the gradients cannot be changed
            for i in range(len(self.all_test)):
                t0 = time()
                pred = checkpoint(torch.Tensor(self.all_test[i])).numpy()  # Make a prediction
                # Apply softmax transforms the output in two actual probabilities, the second one is the committor
                all_comm[i] = softmax(pred,axis=1)[:,1] 
                t1 = time()
                time_test[i] = t1-t0
        return all_comm, time_test

    def process(self, train_traj, train_labels, test_traj, params, cpname="best"):
        """
        The complete Feedforward NN process.

        Parameters
        ----------
        train_traj : the training trajectory, of shape (timesteps, dimension)
        train_labels : the training labels, of shape (timesteps)
        test_traj : the test trajectories, of shape (N_traj, timesteps, dimension)
        params : The parameters needed to build the NN, namely: dimension of the trajectories (number of variables) 
                                                                list of the number of neurons per layer
        cpname : str
            The name to give to the file containing the best state of the NN

        Returns
        -------
        comm : array of shape (N_traj, timesteps), contains the estimated committor for each test trajectory

        """
        self.all_times_train, self.all_times_test = np.zeros(self.nb_runs), np.zeros((self.nb_runs,test_traj.shape[0]))
        self.cpn = cpname
        
        self.k_split(train_traj, train_labels)  # Split the total train set in nb_runs sets to perform k-fold learning
        self.all_test = torch.Tensor(test_traj.copy())  # Build the tensor containing test trajectories
        
        self.loss = np.zeros((self.nb_runs, self.nb_epochs))
        comm = np.zeros((self.nb_runs,self.all_test.shape[0],self.all_test.shape[1]))
        
        for i in tqdm(range(self.nb_runs)):
            self.network = Dense(*params)  #Initialize the NN
            # Initialize the optimizer, here classical Stochastic Gradient Descent
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.LR, momentum=0.9)
            # Initialize the scheduler
            # If the train loss hasn't decreased after 'patience' number of epochs, multiplies LR by 'factor'
            # We gradually decrease the learning rate 
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)
            
            # Estimation of the committor on every test trajectory
            self.loss[i], train_time = self.learning(self.train_data[i], self.valid_data[i])
            comm[i], test_time = self.testing()
            self.all_times_train[i] = train_time
            self.all_times_test[i] = test_time
                
        return comm