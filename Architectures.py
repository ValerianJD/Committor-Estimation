#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:42:00 2022

@author: valerian
"""

import numpy as np
import torch
from torch import nn

def get_output_shape(model, data_dim):
    return model(torch.rand(*(data_dim))).data.shape

def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
        # nn.init.zeros_(m.bias)
    
class Dense(nn.Module):
    def __init__(self, in_neurons, out_feat):
        super(Dense, self).__init__()
        n_layers = len(out_feat)
        in_feat = in_neurons
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_feat, out_feat[i]))
            # layers.append(nn.Dropout(p=0.2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_feat[i]))
            in_feat = out_feat[i]
        layers.append(nn.Linear(out_feat[-1],2))
        
        for l in layers:
            l.apply(init_weights) 
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ConvNet1d(nn.Module):
    def __init__(self, in_neurons, stack, kernel, n_layers, out_feat, nb_neurons=64, pool=True, pool_k=2):
        super(ConvNet1d, self).__init__()
        
        in_feat = in_neurons
        inputsize = (1,in_neurons,stack)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv1d(in_feat, out_feat[i], kernel))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_feat[i]))
            if pool:
                layers.append(nn.MaxPool1d(pool_k))
            in_feat = out_feat[i]    
        outsize = get_output_shape(nn.Sequential(*layers), inputsize)
        flatsize = np.prod(list(outsize))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(flatsize, nb_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(nb_neurons))
        layers.append(nn.Linear(nb_neurons, 2))
        
        for l in layers:
            l.apply(init_weights)
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)   

class RNN(nn.Module):
    def __init__(self, in_neurons, hidden_dim, nb_layers, conv=6):
        super(RNN, self).__init__()
        
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.RNN(in_neurons, self.hidden_dim, self.nb_layers, batch_first=True)
        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv*self.hidden_dim, 2))
        
    def forward(self, x):
        #Initialize hidden state
        x = torch.transpose(x,1,2) #Pytorch convention
        batch_size = x.size(0)
        hidden = torch.zeros(self.nb_layers, batch_size, self.hidden_dim)
        out, hidden = self.rnn(x, hidden)
        out = self.readout(out)
        
        return out 
    
class LSTM(nn.Module):
    def __init__(self, in_neurons, hidden_dim, nb_layers, readout_neurons):
        super(LSTM, self).__init__()
        
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(in_neurons, self.hidden_dim, self.nb_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, readout_neurons),
            nn.ReLU(),
            nn.BatchNorm1d(readout_neurons),
            nn.Linear(readout_neurons, 2))
        
    def forward(self, x):
        #Initialize hidden state
        x = x[:,np.newaxis]
        batch_size = x.size(0)
        h0 = torch.zeros(self.nb_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.nb_layers, batch_size, self.hidden_dim)
        
        out, (hx, cx) = self.lstm(x, (h0,c0))
        out = out.view(-1, self.hidden_dim)
        out = self.relu(out)
        out = self.readout(out)
        
        return out
    