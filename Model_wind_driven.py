#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:42:37 2022

@author: valerian
"""

import os
import numpy as np

class WindDrivenModel():
    def __init__(self, dt, sigma=0.3, beta=0.02, seed=None):
        """
        sigma corresponds to the strenght of the coupling in the model
        beta is the amplitude of the noise

        """
        self.name = "Wind_driven"
        self.capped = False #Is there a limit to the length of trajectories ? 
        
        self.rng = np.random.default_rng(seed)
        
        # Model parameters
        self.c = [0.020736, 0.018337, 0.015617, 0.031977, 0.036673, 0.046850, 0.314802]
        self.l = [0.0128616, 0.0211107, 0.0318615, 0.0427787]
        
        #Loads the steady states, pre-computed for a number of parameters
        self.steady = np.load(os.getcwd()+"/Wind_driven_steady_states.npy", allow_pickle=True).item()

        self.dt = dt
        self.comp_on_off(sigma, beta)
        
    def comp_on_off(self, sigma, beta):
        """
        Defines the values of sigma, beta, the steady on-state, steady off-state
        and epsilon for the whole class.
        
        """
        if sigma is not None and beta is not None:
            self.sigma, self.beta = sigma, beta
            self.up = self.steady[str(np.round(self.sigma,3))][0]
            self.down = self.steady[str(np.round(self.sigma,3))][1]
            self.epsilon = np.linalg.norm(self.up-self.down)/3
        
    def deriv(self, A):
        """
        The equations of the model.
        
        """
        return np.array([
            self.c[0]*A[0]*A[1] + self.c[1]*A[1]*A[2] + self.c[2]*A[2]*A[3] - self.l[0]*A[0],
            self.c[3]*A[1]*A[3] + self.c[4]*A[0]*A[2] - self.c[0]*(A[0]**2) - self.l[1]*A[1] + self.c[6]*self.sigma,
            self.c[5]*A[0]*A[3] - (self.c[1]+self.c[4])*A[0]*A[1] - self.l[2]*A[2],
            -self.c[3]*A[1]**2 - (self.c[2]+self.c[5])*A[0]*A[2] - self.l[3]*A[3]
            ])
    
    def trajectory(self, N_traj, tmax, sigma=None, beta=None, init_state=None):
        """
        Compute N_traj trajectories of length tmax//dt
        Every time you call this trajectory, you can re-define sigma and beta
        /!\ They will be changed for the whole class /!\

        If init-state is not prescribed, then it is set to: steady on-state+0.01

        Returns
        -------
        traj : shape (N_traj, timesteps, 4)

        """
        self.comp_on_off(sigma, beta)
        
        # If init_state is not prescribed, we define it close to the steady on-state
        if np.any(init_state) is None:
            init_state = self.up + 0.01

        N = int(tmax//self.dt)  #Number of timesteps
        traj = np.zeros((N_traj,N,4))
        traj[:,0] = np.repeat(init_state[np.newaxis],N_traj,axis=0)
        
        # Generate noise at once for all trajectories
        noise = self.rng.normal(loc=0., scale=np.sqrt(self.dt), size=(N_traj,N))
        for j in range(N_traj):
            for i in range(1,N):
                A = traj[j,i-1]
                traj[j,i] = A + self.deriv(A)*self.dt + np.array([0,self.beta,0,0])*noise[j,i]  
        return traj
    
    def is_on(self, A):
        """
        A is a trajectory or list of trajectories.
        Returns 1 whenever a point is closer to the steady on-state than epsilon
        
        """
        return np.linalg.norm(A-self.up,axis=-1)<self.epsilon
        
    def is_off(self, A):
        """
        A is a trajectory or list of trajectories.
        Returns 1 whenever a point is closer to the steady off-state than epsilon
        
        """
        return np.linalg.norm(A-self.down,axis=-1)<self.epsilon
    
    def is_transi(self, A):
        """
        A is a trajectory or list of trajectories.
        Returns 1 whenever a point is neither an off nor an on-state
        
        """
        return ~self.is_on(A) & ~self.is_off(A)
    
    def MC_comp_comm(self, N_traj, A0, sigma=None, beta=None):
        """
        Compute trajectories for the Monte-Carlo estimate of the committor.
        From the state init_state, we start N_traj trajectories and count the number of which
        end up in the F-transition zone before ending up in the on-zone.

        Parameters
        ----------
        N_traj : int
            The number of Monte-Carlo runs
        A0 : state, of shape (4)
            The state where to estimate the committor
        sigma : float
            The strength of the coupling in the model
        beta : float
            The amplitude of noise

        Returns
        -------
        float
            The estimated committor on the point init_state
            
        """
        # Initialization
        self.comp_on_off(sigma, beta)
        
        # We compute N_traj trajectories
        # But we halt every trajectory as soon as we reach either the on-zone or off-zone
        # If we reach the off-zone first, we count +1
        proba = 0
        for _ in range(N_traj):
            A = np.copy(A0)
            while not self.is_on(A) and not self.is_off(A):
                A += self.deriv(A)*self.dt + np.array([0,self.beta,0,0])*self.rng.normal(loc=0,scale=np.sqrt(self.dt))
            if self.is_off(A):
                proba += 1
        return proba/N_traj
    
    def check_traj_fine(self, traj):
        """
        For consistance with the AMOC model.
        
        """
        return True
    
    def select_var(self, traj, var):
        """
        Reduce the dimension of traj, by selecting only variables in var.

        Parameters
        ----------
        traj : shape (number of trajectories (if applicable), timesteps, dimension)
        
        var : list of str

        Returns
        -------
        red_traj : shape (number of trajectories (if applicable), timesteps, reduced dimension)

        """
        all_var = np.array(["A1", "A2", "A3", "A4", "A1+A3"])
        red_traj = np.zeros(traj.shape[:-1]+(len(var),))
        ind = [np.where(all_var==var[k])[0][0] for k in range(len(var))]
        for i in range(len(var)):
            if ind[i] == 4:
                red_traj[...,i] = np.copy(traj[...,0]+traj[...,2])
            else:
                red_traj[...,i] = np.copy(traj[...,ind[i]])
        return red_traj
    
#%%    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    sigma = 0.45
    beta = 0.198
    m = WindDrivenModel(1, sigma=sigma, beta=beta)
    print(m.up, m.down)
    
    # Plot of the jet-up and jet-down states
    interval = np.linspace(0,np.pi,1000)
    modes_x = np.array([[np.exp(-2*interval)*np.sin(interval)] for k in range(1,5)])
    modes_y = np.array([np.sin(k*interval) for k in range(1,5)])[...,np.newaxis]
    modes = modes_y @ modes_x
    stream_up, stream_down = np.sum(m.up[:,np.newaxis,np.newaxis]*modes,axis=0), np.sum(m.down[:,np.newaxis,np.newaxis]*modes,axis=0)
    
    im, (ax1, ax2) = plt.subplots(1,2, figsize=(13,5), sharey=True)
    ax1.contourf(interval, interval, stream_up, levels=30, cmap='viridis')
    ax1.set_title("Jet-up state", fontsize=15)
    ax1.set_ylabel("Y", fontsize=13)
    ax1.set_yticklabels([0,0.5,1.0,1.5,2.0,2.5,3.0], fontsize=13)
    ax1.set_xlabel("X", fontsize=13)
    ax1.set_xticklabels([0,0.5,1.0,1.5,2.0,2.5,3.0], fontsize=13)
    f = ax2.contourf(interval, interval, stream_down, levels=30,cmap='viridis')
    ax2.set_title("Jet-down state", fontsize=15)
    ax2.set_xlabel("X", fontsize=13)
    ax2.set_xticklabels([0,0.5,1.0,1.5,2.0,2.5,3.0], fontsize=13)
    im.subplots_adjust(bottom=0.1, top=0.9, left=0.1, wspace=0.03)
    cbar = im.colorbar(f, ax=[ax1,ax2])
    cbar.ax.tick_params(labelsize=13)
    plt.show()
    
    # Plot of the randomly sampled phase space snapshot
    L = 100
    N = 100000
    rng = np.random.default_rng()
    init = np.zeros((N,4))
    
    a3 = np.linspace(m.up[2]-1, m.up[2]+1, num=10)
    for k in range(4):
        if k == 2:
            init[:,k] = np.repeat(a3,N//10)
        else:
            init[:,k] = rng.uniform(m.up[k]-1, m.up[k]+1, N)
    ic_up, ic_down = [], []
    
    for i, ic in enumerate(init):
        if i%(N//10) == 0:
            print(i)
            ic_up.append([])
            ic_down.append([])
        traj = m.trajectory(1, L, init_state=ic)[0]
        if traj[-1,0]+traj[-1,2]<0:
            ic_down[-1].append(i)
        else:
            ic_up[-1].append(i)

    vM, vm = m.up+1, m.up-1
    
    for i in range(10):
        x, y = init[ic_up[i]], init[ic_down[i]]
        
        plt.scatter(x[:,0], x[:,1], s=5, c="y")
        plt.scatter(y[:,0], y[:,1], s=5, c="saddlebrown")
        plt.scatter(m.up[0], m.up[1], s=100, c="k")
        plt.scatter(m.down[0], m.down[1], s=100, c="k")
        plt.xlabel(r"$A_1$", fontsize=13)
        plt.ylabel(r"$A_2$", fontsize=13)
        plt.title(r"$A_3=$"+str(np.round(a3[i],decimals=3)), fontsize=13)
        plt.show()   

