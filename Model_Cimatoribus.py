#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:42:46 2022

@author: valerian
"""

import os
import numpy as np
import scipy.io as sio

class CimatoribusModel():
    """
    CONVENTION: WORK ALL THE TIME WITH NON-DIM VARIABLES
    SO BE CAREFUL TO ALWAYS DIM THEM IN THE METHODS WHEN NEEDED 
    
    """
    
    def __init__(self, dt=1e-3, folder="/Users/valerian/Desktop/Results/", seed=None):
        
        self.name = "Cimatoribus"
        self.capped = True    # Trajectories have a fixed length to avoid the S-transition zone
        
        self.rng = np.random.default_rng(seed)
        self.name_files = folder+"equilibria/"  #Directory where to find info about the steady states
        self.all_eq_files = np.sort(os.listdir(self.name_files))[1:] #Load steady states files
        self.params = np.load(folder+"Castellana_params.npy", allow_pickle=True).item() #Load large number of paramaters

        # The basis for noise computation
        self.noise_amp_base = 1e6*self.params["S0"]/(self.params["Vn"]*self.params["S_dim"])\
            *self.params["t_dim"]\
            *np.array([0,0,-1,self.params["Vn"]/self.params["Vs"],0])

        #System parameters for differential equation solver
        self.toler = 1e-9 #stochastic theta time solver
        self.epsilon = 1e-12 #for the heaviside function
        
        # Traj params
        self.dt = dt
        
    def noise_str(self, s):
        """
        Just a convenience function when creating names to save files.

        """
        return str(s).replace(".",",")
        
    def tmax(self, n_years):
        """
        Takes the number of years in the trajectory, computes its maximum time. 
        
        """
        return 10*n_years*self.dt
        
    def non_dim(self, a):
        """
        Non-dimensionalize any set of trajectories.
        
        """
        D_dim = self.params["D_dim"]
        S_dim = self.params["S_dim"]
        div = [S_dim, S_dim, S_dim, S_dim, D_dim, S_dim]
        return np.divide(a, div)

    def re_dim(self, a):
        """
        Re-dimensionalize any set of trajectories.

        """
        D_dim = self.params["D_dim"]
        S_dim = self.params["S_dim"]
        mult = [S_dim, S_dim, S_dim, S_dim, D_dim, S_dim]
        return np.multiply(a, mult)
    
    def comp_steady(self, idx):
        """
        Load the steady states from the right file (corresponding to the prescribed index of forcing).
        Then puts it in the right shape and non-dimensionalize it.

        """
        eq_point = sio.loadmat(self.name_files+self.all_eq_files[idx])["eq"]
        #St, S_ts, Sn, Ss, D, Sd
        reorder = [0,1,2,3,5,6,4]
        on_dim, off_dim = eq_point[1][reorder], eq_point[2][reorder]
        on_dim = np.array([on_dim[i][0,0] for i in range(7)])
        off_dim = np.array([off_dim[i][0,0] for i in range(7)])
        on = self.non_dim(on_dim[1:])
        off = self.non_dim(off_dim[1:])
        return on, off
    
    def is_on(self, traj):
        """
        Checks every on-state in any set of trajectories.
        traj must contain all 6 variables of the model

        Parameters
        ----------
        traj : shape (Number of trajectories, timesteps, 6)
            TRAJ MUST BE NON-DIMENSIONALIZED

        Returns
        -------
        An array of shape (Number of trajectories, timesteps)
        Every on-state is replaced with 1, all other states are replaced with 0

        """
        traj = self.re_dim(traj) 
        sal, D = traj[... ,2]-traj[... ,1]+1.25, traj[... ,4] 
        a = self.params["Agm"]*self.params["Lx_a"]/(self.params["Ly"]*self.params["beta"]*self.params["eta"])
        b = -self.params["q_Ek"]/(self.params["beta"]*self.params["eta"])
        c = self.params["q_Ek"]
        d = self.params["Agm"]*self.params["Lx_a"]/self.params["Ly"]
        e = self.params["kappa"]*self.params["A"]
        return (sal*D**2 + a*D + b >= 0) & (c*D - d*D**2 - e > 0)
        
    def is_off(self, traj):
        """
        Checks every off-state in any set of trajectories.
        traj must contain all 6 variables of the model

        Parameters
        ----------
        traj : shape (Number of trajectories, timesteps, 6)
            TRAJ MUST BE NON-DIMENSIONALIZED
            
        Returns
        -------
        An array of shape (Number of trajectories, timesteps)
        Every off-state is replaced with 1, all other states are replaced with 0

        """
        traj = self.re_dim(traj)
        return traj[...,2]-traj[...,1] <= -1.25

    def is_transi(self, traj):
        """
        Checks every off-state in any set of trajectories.
        traj must contain all 6 variables of the model

        Parameters
        ----------
        traj : shape (Number of trajectories, timesteps, 6)
            TRAJ MUST BE NON-DIMENSIONALIZED

        Returns
        -------
        Transition state = state that is not an on-state nor an off-state

        """
        return ~self.is_on(traj) & ~self.is_off(traj)

    def is_off_tot(self, traj):
        """
        Similar to is_off.
        But is_off only checks the F-transitions.
        This function only checks S-transitions.

        """
        traj = self.re_dim(traj)
        p = self.params["Ly"]*self.params["q_Ek"]/(self.params["Agm"]*self.params["Lx_a"])
        return np.array(traj[:,2]-traj[:,1] <= -1.25) & np.array(traj[:,4] >= p)

    def system(self, state):
        """
        Computes the differential equations at each timestep.
        State is the current state and this function returns what is needed to compute the next.

        """
        D_dim, S_dim, t_dim = self.params["D_dim"], self.params["S_dim"], self.params["t_dim"]
        Vn, Vs = self.params["Vn"], self.params["Vs"]
        St, S_ts, Sn, Ss, D, Sd = state
        Lxa, Ly, A = self.params["Lx_a"], self.params["Ly"], self.params["A"]
        
        Vt = A*D_dim
        V_ts = (Lxa*Ly/2)*D_dim
        Vr = (A + Lxa*Ly/2)
        
        heaviside = lambda x: 0.5*(np.tanh(x/self.epsilon)+1)
        
        qEk = self.params["tau"]*self.params["Lx_s"]/(self.params["rho_0"]*self.params["fs"])
        qe = self.params["Agm"]*Lxa*D*D_dim/Ly
        qs = qEk-qe
        qu = self.params["kappa"]*A/(D*D_dim)
        qn1 = self.params["eta"]*self.params["alpha"]*(self.params["T_ts"]-self.params["Tn"])*(D*D_dim)**2
        qn2 = self.params["eta"]*self.params["beta"]*(D*D_dim)**2
        qn = qn1 + qn2*(Sn-S_ts)*S_dim
        
        deriv_D = (qu + qs - heaviside(qn)*qn) * t_dim/(Vr*D_dim)
        
        f = np.zeros(5)
        f[0] = qs*(heaviside(qs)*S_ts + heaviside(-qs)*St) + qu*Sd \
            - heaviside(qn)*qn*St + self.params["r_s"]*(S_ts-St) + self.params["r_n"]*(Sn-St) \
            + 2*self.params["Es"]
        f[0] *= t_dim/Vt
        f[0] -= St*deriv_D
        f[0] /= D
        
        f[1] = qEk*Ss - qe*S_ts - qs*(heaviside(qs)*S_ts + heaviside(-qs)*St) \
            + self.params["r_s"]*(St - S_ts)
        f[1] *= t_dim/V_ts
        f[1] -= S_ts*deriv_D
        f[1] /= D
        
        f[2] = (qn*heaviside(qn) + self.params["r_n"])*(St - Sn) - (self.params["Es"]+self.Ea*1e6)
        f[2] *= t_dim/Vn
        
        f[3] = qs*(heaviside(qs)*Sd + heaviside(-qs)*Ss) + qe*S_ts - qEk*Ss - (self.params["Es"]-self.Ea*1e6)
        f[3] *= t_dim/Vs
        
        f[4] = deriv_D
        
        return f
        
    def algebraeic_constraint(self, state, jac=False):
        #Define the algebraeic constraint on the system
        #If jac, return the associated Jacobian
        D_dim, S_dim = self.params["D_dim"], self.params["S_dim"]
        S0, V0, Vn, Vs = self.params["S0"], self.params["V0"], self.params["Vn"], self.params["Vs"]
        St, S_ts, Sn, Ss, D, Sd = state
        
        b1 = S0/S_dim
        b2 = -Vn/V0
        b3 = -1 + (Vn+Vs)/V0
        b4 = (D_dim/V0) * (self.params["A"]+self.params["Lx_a"]*self.params["Ly"]/2)
        b5 = - self.params["A"]*D_dim/V0
        b6 = - self.params["Lx_a"]*self.params["Ly"]*D_dim/(2*V0)
        b7 = - Vs/V0
        
        if not jac:
            return b1 + b2*Sn + b3*Sd + b4*D*Sd + b5*D*St + b6*D*S_ts + b7*Ss
        return np.array([b5*D,
                         b6*D,
                         b2,
                         b7,
                         b4*Sd+b5*St+b6*S_ts,
                         b3+b4*D])

    def residual(self, z, y, B):
        #Compute residual for stochastic theta method
        R = z[:-1] - y[:-1] - 0.5*self.dt*(self.system(z)+self.system(y)) - B*np.sqrt(self.dt)
        return R

    def stochastic_theta(self, z, noise, dW):
        #y: solution of the former timestesp
        #F: function of the system, J its jacobian
        #noise: diagonal noise vector
        #dW: stochastic matrix (dimension x times)
        #dt: timestep
        #toler: tolerance for the implicit scheme
        #Returns z, solution at the new timestep
        
        #Semi-implicit scheme:
        #x(n+1) = x(n) + 1/2 * dt * ( f(x(n+1)) + f(x(n)) ) + B*sqrt(dt8)
        y = np.copy(z)
        B = noise * dW
        Rz1 = self.residual(z, y, B)
        Rz2 = self.algebraeic_constraint(z)
        Rz = np.concatenate((Rz1,[Rz2]))
        while np.linalg.norm(Rz) > self.toler:
            A1 = np.concatenate((np.eye(z.shape[0]-1),np.zeros((z.shape[0]-1,1))),axis=1) - 0.5*self.dt*Rz2
            A2 = np.expand_dims(self.algebraeic_constraint(z, jac=True), axis=0)
            A = np.concatenate((A1,A2))
        
            D = np.expand_dims(-Rz,axis=1) #D = -R(z)
            dz = np.linalg.solve(A,D) #Solve A*dz = D
            z += dz.reshape(6) #Update z
            #Update residual
            Rz1 = self.residual(z, y, B)
            Rz2 = self.algebraeic_constraint(z)
            Rz = np.concatenate((Rz1,[Rz2]))

        return z

    def MC_comp_comm(self, N_traj, init_state, idx, noise_ratio):
        """
        Compute trajectories for the Monte-Carlo estimate of the committor.
        From the state init_state, we start N_traj trajectories and count the number of which
        end up in the F-transition zone before ending up in the on-zone.

        Parameters
        ----------
        N_traj : int
            The number of Monte-Carlo runs
        init_state : state, of shape (6)
            The state where to estimate the committor
            IT MUST BE NON-DIMENSIONALIZED
        idx : int in [0,99]
            The index of the forcing to be used.
            0 corresponds to the low-end of the bistability zone, 99 to its high-end
        noise_ratio : float
            The amount of noise to add to the system

        Returns
        -------
        float
            The estimated committor on the point init_state
            
        """
        # Initialization
        self.idx = idx
        self.Ea = float(self.all_eq_files[self.idx][13:-4])
        noise = self.Ea * self.noise_amp_base * noise_ratio
        
        # We compute N_traj trajectories
        # But we halt every trajectory as soon as we reach either the on-zone or off-zone
        # If we reach the off-zone first, we count +1
        proba = 0
        for _ in range(N_traj): 
            z = np.copy(init_state)
            while not self.is_on(z) and not self.is_off(z):
                z = self.stochastic_theta(z, noise, self.rng.normal())
            if self.is_off(z):
                proba += 1
        return proba/N_traj

    def trajectory(self, N_traj, N_years, idx, noise_ratio, t_init=0, init_state=None):  
        """
        Compute a trajectory of the system.

        Parameters
        ----------
        N_traj : int
            The number of trajectories to compute.
        N_years : int
            The number of years these traj should last.
            With the default value of dt: number of timesteps = nb_years * 10
        idx : int in [0,99]
            The index of the forcing to be used.
            0 corresponds to the low-end of the bistability zone, 99 to its high-end
        noise_ratio : float
            The amount of noise to add to the system
        t_init : int or float
            The time at which the trajectory should start
        init_state : state, thus of shape (6)
            The initial condition of the traj. If None (default), will start close to the steady on-state

        Returns
        -------
        z_trajectory : traj of shape (N_traj, 10*N_years, 6)
            The computed trajectories.

        """
        # Define the forcing and the noise
        self.Ea = float(self.all_eq_files[idx][13:-4])
        noise = self.Ea * self.noise_amp_base * noise_ratio
        
        # Define the initial condition
        if init_state == None:
            init_state = self.comp_steady(idx)[0] + 0.01
        
        # Initialize the trajectories, the noise is computed in advance
        T = self.tmax(N_years)
        M = int(np.ceil((T-t_init)/self.dt))
        dW = self.rng.normal(size=(N_traj,M))
        z_trajectory = np.zeros((N_traj, M, init_state.shape[0]))
        z_trajectory[:,0] = np.repeat(init_state[np.newaxis],N_traj,axis=0)

        # Actually compute the trajectories
        for i in range(N_traj):
            z = np.copy(init_state)
            for j in range(M-1):
                z = self.stochastic_theta(z, noise, dW[i,j])
                z_trajectory[i,j+1] = np.copy(z)

        return z_trajectory
    
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
        all_var = np.array(["St", "Sts", "Sn", "Ss", "D", "Sd", "Sdiff"])
        red_traj = np.zeros(traj.shape[:-1]+(len(var),))
        ind = [np.where(all_var==var[k])[0][0] for k in range(len(var))]
        for i in range(len(var)):
            if ind[i] == 6:
                red_traj[...,i] = np.copy(traj[...,2]-traj[...,1])
            else:
                red_traj[...,i] = np.copy(traj[...,ind[i]])
        return red_traj

    def check_traj_fine(self, traj):
        """
        Check whether trajectory hits the S-transition zone.
        Returns True if the traj always stays away from that zone.

        """
        return np.all(np.logical_not(self.is_off_tot(traj)))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time
    
    C = CimatoribusModel()
    idx, noise = 60, 0.4
    
    t0 = time()
    t = C.trajectory(1, 500, idx, noise)[0]
    t = C.re_dim(t)
    t1 = time()
    print(t1-t0)
    
    on, off = C.comp_steady(60)
    on, off = C.re_dim(on), C.re_dim(off)
    
    on_states, off_states = C.is_on(C.non_dim(t)), C.is_off(C.non_dim(t))

    x_m, x_M, y_m, y_M = -6, 1.5, 700, 1800
    interval = np.linspace(-1.24,x_M,100)
    a = C.params['Agm']*C.params['Lx_a']/(C.params['Ly']*C.params['eta']*C.params['beta'])
    b = C.params['q_Ek']/(C.params['eta']*C.params['beta'])
    lim_D = (-a + np.sqrt(a**2 + 4*b*(interval+1.25)))/(2*(interval+1.25))
    plt.scatter(t[:,2]-t[:,1], t[:,4], s=2)
    plt.scatter(t[on_states,2]-t[on_states,1], t[on_states,4], s=2, c="g")
    plt.scatter(t[off_states,2]-t[off_states,1], t[off_states,4], s=2, c="r")
    plt.scatter(on[2]-on[1], on[4], color="m", s=100)
    plt.scatter(off[2]-off[1], off[4], color="m", s=100)    
    plt.axvline(x=-1.25, c="r")
    plt.hlines(y=1682.5, xmin=-1.25, xmax=x_M, color="r")
    plt.hlines(y=1717, xmax=-1.25, xmin=x_m, color="r")
    plt.fill_between(interval,lim_D, np.ones(100)*1682.5, color="g", alpha=0.2)
    plt.fill_between(np.linspace(x_m,-1.25,100), y_m, np.ones(100)*1717, color="r", alpha=0.2)
    plt.fill_between(np.linspace(x_m,-1.25,100), 1717, np.ones(100)*y_M, color="k", alpha=0.2)
    plt.plot(interval, lim_D, c="r")
    plt.xlim(x_m, x_M)
    plt.ylim(y_m, y_M)
    plt.title("Ea="+str(idx)+", noise="+str(noise), fontsize=13)
    plt.show()
    
#%%

    import h5py
    f = h5py.File("/Users/Valerian/Desktop/Traj_Julia.jld", "r")
    traj_julia = f["traj"]
    print(traj_julia.shape)
    traj_julia = C.re_dim(traj_julia)
    print(t.shape, traj_julia.shape)
    print(np.count_nonzero(t!=traj_julia[:10],axis=2))
    
    for i in range(10):
        plt.plot(t[i,:,2]-t[i,:,1], t[i,:,4])
        plt.plot(traj_julia[i,:,2]-traj_julia[i,:,1], traj_julia[i,:,4], alpha=0.5)
        plt.show()
    
### DEPRECATED

# def system(self, state, jac=False):
#     """
#     Computes the differential equations at each timestep.
#     State is the current state and this function returns what is needed to compute the next.

#     """
#     D_dim, S_dim, t_dim = self.params["D_dim"], self.params["S_dim"], self.params["t_dim"]
#     Vn, Vs = self.params["Vn"], self.params["Vs"]
#     St, S_ts, Sn, Ss, D, Sd = state
#     Lxa, Ly, A = self.params["Lx_a"], self.params["Ly"], self.params["A"]
    
#     Vt = A*D_dim
#     V_ts = (Lxa*Ly/2)*D_dim
#     Vr = (A + Lxa*Ly/2)
    
#     heaviside = lambda x: 0.5*(np.tanh(x/self.epsilon)+1)
    
#     qEk = self.params["tau"]*self.params["Lx_s"]/(self.params["rho_0"]*self.params["fs"])
#     qe = self.params["Agm"]*Lxa*D*D_dim/Ly
#     qs = qEk-qe
#     qu = self.params["kappa"]*A/(D*D_dim)
#     qn1 = self.params["eta"]*self.params["alpha"]*(self.params["T_ts"]-self.params["Tn"])*(D*D_dim)**2
#     qn2 = self.params["eta"]*self.params["beta"]*(D*D_dim)**2
#     qn = qn1 + qn2*(Sn-S_ts)*S_dim
    
#     deriv_D = (qu + qs - heaviside(qn)*qn) * t_dim/(Vr*D_dim)
    
#     if not jac:
#         f = np.zeros(5)
#         f[0] = qs*(heaviside(qs)*S_ts + heaviside(-qs)*St) + qu*Sd \
#             - heaviside(qn)*qn*St + self.params["r_s"]*(S_ts-St) + self.params["r_n"]*(Sn-St) \
#             + 2*self.params["Es"]
#         f[0] *= t_dim/Vt
#         f[0] -= St*deriv_D
#         f[0] /= D
        
#         f[1] = qEk*Ss - qe*S_ts - qs*(heaviside(qs)*S_ts + heaviside(-qs)*St) \
#             + self.params["r_s"]*(St - S_ts)
#         f[1] *= t_dim/V_ts
#         f[1] -= S_ts*deriv_D
#         f[1] /= D
        
#         f[2] = (qn*heaviside(qn) + self.params["r_n"])*(St - Sn) - (self.params["Es"]+self.Ea*1e6)
#         f[2] *= t_dim/Vn
        
#         f[3] = qs*(heaviside(qs)*Sd + heaviside(-qs)*Ss) + qe*S_ts - qEk*Ss - (self.params["Es"]-self.Ea*1e6)
#         f[3] *= t_dim/Vs
        
#         f[4] = deriv_D
        
#         return f
    
#     jac = np.zeros((5,6))
    
#     jac[4,1] = qn2*S_dim*heaviside(qn)
#     jac[4,2] = - qn2*S_dim*heaviside(qn)
#     jac[4,4] = -(qu + qe + 2*qn*heaviside(qn)) / D
#     jac[4] *= t_dim/(Vr*D_dim)
       
#     jac[0,0] = (qs*heaviside(-qs) - qn*heaviside(qn) - self.params["r_s"] - self.params["r_n"]) * t_dim/Vt \
#         - deriv_D
#     jac[0,1] = (qs*heaviside(qs) + qn2*S_dim*heaviside(qn)*St + self.params["r_s"]) * t_dim/Vt - St*jac[4,1]
#     jac[0,2] = (-qn2*S_dim*heaviside(qn)*St + self.params["r_n"]) * t_dim/Vt - St*jac[4,2]
#     jac[0,4] = ( (-qe*(heaviside(qs)*S_ts + heaviside(-qs)*St)/D - qu*Sd/D - 2*qn*heaviside(qn)*St/D) * t_dim/Vt - St*jac[4,4]) \
#         - ((qs*(heaviside(qs)*S_ts + heaviside(-qs)*St) + qu*Sd \
#         - heaviside(qn)*qn*St + self.params["r_s"]*(S_ts-St) + self.params["r_n"]*(Sn-St) \
#         + 2*self.params["Es"])*t_dim/Vt - St*deriv_D) / D
#     jac[0,5] = qu*t_dim/Vt
#     jac[0] /= D
    
#     jac[1,0] = (self.params["r_s"] - qs*heaviside(-qs)) * t_dim/V_ts
#     jac[1,1] = -(qe + qs*heaviside(qs) + self.params["r_s"]) * t_dim/V_ts - deriv_D - S_ts*jac[4,1]
#     jac[1,2] = -S_ts*jac[4,2]
#     jac[1,3] = qEk*t_dim/V_ts
#     jac[1,4] = (-qe*S_ts/D + qe*(heaviside(qs)*S_ts+heaviside(-qs)*St)/D) * t_dim/V_ts \
#         - S_ts*jac[4,4] - ((qEk*Ss - qe*S_ts - qs*(heaviside(qs)*S_ts + heaviside(-qs)*St) \
#             + self.params["r_s"]*(St - S_ts)) * t_dim/V_ts - S_ts*deriv_D) / D
#     jac[1] /= D
    
#     jac[2,0] = qn*heaviside(qn) + self.params["r_n"]
#     jac[2,1] = -qn2*S_dim*heaviside(qn)*(St-Sn)
#     jac[2,2] = -qn*heaviside(qn) - self.params["r_n"] + qn2*S_dim*heaviside(qn)*(St-Sn)
#     jac[2,4] = 2*qn*heaviside(qn)*(St-Sn)/D
#     jac[2] *= t_dim/Vn
    
#     jac[3,1] = qe
#     jac[3,3] = qs*heaviside(-qs) - qEk
#     jac[3,4] = (S_ts - heaviside(qs)*Sd - heaviside(-qs)*Ss) * qe/D
#     jac[3,5] = qs*heaviside(qs)
#     jac[3] *= t_dim/Vs
    
#     return jac