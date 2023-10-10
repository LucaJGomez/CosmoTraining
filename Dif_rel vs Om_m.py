# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 20:24:34 2023

@author: Luca
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
import torch

# Load the networks:
nets = torch.load('nets_LCDM_proof.ph',
                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
                  )
Om_r=9.24*10**(-5)
a_0 = 10**(-3)
a_f = 1
N_0 = np.log(a_0)
N_f= np.log(a_f)

condition = [BundleIVP(N_0, a_0),
             BundleIVP(N_0, a_0)]

x = BundleSolution1D(nets, condition)


# The Hubble parameter as a function of the dependent variables of the system:

def delta(N, Om_m_0):
    deltas = x(N, Om_m_0, to_numpy=True)[0]
    return deltas

def delta_pann(N, Om_m_0,):
    deltas_p = x(N, Om_m_0, to_numpy=True)[1]
    return deltas_p


N_vec = np.linspace(N_0, N_f,200)

Om_m_vec=np.linspace(0.2,0.4,40)

err_porc=[]
err_porc_p=[]

for i in range(len(Om_m_vec)):
    
    Om_m_0_vec=Om_m_vec[i]*np.ones(len(N_vec))
    delta_ann=delta(N_vec,Om_m_0_vec)
    delta_p_ann=delta_pann(N_vec,Om_m_0_vec)
    
    Om_m_0=Om_m_vec[i]
    Om_r=9.24*10**(-5)
    Om_L=1-Om_r-Om_m_0
    a_eq=Om_r/Om_m_0 #Equality between matter and radiation
    alpha=a_eq**3*Om_L/Om_m_0
    
    def F(N,X):
        
        f1=X[1] 

        term1=(3*np.exp(N)/(2*a_eq*(1+(np.exp(N)/a_eq)+alpha*(np.exp(N)/a_eq)**4)))*X[0]
        
        term2=-((1+4*alpha*(np.exp(N)/a_eq)**3)/(2*(1+(a_eq/np.exp(N))+alpha*(np.exp(N)/a_eq)**3)))*X[1]
        
        f2=term1+term2
        
        return np.array([f1,f2])


    atol, rtol = 1e-15, 1e-12
    #Perform the backwards-in-time integration
    out2 = solve_ivp(fun = F, t_span = [N_0,N_f], y0 = np.array([a_0,a_0]),
                    t_eval = N_vec, method = 'RK45',rtol=rtol, atol=atol)

    delta_num=out2.y[0]
    delta_p_num=out2.y[1]
    
    dif_rel=[]
    dif_rel_p=[]
    for i in range(len(N_vec)):
        dif_rel.append(100*np.abs(delta_ann[i]-delta_num[i])/np.abs(delta_num[i]))
        dif_rel_p.append(100*np.abs(delta_p_ann[i]-delta_p_num[i])/np.abs(delta_p_num[i]))
    
        
    err_porc.append((dif_rel[-1]))
    err_porc_p.append(np.mean(dif_rel_p))
    
    #plt.plot(N_vec,dif_rel)
    

plt.scatter(Om_m_vec,err_porc,label='delta (today)')
plt.scatter(Om_m_vec,err_porc_p,label='delta_p (mean)')
plt.xlabel(r'$\Omega_{m0}$')
plt.ylabel('err%')
plt.legend()