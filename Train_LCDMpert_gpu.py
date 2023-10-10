# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:51:16 2023

@author: Luca
"""


import matplotlib.pyplot as plt
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq.callbacks import ActionCallback
from neurodiffeq import diff  # the differentiation operation
import torch
from neurodiffeq.generators import Generator1D
import numpy as np
import torch.nn as nn
from neurodiffeq.networks import FCNN


# Set a fixed random seed:
    
torch.manual_seed(42)


# Set the parameters of the problem

Om_r_0=9.24*10**(-5)



# Set the range of the independent variable:

a_0 = 10**(-3)
a_f = 1

N_0 = np.log(a_0)
N_f = np.log(a_f)

Om_m_0_min=0.2
Om_m_0_max=0.4

# Define the differential equation:
    
def ODE_LCDM(delta, delta_prime, N, Om_m_0):
    
    a_eq=Om_r_0/Om_m_0
    Om_L_0=1-Om_m_0-Om_r_0
    alpha=a_eq**3*Om_L_0/Om_m_0
    
    res1 = diff(delta, N) - delta_prime
    res2 = diff(delta_prime, N) - (3*torch.exp(N)/(2*a_eq*(1+(torch.exp(N)/a_eq)+alpha*(torch.exp(N)/a_eq)**4)))*delta + ((1+4*alpha*(torch.exp(N)/a_eq)**3)/(2*(1+(a_eq/torch.exp(N))+alpha*(torch.exp(N)/a_eq)**3)))*delta_prime
    return [res1 , res2]

# Define the initial condition:

condition = [BundleIVP(N_0, a_0),
             BundleIVP(N_0, a_0)]

# Define a custom loss function:

def weighted_loss_LCDM(res, x, t):
    
    N = t[0]
    w = 2

    loss = (res ** 2) * torch.exp(-w * (N - N_0))
    
    return loss.mean()

# Define the optimizer (this is commented in the solver)

nets = [FCNN(n_input_units=2,  hidden_units=(32,32,)) for _ in range(2)]

#nets = torch.load('nets_LCDM_proof.ph')


adam = torch.optim.Adam(set([p for net in nets for p in net.parameters()]),
                        lr=5e-5)


tgz = Generator1D(256, t_min=N_0, t_max=N_f)#, method='log-spaced-noisy')

vgz = Generator1D(256, t_min=N_0, t_max=N_f)#, method='log-spaced')

tg0 = Generator1D(256, t_min=Om_m_0_min, t_max=Om_m_0_max)#, method='log-spaced-noisy')

vg0 = Generator1D(256, t_min=Om_m_0_min, t_max=Om_m_0_max)#, method='log-spaced')

train_gen = tgz ^ tg0

valid_gen = vgz ^ vg0


# Define the ANN based solver:
    
solver = BundleSolver1D(ode_system=ODE_LCDM,
                        nets=nets,
                        conditions=condition,
                        t_min=N_0, t_max=N_f,
                        theta_min=Om_m_0_min,
                        theta_max=Om_m_0_max,
                        eq_param_index=(0,),
                        optimizer=adam,
                        train_generator=train_gen,
                        valid_generator=valid_gen,
                        loss_fn=weighted_loss_LCDM,
                        )

# Set the amount of interations to train the solver:
iterations = 3000000

# Start training:
solver.fit(iterations)

# Plot the loss during training, and save it:
loss = solver.metrics_history['train_loss']
plt.plot(loss, label='training loss')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.suptitle('Total loss during training')
plt.savefig('loss_LCDM.png')

# Save the neural network:
torch.save(solver._get_internal_variables()['best_nets'], 'nets_LCDM.ph')
