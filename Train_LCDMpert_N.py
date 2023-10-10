# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:21:07 2023

@author: Luca
"""

import matplotlib.pyplot as plt
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
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
Om_m_0=0.272
a_eq=Om_r_0/Om_m_0
Om_L_0=1-Om_m_0-Om_r_0
alpha=a_eq**3*Om_L_0/Om_m_0


# Set the range of the independent variable:

a_0 = 10**(-3)
a_f = 1

N_0 = np.log(a_0)
N_f = np.log(a_f)

# Define the differential equation:
    
def ODE_LCDM(delta, delta_prime, N):

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

nets = [FCNN(n_input_units=1,  hidden_units=(32,32,)) for _ in range(2)]

#nets = torch.load('nets_LCDM.ph',
#                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
#                  )


adam = torch.optim.Adam(set([p for net in nets for p in net.parameters()]),
                        lr=0.0001)
    

tgz = Generator1D(128, t_min=N_0, t_max=N_f)#, method='log-spaced-noisy')

vgz = Generator1D(128, t_min=N_0, t_max=N_f)#, method='log-spaced')


# Define the ANN based solver:
    
solver = BundleSolver1D(ode_system=ODE_LCDM,
                        nets=nets,
                        conditions=condition,
                        t_min=N_0, t_max=N_f,
                        optimizer=adam,
                        train_generator=tgz,
                        valid_generator=vgz,
                        loss_fn=weighted_loss_LCDM,
                        )

# Set the amount of interations to train the solver:
iterations = 1000

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
