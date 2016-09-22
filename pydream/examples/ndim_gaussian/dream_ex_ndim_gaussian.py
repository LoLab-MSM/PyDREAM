# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:34 2015

@author: Erin
"""

# An implementation of example 2 from MT-DREAM(ZS) original Matlab code. (see Laloy and Vrugt 2012)
# 100 dimensional Gaussian distribution

import numpy as np
import os
from parameters import FlatParam
from core import run_dream

def Latin_hypercube(minn, maxn, N):
    y = np.random.rand(N, len(minn))
    x = np.zeros((N, len(minn)))
    
    for j in range(len(minn)):
        idx = np.random.permutation(N)
        P = (idx - y[:,j])/N
        x[:,j] = minn[j] + P * (maxn[j] - minn[j])
    
    return x


d = 100
A = .5 * np.identity(d) + .5 * np.ones((d,d))
C = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        C[i][j] = A[i][j] * np.sqrt((i+1)*(j+1))

invC = np.linalg.inv(C)
mu = np.zeros(d)

if d > 150:
    log_F = 0
else:
    log_F = np.log(((2 * np.pi)**(-d/2))*np.linalg.det(C)**(- 1./2))

#Create initial samples matrix m that will be loaded in as DREAM history file
m = Latin_hypercube(np.linspace(-5, -5, num=d), np.linspace(15, 15, num=d), 1000)

np.save('ndim_gaussian_seed.npy', m)

def likelihood(param_vec):
    logp = log_F - .5 * np.sum(param_vec*np.dot(invC, param_vec))
    
    return logp

starts = [m[chain] for chain in range(3)]

params = FlatParam('params', value=mu)       
    
sampled_params, log_ps = run_dream([params], likelihood, niterations=50000, nchains=3, start=starts, start_random=False, save_history=True, adapt_gamma=False, gamma_levels=1, tempering=False, history_file='ndim_gaussian_seed.npy', multitry=5, parallel=False)
    
for chain in range(len(sampled_params)):
    np.save('ndimgauss_mtdreamzs_3chain_a1_sampled_params_chain_'+str(chain), sampled_params[chain])
    np.save('ndimgauu_mtdreamzs_3chain_a1_logps_chain_'+str(chain), log_ps[chain])

os.remove('ndim_gaussian_seed.npy')

    
    


