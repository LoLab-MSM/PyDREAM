# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:34 2015

@author: Erin
"""

# An implementation of example 3 from MT-DREAM(ZS) original Matlab code.
# Mixture Model

import numpy as np
import os
import scipy.linalg
from pydream.parameters import FlatParam
from pydream.core import run_dream
from pydream.convergence import  Gelman_Rubin

log_F = np.array([-10.2880, -9.5949])

d = 10
k = 2
log_prior = np.log(np.array([.3333, .6666]))
mu1 = np.linspace(-5, -5, num=d)
mu2 = np.linspace(5, 5, num=d)
mu = np.array([mu1,mu2])
C = np.identity(d)
L = scipy.linalg.cholesky(C, lower=False)
diagL = np.diag(L)
logDetSigma = 2 * np.sum(np.log(diagL))
cov = np.identity(10)
mean = np.linspace(0, 0, num=10)
#Create initial samples matrix m that will be loaded in as DREAM history file
m = np.random.multivariate_normal(mean, cov, size=100)

np.save('mixturemodel_seed.npy', m)

def likelihood(params):
    log_lh = np.zeros((k))
    for j in range(2):
        log_lh[j] = -.5 * np.sum((params - mu[j,:])**2) + log_F[j]
    maxll = np.max(log_lh)
    post = np.array(np.exp(log_lh - maxll), dtype='float64')
    density = np.sum(post)
    post = post/float(density)
    log_L = np.log(density) + maxll
    #print 'params: ',params,'log_L: ',log_L,'log_lh: ',log_lh,'maxll: ',maxll,'post: ',post,'density: ',density
    
    return log_L
    
params = FlatParam(test_value=mean)
 
starts = [m[chain] for chain in range(3)]

if __name__ == '__main__':

    niterations=50000
    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = niterations
    nchains=3

    sampled_params, log_ps = run_dream([params], likelihood, niterations=niterations, nchains=nchains, start=starts, start_random=False, save_history=True, history_file='mixturemodel_seed.npy', multitry=5, parallel=False)
    
    for chain in range(len(sampled_params)):
        np.save('mixmod_mtdreamzs_3chain_sampled_params_chain_'+str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('mixmod_mtdreamzs_3chain_logps_chain_'+str(chain)+'_'+str(total_iterations), log_ps[chain])

    os.remove('mixturemodel_seed.npy')

    try:
        #Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt
        total_iterations = len(sampled_params[0])
        burnin = total_iterations/2
        samples = np.concatenate((sampled_params[0][burnin:, :], sampled_params[1][burnin:, :], sampled_params[2][burnin:, :]))

        ndims = len(sampled_params[0][0])
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim])
            fig.savefig('PyDREAM_example_MixtureModel_dimension_'+str(dim))

    except ImportError:
        pass

else:
    run_kwargs = {'parameters':params, 'likelihood':likelihood, 'niterations':50000, 'nchains':3, 'start':starts, 'start_random':False,\
                  'multitry':5, 'adapt_gamma':True, 'history_thin':1, 'model_name':'corm_dreamzs_5chain', 'verbose':True, \
                  'save_history':True, 'history_file':'mixturemodel_seed.npy', 'parallel':False}

    
    


