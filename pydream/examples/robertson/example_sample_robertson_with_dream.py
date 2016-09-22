# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:58:34 2016

@author: Erin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:26:46 2014
@author: Erin
"""

from core import run_dream
from pysb.integrate import Solver
import numpy as np
from parameters import UniformParam
from scipy.stats import norm

from pysb.examples.robertson import model

#Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0,40)
solver = Solver(model, tspan)

#Load experimental data to which Robertson model will be fit here.  The "experimental data" in this case is just the total C trajectory at the default model parameters with a standard deviation of .01.
location= 'exp_data/'
exp_data_ctot = np.loadtxt(location+'exp_data_ctotal.txt')

exp_data_sd_ctot = np.loadtxt(location+'exp_data_sd_ctotal.txt')

#Create scipy normal probability distributions for data likelihoods
like_ctot = norm(loc=exp_data_ctot, scale=exp_data_sd_ctot)

#Create lists of sampled pysb parameter names to use for subbing in parameter values in likelihood function.
pysb_sampled_parameter_names = [param.name for param in model.parameters_rules()]

#Define likelihood function to generate simulated data that corresponds to experimental time points.  
#This function should take as input a parameter vector (parameter values are in the order dictated by first argument to run_dream function below).
#The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector):    
    
    param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, parameter_vector)}
  
    for pname, pvalue in param_dict.items():   
        
        #Change model parameter values to current location in parameter space
        
        model.parameters[pname].value = 10**(pvalue)
    
    #Simulate "experimentally measured" Ctotal values. 
    
    solver.run()
    
    #Calculate log probability contribution from simulated "experimental" values.
    
    logp_ctotal = np.sum(like_ctot.logpdf(solver.yobs['C_total']))
    
    #If model simulation failed due to integrator errors, return a log probability of -inf.
    if np.isnan(logp_ctotal):
        logp_ctotal = -np.inf
      
    return logp_ctotal


# Add vector of PySB rate parameters to be sampled as unobserved random variables to DREAM with uniform priors.
  
original_params = np.log10([param.value for param in model.parameters_rules()])
#Set upper and lower limits for uniform prior to be 3 orders of magnitude above and below original parameter values.
lower_limits   = original_params - 3
upper_limits = original_params + 3 

parameters_to_sample = UniformParam('parameters_to_sample', value = [1]*len(lower_limits), lower=lower_limits, upper=upper_limits)  

sampled_parameter_names = [parameters_to_sample]

#DREAM should be run with at least 3 chains.
nchains = 5

#Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=100000, nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='robertson_dreamzs_5chain_', verbose=True)
    
#Save sampling output (sampled parameter values and their corresponding logps).
for chain in range(len(sampled_params)):
    np.save('robertson_dreamzs_5chain_sampled_params_chain_'+str(chain), sampled_params[chain])
    np.save('robertson_dreamzs_5chain_logps_chain_'+str(chain), log_ps[chain])
