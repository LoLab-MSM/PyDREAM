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
from parameters import NormalParam
from scipy.stats import norm

from corm import model as cox2_model

#Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0,10, num=100)
solver = Solver(cox2_model, tspan)

#Load experimental data to which CORM model will be fit here
location= 'exp_data/'
exp_data_PG = np.loadtxt(location+'exp_data_pg.txt')
exp_data_PGG = np.loadtxt(location+'exp_data_pgg.txt')

exp_data_sd_PG = np.loadtxt(location+'exp_data_sd_pg.txt')
exp_data_sd_PGG = np.loadtxt(location+'exp_data_sd_pgg.txt')

#Experimental starting values of AA and 2-AG species (all in microM).
exp_cond_AA = [0, .5, 1, 2, 4, 8, 16]
exp_cond_AG = [0, .5, 1, 2, 4, 8, 16]

#Experimentally measured parameter values. These will not be sampled with DREAM.
KD_AA_cat1 = np.log10(cox2_model.parameters['kr_AA_cat1'].value/cox2_model.parameters['kf_AA_cat1'].value)
kcat_AA1 = np.log10(cox2_model.parameters['kcat_AA1'].value)
KD_AG_cat1 = np.log10(cox2_model.parameters['kr_AG_cat1'].value/cox2_model.parameters['kf_AG_cat1'].value)
kcat_AG1 = np.log10(cox2_model.parameters['kcat_AG1'].value)
KD_AG_allo3 = np.log10(cox2_model.parameters['kr_AG_allo3'].value/cox2_model.parameters['kf_AG_allo3'].value)

#generic kf in units of inverse microM*s (matches model units).  All kf reactions are assumed to be diffusion limited.
generic_kf = np.log10(1.5e4)

#Create scipy normal probability distributions for data likelihoods
like_PGs = norm(loc=exp_data_PG, scale=exp_data_sd_PG)
like_PGGs = norm(loc=exp_data_PGG, scale=exp_data_sd_PGG)
like_thermobox = norm(loc=1, scale=1e-2)

#Create lists of sampled pysb parameter names to use for subbing in parameter values in likelihood function and for setting all kfs to diffusion limited value.
pysb_sampled_parameter_names = ['kr_AA_cat2', 'kcat_AA2', 'kr_AA_cat3', 'kcat_AA3', 'kr_AG_cat2', 'kr_AG_cat3', 'kcat_AG3', 'kr_AA_allo1', 'kr_AA_allo2', 'kr_AA_allo3', 'kr_AG_allo1', 'kr_AG_allo2']
kfs_to_change = ['kf_AA_cat2', 'kf_AA_cat3', 'kf_AG_cat2', 'kf_AG_cat3', 'kf_AA_allo1', 'kf_AA_allo2', 'kf_AA_allo3', 'kf_AG_allo1', 'kf_AG_allo2']
kf_idxs = [i for i, param in enumerate(cox2_model.parameters) if param.name in kfs_to_change]

#Define likelihood function to generate simulated data that corresponds to experimental time points.  
#This function should take as input a parameter vector (parameter values are in the order dictated by first argument to run_dream function below).
#The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector):    
    
    param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, parameter_vector)}
  
    for pname, pvalue in param_dict.items():   
        
        #Change model parameter values to current location in parameter space
        
        if 'kr' in pname:
            cox2_model.parameters[pname].value = 10**(pvalue + generic_kf)
        
        elif 'kcat' in pname:
            cox2_model.parameters[pname].value = 10**pvalue
    
    #Simulate experimentally measured PG and PGG values at all experimental AA and 2-AG starting conditions.    
    
    PG_array = np.zeros((7,7), dtype='float64')
    PGG_array = np.zeros((7,7), dtype='float64')    
    
    arr_row = 0
    arr_col = 0
    
    for AA_init in exp_cond_AA:
        for AG_init in exp_cond_AA:
            cox2_model.parameters['AA_0'].value = AA_init
            cox2_model.parameters['AG_0'].value = AG_init
            solver.run()
            PG_array[arr_row, arr_col] = solver.yobs['obsPG'][-1]
            PGG_array[arr_row, arr_col] = solver.yobs['obsPGG'][-1]
            if arr_col < 6:        
                arr_col += 1
            else:
                arr_col = 0
        arr_row += 1
    
    #Calculate log probability contribution from simulated PG and PGG values.
    
    logp_PG = np.sum(like_PGs.logpdf(PG_array))
    logp_PGG = np.sum(like_PGGs.logpdf(PGG_array))
    
    #Calculate conservation for thermodynamic boxes in enyzme-substrate interaction diagram.    
    
    box1 = (1/(10**KD_AA_cat1))*(1/(10**param_dict['kr_AA_allo2']))*(10**param_dict['kr_AA_cat3'])*(10**param_dict['kr_AA_allo1'])
    box2 = (1/(10**param_dict['kr_AA_allo1']))*(1/(10**param_dict['kr_AG_cat3']))*(10**param_dict['kr_AA_allo3'])*(10**KD_AG_cat1)
    box3 = (1/(10**param_dict['kr_AG_allo1']))*(1/(10**param_dict['kr_AA_cat2']))*(10**param_dict['kr_AG_allo2'])*(10**KD_AA_cat1)
    box4 = (1/(10**KD_AG_cat1))*(1/(10**KD_AG_allo3))*(10**param_dict['kr_AG_cat2'])*(10**param_dict['kr_AG_allo1'])
   
    #Calculate log probability contribution from thermodynamic box energy conservation. 
    
    logp_box1 = like_thermobox.logpdf(box1)
    logp_box2 = like_thermobox.logpdf(box2)
    logp_box3 = like_thermobox.logpdf(box3)
    logp_box4 = like_thermobox.logpdf(box4)
    
    
    total_logp = logp_PG + logp_PGG + logp_box1 + logp_box2 + logp_box3 + logp_box4
    
    #If model simulation failed due to integrator errors, return a log probability of -inf.
    if np.isnan(total_logp):
        total_logp = -np.inf
      
    return total_logp


# Add PySB rate parameters to be sampled as unobserved random variables to DREAM with normal priors.
   
kd_AA_cat2 = NormalParam('KD_AA_cat2', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_cat2'].value/cox2_model.parameters['kf_AA_cat2'].value), sd=1.5)
kcat_AA2 = NormalParam('kcat_AA2', value = 1, mu=np.log10(cox2_model.parameters['kcat_AA2'].value), sd=.66)
kd_AA_cat3 = NormalParam('KD_AA_cat3', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_cat3'].value/cox2_model.parameters['kf_AA_cat3'].value), sd=1.5)
kcat_AA3 = NormalParam('kcat_AA3', value = 1, mu=np.log10(cox2_model.parameters['kcat_AA1'].value), sd=.66) 
kd_AG_cat2 = NormalParam('KD_AG_cat2', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_cat2'].value/cox2_model.parameters['kf_AG_cat2'].value), sd=1.5)
kd_AG_cat3 = NormalParam('KD_AG_cat3', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_cat3'].value/cox2_model.parameters['kf_AG_cat3'].value), sd=1.5)
kcat_AG3 = NormalParam('kcat_AG3', value = 1, mu=np.log10(cox2_model.parameters['kcat_AG3'].value), sd=.66)
kd_AA_allo1 = NormalParam('KD_AA_allo1', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_allo1'].value/cox2_model.parameters['kf_AA_allo1'].value), sd=1)
kd_AA_allo2 = NormalParam('KD_AA_allo2', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_allo2'].value/cox2_model.parameters['kf_AA_allo2'].value), sd=1)
kd_AA_allo3 = NormalParam('KD_AA_allo3', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_allo3'].value/cox2_model.parameters['kf_AA_allo3'].value), sd=1)
kd_AG_allo1 = NormalParam('KD_AG_allo1', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_allo1'].value/cox2_model.parameters['kf_AG_allo1'].value), sd=1)
kd_AG_allo2 = NormalParam('KD_AG_allo2', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_allo2'].value/cox2_model.parameters['kf_AG_allo2'].value), sd=1)       

sampled_parameter_names = [kd_AA_cat2, kcat_AA2, kd_AA_cat3, kcat_AA3, kd_AG_cat2, kd_AG_cat3, kcat_AG3, kd_AA_allo1, kd_AA_allo2, kd_AA_allo3, kd_AG_allo1, kd_AG_allo2]

#DREAM should be run with at least 3 chains.
nchains = 5

#Change model kf values to be assumed diffusion-limited value.
for idx in kf_idxs:
    cox2_model.parameters[idx].value = 10**generic_kf

#Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=100000, nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='corm_dreamzs_5chain_', verbose=True)
    
#Save sampling output (sampled parameter values and their corresponding logps).
for chain in range(len(sampled_params)):
    np.save('corm_dreamzs_5chain_sampled_params_chain_'+str(chain), sampled_params[chain])
    np.save('corm_dreamzs_5chain_logps_chain_'+str(chain), log_ps[chain])
