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

from pydream.core import run_dream
from pysb.integrate import Solver
import numpy as np
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm
import inspect
import os.path

from .corm import model as cox2_model

pydream_path = os.path.dirname(inspect.getfile(run_dream))

#Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0,10, num=100)
solver = Solver(cox2_model, tspan)
solver.run()

#Load experimental data to which CORM model will be fit here
location= pydream_path+'/examples/corm/exp_data/'
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

# Add PySB rate parameters to be sampled as unobserved random variables to DREAM with normal priors.

kd_AA_cat2 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AA_cat2'].value / cox2_model.parameters['kf_AA_cat2'].value), scale=1.5)
kcat_AA2 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kcat_AA2'].value), scale=.66)
kd_AA_cat3 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AA_cat3'].value / cox2_model.parameters['kf_AA_cat3'].value), scale=1.5)
kcat_AA3 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kcat_AA1'].value), scale=.66)
kd_AG_cat2 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AG_cat2'].value / cox2_model.parameters['kf_AG_cat2'].value), scale=1.5)
kd_AG_cat3 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AG_cat3'].value / cox2_model.parameters['kf_AG_cat3'].value), scale=1.5)
kcat_AG3 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kcat_AG3'].value), scale=.66)
kd_AA_allo1 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AA_allo1'].value / cox2_model.parameters['kf_AA_allo1'].value), scale=1)
kd_AA_allo2 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AA_allo2'].value / cox2_model.parameters['kf_AA_allo2'].value), scale=1)
kd_AA_allo3 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AA_allo3'].value / cox2_model.parameters['kf_AA_allo3'].value), scale=1)
kd_AG_allo1 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AG_allo1'].value / cox2_model.parameters['kf_AG_allo1'].value), scale=1)
kd_AG_allo2 = SampledParam(norm, loc=np.log10(cox2_model.parameters['kr_AG_allo2'].value / cox2_model.parameters['kf_AG_allo2'].value), scale=1)

sampled_parameter_names = [kd_AA_cat2, kcat_AA2, kd_AA_cat3, kcat_AA3, kd_AG_cat2, kd_AG_cat3, kcat_AG3, kd_AA_allo1, kd_AA_allo2, kd_AA_allo3, kd_AG_allo1, kd_AG_allo2]

# DREAM should be run with at least 3 chains.
# Set these options so script can be called with command line arguments (needed for tests)
nchains = 5

niterations = 10000

# Change model kf values to be assumed diffusion-limited value.
for idx in kf_idxs:
    cox2_model.parameters[idx].value = 10 ** generic_kf

# Define likelihood function to generate simulated data that corresponds to experimental time points.
# This function should take as input a parameter vector (parameter values are in the order dictated by first argument to run_dream function below).
# The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector):

    param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, parameter_vector)}

    for pname, pvalue in list(param_dict.items()):

        # Change model parameter values to current location in parameter space

        if 'kr' in pname:
            cox2_model.parameters[pname].value = 10 ** (pvalue + generic_kf)

        elif 'kcat' in pname:
            cox2_model.parameters[pname].value = 10 ** pvalue

    # Simulate experimentally measured PG and PGG values at all experimental AA and 2-AG starting conditions.

    PG_array = np.zeros((7, 7), dtype='float64')
    PGG_array = np.zeros((7, 7), dtype='float64')

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

    # Calculate log probability contribution from simulated PG and PGG values.

    logp_PG = np.sum(like_PGs.logpdf(PG_array))
    logp_PGG = np.sum(like_PGGs.logpdf(PGG_array))

    # Calculate conservation for thermodynamic boxes in enyzme-substrate interaction diagram.

    box1 = (1 / (10 ** KD_AA_cat1)) * (1 / (10 ** param_dict['kr_AA_allo2'])) * (10 ** param_dict['kr_AA_cat3']) * (
    10 ** param_dict['kr_AA_allo1'])
    box2 = (1 / (10 ** param_dict['kr_AA_allo1'])) * (1 / (10 ** param_dict['kr_AG_cat3'])) * (
    10 ** param_dict['kr_AA_allo3']) * (10 ** KD_AG_cat1)
    box3 = (1 / (10 ** param_dict['kr_AG_allo1'])) * (1 / (10 ** param_dict['kr_AA_cat2'])) * (
    10 ** param_dict['kr_AG_allo2']) * (10 ** KD_AA_cat1)
    box4 = (1 / (10 ** KD_AG_cat1)) * (1 / (10 ** KD_AG_allo3)) * (10 ** param_dict['kr_AG_cat2']) * (
    10 ** param_dict['kr_AG_allo1'])

    # Calculate log probability contribution from thermodynamic box energy conservation.

    logp_box1 = like_thermobox.logpdf(box1)
    logp_box2 = like_thermobox.logpdf(box2)
    logp_box3 = like_thermobox.logpdf(box3)
    logp_box4 = like_thermobox.logpdf(box4)

    total_logp = logp_PG + logp_PGG + logp_box1 + logp_box2 + logp_box3 + logp_box4

    # If model simulation failed due to integrator errors, return a log probability of -inf.
    if np.isnan(total_logp):
        total_logp = -np.inf

    return total_logp


if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = niterations
    sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood, niterations=niterations, nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='corm_dreamzs_5chain', verbose=True)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('corm_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('corm_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

    #Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    np.savetxt('corm_dreamzs_5chain_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)

    old_samples = sampled_params
    if np.any(GR>1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations
            sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood,
                                               niterations=niterations, nchains=nchains, start=starts, multitry=False, gamma_levels=4,
                                               adapt_gamma=True, history_thin=1, model_name='corm_dreamzs_5chain',
                                               verbose=True, restart=True)


            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save('corm_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
                np.save('corm_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ',total_iterations,' GR = ',GR)
            np.savetxt('corm_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

            if np.all(GR<1.2):
                converged = True

    try:
        #Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt
        total_iterations = len(old_samples[0])
        burnin = total_iterations/2
        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :], old_samples[3][burnin:, :], old_samples[4][burnin:, :]))

        ndims = len(sampled_parameter_names)
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
            fig.savefig('PyDREAM_example_CORM_dimension_'+str(dim))

    except ImportError:
        pass

else:

    run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':niterations, 'nchains':nchains, \
                  'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'corm_dreamzs_5chain', 'verbose':False}
