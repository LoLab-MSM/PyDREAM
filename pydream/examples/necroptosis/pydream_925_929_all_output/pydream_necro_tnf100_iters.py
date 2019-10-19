import sys
sys.path.append('..')
sys.path.insert(0,'/home/ildefog/tropical')
# from necroptosismodule import model
from pylab import *
from pysb.core import *
from pysb.bng import *
from pysb.integrate import *
import matplotlib.pyplot as plt
import numpy as np
from pysb.util import alias_model_components
from pysb.simulator import ScipyOdeSimulator
from necro_uncal_new import model
import pandas as pd
alias_model_components(model)

chain0 = np.load('necro_smallest_dreamzs_5chain_sampled_params_chain925_0_50000.npy')
chain1 = np.load('necro_smallest_dreamzs_5chain_sampled_params_chain925_1_50000.npy')
chain2 = np.load('necro_smallest_dreamzs_5chain_sampled_params_chain925_2_50000.npy')
chain3 = np.load('necro_smallest_dreamzs_5chain_sampled_params_chain925_3_50000.npy')
chain4 = np.load('necro_smallest_dreamzs_5chain_sampled_params_chain925_4_50000.npy')

total_iterations = chain0.shape[0]
burnin = int(total_iterations/2)
samples = np.concatenate((chain0[burnin:30000, :], chain1[burnin:30000, :], chain2[burnin:30000, :], chain3[burnin:30000, :], chain4[burnin:30000, :]))
np.save('necro_pydream_5chns_929_100tnf.npy', samples)


par_files = np.load('necro_pydream_5chns_929_100tnf.npy')
n_pars = len(par_files)
all_pars = np.zeros((n_pars, len(model.parameters)))

rate_params = model.parameters_rules() # these are only the parameters involved in the rules
param_values = np.array([p.value for p in model.parameters]) # these are all the parameters
rate_mask = np.array([p in rate_params for p in model.parameters])

for i in range(n_pars):
    par = par_files[i]
    param_values[rate_mask] = 10 ** par
    all_pars[i] = param_values
obs_y_range = ['MLKLa_obs']


tnf = [2326]
t = np.array([0, 30, 90, 270, 480, 600, 720, 840, 960])
tspan = np.linspace(0, 1440, 100)
solver100 = ScipyOdeSimulator(model, tspan=tspan, verbose = True)
result100 = solver100.run(initials= {TNF(brec=None): tnf}, param_values=all_pars[:], num_processors=20)
df = result100.dataframe
result100.save('necro_25000params_100TNF_pydream_5chns.h5')
