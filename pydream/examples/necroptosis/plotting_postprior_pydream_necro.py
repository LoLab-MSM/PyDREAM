from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm,uniform
from necro_uncal_new import model
import seaborn as sns
from matplotlib import pyplot as plt
import random
sns.set()
random.seed(0)

idx = list(range(14, 54,1)) #index for parameter values after initial conditions 14-50
counter = 0

#CHAINS
chain0 = np.load('dreamzs_5chain_sampled_params_chain_922_0_50000.npy')
chain1 = np.load('dreamzs_5chain_sampled_params_chain_922_1_50000.npy')
chain2 = np.load('dreamzs_5chain_sampled_params_chain_922_2_50000.npy')
chain3 = np.load('dreamzs_5chain_sampled_params_chain_922_3_50000.npy')
chain4 = np.load('dreamzs_5chain_sampled_params_chain_922_4_50000.npy')

logps = np.load('dreamzs_5chain_logps_chain_922_4_50000.npy')
# print(logps)
# quit()

total_iterations = chain0.shape[0]
burnin = int(total_iterations/2)
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :], chain2[burnin:, :], chain3[burnin:, :], chain4[burnin:, :]))
# print(10 ** samples[:2])
# quit()
samples_nonlog = 10 ** samples
ndims = len(idx)

plt.figure(figsize=(15, 10))
for i in range(1,40): #range of how many params in model
    plt.subplot(8, 5, i) #number of rows/columns
    sns.distplot(samples_nonlog[:, counter])
    plt.title(model.parameters[idx[counter]].name, fontdict={'fontsize': 10})
    counter += 1
    plt.xlabel("Log(10) Value", fontsize=10)
    plt.ylabel("Probability", fontsize=9, labelpad=15)
# plt.savefig('pydream_priorpost_traceplot_629_all_chains.pdf', format='pdf', bbox_inches="tight")
plt.show()
