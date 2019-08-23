from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm,uniform
from necro import model
import seaborn as sns
from matplotlib import pyplot as plt
import random
sns.set()
random.seed(0)

idx = list(range(14, 51,1)) #index for parameter values after initial conditions 14-50
counter = 0

#CHAINS
chain0 = np.load('new_wseed_50000_6_29/newdreamzs_5chain_sampled_params_chainnew_0_50000.npy')
chain1 = np.load('new_wseed_50000_6_29/newdreamzs_5chain_sampled_params_chainnew_1_50000.npy')
chain2 = np.load('new_wseed_50000_6_29/newdreamzs_5chain_sampled_params_chainnew_2_50000.npy')
chain3 = np.load('new_wseed_50000_6_29/newdreamzs_5chain_sampled_params_chainnew_3_50000.npy')
chain4 = np.load('new_wseed_50000_6_29/newdreamzs_5chain_sampled_params_chainnew_4_50000.npy')

total_iterations = chain0.shape[0]
burnin = int(total_iterations/2)
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :], chain2[burnin:, :], chain3[burnin:, :], chain4[burnin:, :]))
ndims = len(idx)

plt.figure(figsize=(15, 10))
for i in range(1,38): #range of how many params in model
    plt.subplot(8, 5, i) #number of rows/columns
    sns.distplot(samples[:, counter])
    plt.title(model.parameters[idx[counter]].name, fontdict={'fontsize': 10})
    counter += 1
    plt.xlabel("Log(10) Value", fontsize=10)
    plt.ylabel("Probability", fontsize=9, labelpad=15)
# plt.savefig('pydream_priorpost_traceplot_629_all_chains.pdf', format='pdf', bbox_inches="tight")
plt.show()
