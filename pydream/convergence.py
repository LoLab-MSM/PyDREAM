import numpy as np
import math

def Gelman_Rubin(sampled_parameters, fburnin=0.5):
    nsamples = len(sampled_parameters[0])
    nchains = len(sampled_parameters)
    nburnin = int(math.floor(nsamples * fburnin))  # math.floor returns an integer

    chain_var = [np.var(sampled_parameters[chain][nburnin:,:], axis=0) for chain in range(nchains)]

    W = np.mean(chain_var, axis=0)

    chain_means = [np.mean(sampled_parameters[chain][nburnin:,:], axis=0) for chain in range(nchains)]

    B = np.var(chain_means, axis=0)

    var_est = W * (1 - (1. / (nsamples - nburnin))) + B

    Rhat = np.sqrt(np.divide(var_est, W))

    return Rhat
