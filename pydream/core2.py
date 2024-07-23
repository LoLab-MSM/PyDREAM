# -*- coding: utf-8 -*-

from pydream.dream_2 import SharedVariables
from pydream.model import Model
import numpy as np


def run_dream2(parameters, likelihood, nchains=5, niterations=50000, start=None, restart=False, verbose=True,
               nverbose=10, mp_context=None, adapt_gamma=False, history_thin=1, gamma_levels=3,
               multitry=1, model_name='chain', **kwargs):
    if restart:
        if start is None:
            raise Exception('Restart run specified but no start positions given.')

    if type(parameters) is not list:
        parameters = [parameters]

    model = Model(likelihood=likelihood, sampled_parameters=parameters)

    if restart:
        crossover_file = model_name + '_DREAM_chain_adapted_crossoverprob.npy'
        history_file = model_name + '_DREAM_chain_history.npy'
        gamma_file = model_name + '_DREAM_chain_adapted_gammalevelprob.npy'
    else:
        crossover_file = kwargs.get('crossover_file', None)
        history_file = kwargs.get('history_file', None)
        gamma_file = kwargs.get('gamma_file', None)
    if type(start) is list:
        assert len(start) == nchains
        pass
    else:
        start = [start] * nchains
    shared_vars = SharedVariables(model=model, parameters=parameters, number_chains=nchains,
                                  num_iterations=niterations,
                                  crossover_file=crossover_file, history_file=history_file, gamma_file=gamma_file,
                                  save_history=True, n_seed_chains=None, num_cr=3,
                                  mp_context=mp_context, crossover_burnin=None,
                                  starting_positions=start, verbose=verbose, nverbose=nverbose,
                                  adapt_gamma=adapt_gamma,
                                  history_thin=history_thin, gamma_levels=gamma_levels, multitry=multitry,
                                  model_name=model_name
                                  )

    for i in range(niterations):
        shared_vars.run_step(temperature=1)

    sampled_params = np.array([chain.sampled_params for chain in shared_vars.chains])
    log_ps = np.array([chain.log_ps for chain in shared_vars.chains])
    return sampled_params, log_ps
