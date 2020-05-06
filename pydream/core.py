# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from . import Dream_shared_vars
from .Dream import Dream, DreamPool
from .model import Model
import traceback


def run_dream(parameters, likelihood, nchains=5, niterations=50000, start=None, restart=False, verbose=True, nverbose=10, tempering=False, mp_context=None, **kwargs):
    """Run DREAM given a set of parameters with priors and a likelihood function.

    Parameters
    ----------
    parameters: iterable of SampledParam class
        A list of parameter priors
    likelihood: function
        A user-defined likelihood function
    nchains: int, optional
        The number of parallel DREAM chains to run.  Default = 5
    niterations: int, optional
        The number of algorithm iterations to run. Default = 50,000
    start: iterable of arrays or single array, optional
        Either a list of start locations to initialize chains in, or a single start location to initialize all chains in. Default: None
    restart: Boolean, optional
        Whether run is a continuation of an earlier run.  Pass this with the model_name argument to automatically load previous history and crossover probability files.  Default: False
    verbose: Boolean, optional
        Whether to print verbose output (including acceptance or rejection of moves and the current acceptance rate).  Default: True
    tempering: Boolean, optional
        Whether to use parallel tempering for the DREAM chains.  Warning: this feature is untested.  Use at your own risk! Default: False
    mp_context: multiprocessing context or None.
        Method used to to start the processes. If it's None, the default context, which depends in Python version and OS, is used.
        For more information please check: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    kwargs:
        Other arguments that will be passed to the Dream class on initialization.  For more information, see Dream class.

    Returns
    -------
    sampled_params : list of arrays
        Sampled parameters for each chain
    log_ps : list of arrays
        Log probability for each sampled point for each chain
        """

    if restart:
        if start == None:
            raise Exception('Restart run specified but no start positions given.')
        if 'model_name' not in kwargs:
            raise Exception('Restart run specified but no model name to load history and crossover value files from given.')

    if type(parameters) is not list:
        parameters = [parameters]

    model = Model(likelihood=likelihood, sampled_parameters=parameters)
    
    if restart:
        step_instance = Dream(model=model, variables=parameters,
                              history_file=kwargs['model_name'] + '_DREAM_chain_history.npy',
                              crossover_file=kwargs['model_name'] + '_DREAM_chain_adapted_crossoverprob.npy',
                              gamma_file=kwargs['model_name'] + '_DREAM_chain_adapted_gammalevelprob.npy',
                              verbose=verbose, mp_context=mp_context, **kwargs)
    else:
        step_instance = Dream(model=model, variables=parameters, verbose=verbose, mp_context=mp_context, **kwargs)

    pool = _setup_mp_dream_pool(nchains, niterations, step_instance, start_pt=start, mp_context=mp_context)
    try:
        if tempering:

            sampled_params, log_ps = _sample_dream_pt(nchains, niterations, step_instance, start, pool, verbose=verbose)

        else:

            if type(start) is list:
                args = zip([step_instance]*nchains, [niterations]*nchains, start, [verbose]*nchains, [nverbose]*nchains)

            else:
                args = list(zip([step_instance]*nchains, [niterations]*nchains, [start]*nchains, [verbose]*nchains, [nverbose]*nchains))

            returned_vals = pool.map(_sample_dream, args)
            sampled_params = [val[0] for val in returned_vals]
            log_ps = [val[1] for val in returned_vals]
    finally:
        pool.close()
        pool.join()
    return sampled_params, log_ps


def _sample_dream(args):

    try: 
        dream_instance = args[0]
        iterations = args[1]
        start = args[2]
        verbose = args[3]
        nverbose = args[4]
        step_fxn = getattr(dream_instance, 'astep')
        sampled_params = np.empty((iterations, dream_instance.total_var_dimension))
        log_ps = np.empty((iterations, 1))
        q0 = start
        naccepts = 0
        naccepts100win = 0
        for iteration in range(iterations):
            if iteration%nverbose == 0:
                acceptance_rate = float(naccepts)/(iteration+1)
                if verbose:
                    print('Iteration: ',iteration,' acceptance rate: ',acceptance_rate)
                if iteration%100 == 0:
                    acceptance_rate_100win = float(naccepts100win)/100
                    if verbose:
                        print('Iteration: ',iteration,' acceptance rate over last 100 iterations: ',acceptance_rate_100win)
                    naccepts100win = 0
            old_params = q0
            sampled_params[iteration], log_prior , log_like = step_fxn(q0)
            log_ps[iteration] = log_like + log_prior
            q0 = sampled_params[iteration]
            if old_params is None:
                old_params = q0

            if np.any(q0 != old_params):
                naccepts += 1
                naccepts100win += 1
            
    except Exception as e:
        traceback.print_exc()
        print()
        raise e

    return sampled_params, log_ps

def _sample_dream_pt(nchains, niterations, step_instance, start, pool, verbose):
    
    T = np.zeros((nchains))
    T[0] = 1.
    for i in range(nchains):
        T[i] = np.power(.001, (float(i)/nchains))
    
    step_instances = [step_instance]*nchains   
    
    if type(start) is list:
        args = list(zip(step_instances, start, T, [None]*nchains, [None]*nchains))
    else:
        args = list(zip(step_instances, [start]*nchains, T, [None]*nchains, [None]*nchains))
        
    sampled_params = np.zeros((nchains, niterations*2, step_instance.total_var_dimension))
    log_ps = np.zeros((nchains, niterations*2, 1))
    
    q0 = start
    naccepts = np.zeros((nchains))
    naccepts100win = np.zeros((nchains))
    nacceptsT = np.zeros((nchains))
    nacceptsT100win = np.zeros((nchains))
    ttestsper100 = 100./nchains
    
    for iteration in range(niterations):
        itidx = iteration*2
        if iteration%10 == 0:
            ttests = iteration/float(nchains)
            ntests = ttests + iteration
            acceptance_rate = naccepts/(ntests+1)
            Tacceptance_rate = nacceptsT/(ttests+1)
            overall_Tacceptance_rate = np.sum(nacceptsT)/(iteration+1)
            if verbose:
                print('Iteration: ',iteration,' overall acceptance rate: ',acceptance_rate,' temp swap acceptance rate per chain: ',Tacceptance_rate,' and overall temp swap acceptance rate: ',overall_Tacceptance_rate)
            if iteration%100 == 0:
                acceptance_rate_100win = naccepts100win/(100 + ttestsper100)
                Tacceptance_rate_100win = nacceptsT100win/ttestsper100
                overall_Tacceptance_rate_100win = np.sum(nacceptsT100win)/100
                if verbose:
                    print('Iteration: ',iteration,' overall acceptance rate over last 100 iterations: ',acceptance_rate_100win,' temp swap acceptance rate: ',Tacceptance_rate_100win,' and overall temp swap acceptance rate: ',overall_Tacceptance_rate_100win)
                naccepts100win = np.zeros((nchains))
                nacceptsT100win = np.zeros((nchains))

        returned_vals = pool.map(_sample_dream_pt_chain, args)
        qnews = [val[0] for val in returned_vals]
        logprinews = [val[1] for val in returned_vals]
        loglikenews = [val[2] for val in returned_vals]
        dream_instances = [val[3] for val in returned_vals]
        logpnews = [T[i]*loglikenews[i] + logprinews[i] for i in range(nchains)]             
        
        for chain in range(nchains):
            sampled_params[chain][itidx] = qnews[chain]
            log_ps[chain][itidx] = logpnews[chain]

        random_chains = np.random.choice(nchains, 2, replace=False)
        loglike1 = loglikenews[random_chains[0]]
        T1 = T[random_chains[0]]
        loglike2 = loglikenews[random_chains[1]]
        T2 = T[random_chains[1]]
        logp1 = logpnews[random_chains[0]]
        logp2 = logpnews[random_chains[1]]
        
            
        alpha = ((T1*loglike2)+(T2*loglike1))-((T1*loglike1)+(T2*loglike2))
        
        if np.log(np.random.uniform()) < alpha:
            if verbose:
                print('Accepted temperature swap of chains: ',random_chains,' at temperatures: ',T1,' and ',T2,' and logps: ',logp1,' and ',logp2)
            nacceptsT[random_chains[0]] += 1
            nacceptsT[random_chains[1]] += 1
            nacceptsT100win[random_chains[0]] += 1
            nacceptsT100win[random_chains[1]] += 1    
            old_qs = list(qnews)
            old_logps = list(logpnews)
            old_loglikes = list(loglikenews)
            old_logpri = list(logprinews)
            qnews[random_chains[0]] = old_qs[random_chains[1]]
            qnews[random_chains[1]] = old_qs[random_chains[0]]
            logpnews[random_chains[0]] = old_logps[random_chains[1]]
            logpnews[random_chains[1]] = old_logps[random_chains[0]]
            loglikenews[random_chains[0]] = old_loglikes[random_chains[1]]
            loglikenews[random_chains[1]] = old_loglikes[random_chains[0]]
            logprinews[random_chains[0]] = old_logpri[random_chains[1]]
            logprinews[random_chains[1]] = old_logpri[random_chains[0]]
        else:
            if verbose:
                print('Did not accept temperature swap of chains: ',random_chains,' at temperatures: ',T1,' and ',T2,' and logps: ',logp1,' and ',logp2)
        
        for chain in range(nchains):
            sampled_params[chain][itidx+1] = qnews[chain]
            log_ps[chain][itidx+1] = logpnews[chain]
                
        for i, q in enumerate(qnews):
            try:
                if not np.all(q == q0[i]):
                    naccepts[i] += 1
                    naccepts100win[i] += 1
            except TypeError:
                #On first iteration without starting points this will fail because q0 == None
                pass
            
        args = list(zip(dream_instances, qnews, T, loglikenews, logprinews))
        q0 = qnews
    
    return sampled_params, log_ps
            

def _sample_dream_pt_chain(args):

    dream_instance = args[0]
    start = args[1]
    T = args[2]
    last_loglike = args[3]
    last_logpri = args[4]
    step_fxn = getattr(dream_instance, 'astep')
    q1, logprior1, loglike1 = step_fxn(start, T, last_loglike, last_logpri)
    
    return q1, logprior1, loglike1, dream_instance

def _setup_mp_dream_pool(nchains, niterations, step_instance, start_pt=None, mp_context=None):
    
    min_njobs = (2*len(step_instance.DEpairs))+1
    if nchains < min_njobs:
        raise Exception('Dream should be run with at least (2*DEpairs)+1 number of chains.  For current algorithmic settings, set njobs>=%s.' %str(min_njobs))
    if step_instance.history_file != False:
        old_history = np.load(step_instance.history_file)
        len_old_history = len(old_history.flatten())
        nold_history_records = len_old_history/step_instance.total_var_dimension
        step_instance.nseedchains = nold_history_records
        if niterations < step_instance.history_thin:
            arr_dim = ((np.floor(nchains*niterations/step_instance.history_thin)+nchains)*step_instance.total_var_dimension)+len_old_history
        else:
            arr_dim = np.floor((((nchains*niterations)*step_instance.total_var_dimension)/step_instance.history_thin))+len_old_history
    else:
        if niterations < step_instance.history_thin:
            arr_dim = ((np.floor(nchains*niterations/step_instance.history_thin)+nchains)*step_instance.total_var_dimension)+(step_instance.nseedchains*step_instance.total_var_dimension)
        else:
            arr_dim = np.floor(((nchains*niterations/step_instance.history_thin)*step_instance.total_var_dimension))+(step_instance.nseedchains*step_instance.total_var_dimension)
            
    min_nseedchains = 2*len(step_instance.DEpairs)*nchains
    
    if step_instance.nseedchains < min_nseedchains:
        raise Exception('The size of the seeded starting history is insufficient.  Increase nseedchains>=%s.' %str(min_nseedchains))
        
    current_position_dim = nchains*step_instance.total_var_dimension
    # Get context to define arrays
    if mp_context is None:
        ctx = mp.get_context(mp_context)
    else:
        ctx = mp_context
    history_arr = ctx.Array('d', [0] * int(arr_dim))
    if step_instance.history_file != False:
        history_arr[0:len_old_history] = old_history.flatten()
    nCR = step_instance.nCR
    ngamma = step_instance.ngamma
    crossover_setting = step_instance.CR_probabilities
    crossover_probabilities = ctx.Array('d', crossover_setting)
    ncrossover_updates = ctx.Array('d', [0] * nCR)
    delta_m = ctx.Array('d', [0] * nCR)
    gamma_level_setting = step_instance.gamma_probabilities
    gamma_probabilities = ctx.Array('d', gamma_level_setting)
    ngamma_updates = ctx.Array('d', [0] * ngamma)
    delta_m_gamma = ctx.Array('d', [0] * ngamma)
    current_position_arr = ctx.Array('d', [0] * current_position_dim)
    shared_nchains = ctx.Value('i', nchains)
    n = ctx.Value('i', 0)
    tf = ctx.Value('c', b'F')
    
    if step_instance.crossover_burnin == None:
        step_instance.crossover_burnin = int(np.floor(niterations/10))
        
    if start_pt != None:
        if step_instance.start_random:
            print('Warning: start position provided but random_start set to True.  Overrode random_start value and starting walk at provided start position.')
            step_instance.start_random = False

    p = DreamPool(nchains, context=ctx, initializer=_mp_dream_init,
                  initargs=(history_arr, current_position_arr, shared_nchains,
                            crossover_probabilities, ncrossover_updates, delta_m,
                            gamma_probabilities, ngamma_updates, delta_m_gamma, n, tf,))
    # p = mp.pool.ThreadPool(nchains, initializer=_mp_dream_init, initargs=(history_arr, current_position_arr, shared_nchains, crossover_probabilities, ncrossover_updates, delta_m, gamma_probabilities, ngamma_updates, delta_m_gamma, n, tf, ))
    # p = mp.Pool(nchains, initializer=_mp_dream_init, initargs=(history_arr, current_position_arr, shared_nchains, crossover_probabilities, ncrossover_updates, delta_m, gamma_probabilities, ngamma_updates, delta_m_gamma, n, tf, ))

    return p

def _mp_dream_init(arr, cp_arr, nchains, crossover_probs, ncrossover_updates, delta_m, gamma_probs, ngamma_updates, delta_m_gamma, val, switch):
      Dream_shared_vars.history = arr
      Dream_shared_vars.current_positions = cp_arr
      Dream_shared_vars.nchains = nchains
      Dream_shared_vars.cross_probs = crossover_probs
      Dream_shared_vars.ncr_updates = ncrossover_updates
      Dream_shared_vars.delta_m = delta_m
      Dream_shared_vars.gamma_level_probs = gamma_probs
      Dream_shared_vars.ngamma_updates = ngamma_updates
      Dream_shared_vars.delta_m_gamma = delta_m_gamma
      Dream_shared_vars.count = val
      Dream_shared_vars.history_seeded = switch
