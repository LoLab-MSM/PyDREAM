# -*- coding: utf-8 -*-
import os
import logging

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

from contextlib import contextmanager
import numpy as np
import random
from . import Dream_shared_vars
from datetime import datetime
import multiprocessing as mp
from multiprocessing import pool
import time


@contextmanager
def acquire_locks(*locks):
    # Sort locks by memory address to avoid deadlocks by always acquiring locks in the same order
    sorted_locks = sorted(locks, key=id)
    acquired_locks = []

    try:
        for lock in sorted_locks:
            lock.acquire()
            acquired_locks.append(lock)
        yield
    finally:
        # Release locks in reverse order of acquisition
        for lock in reversed(acquired_locks):
            lock.release()


class Dream(object):
    """An implementation of the MT-DREAM\ :sub:`(ZS)`\  algorithm introduced in:
        Laloy, E. & Vrugt, J. A. High-dimensional posterior exploration of hydrologic models using multiple-try DREAM\ :sub:`(ZS)`\  and high-performance computing. Water Resources Research 48, W01526 (2012).

    Parameters
    ----------
    variables : iterable of instance(s) of SampledParam class
        Model parameters to be sampled with specified prior.
    nseedchains : int
        Number of draws with which to initialize the DREAM history.  Default = 10 * n dimensions
    nCR : int
        Number of crossover values to sample from during run (and to fit during crossover burn-in period).  Default = 3
    adapt_crossover : bool
        Whether to adapt crossover values during the burn-in period.  Default is to adapt.
    crossover_burnin : int
        Number of iterations to fit the crossover values.  Defaults to 10% of total iterations.
    DEpairs : int or list
        Number of chain pairs to use for crossover and selection of next point.  Default = 1.  Can pass a list to have a random number of pairs selected every iteration.
    lamb : float
        e sub d in DREAM papers.  Random error for ergodicity.  Default = .05
    zeta : float
        Epsilon in DREAM papers.  Randomization term. Default = 1e-12
    history_thin : int
        Thinning rate for history to reduce storage requirements.  Every n-th iteration will be added to the history.
    snooker : float
        Probability of proposing a snooker update.  Default is .1.  To forego snooker updates, set to 0.
    p_gamma_unity : float
        Probability of proposing a point with gamma=unity (i.e. a point relatively far from the current point to enable jumping between disconnected modes).  Default = .2.
    start_random : bool
        Whether to intialize chains from a random point in parameter space drawn from the prior (default = yes).  Will override starting position set when sample was called, if any.
    save_history : bool
        Whether to save the history to file at the end of the run (essential if you want to continue the run).  Default is yes.
    history_file : str
        Name of history file to be loaded.  Assumed to be in directory you ran the script from.  If False, no file to be loaded.
    crossover_file : str
        Name of crossover file to be loaded. Assumed to be in directory you ran the script from.  If False, no file to be loaded.
    multitry : bool
        Whether to utilize multi-try sampling.  Default is no.  If set to True, will be set to 5 multiple tries.  Can also directly specify an integer if desired.
    parallel : bool
        Whether to run multi-try samples in parallel (using multiprocessing).  Default is false.  Irrelevant if multitry is set to False.
    verbose : bool
        Whether to print verbose progress.  Default is false.
    model_name : str
        A model name to be used as a prefix when saving history and crossover value files.
    hardboundaries : bool
        Whether to relect point back into bounds of hard prior (i.e., if using a uniform prior, reflect points outside of boundaries back in, so you don't waste time looking at points with logpdf = -inf).
    mp_context : multiprocessing context or None.
        Method used to start the processes. If it's None, the default context, which depends in Python version and OS, is used.
        For more information please check: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    """

    def __init__(self, model, variables=None, nseedchains=None, nCR=3, adapt_crossover=True, adapt_gamma=False,
                 crossover_burnin=None, DEpairs=1, lamb=.05, zeta=1e-12, history_thin=10, snooker=.10,
                 p_gamma_unity=.20, gamma_levels=1, start_random=True, save_history=True, history_file=False,
                 crossover_file=False, gamma_file=False, multitry=False, parallel=False, verbose=False,
                 model_name=False, hardboundaries=True, mp_context=None, **kwargs):

        if isinstance(verbose, bool):
            verbose = logging.DEBUG
        elif not isinstance(verbose, int):
            raise ValueError('log_level must be a boolean, integer or None')
        if logger.getEffectiveLevel() != verbose:
            logger.debug('Changing log_level from %d to %d' % (
                logger.getEffectiveLevel(), verbose))
            logger.setLevel(verbose)
        # Set Dream multiprocessing context
        self.last_like = None
        self.last_prior = None
        self.mp_context = mp_context
        # Set model and variable attributes (if no variables passed, set to all parameters)
        self.model = model
        self.model_name = model_name
        if variables is None:
            self.variables = self.model.sampled_parameters
        else:
            self.variables = variables

        # Calculate total variable dimension and set boundaries
        self.boundaries = hardboundaries
        self.total_var_dimension = 0
        for var in self.variables:
            self.total_var_dimension += var.dsize

        # Set min and max values for boundaries
        if self.boundaries:
            if self.total_var_dimension == 1:
                self.boundary_mask = True
            else:
                self.boundary_mask = np.ones(self.total_var_dimension, dtype=bool)
            self.mins = []
            self.maxs = []
            n = 0
            for var in self.variables:
                interval = var.interval(1)

                if var.dsize > 1:
                    self.mins += list(interval[0])
                    self.maxs += list(interval[1])
                else:
                    self.mins.append(interval[0])
                    self.maxs.append(interval[1])
                n += var.dsize
            self.mins = np.array(self.mins)
            self.maxs = np.array(self.maxs)

        self.nseedchains = nseedchains
        self.nCR = nCR

        # If the number of crossover values is greater than the total variable dimension,
        # set it to be the total variable dimension
        if self.nCR > self.total_var_dimension:
            self.nCR = self.total_var_dimension
            logger.info(f'Warning: the total number of crossover values specified ({nCR}) is less than the total '
                        f'dimension of all variables ({self.total_var_dimension}).  Setting the number of crossover '
                        f'values to be equal to the total variable dimension.')

        # If there is only one variable dimension, don't adapt crossover values
        if self.total_var_dimension == 1 and adapt_crossover:
            adapt_crossover = False
            logger.info('Warning: the total variable dimension = 1, so crossover values will not'
                        ' be adapted, even though crossover adaptation was requested.')

        self.ngamma = gamma_levels
        self.njoint_cr_gamma_probs = nCR * gamma_levels
        self.crossover_burnin = crossover_burnin
        self.crossover_file = crossover_file

        self.adapt_crossover = adapt_crossover

        # Load crossover values from file if given, else set to 1/nCR for all and adapt if requested
        if crossover_file:
            self.CR_probabilities = np.load(crossover_file)
            self.nCR = len(self.CR_probabilities)
            if self.adapt_crossover:
                logger.info('Warning: Crossover values loaded and adapt_crossover = True.  '
                            'Crossover values will be further adapted.')
        else:
            self.CR_probabilities = [1 / float(self.nCR) for _ in range(self.nCR)]

        # Load gamma values from file if given, otherwise set to 1/ngamma for all
        self.adapt_gamma = adapt_gamma
        if gamma_file:
            self.gamma_probabilities = np.load(gamma_file)
            if adapt_gamma:
                logger.info('Warning: Gamma values loaded and adapt gamma = True.  '
                            'Gamma values will be further adapted.')
        else:
            self.gamma_probabilities = [1 / float(self.ngamma) for i in range(self.ngamma)]

        # Set crossover values and gamma (the proportion of dimensions to crossover/gamma level to choose)
        self.CR_values = np.array([m / float(self.nCR) for m in range(1, self.nCR + 1)])
        self.gamma_level_values = np.array([m for m in range(1, self.ngamma + 1)])

        # Set number of pairs to use for determining distance between points for proposals
        self.DEpairs = np.linspace(1, DEpairs, num=DEpairs, dtype=int)  # This is delta in original Matlab code

        self.snooker = snooker
        self.p_gamma_unity = p_gamma_unity

        # If no multitry requested, set value to 1,
        # if requested without a value, set to 5, else set to the value passed
        if not multitry:
            self.num_tries = 1
        elif multitry > 1:
            self.num_tries = multitry
        else:
            self.num_tries = 5

        self.parallel = parallel
        self.lamb = lamb  # This is e sub d in DREAM papers
        self.zeta = zeta  # This is epsilon in DREAM papers
        self.last_logp = None

        # Set the number of seedchains to 10*dimensions to fit
        if self.nseedchains is None:
            self.nseedchains = self.total_var_dimension * 10

        # Set array of gamma values (decreasing step size with increasing level)
        gamma_array = np.zeros((self.ngamma, DEpairs, self.total_var_dimension))
        gamma_level_decrease = 1
        for gamma_level in range(1, self.ngamma + 1):
            for delta in range(1, DEpairs + 1):
                gamma_array[gamma_level - 1, delta - 1, :] = (2.38 / np.sqrt(
                    2 * delta * np.linspace(1, self.total_var_dimension,
                                            num=self.total_var_dimension))) / gamma_level_decrease
            gamma_level_decrease = gamma_level_decrease * 2
        self.gamma_arr = gamma_array
        self.gamma = None

        self.iter = 0
        self.chain_n = None
        self.nchains = None
        self.len_history = 0
        self.save_history = save_history
        self.history_file = history_file
        self.history_thin = history_thin
        self.start_random = start_random
        self.verbose = verbose
        self.logp = self.model.total_logp

    def init_zero(self, q0):

        self.chain_n = Dream_shared_vars.nchains.value - 1
        Dream_shared_vars.nchains.value = Dream_shared_vars.nchains.value - 1

        # Assuming the shared variables exist, seed the history with nseedchain draws from the prior

        if not self.history_file:
            if self.verbose:
                logger.info('History file not loaded.')
            if Dream_shared_vars.history_seeded.value == b'F':
                if self.verbose:
                    logger.info(f'Seeding history with {self.nseedchains} draws from prior.')
                for i in range(self.nseedchains):
                    start_loc = i * self.total_var_dimension
                    end_loc = start_loc + self.total_var_dimension
                    Dream_shared_vars.history[start_loc:end_loc] = self.draw_from_prior(self.variables)
                Dream_shared_vars.history_seeded.value = b'T'
        else:
            if self.verbose:
                logger.info('History file loaded.')
        if self.verbose:
            logger.info('Setting crossover probability starting values.')
            logger.info(f'Set probability of different crossover values to: {self.CR_probabilities}')
        if self.start_random:
            if self.verbose:
                logger.info('Setting start to random draw from prior.')

            q0 = self.draw_from_prior(self.variables, random_seed=True)

        if self.verbose:
            logger.info(f'Start: {q0}')

        # Also get length of history array so we know when to save it at end of run.
        if self.save_history:
            self.len_history = len(np.frombuffer(Dream_shared_vars.history.get_obj()))
        return q0

    def astep(self, q0, temperature=1., last_loglike=None, last_logprior=None):
        self.verbose = False
        self.save_history = False
        cpu_name = mp.current_process().name
        # On first iteration, check that shared variables have been initialized
        # (which only occurs if multiple chains have been started).
        if self.iter == 0:
            logger.debug(f'{cpu_name} init_zero')
            with acquire_locks(Dream_shared_vars.nchains.get_lock(),
                               Dream_shared_vars.history_seeded.get_lock(),
                               Dream_shared_vars.history.get_lock()):
                q0 = self.init_zero(q0)

            logger.debug(f'{cpu_name} done init_zero')

        if last_loglike is not None:
            self.last_like = last_loglike
            self.last_prior = last_logprior
            self.last_logp = temperature * self.last_like + self.last_prior

        if self.last_logp is None:
            self.last_prior, self.last_like = self.logp(q0)
            self.last_logp = temperature * self.last_like + self.last_prior

        logger.debug(f'{cpu_name} set_snooker')
        run_snooker = self.set_snooker()
        run_snooker = True
        logger.debug(f'{cpu_name} set_CR')
        crossover_rates = self.set_crossover_rates(self.CR_probabilities, self.CR_values)

        logger.debug(f'{cpu_name} set_DEpair')
        de_pair_choice = self.set_de_pair(self.DEpairs)

        logger.debug(f'{cpu_name} set_gamma_level')
        gamma_level = self.set_gamma_level(self.gamma_probabilities, self.gamma_level_values)

        # Generate proposal points
        with acquire_locks(Dream_shared_vars.count.get_lock(), Dream_shared_vars.history.get_lock()):
            logger.debug(f'{cpu_name} generate_proposal_points')
            proposed_pts, snooker_logp_prop, z = self.generate_proposal_points(self.num_tries, q0, crossover_rates,
                                                                               de_pair_choice, gamma_level,
                                                                               snooker=run_snooker)

        # Evaluate logp(s)
        if self.num_tries == 1:
            logger.debug(f'{cpu_name} logp')
            q_prior, q_loglike_noT = self.logp(np.squeeze(proposed_pts))
            logger.debug(f'{cpu_name} done logp')
            q_logp_noT = q_prior + q_loglike_noT
            q_logp = temperature * q_loglike_noT + q_prior
            q = np.squeeze(proposed_pts)
            if run_snooker:
                total_proposed_logp = q_logp + snooker_logp_prop
                norm = np.linalg.norm(q0 - z)
                snooker_current_logp = np.log(norm, where=norm != 0) * (self.total_var_dimension - 1)
                total_old_logp = self.last_logp + snooker_current_logp
                q_new, accepted = metrop_select(np.nan_to_num(total_proposed_logp - total_old_logp), q, q0)
            else:
                q_new, accepted = metrop_select(np.nan_to_num(q_logp) - np.nan_to_num(self.last_logp), q, q0)
        # multiple tries
        else:
            logger.debug(f'{cpu_name} multi logp')
            log_priors, log_likes = self.mt_evaluate_logps(self.parallel, self.num_tries, proposed_pts, self.logp,
                                                           ref=False)
            log_ps = temperature * log_likes + log_priors

            # Check if all logps are -inf, in which case they'll all be
            # impossible, and we need to generate more proposal points
            while not np.any(np.isfinite(log_ps)):
                if self.verbose:
                    logger.debug('All proposed points have logps of -inf.  Generating new proposal points.')
                with acquire_locks(Dream_shared_vars.count.get_lock(), Dream_shared_vars.history.get_lock()):
                    proposed_pts, snooker_logp_prop, z = self.generate_proposal_points(
                        self.num_tries, q0, crossover_rates, de_pair_choice, gamma_level, snooker=run_snooker
                    )

                log_priors, log_likes = self.mt_evaluate_logps(self.parallel, self.num_tries, proposed_pts,
                                                               self.logp, ref=False)
                log_ps = temperature * log_likes + log_priors

            q_proposal, q_logp, q_logp_noT, q_loglike_noT, q_prior = self.mt_choose_proposal_pt(
                log_priors,
                log_likes,
                proposed_pts,
                temperature
            )

            # Draw reference points around the randomly selected proposal point

            with acquire_locks(Dream_shared_vars.count.get_lock(), Dream_shared_vars.history.get_lock()):
                reference_pts, snooker_logp_ref, z_ref = self.generate_proposal_points(
                    self.num_tries - 1, q_proposal, crossover_rates, de_pair_choice,
                    gamma_level, snooker=run_snooker
                )

            # Compute posterior density at reference points.
            ref_log_priors, ref_log_likes = self.mt_evaluate_logps(self.parallel, self.num_tries - 1, reference_pts,
                                                                   self.logp, ref=True)
            ref_log_ps = temperature * ref_log_likes + ref_log_priors

            if run_snooker:
                total_proposal_logp = log_ps + snooker_logp_prop
                # Goal is to determine the ratio =  p(y) * p(y --> X) / p(Xref) * p(Xref --> X)
                # where y = proposal point, X = current point, and Xref = reference point
                # First determine p(y --> X) (i.e. moving from proposed point y to original point X)
                # p(y --> X) equals ||y - z||^(n-1), i.e. the snooker_logp for the proposed point
                # p(Xref --> X) is equal to p(Xref --> y) * p(y --> X)
                # (i.e. moving from Xref to proposed point y to original point X)
                snooker_logp_ref = np.append(snooker_logp_ref, 0)
                total_reference_logp = ref_log_ps + snooker_logp_ref + snooker_logp_prop

            else:
                total_proposal_logp = log_ps
                total_reference_logp = ref_log_ps

            # Determine max logp for all proposed and reference points
            max_logp = np.amax(np.concatenate((total_proposal_logp, total_reference_logp)))
            weight_proposed = np.exp(total_proposal_logp - max_logp)
            weight_reference = np.exp(total_reference_logp - max_logp)
            q_new, accepted = metrop_select(np.nan_to_num(np.log(np.sum(weight_proposed) / np.sum(weight_reference))),
                                            q_proposal, q0)

        if accepted:
            self.last_logp = q_logp_noT
            self.last_prior = q_prior
            self.last_like = q_loglike_noT
            if self.verbose:
                if self.num_tries == 1:
                    message = f'Accepted point.  New logp: {q_logp}, old logp: {self.last_logp}, at temperature: {temperature}'

                else:
                    message = (f'Accepted point.  New logp: {q_logp}, old logp: {self.last_logp},'
                               f' weight proposed: {log_ps}, weight ref: {ref_log_ps},'
                               f' ratio: {np.sum(weight_proposed) / np.sum(weight_reference),}'
                               f'at temp: {temperature}')

                logger.info(message)

        else:
            if self.verbose:
                if self.num_tries == 1:
                    message = (f'Did not accept point.  Kept old logp: {self.last_logp}, Tested logp: {q_logp},'
                               f' at temp: {temperature}')
                else:
                    message = (f'Did not accept point.  Kept old logp: {self.last_logp},  Tested logp: {q_logp},'
                               f' weight proposed: {log_ps}, weight ref: {ref_log_ps}, ratio:'
                               f' {np.sum(weight_proposed) / np.sum(weight_reference),} '
                               f'at temperature: {temperature}')
                logger.info(message)

        # Place new point in history given history thinning rate
        if self.iter % self.history_thin == 0:
            logger.debug(f'{cpu_name} update_history')
            with acquire_locks(Dream_shared_vars.count.get_lock(), Dream_shared_vars.history.get_lock()):
                self.record_history(self.nseedchains, self.total_var_dimension, q_new, self.len_history)

        # We only need to have the current position of all chains for estimating the crossover probabilities
        # during burn-in so don't bother updating after that
        if self.iter < self.crossover_burnin + 1:
            logger.debug(f'{cpu_name} set_current_position')
            with acquire_locks(Dream_shared_vars.current_positions.get_lock(), Dream_shared_vars.nchains.get_lock()):
                self.set_current_position_arr(self.total_var_dimension, q_new)

        def update_gammas(total_var_dimension, q0, q_new, gamma_level):
            with acquire_locks(
                    Dream_shared_vars.gamma_level_probs.get_lock(),
                    Dream_shared_vars.count.get_lock(),
                    Dream_shared_vars.ngamma_updates.get_lock(),
                    Dream_shared_vars.current_positions.get_lock(),
                    Dream_shared_vars.delta_m_gamma.get_lock()
            ):
                return self.estimate_gamma_level_probs(total_var_dimension, q0, q_new, gamma_level)

        def update_cross_probs(total_var_dimension, q0, q_new, cr):
            with acquire_locks(
                    Dream_shared_vars.cross_probs.get_lock(),
                    Dream_shared_vars.count.get_lock(),
                    Dream_shared_vars.ncr_updates.get_lock(),
                    Dream_shared_vars.current_positions.get_lock(),
                    Dream_shared_vars.delta_m.get_lock()
            ):
                return self.estimate_crossover_probabilities(total_var_dimension, q0, q_new, CR=cr)

        # If adapting crossover values, estimate ideal
        # crossover probabilities for each dimension during burn-in.
        # Don't do this for the first 10 iterations to give all
        # chains a chance to fill in the shared current position array
        # Don't count iterations where gamma was set to 1 in crossover adaptation calculations
        is_iter_in_range = 10 < self.iter < self.crossover_burnin
        does_gamma_contain_one = not np.any(np.array(self.gamma) == 1.0)
        if self.adapt_crossover and is_iter_in_range and does_gamma_contain_one:
            logger.debug(f'{cpu_name} update_crossover')
            # If a snooker update was run, then regardless of the originally selected CR, a CR=1.0 was used.
            cr = 1 if run_snooker else crossover_rates
            self.CR_probabilities = update_cross_probs(self.total_var_dimension, q0, q_new, cr=cr)

        if self.adapt_gamma and is_iter_in_range and does_gamma_contain_one and not run_snooker:
            logger.debug(f'{cpu_name} update_gamma')
            self.gamma_probabilities = update_gammas(self.total_var_dimension, q0, q_new, gamma_level)

        if self.iter == self.crossover_burnin:
            logger.debug(f'{cpu_name} crossing_over ')
            # To ensure all chains use the same fitted shared probability values,
            # wait for all parallel chains to reach end of burnin period before grabbing shared probabilities
            with acquire_locks(Dream_shared_vars.nchains.get_lock()):
                Dream_shared_vars.nchains.value = Dream_shared_vars.nchains.value + 1
                nchains_finished_burnin = Dream_shared_vars.nchains.value

            if self.adapt_gamma:
                self.gamma_probabilities = update_gammas(self.total_var_dimension, q0, q_new, gamma_level)

            if self.adapt_crossover:
                cr = 1 if run_snooker else crossover_rates
                self.CR_probabilities = update_cross_probs(self.total_var_dimension, q0, q_new, cr=cr)

            while nchains_finished_burnin != self.nchains:
                with acquire_locks(Dream_shared_vars.nchains.get_lock()):
                    nchains_finished_burnin = Dream_shared_vars.nchains.value

            if self.adapt_gamma:
                with acquire_locks(Dream_shared_vars.gamma_level_probs.get_lock()):
                    self.gamma_probabilities = Dream_shared_vars.gamma_level_probs[0:self.ngamma]

            if self.adapt_crossover:
                with acquire_locks(Dream_shared_vars.cross_probs.get_lock()):
                    self.CR_probabilities = Dream_shared_vars.cross_probs[0:self.nCR]

        self.iter += 1
        # with Dream_shared_vars.sync_counter.get_lock():
        #     Dream_shared_vars.sync_counter[self.chain_n-1] = self.iter
        #
        # def check():
        #     with Dream_shared_vars.sync_counter.get_lock():
        #         tmp = np.frombuffer(Dream_shared_vars.sync_counter.get_obj())
        #         return np.all(tmp == self.iter)
        #
        # progress = check()
        # while not progress:
        #     time.sleep(5)
        #     progress = check()
        logger.debug(f"{self.iter} chains synced")
        return q_new, self.last_prior, self.last_like

    def set_current_position_arr(self, ndimensions, q_new):
        """Add current position of chain to shared array available to other chains.

        Parameters
        ----------
        ndimensions : int
            number of dimensions in a draw

        q_new : numpy array
            accepted point in parameter space
        """

        if self.nchains is None:
            current_positions = np.frombuffer(Dream_shared_vars.current_positions.get_obj())
            self.nchains = len(current_positions) // ndimensions

        if self.chain_n is None:
            self.chain_n = Dream_shared_vars.nchains.value - 1
            Dream_shared_vars.nchains.value = Dream_shared_vars.nchains.value - 1

        start_cp = int(self.chain_n * ndimensions)
        end_cp = int(start_cp + ndimensions)
        Dream_shared_vars.current_positions[start_cp:end_cp] = np.array(q_new).flatten()

    def estimate_crossover_probabilities(self, ndim, q0, q_new, CR):
        """Adapt crossover probabilities during crossover burn-in period.

        Parameters
        ----------
        ndim : int
            number of dimensions in a draw
        q0 : numpy array
            original point in parameter space
        q_new : numpy array
            new point in parameter space
        CR : float
            selected crossover probability for this step"""

        cross_probs = Dream_shared_vars.cross_probs[0:self.nCR]

        # Compute squared normalized jumping distance
        m_loc = int(np.where(self.CR_values == CR)[0])

        Dream_shared_vars.ncr_updates[m_loc] = Dream_shared_vars.ncr_updates[m_loc] + 1

        current_positions = np.frombuffer(Dream_shared_vars.current_positions.get_obj())
        current_positions = current_positions.reshape((self.nchains, ndim))

        sd_by_dim = np.std(current_positions, axis=0)

        # Replace any zeros in sd array with a very small number to avoid division by zero errors
        sd_by_dim[sd_by_dim == 0] = 1e-12

        change = np.nan_to_num(np.sum(((q_new - q0) / sd_by_dim) ** 2))

        Dream_shared_vars.delta_m[m_loc] = Dream_shared_vars.delta_m[m_loc] + change
        # Update probabilities of tested crossover value
        # Leave probabilities unchanged until all possible crossover values have had at least one successful move
        # so that a given value's probability isn't prematurely set to 0, preventing further testing.
        delta_ms = np.array(Dream_shared_vars.delta_m[0:self.nCR])

        if np.all(delta_ms != 0):
            for m in range(self.nCR):
                cross_probs[m] = (Dream_shared_vars.delta_m[m] / Dream_shared_vars.ncr_updates[m]) * self.nchains
            cross_probs = cross_probs / np.sum(cross_probs)

        Dream_shared_vars.cross_probs[0:self.nCR] = cross_probs

        self.CR_probabilities = cross_probs

        return cross_probs

    def estimate_gamma_level_probs(self, ndim, q0, q_new, gamma_level):
        """Adapt gamma level probabilities during burn-in

        Parameters
        ----------
        ndim : int
            number of dimensions in a draw
        q0 : numpy array
            original point in parameter space
        q_new : numpy array
            new point in parameter space
        gamma_level : int
            gamma level selected for this step"""

        current_positions = np.frombuffer(Dream_shared_vars.current_positions.get_obj())
        current_positions = current_positions.reshape((self.nchains, ndim))

        sd_by_dim = np.std(current_positions, axis=0)
        gamma_loc = int(np.where(self.gamma_level_values == gamma_level)[0])

        Dream_shared_vars.ngamma_updates[gamma_loc] = Dream_shared_vars.ngamma_updates[gamma_loc] + 1

        Dream_shared_vars.delta_m_gamma[gamma_loc] = Dream_shared_vars.delta_m_gamma[gamma_loc] + \
                                                     np.nan_to_num(np.sum(((q_new - q0) / sd_by_dim) ** 2))

        delta_ms_gamma = np.array(Dream_shared_vars.delta_m_gamma[0:self.ngamma])
        gamma_level_probs = Dream_shared_vars.gamma_level_probs[0:self.ngamma]
        if np.all(delta_ms_gamma != 0):
            for m in range(self.ngamma):
                gamma_level_probs[m] = (Dream_shared_vars.delta_m_gamma[m] /
                                        Dream_shared_vars.ngamma_updates[m]) * self.nchains

            gamma_level_probs = gamma_level_probs / np.sum(gamma_level_probs)

        Dream_shared_vars.gamma_level_probs[0:self.ngamma] = gamma_level_probs

        return gamma_level_probs

    def set_snooker(self):
        """Choose to run a snooker update on a given iteration or not."""
        run_snooker = False
        if self.snooker != 0:
            snooker_choice = np.where(np.random.multinomial(1, [self.snooker, 1 - self.snooker]) == 1)
            if snooker_choice[0] == 0:
                run_snooker = True
        return run_snooker

    @staticmethod
    def set_crossover_rates(crossover_probabilities, crossover_values):
        """Select crossover value for a given iteration.

        Parameters
        ----------
        crossover_probabilities : numpy array
            current probabilities of selecting given crossover values
        crossover_values : numpy array
            possible crossover values"""
        return crossover_values[np.where(np.random.multinomial(1, crossover_probabilities) == 1)]

    @staticmethod
    def set_de_pair(DEpairs):
        """Select the number of pairs of chains to be used for creating the next proposal point for a given iteration.

        Parameters
        ----------
        DEpairs : numpy array
            possible values for the number of chain pairs to be used for proposing the next point"""

        return np.squeeze(np.random.randint(1, len(DEpairs) + 1, size=1)) if len(DEpairs) > 1 else 1

    def set_gamma(self, DEpairs, snooker_choice, gamma_level_choice, d_prime):
        """Select gamma value for a given iteration.

        Parameters
        ----------
        DEpairs : int
            selected number of chain pairs to be used for proposing the next point
        snooker_choice : bool
            whether to use a snooker update scheme on this iteration
        gamma_level_choice : int
            selected level of gamma values to be used this iteration
        d_prime : int
            number of parameter dimensions to be updated on this step."""

        gamma_unity_choice = np.where(np.random.multinomial(1, [self.p_gamma_unity, 1 - self.p_gamma_unity]) == 1)
        if snooker_choice:
            gamma = np.random.uniform(1.2, 2.2)
        elif gamma_unity_choice[0] == 0:
            gamma = 1.0
        else:
            gamma = self.gamma_arr[gamma_level_choice - 1][DEpairs - 1][d_prime - 1]
        return gamma

    def generate_proposal_points(self, n_proposed_pts, q0, CR, DEpairs, gamma_level, snooker):
        """Generate proposal points.

        Parameters
        ----------
        n_proposed_pts : int
            Number of points to propose this iteration (greater than one if using multi-try update scheme)
        q0 : numpy array
            Original point in parameter space
        CR : float
            Crossover value selected for this iteration
        DEpairs : int
            Number of chain pairs to use for proposing the next point for this iteration
        gamma_level : int
            Level of gamma values to use for this iteration
        snooker : bool
            Whether to use a snooker update on this iteration."""
        snooker_logp, z = None, None
        if snooker:
            # With a snooker update all CR always equals 1 (i.e. all parameter dimensions are changed).
            self.gamma = self.set_gamma(DEpairs, snooker, gamma_level, self.total_var_dimension)
            proposed_pts, snooker_logp, z = self.snooker_update(n_proposed_pts, q0)
        else:
            sampled_history_pts = np.array(
                [self.sample_from_history(self.nseedchains, DEpairs, self.total_var_dimension) for _ in
                 range(n_proposed_pts)])

            chain_differences = np.array([np.sum(sampled_history_pts[i][0:DEpairs], axis=0) - np.sum(
                sampled_history_pts[i][DEpairs:DEpairs * 2], axis=0) for i in range(len(sampled_history_pts))])

            zeta = np.array([np.random.normal(0, self.zeta, self.total_var_dimension) for _ in range(n_proposed_pts)])

            e = np.array(
                [np.random.uniform(-self.lamb, self.lamb, self.total_var_dimension) for _ in range(n_proposed_pts)])
            e += 1

            U = np.random.uniform(0, 1, size=chain_differences.shape)

            if n_proposed_pts > 1:
                # Select gamma values given number of parameter dimensions to be changed (d_prime).
                d_prime = [len(U[point][np.where(U[point] < CR)]) for point in range(n_proposed_pts)]
                self.gamma = [self.set_gamma(DEpairs, snooker, gamma_level, d_p) for d_p in d_prime]

                # Generate proposed points given gamma values.
                proposed_pts = [q0 + e[point] * gamma * chain_differences[point] + zeta[point] for point, gamma in
                                zip(range(n_proposed_pts), self.gamma)]
            else:
                d_prime = len(U[np.where(U < CR)])
                self.gamma = self.set_gamma(DEpairs, snooker, gamma_level, d_prime)

                proposed_pts = q0 + e * self.gamma * chain_differences + zeta

            # Crossover proposed points based on number of parameter dimensions to be changed.
            if np.any(d_prime != self.total_var_dimension):
                if n_proposed_pts > 1:
                    for point, pt_num in zip(proposed_pts, range(n_proposed_pts)):
                        proposed_pts[pt_num][np.where(U[pt_num] > CR)] = q0[np.where(U[pt_num] > CR)]

                else:
                    proposed_pts[np.where(U > CR)] = q0[np.where(U > CR)[1]]

        # If uniform priors were used, check that proposed points are within bounds and reflect if not.
        if self.boundaries:

            if n_proposed_pts > 1:
                for pt_num in range(n_proposed_pts):
                    masked_point = proposed_pts[pt_num][self.boundary_mask]
                    corrected_point = reflect_or_correct(masked_point, self.mins, self.maxs)
                    proposed_pts[pt_num][self.boundary_mask] = corrected_point

            else:
                masked_point = np.squeeze(proposed_pts)[self.boundary_mask]
                corrected_point = reflect_or_correct(masked_point, self.mins, self.maxs)
                if snooker:
                    try:
                        proposed_pts[self.boundary_mask] = corrected_point

                    except IndexError:
                        # Raised in the unusual case when total variable dimension = 1
                        if self.boundary_mask:
                            proposed_pts = np.array([corrected_point])
                else:
                    try:
                        proposed_pts[0][self.boundary_mask] = corrected_point

                    except IndexError:
                        # Raised in the unusual case when total variable dimension = 1
                        if self.boundary_mask:
                            proposed_pts = np.array([corrected_point])

        return proposed_pts, snooker_logp, z

    def snooker_update(self, n_proposed_pts, q0):
        """Generate a proposed point with snooker updating scheme.

        Parameters
        ----------
        n_proposed_pts : int
            Number of points to propose this iteration (greater than one if using multi-try update scheme)
        q0 : numpy array
            Original point in parameter space"""

        sampled_history_pt = [
            self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True)
            for i in range(n_proposed_pts)
        ]

        chains_to_be_projected = np.squeeze(
            [np.array(
                [self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True)
                 for _ in range(2)])
                for _ in range(n_proposed_pts)]
        )

        # Define projection vector
        proj_vec_diff = np.squeeze(q0-sampled_history_pt)

        if n_proposed_pts > 1:
            D = [np.dot(proj_vec_diff[point], proj_vec_diff[point]) for point in range(len(proj_vec_diff))]

            # Orthogonal projection of chains_to_projected onto projection vector
            diff_chains_to_be_projected = [(chains_to_be_projected[point][0] - chains_to_be_projected[point][1])
                                           for point in range(n_proposed_pts)]
            z_p = np.nan_to_num(
                np.array(
                    [(np.sum(diff_chains_to_be_projected[point] * proj_vec_diff[point]) / D[point] * proj_vec_diff[
                        point])
                     for point in range(n_proposed_pts)]
                )
            )
            dx = self.gamma * z_p
            proposed_pts = [q0 + dx[point] for point in range(n_proposed_pts)]
            norms = [np.linalg.norm(proposed_pts[point] - sampled_history_pt[point]) for point in range(n_proposed_pts)]
            snooker_logp = [np.log(norm, where= norm != 0)*(self.total_var_dimension-1) for norm in norms]

        else:
            D = np.dot(proj_vec_diff, proj_vec_diff)

            # Orthogonal projection of chains_to_projected onto projection vector
            diff_chains_to_be_projected = chains_to_be_projected[0]-chains_to_be_projected[1]
            z_p = np.nan_to_num(np.array(
                [np.sum(np.divide((diff_chains_to_be_projected * proj_vec_diff), D, where=D != 0))])) * proj_vec_diff
            dx = self.gamma * z_p
            proposed_pts = q0 + dx
            norm = np.linalg.norm(proposed_pts-sampled_history_pt)
            snooker_logp = np.log(norm, where=norm != 0) * (self.total_var_dimension - 1)

        return proposed_pts, snooker_logp, sampled_history_pt

    def mt_evaluate_logps(self, parallel, multitry, proposed_pts, pfunc, ref=False):
        """
        Evaluate the log probability for multiple points in serial or parallel when using multi-try.
        """
        if parallel:
            args = list(zip([self] * multitry, np.squeeze(proposed_pts)))
            with pool.Pool(multitry, context=self.mp_context) as p:
                logps = p.map(call_logp, args)
            log_priors = np.array([val[0] for val in logps])
            log_likes = np.array([val[1] for val in logps])
        else:
            log_priors = []
            log_likes = []
            for pt in np.squeeze(proposed_pts):
                x, y = pfunc(pt)
                log_priors.append(x)
                log_likes.append(y)

        log_priors = np.array(log_priors)
        log_likes = np.array(log_likes)

        if ref:
            # Ensure last_like and last_prior are scalars or 0-d arrays; use np.array to ensure consistent shape
            log_likes = np.append(log_likes, self.last_like)
            log_priors = np.append(log_priors, self.last_prior)

        return log_priors, log_likes

    def record_history(self, nseedchains, ndimensions, q_new, len_history):
        """Record accepted point in history.

        Parameters
        ----------
        nseedchains : int
            Number of points in parameter space with which the original history was seeded
        ndimensions : int
            Number of parameter dimensions being sampled
        q_new : numpy array
            Accepted point
        len_history : int
            The total dimension of the history when completely filled"""

        nhistoryrecs = Dream_shared_vars.count.value + nseedchains
        Dream_shared_vars.count.value = Dream_shared_vars.count.value + 1
        start_loc = int(nhistoryrecs * ndimensions)
        end_loc = int(start_loc + ndimensions)

        Dream_shared_vars.history[start_loc:end_loc] = np.array(q_new).flatten()
        if self.save_history and len_history == (nhistoryrecs + 1) * ndimensions:
            if not self.model_name:
                prefix = datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '_'
            else:
                prefix = self.model_name + '_'
            self.save_history_to_disc(np.frombuffer(Dream_shared_vars.history.get_obj()), prefix)

    def save_history_to_disc(self, history, prefix):
        """Save history and crossover probabilities to files at end of run.

        Parameters
        ----------
        history : numpy array
            History array
        prefix : str
            Prefix to add to history filename when saving"""

        filename = f'{prefix}DREAM_chain_history.npy'
        logger.info(f'Saving history to file: {filename}')
        np.save(filename, history)

        # Also save crossover probabilities if adapted
        filename = f'{prefix}DREAM_chain_adapted_crossoverprob.npy'
        logger.info(f'Saving fitted crossover values: {self.CR_probabilities} to file: {filename}')
        np.save(filename, self.CR_probabilities)

        # Also save gamma level probabilities
        filename = f'{prefix}DREAM_chain_adapted_gammalevelprob.npy'
        logger.info(f'Saving fitted gamma level values: {self.gamma_probabilities} to file: {filename}')
        np.save(filename, self.gamma_probabilities)

    @staticmethod
    def draw_from_prior(model_vars, random_seed=False):
        """Draw from a parameter's prior to seed history array.

        Parameters
        ----------
        model_vars : iterable of instance(s) of SampledParam class
            Model parameters to be sampled with their previously specified prior
        """

        draw = np.array([])
        for variable in model_vars:
            try:
                var_draw = variable.random(reseed=random_seed)
            except AttributeError:
                raise Exception('Random draw from distribution for variable %s not implemented yet.' % variable)
            draw = np.append(draw, var_draw)
        return draw.flatten()

    @staticmethod
    def sample_from_history(nseedchains, DEpairs, ndimensions, snooker=False):
        """Draw random point from the history array.

        Parameters
        ----------
        nseedchains : int
            number of points with which the history was initially seeded
        DEpairs : int
            number of pairs of chains to be used for proposing the next point
        ndimensions : int
            number of dimensions in a draw
        snooker : bool
            whether to use a snooker update at this iteration. Default = False
        """

        current_val = int(Dream_shared_vars.count.value + nseedchains)
        if not snooker:
            chain_num = random.sample(range(current_val), DEpairs * 2)
        else:
            chain_num = random.sample(range(current_val), 1)
        start_locs = [int(i * ndimensions) for i in chain_num]
        end_locs = [int(i + ndimensions) for i in start_locs]

        return [Dream_shared_vars.history[start_loc:end_loc] for start_loc, end_loc in zip(start_locs, end_locs)]

    @staticmethod
    def set_gamma_level(gamma_level_probs, gamma_level_vals):
        """Set gamma level value given current probabilities and possible values.

        Parameters
        ----------
        gamma_level_probs : numpy array
            current probabilities of selecting possible gamma levels
        gamma_level_vals : numpy array
            possible values of gamma level
        """
        gamma_loc = np.where(np.random.multinomial(1, gamma_level_probs) == 1)
        return np.squeeze(gamma_level_vals[gamma_loc])

    @staticmethod
    def mt_choose_proposal_pt(log_priors, log_likes, proposed_pts, T):
        """Select a proposed point with probability proportional to the probability density at that point.

        Parameters
        ----------
        log_priors : numpy array
            Values of the log prior probability for all proposed multi-try points
        log_likes : numpy array
            Values of the log likelihood probability for all proposed multi-try points
        proposed_pts : numpy 2D array nmulti-tries x nparameterdims
            Proposed points
        T : float
            Temperature (only used if using parallel tempering)"""

        # Subtract largest logp from all logps (this from original Matlab code)
        org_log_likes = log_likes
        log_likes = T * log_likes
        log_ps = log_priors + log_likes
        noT_logps = org_log_likes + log_priors
        max_logp = np.amax(log_ps)
        log_ps_sub = np.exp(log_ps - max_logp)

        # Calculate probabilities
        sum_proposal_logps = np.sum(log_ps_sub)
        logp_prob = log_ps_sub / sum_proposal_logps
        best_logp_loc = int(np.squeeze(np.where(np.random.multinomial(1, logp_prob) == 1)[0]))

        # Randomly select one of the tested points with probability proportional
        # to the probability density at the point
        q_proposal = np.squeeze(proposed_pts[best_logp_loc])
        q_logp = log_ps[best_logp_loc]
        q_prior = log_priors[best_logp_loc]
        noT_loglike = org_log_likes[best_logp_loc]
        noT_logp = noT_logps[best_logp_loc]

        return q_proposal, q_logp, noT_logp, noT_loglike, q_prior


def reflect_or_correct(point, lower_bound, upper_bound):
    out_of_bounds_lower = point < lower_bound
    out_of_bounds_upper = point > upper_bound
    point[out_of_bounds_lower] = 2 * lower_bound[out_of_bounds_lower] - point[out_of_bounds_lower]
    point[out_of_bounds_upper] = 2 * upper_bound[out_of_bounds_upper] - point[out_of_bounds_upper]
    # Correct points that are still out of bounds after reflection
    still_out_of_bounds_lower = point < lower_bound
    still_out_of_bounds_upper = point > upper_bound
    point[still_out_of_bounds_lower] = (
            lower_bound[still_out_of_bounds_lower] +
            np.random.rand(np.sum(still_out_of_bounds_lower)) * (upper_bound[still_out_of_bounds_lower] -
                                                                 lower_bound[still_out_of_bounds_lower])
    )
    point[still_out_of_bounds_upper] = (
            lower_bound[still_out_of_bounds_upper] +
            np.random.rand(np.sum(still_out_of_bounds_upper)) * (upper_bound[still_out_of_bounds_upper] -
                                                                 lower_bound[still_out_of_bounds_upper])
    )
    return point


def call_logp(args):
    # Defined at top level so it can be pickled.
    instance = args[0]
    tested_point = args[1]

    logp_fxn = getattr(instance, 'logp')

    return logp_fxn(tested_point)


def metrop_select(mr, q, q0):
    """Perform Metropolis rejection/acceptance

    Parameters
    ----------
    mr : float
        Metropolis ratio
    q : numpy array
        Proposed point
    q0 : numpy array
        Original point"""

    # Compare acceptance ratio to uniform random number
    if np.isfinite(mr) and np.log(np.random.uniform()) < mr:
        # Accept proposed value
        return q, True
    else:
        # Reject proposed value
        return q0, False


# The following part uses source code from the file legacymultiproc.py from https://github.com/nipy/nipype
# copyright NIPY developers, licensed under the Apache 2.0 license.

# Pythons 3.4-3.7.0, and 3.7.1 have three different implementations of
# pool.Pool().Process(), and the type of the result varies based on the default
# multiprocessing context, so we need to dynamically patch the daemon property


class NonDaemonMixin(object):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


from multiprocessing import context


# Exists on all platforms
class NonDaemonSpawnProcess(NonDaemonMixin, context.SpawnProcess):
    pass


class NonDaemonSpawnContext(context.SpawnContext):
    Process = NonDaemonSpawnProcess


_nondaemon_context_mapper = {
    'spawn': NonDaemonSpawnContext()
}

# POSIX only
try:
    class NonDaemonForkProcess(NonDaemonMixin, context.ForkProcess):
        pass


    class NonDaemonForkContext(context.ForkContext):
        Process = NonDaemonForkProcess


    _nondaemon_context_mapper['fork'] = NonDaemonForkContext()
except AttributeError:
    pass
# POSIX only
try:
    class NonDaemonForkServerProcess(NonDaemonMixin, context.ForkServerProcess):
        pass


    class NonDaemonForkServerContext(context.ForkServerContext):
        Process = NonDaemonForkServerProcess


    _nondaemon_context_mapper['forkserver'] = NonDaemonForkServerContext()
except AttributeError:
    pass
from concurrent.futures import ProcessPoolExecutor, Executor, Future


class DreamPool(ProcessPoolExecutor):
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        if context is None:
            context = mp.get_context()
        context = _nondaemon_context_mapper[context._name]
        super(DreamPool, self).__init__(max_workers=processes,
                                        initializer=initializer,
                                        initargs=initargs,
                                        # maxtasksperchild=maxtasksperchild,
                                        mp_context=context)
