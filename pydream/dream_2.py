# -*- coding: utf-8 -*-
from multiprocessing import context
from concurrent.futures import ProcessPoolExecutor
import logging
from copy import deepcopy

# create logger
logger = logging.getLogger('pydream')
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

import numpy as np
import random
from datetime import datetime
import multiprocessing as mp
from multiprocessing import pool
from functools import partial


class Dream2(object):

    def __init__(self, model, variables, num_cr=3, num_de_pairs=1, lamb=.05, zeta=1e-12, snooker=.10,
                 p_gamma_unity=.20, gamma_levels=1, start_random=True, cr_probabilities=None, gamma_probabilities=None,
                 multitry=False, parallel=False, verbose=0, model_name=False, hard_boundaries=True,
                 mp_context=None, starting_position=None, chain_number=0):
        self.last_position = None
        self.gamma_level = None
        self.de_pair_choice = None
        self.crossover_rates = None
        self.q_new = None

        self.chain_n = chain_number
        self.run_snooker = None
        self.cpu_name = self.chain_n

        self.log_ps = []
        self.sampled_params = []

        logger.info(f"{self.cpu_name} started")
        # Set Dream multiprocessing context
        self.last_like = None
        self.last_prior = None
        self.last_logp = None
        self.mp_context = mp_context
        # Set model and variable attributes (if no variables passed, set to all parameters)
        self.model = model
        self.model_name = model_name
        self.variables = variables
        self.cr_probabilities = cr_probabilities
        self.gamma_probabilities = gamma_probabilities

        # Calculate total variable dimension and set boundaries
        self.boundaries = hard_boundaries
        self.total_var_dimension = sum(var.dsize for var in variables)
        self.boundary_mask = np.ones(self.total_var_dimension,
                                     dtype=bool) if self.boundaries and self.total_var_dimension > 1 else True
        self.num_cr = num_cr

        # If the number of crossover values is greater than the total variable dimension,
        # set it to be the total variable dimension
        if self.num_cr > self.total_var_dimension:
            self.num_cr = self.total_var_dimension
            logger.info(
                f'Warning: the total number of crossover values specified ({self.num_cr}) is less than the total '
                f'dimension of all variables ({self.total_var_dimension}).  Setting the number of crossover '
                f'values to be equal to the total variable dimension.')

        self.n_gamma = gamma_levels
        self.n_joint_cr_gamma_probs = self.num_cr * gamma_levels

        # Set crossover values and gamma (the proportion of dimensions to crossover/gamma level to choose)
        self.CR_values = np.array([m / float(self.num_cr) for m in range(1, self.num_cr + 1)])

        self.gamma_level_values = np.array([m for m in range(1, self.n_gamma + 1)])

        # Set number of pairs to use for determining distance between points for proposals
        self.de_pairs = np.linspace(1, num_de_pairs, num=num_de_pairs,
                                    dtype=int)  # This is delta in original Matlab code

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

        # Set array of gamma values (decreasing step size with increasing level)
        gamma_array = np.zeros((self.n_gamma, num_de_pairs, self.total_var_dimension))
        gamma_level_decrease = 1
        for gamma_level in range(1, self.n_gamma + 1):
            for delta in range(1, num_de_pairs + 1):
                gamma_array[gamma_level - 1, delta - 1, :] = (2.38 / np.sqrt(
                    2 * delta * np.linspace(1, self.total_var_dimension,
                                            num=self.total_var_dimension))) / gamma_level_decrease
            gamma_level_decrease = gamma_level_decrease * 2
        self.gamma_arr = gamma_array
        self.gamma = None
        self.start_random = start_random
        self.verbose = verbose
        self.logp = self.model.total_logp
        self.min_boundary, self.max_boundary = self._set_boundaries()
        self.current_position = starting_position
        self.num_accepted = 0
        self.iter = 0
        self.init_zero()

    def _set_boundaries(self):
        min_boundary, max_boundary = [], []
        for var in self.variables:
            interval = var.interval(1)
            min_boundary.extend(interval[0] if var.dsize > 1 else [interval[0]])
            max_boundary.extend(interval[1] if var.dsize > 1 else [interval[1]])
        return np.array(min_boundary), np.array(max_boundary)

    def init_zero(self):
        logger.debug(f'{self.cpu_name} init_zero')

        # Also get length of history array we know when to save it at end of run.
        if self.current_position is None:
            self.current_position = self.draw_from_prior(self.variables)
        last_prior, last_like = self.logp(self.current_position)
        self.set_logp(1, last_like, last_prior)

        if self.verbose:
            logger.info(f'Start: {self.current_position}')

    def set_logp(self, temperature, last_like, last_prior):
        self.last_like = last_like
        self.last_prior = last_prior
        self.last_logp = temperature * self.last_like + self.last_prior
        logger.info(f"Setting last_logp : {self.last_like} , {self.last_logp}")

    def update_params(self):
        logger.debug(f'{self.cpu_name} set_snooker')
        self.run_snooker = self.set_snooker()

        logger.debug(f'{self.cpu_name} set_CR')
        self.crossover_rates = self.set_crossover_rates(self.cr_probabilities, self.CR_values)
        logger.debug(f'{self.cpu_name} set_DEpair')
        self.de_pair_choice = self.set_de_pair(self.de_pairs)

        logger.debug(f'{self.cpu_name} set_gamma_level')
        self.gamma_level = self.set_gamma_level(self.gamma_probabilities, self.gamma_level_values)

        logger.debug(f'{self.cpu_name} generate_proposal_points')

    def step(self, temperature, history, current_point):
        q0 = self.current_position
        self.last_position = q0
        self.update_params()

        if self.num_tries == 1:

            q_new = self.test_1_position(q0, temperature, history, current_point)
        else:
            q_new = self.test_multi_positions(q0, temperature, history, current_point)

        self.log_ps.append(self.last_logp)
        self.sampled_params.append(q_new)
        self.iter += 1

        if self.iter % 10 == 0:
            acceptance_rate = float(self.num_accepted) / (self.iter + 1)
            if self.verbose:
                logger.info(f'{self.cpu_name} Iteration: {self.iter} acceptance rate: {acceptance_rate}')
        self.current_position = q_new

    def test_1_position(self, q0, temperature, history, current_point):

        proposed_pts, snooker_logp_prop, z = self.generate_proposal_points(
            self.num_tries, q0, history=history, current_point=current_point
        )
        q_prior, q_loglike_not = self.logp(np.squeeze(proposed_pts))
        logger.debug(f'{self.cpu_name} done logp')
        q_logp_no_t = q_prior + q_loglike_not
        q_logp = temperature * q_loglike_not + q_prior
        q = np.squeeze(proposed_pts)
        if self.run_snooker:
            total_proposed_logp = q_logp + snooker_logp_prop
            norm = np.linalg.norm(q0 - z)
            snooker_current_logp = np.log(norm, where=norm != 0) * (self.total_var_dimension - 1)
            total_old_logp = self.last_logp + snooker_current_logp
            q_new, accepted = metrop_select(np.nan_to_num(total_proposed_logp - total_old_logp), q, q0)
        else:
            q_new, accepted = metrop_select(np.nan_to_num(q_logp) - np.nan_to_num(self.last_logp), q, q0)

        if accepted:
            self.num_accepted += 1
            if self.verbose:
                message = f'Accept.  logp::New: {q_logp}, old: {self.last_logp}, at temperature: {temperature}'
                logger.debug(message)
                self.last_logp = q_logp_no_t
                self.last_prior = q_prior
                self.last_like = q_loglike_not
        else:
            if self.verbose:
                message = f'Reject.  logp::old: {self.last_logp}, current: {q_logp}, at temp: {temperature}'
                logger.debug(message)

        return q_new

    def test_multi_positions(self, q0, temperature, history, current_point):

        proposed_pts, snooker_logp_prop, z = self.generate_proposal_points(
            self.num_tries, q0, history=history, current_point=current_point
        )
        logger.debug(f'{self.cpu_name} multi logp')
        log_priors, log_likes = self.mt_evaluate_logps(proposed_pts, self.logp, ref=False)
        log_ps = temperature * log_likes + log_priors

        # Check if all logps are -inf, in which case they'll all be
        # impossible, and we need to generate more proposal points
        while not np.any(np.isfinite(log_ps)):
            if self.verbose:
                logger.debug('All proposed points have logps of -inf.  Generating new proposal points.')

            proposed_pts, snooker_logp_prop, z = self.generate_proposal_points(
                self.num_tries, q0, history=history, current_point=current_point
            )

            log_priors, log_likes = self.mt_evaluate_logps(proposed_pts, self.logp, ref=False)
            log_ps = temperature * log_likes + log_priors

        q_proposal, q_logp, q_logp_no_t, q_loglike_no_t, q_prior = self.mt_choose_proposal_pt(
            log_priors,
            log_likes,
            proposed_pts,
            temperature
        )

        # Draw reference points around the randomly selected proposal point
        reference_pts, snooker_logp_ref, z_ref = self.generate_proposal_points(
            self.num_tries - 1, q_proposal, history=history, current_point=current_point
        )

        # Compute posterior density at reference points.
        ref_log_priors, ref_log_likes = self.mt_evaluate_logps(reference_pts, self.logp, ref=True)
        ref_log_ps = temperature * ref_log_likes + ref_log_priors

        if self.run_snooker:
            total_proposal_logp = log_ps + snooker_logp_prop
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
            self.num_accepted += 1
            self.last_logp = q_logp_no_t
            self.last_prior = q_prior
            self.last_like = q_loglike_no_t
            if self.verbose:
                message = (f'Accepted point.  New logp: {q_logp}, old logp: {self.last_logp},'
                           f' weight proposed: {log_ps}, weight ref: {ref_log_ps},'
                           f' ratio: {np.sum(weight_proposed) / np.sum(weight_reference),}'
                           f'at temp: {temperature}')

                logger.debug(message)

        else:
            if self.verbose:
                message = (f'Did not accept point.  Kept old logp: {self.last_logp},  Tested logp: {q_logp},'
                           f' weight proposed: {log_ps}, weight ref: {ref_log_ps}, ratio:'
                           f' {np.sum(weight_proposed) / np.sum(weight_reference),} '
                           f'at temperature: {temperature}')
                logger.debug(message)
        return q_new

    def set_snooker(self):
        run_snooker = False
        if self.snooker != 0:
            snooker_choice = np.where(np.random.multinomial(1, [self.snooker, 1 - self.snooker]) == 1)
            if snooker_choice[0] == 0:
                run_snooker = True
        return run_snooker

    @staticmethod
    def set_crossover_rates(crossover_probabilities, crossover_values):
        return crossover_values[np.where(np.random.multinomial(1, crossover_probabilities) == 1)]

    @staticmethod
    def set_de_pair(de_pairs):
        return np.squeeze(np.random.randint(1, len(de_pairs) + 1, size=1)) if len(de_pairs) > 1 else 1

    def set_gamma(self, de_pairs, snooker_choice, gamma_level_choice, d_prime):
        gamma_unity_choice = np.where(np.random.multinomial(1, [self.p_gamma_unity, 1 - self.p_gamma_unity]) == 1)
        if snooker_choice:
            gamma = np.random.uniform(1.2, 2.2)
        elif gamma_unity_choice[0] == 0:
            gamma = 1.0
        else:
            gamma = self.gamma_arr[gamma_level_choice - 1][de_pairs - 1][d_prime - 1]
        return gamma

    def generate_proposal_points(self, n_proposed_pts, q0, history, current_point):
        snooker_logp, z = None, None
        if self.run_snooker:
            # With a snooker update all CR always equals 1 (i.e. all parameter dimensions are changed).
            self.gamma = self.set_gamma(self.de_pair_choice, True, self.gamma_level, self.total_var_dimension)
            proposed_pts, snooker_logp, z = self.snooker_update(n_proposed_pts, q0, history, current_point)
        else:
            sampled_history_pts = np.array([self.sample_from_history(self.de_pair_choice, history, current_point)
                                            for _ in range(n_proposed_pts)])

            chain_differences = np.array([np.sum(sampled_history_pts[i][0:self.de_pair_choice], axis=0) -
                                          np.sum(sampled_history_pts[i][self.de_pair_choice:self.de_pair_choice * 2],
                                                 axis=0)
                                          for i in range(len(sampled_history_pts))])

            zeta = np.array([np.random.normal(0, self.zeta, self.total_var_dimension) for _ in range(n_proposed_pts)])

            e = np.array(
                [np.random.uniform(-self.lamb, self.lamb, self.total_var_dimension) for _ in range(n_proposed_pts)]
            )
            e += 1

            U = np.random.uniform(0, 1, size=chain_differences.shape)

            if n_proposed_pts > 1:
                # Select gamma values given number of parameter dimensions to be changed (d_prime).
                d_prime = [len(U[point][np.where(U[point] < self.crossover_rates)]) for point in range(n_proposed_pts)]
                self.gamma = [self.set_gamma(self.de_pair_choice, self.run_snooker, self.gamma_level, d_p) for d_p in
                              d_prime]

                # Generate proposed points given gamma values.
                proposed_pts = [q0 + e[point] * gamma * chain_differences[point] + zeta[point] for point, gamma in
                                zip(range(n_proposed_pts), self.gamma)]
            else:
                d_prime = len(U[np.where(U < self.crossover_rates)])
                self.gamma = self.set_gamma(self.de_pair_choice, self.run_snooker, self.gamma_level, d_prime)

                proposed_pts = q0 + e * self.gamma * chain_differences + zeta

            # Crossover proposed points based on number of parameter dimensions to be changed.
            if np.any(d_prime != self.total_var_dimension):
                if n_proposed_pts > 1:
                    for point, pt_num in zip(proposed_pts, range(n_proposed_pts)):
                        proposed_pts[pt_num][np.where(U[pt_num] > self.crossover_rates)] = q0[
                            np.where(U[pt_num] > self.crossover_rates)]

                else:
                    proposed_pts[np.where(U > self.crossover_rates)] = q0[np.where(U > self.crossover_rates)[1]]

        # If uniform priors were used, check that proposed points are within bounds and reflect if not.
        if self.boundaries:
            if n_proposed_pts > 1:
                for pt_num in range(n_proposed_pts):
                    masked_point = proposed_pts[pt_num][self.boundary_mask]
                    corrected_point = reflect_or_correct(masked_point, self.min_boundary, self.max_boundary)
                    proposed_pts[pt_num][self.boundary_mask] = corrected_point

            else:
                masked_point = np.squeeze(proposed_pts)[self.boundary_mask]
                corrected_point = reflect_or_correct(masked_point, self.min_boundary, self.max_boundary)
                if self.run_snooker:
                    proposed_pts[self.boundary_mask] = corrected_point
                else:
                    proposed_pts[0][self.boundary_mask] = corrected_point

        return proposed_pts, snooker_logp, z

    def snooker_update(self, n_proposed_pts, q0, history, current_point):
        """Generate a proposed point with snooker updating scheme.

        Parameters
        ----------
        n_proposed_pts : int
            Number of points to propose this iteration (greater than one if using multi-try update scheme)
        q0 : numpy array
            Original point in parameter space"""

        sampled_history_pt = [self.sample_from_history(self.de_pair_choice, history, current_point, snooker=True)
                              for _ in range(n_proposed_pts)]

        chains_to_be_projected = np.squeeze(
            [np.array(
                [self.sample_from_history(self.de_pair_choice, history, current_point, snooker=True)
                 for _ in range(2)])
                for _ in range(n_proposed_pts)]
        )

        # Define projection vector
        proj_vec_diff = np.squeeze(q0 - sampled_history_pt)

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
            norms = [np.linalg.norm(proposed_pts[point] - sampled_history_pt[point]) for point in
                     range(n_proposed_pts)]
            snooker_logp = [np.log(norm, where=norm != 0) * (self.total_var_dimension - 1) for norm in norms]

        else:
            D = np.dot(proj_vec_diff, proj_vec_diff)

            # Orthogonal projection of chains_to_projected onto projection vector
            diff_chains_to_be_projected = chains_to_be_projected[0] - chains_to_be_projected[1]
            z_p = np.nan_to_num(np.array(
                [np.sum(np.divide((diff_chains_to_be_projected * proj_vec_diff), D, where=D != 0))])) * proj_vec_diff
            dx = self.gamma * z_p
            proposed_pts = q0 + dx
            norm = np.linalg.norm(proposed_pts - sampled_history_pt)
            snooker_logp = np.log(norm, where=norm != 0) * (self.total_var_dimension - 1)

        return proposed_pts, snooker_logp, sampled_history_pt

    def mt_evaluate_logps(self, proposed_pts, pfunc, ref=False):
        """
        Evaluate the log probability for multiple points in serial or parallel when using multi-try.
        """
        n_tries = self.num_tries - 1 if ref else self.num_tries
        if self.parallel:
            args = list(zip([self] * n_tries, np.squeeze(proposed_pts)))
            with pool.Pool(n_tries, context=self.mp_context) as p:
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

    def draw_from_prior(self, random_seed=True):
        """Draw from a parameter's prior to seed history array."""
        return np.array([variable.random(reseed=random_seed) for variable in self.variables])

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
        return np.squeeze(gamma_level_vals[np.where(np.random.multinomial(1, gamma_level_probs) == 1)])

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

    def sample_from_history(self, DEpairs, history, current_val, snooker=False):
        if not snooker:
            chain_num = random.sample(range(current_val), DEpairs * 2)
        else:
            chain_num = random.sample(range(current_val), 1)
        start_locs = [int(i * self.total_var_dimension) for i in chain_num]
        end_locs = [int(i + self.total_var_dimension) for i in start_locs]

        return [history[start_loc:end_loc] for start_loc, end_loc in zip(start_locs, end_locs)]


class SharedVariables(object):
    def __init__(self, model, parameters, number_chains, num_iterations, model_name=False,
                 crossover_file=False, history_file=False, gamma_file=False, adapt_crossover=True, adapt_gamma=False,
                 crossover_burnin=None, multitry=False, parallel=False, verbose=False, hardboundaries=True,
                 mp_context=None, save_history=True, gamma_levels=1, n_seed_chains=None, num_cr=3, history_thin=1,
                 starting_positions=None, nverbose=1
                 ):
        if isinstance(verbose, bool):
            verbose = logging.DEBUG
        elif not isinstance(verbose, int):
            raise ValueError('log_level must be a boolean, integer or None')
        if logger.getEffectiveLevel() != verbose:
            logger.debug('Changing log_level from %d to %d' % (
                logger.getEffectiveLevel(), verbose))
            logger.setLevel(verbose)

        self.model = model
        self.parameters = parameters if parameters is None else self.model.sampled_parameters
        self.num_iterations = num_iterations
        self.crossover_burnin = None
        self.history_thin = history_thin if history_thin > 0 else 1
        self.iter = 0
        self.model_name = model_name
        self.save_history = save_history

        self.n_gamma = gamma_levels
        self.num_cr = num_cr
        self.adapt_crossover = adapt_crossover
        self.adapt_gamma = adapt_gamma
        self.cpu_name = mp.current_process().name
        self.number_chains = number_chains
        self.multitry = multitry
        self.parallel = parallel
        self.verbose = verbose
        self.nverbose = nverbose
        self.hardboundaries = hardboundaries

        if crossover_file:
            cr_probabilities = np.load(crossover_file)
            self.num_cr = len(cr_probabilities)
            if self.adapt_crossover:
                logger.warning('Crossover values loaded and adapt_crossover = True.'
                               'Crossover values will be further adapted.')
        else:
            cr_probabilities = np.array([1 / float(self.num_cr) for _ in range(self.num_cr)], dtype=float)

        variable_size = 0
        for var in self.parameters:
            variable_size += var.dsize

        if n_seed_chains is None:
            n_seed_chains = variable_size * 10

        if history_file:
            old_history = np.load(history_file).flatten()
            n_seed_chains = len(old_history) / variable_size

            if num_iterations < self.history_thin:
                arr_dim = ((np.floor(
                    number_chains * num_iterations / self.history_thin) + number_chains) * variable_size) + len(
                    old_history)
            else:
                arr_dim = (np.floor((((number_chains * num_iterations) * variable_size) / self.history_thin))
                           + len(old_history))
        else:
            if num_iterations < self.history_thin:
                arr_dim = (((np.floor(
                    number_chains * num_iterations / self.history_thin) + number_chains) * variable_size) + (
                                   n_seed_chains * variable_size))
            else:
                arr_dim = (np.floor(((number_chains * num_iterations / self.history_thin) * variable_size))
                           + (n_seed_chains * variable_size))
        self.n_gamma = gamma_levels
        if gamma_file:
            gamma_probabilities = np.load(gamma_file)
        else:
            gamma_probabilities = np.array([1 / float(self.n_gamma) for _ in range(self.n_gamma)])

        if mp_context is None:
            ctx = mp.get_context(mp_context)
        else:
            ctx = mp_context

        if crossover_burnin is None:
            self.crossover_burnin = int(np.floor(num_iterations / 10))
        else:
            self.crossover_burnin = crossover_burnin

        # shared memory
        self.mp_context = ctx
        self.global_count = 0

        self.n_seed_chains = n_seed_chains
        self.total_var_dimension = variable_size
        self.global_history_seeded = False
        self.global_history = np.array([0] * int(arr_dim), dtype=float)

        if history_file:
            self.global_history_seeded = True
            self.global_history[0:len(old_history)] = old_history

        self.global_current_positions = np.array([0] * self.number_chains * self.total_var_dimension, dtype=float)
        self.global_ncr_updates = np.array([0] * self.num_cr)
        self.global_delta_m = np.array([0] * self.num_cr, dtype=float)
        self.global_gamma_probabilities = gamma_probabilities
        self.global_cr_probabilities = cr_probabilities
        self.global_ngamma_updates = np.array([0] * self.n_gamma)
        self.global_delta_m_gamma = np.array([0] * self.n_gamma, dtype=float)

        if self.num_cr > self.total_var_dimension:
            self.num_cr = self.total_var_dimension
            logger.info(
                f'Warning: the total number of crossover values specified ({self.num_cr}) is less than the total '
                f'dimension of all variables ({self.total_var_dimension}).  Setting the number of crossover '
                f'values to be equal to the total variable dimension.')

        if not self.global_history_seeded:
            self.seed_shared_history()
        chains = []
        for i in range(self.number_chains):
            chains.append(
                Dream2(model=self.model, variables=self.parameters, verbose=self.verbose, mp_context=self.mp_context,
                       chain_number=i, gamma_levels=self.n_gamma,
                       cr_probabilities=cr_probabilities, gamma_probabilities=gamma_probabilities,
                       starting_position=starting_positions[i], )
            )
        self.chains = chains

    def add_to_history(self, q_new):
        nhistoryrecs = self.global_count + self.n_seed_chains
        self.global_count += 1
        start_loc = int(nhistoryrecs * self.total_var_dimension)
        end_loc = int(start_loc + self.total_var_dimension)

        self.global_history[start_loc:end_loc] = np.array(q_new).flatten()
        if self.save_history and self.iter == self.num_iterations:
            if not self.model_name:
                prefix = datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '_'
            else:
                prefix = self.model_name + '_'
            self.save_history_to_disc(self.global_history, prefix)

    def save_history_to_disc(self, history, prefix):

        filename = f'{prefix}DREAM_chain_history.npy'
        logger.info(f'Saving history to file: {filename}')
        np.save(filename, history)

        # Also save crossover probabilities if adapted
        filename = f'{prefix}DREAM_chain_adapted_crossoverprob.npy'
        logger.info(f'Saving fitted crossover values: {self.global_cr_probabilities} to file: {filename}')
        np.save(filename, self.global_cr_probabilities)

        # Also save gamma level probabilities
        filename = f'{prefix}DREAM_chain_adapted_gammalevelprob.npy'
        logger.info(f'Saving fitted gamma level values: {self.global_gamma_probabilities} to file: {filename}')
        np.save(filename, self.global_gamma_probabilities)

    def set_current_position_arr(self, q_new, chain_number):
        start_cp = int(chain_number * self.total_var_dimension)
        end_cp = int(start_cp + self.total_var_dimension)
        self.global_current_positions[start_cp:end_cp] = np.array(q_new).flatten()

    def seed_shared_history(self):
        if self.verbose:
            logger.info(f'Seeding history with {self.n_seed_chains} draws from prior.')

        def draw_from_prior(random_seed=True):
            """Draw from a parameter's prior to seed history array."""
            return [variable.random(reseed=random_seed) for variable in self.parameters]

        for i in range(self.n_seed_chains):
            start_loc = i * self.total_var_dimension
            end_loc = start_loc + self.total_var_dimension
            self.global_history[start_loc:end_loc] = draw_from_prior(True)

    def estimate_crossover_probabilities(self, chain):
        cr = 1 if chain.run_snooker else chain.crossover_rates
        # Compute squared normalized jumping distance
        m_loc = int(np.where(chain.CR_values == cr)[0])
        self.global_ncr_updates[m_loc] += 1

        current_positions = deepcopy(self.global_current_positions)
        current_positions = current_positions.reshape((self.number_chains, self.total_var_dimension))

        sd_by_dim = np.std(current_positions, axis=0)
        sd_by_dim[sd_by_dim == 0] = 1e-12  # Avoid division by zero

        change = np.nan_to_num(np.sum(((chain.current_position - chain.last_position) / sd_by_dim) ** 2))
        self.global_delta_m[m_loc] += change

        # Update probabilities of tested crossover value
        delta_ms = self.global_delta_m
        cross_probs = deepcopy(self.global_cr_probabilities)
        if np.all(delta_ms != 0):
            for m in range(self.num_cr):
                cross_probs[m] = (self.global_delta_m[m] / self.global_ncr_updates[m]) * self.number_chains
            cross_probs = cross_probs / np.sum(cross_probs)

        self.global_cr_probabilities = cross_probs
        chain.cr_probabilities = cross_probs
        return cross_probs

    def estimate_gamma_level_probs(self, chain):

        current_positions = deepcopy(self.global_current_positions)
        current_positions = current_positions.reshape((self.number_chains, self.total_var_dimension))

        sd_by_dim = np.std(current_positions, axis=0)
        sd_by_dim[sd_by_dim == 0] = 1e-12
        gamma_loc = int(np.where(chain.gamma_level_values == chain.gamma_level)[0])

        self.global_ngamma_updates[gamma_loc] += 1

        self.global_delta_m_gamma[gamma_loc] += np.nan_to_num(
            np.sum(((chain.current_position - chain.last_position) / sd_by_dim) ** 2)
        )

        delta_ms_gamma = self.global_delta_m_gamma
        gamma_level_probs = deepcopy(self.global_gamma_probabilities)
        if np.all(delta_ms_gamma != 0):
            for m in range(self.n_gamma):
                gamma_level_probs[m] = ((self.global_delta_m_gamma[m] / self.global_ngamma_updates[m])
                                        * self.number_chains)

            gamma_level_probs = gamma_level_probs / np.sum(gamma_level_probs)
        self.global_gamma_probabilities = gamma_level_probs
        return gamma_level_probs

    def run_step(self, temperature):
        logger.info(f'{self.cpu_name} running step {self.iter}')

        # Run parallel simulations for each chain
        sim_partial = partial(_run_parallel, temperature=temperature, history=self.global_history,
                              current_count=int(self.global_count + self.n_seed_chains))
        with ProcessPoolExecutor(max_workers=self.number_chains) as executor:
            futures = [executor.submit(sim_partial, chain) for chain in self.chains]
            self.chains = [future.result() for future in futures]

            # Place new point in history given history thinning rate
        if self.iter % self.history_thin == 0:
            for chain in self.chains:
                self.add_to_history(chain.current_position)

        # We only need to have the current position of all chains for estimating the crossover probabilities
        # during burn-in so don't bother updating after that
        # update chain positioin array
        if self.iter < self.crossover_burnin + 1:
            for chain in self.chains:
                self.set_current_position_arr(chain.current_position, chain.chain_n)

        # If adapting crossover values, estimate ideal
        # crossover probabilities for each dimension during burn-in.
        # Don't do this for the first 10 iterations to give all
        # chains a chance to fill in the shared current position array
        # Don't count iterations where gamma was set to 1 in crossover adaptation calculations
        is_iter_in_range = 10 < self.iter < self.crossover_burnin

        if self.adapt_crossover and is_iter_in_range:
            logger.debug(f'{self.cpu_name} update_crossover')
            for chain in self.chains:
                does_gamma_contain_one = not np.any(np.array(chain.gamma) == 1.0)
                if does_gamma_contain_one:
                    chain.cr_probabilities = self.estimate_crossover_probabilities(chain)
                    if not chain.run_snooker:
                        logger.debug(f'{self.cpu_name} update_gamma')
                        chain.gamma_probabilities = self.estimate_gamma_level_probs(chain)

        if self.iter == self.crossover_burnin:
            for chain in self.chains:
                self.do_cross_over(chain)
            for chain in self.chains:
                if self.adapt_gamma:
                    chain.gamma_probabilities = self.global_gamma_probabilities
                if self.adapt_crossover:
                    chain.cr_probabilities = self.global_cr_probabilities

        self.iter += 1

    def do_cross_over(self, chain):

        logger.debug(f'{self.cpu_name} crossing_over ')
        # To ensure all chains use the same fitted shared probability values,
        # wait for all parallel chains to reach end of burnin period before grabbing shared probabilities
        if self.adapt_gamma:
            self.global_gamma_probabilities = self.estimate_gamma_level_probs(chain)

        if self.adapt_crossover:
            self.global_cr_probabilities = self.estimate_crossover_probabilities(chain)


def _run_parallel(chain, temperature, history, current_count):
    chain.step(temperature, history, current_count)
    return chain


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
    """Perform Metropolis rejection/acceptance."""
    return (q, True) if np.log(np.random.uniform()) < mr else (q0, False)


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


class DreamPool(ProcessPoolExecutor):
    def __init__(self, processes=None, initializer=None, initargs=(), context=None):
        if context is None:
            context = mp.get_context()
        context = _nondaemon_context_mapper[context._name]
        super(DreamPool, self).__init__(max_workers=processes,
                                        initializer=initializer,
                                        initargs=initargs,
                                        mp_context=context)
