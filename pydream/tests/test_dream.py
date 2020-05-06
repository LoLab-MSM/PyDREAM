# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:50:59 2014

@author: Erin
"""

import unittest
from os import remove

import multiprocessing as mp
import numpy as np
import pydream.Dream_shared_vars
from pydream.Dream import Dream
from pydream.core import run_dream, _setup_mp_dream_pool, _sample_dream, _sample_dream_pt, _sample_dream_pt_chain
from pydream.model import Model
from pydream.tests.test_models import onedmodel, multidmodel, multidmodel_uniform
from pydream.examples.corm.example_sample_corm_with_dream import likelihood as corm_like
from pydream.examples.corm.example_sample_corm_with_dream import run_kwargs as corm_kwargs
from pydream.examples.mixturemodel.mixturemodel import run_kwargs as mix_kwargs
from pydream.examples.mixturemodel.mixturemodel import likelihood as mix_like
from pydream.examples.ndim_gaussian.dream_ex_ndim_gaussian import run_kwargs as ndimgauss_kwargs
from pydream.examples.ndim_gaussian.dream_ex_ndim_gaussian import likelihood as ndimgauss_like
from pydream.examples.robertson.example_sample_robertson_with_dream import run_kwargs as robertson_kwargs
from pydream.examples.robertson.example_sample_robertson_with_dream import likelihood as robertson_like
from pydream.examples.robertson_nopysb.example_sample_robertson_nopysb_with_dream import run_kwargs as rob_nop_kwargs
from pydream.examples.robertson_nopysb.example_sample_robertson_nopysb_with_dream import likelihood as rob_nop_like

import numbers
import sys

class Test_Dream_Initialization(unittest.TestCase):
    
    def test_fail_with_one_chain(self):
        """Test that DREAM fails if run with only one chain."""
        self.param, self.like = onedmodel()
        assertRaisesRegex = self.assertRaisesRegexp if sys.version_info[0] < 3 else self.assertRaisesRegex
        assertRaisesRegex(Exception, 'Dream should be run with at least ', run_dream, self.param, self.like, nchains=1)
    
    def test_total_var_dimension_init(self):
        """Test that DREAM correctly identifies the total number of dimensions in all sampled parameters for a few test cases."""
        self.param, self.like = onedmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        step = Dream(model=model, variables=self.param)
        self.assertEqual(step.total_var_dimension, 1)
        
        self.param, self.like = multidmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        step = Dream(model=model, variables=self.param)
        self.assertEqual(step.total_var_dimension, 4)

class Test_Dream_Algorithm_Components(unittest.TestCase):
    
    def test_gamma_unityfraction(self):
        """Test that gamma value is set to 1 the fraction of times indicated by the p_gamma_unity DREAM parameter."""
        self.param, self.like = onedmodel()
        n_unity_choices = 0
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        step = Dream(model=model)
        fraction = step.p_gamma_unity
        for iteration in range(10000):
           choice = step.set_gamma(DEpairs=1, snooker_choice=False, gamma_level_choice=1, d_prime=step.total_var_dimension)
           if choice == 1:
               n_unity_choices += 1
        emp_frac = n_unity_choices/10000.0
        self.assertAlmostEqual(emp_frac, fraction, places=1)
    
    def test_gamma_array(self):
        """Test assigned value of gamma array matches for test data."""
        true_gamma_array = np.array([[1.683, 1.19, .972, .841, .753]])
        self.param, self.like = onedmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        dream = Dream(model=model, DEpairs=5, p_gamma_unity=0)
        for d_prime in range(1, dream.total_var_dimension+1):
            for n_DEpair in range(1, 6):
                self.assertAlmostEqual(true_gamma_array[d_prime-1][n_DEpair-1], dream.set_gamma(DEpairs=n_DEpair, snooker_choice=False, gamma_level_choice=1, d_prime=d_prime), places=3)
    
    def test_gamma_snooker_choice(self):
        """Test that when a snooker move is made, gamma is set to a random value between 1.2 and 2.2."""
        self.param, self.like = onedmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        step = Dream(model=model)
        self.assertGreaterEqual(step.set_gamma(DEpairs=1, snooker_choice=True, gamma_level_choice=1, d_prime=3), 1.2)
        self.assertLess(step.set_gamma(DEpairs=1, snooker_choice=True, gamma_level_choice=1, d_prime=3), 2.2)
    
    def test_snooker_fraction(self):
        """Test that the fraction of snooker moves corresponds to the snooker parameter."""
        self.param, self.like = onedmodel()
        n_snooker_choices = 0
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        step = Dream(model=model)
        fraction = step.snooker
        for iteration in range(10000):
           choice = step.set_snooker()
           if choice == True:
               n_snooker_choices += 1
        emp_frac = n_snooker_choices/10000.0
        self.assertAlmostEqual(emp_frac, fraction, places=1)   
        
    def test_CR_fraction(self):
        """Test that the crossover values chosen match with the crossover probability values for test data."""
        self.param, self.like = onedmodel()
        nCR1 = 0
        nCR2 = 0
        nCR3 = 0
        crossoverprobs = np.array([.10, .65, .25])
        crossovervals = np.array([.33, .66, 1.0])
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        step = Dream(model=model, variables=self.param)
        for iteration in range(10000):
            choice = step.set_CR(crossoverprobs, crossovervals)
            if choice == crossovervals[0]:
                nCR1 += 1
            elif choice == crossovervals[1]:
                nCR2 += 1
            else:
                nCR3 += 1
        emp_frac1 = nCR1/10000.0
        emp_frac2 = nCR2/10000.0
        emp_frac3 = nCR3/10000.0
        self.assertAlmostEqual(emp_frac1, crossoverprobs[0], places=1)
        self.assertAlmostEqual(emp_frac2, crossoverprobs[1], places=1)
        self.assertAlmostEqual(emp_frac3, crossoverprobs[2], places=1)
    
    def test_DEpair_selec(self):
        """Test that fraction for selected DEpair value is consistent with number of specified DEPair value."""
        self.param, self.like = onedmodel()
        single_DEpair = np.array([1])
        multi_DEpair = np.array([1, 2, 3])
        nDE1 = 0
        nDE2 = 0
        nDE3 = 0
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        step = Dream(model=model, variables=self.param)
        self.assertEqual(step.set_DEpair(single_DEpair), 1)
        for iteration in range(10000):
            choice = step.set_DEpair(multi_DEpair)
            if choice == multi_DEpair[0]:
                nDE1 += 1
            elif choice == multi_DEpair[1]:
                nDE2 += 1
            else:
                nDE3 += 1
        emp_frac1 = nDE1/10000.0
        emp_frac2 = nDE2/10000.0
        emp_frac3 = nDE3/10000.0
        self.assertAlmostEqual(emp_frac1, .3, places=1)
        self.assertAlmostEqual(emp_frac2, .3, places=1)
        self.assertAlmostEqual(emp_frac3, .3, places=1)
    
    def test_prior_draw(self):
        """Test random draw from prior for normally distributed priors in test models."""
        self.param, self.like = onedmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        self.assertEqual(len(Dream(model=model).draw_from_prior(model.sampled_parameters)), 1)
        self.param, self.like = multidmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        self.assertEqual(len(Dream(model=model).draw_from_prior(model.sampled_parameters)), 4)

    def test_chain_sampling_simple_model(self):
        """Test that sampling from DREAM history for one dimensional model when the history is known matches with expected possible samples."""
        self.param, self.like = onedmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.param)
        dream = Dream(model=model)
        history_arr = mp.Array('d', [0]*2*dream.total_var_dimension)
        n = mp.Value('i', 0)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        chains_added_to_history = []
        for i in range(2):
            start = i*dream.total_var_dimension
            end = start+dream.total_var_dimension
            chain = dream.draw_from_prior(dream.variables)
            pydream.Dream_shared_vars.history[start:end] = chain
            chains_added_to_history.append(chain)
        sampled_chains = dream.sample_from_history(nseedchains=2, DEpairs=1, ndimensions=dream.total_var_dimension)
        sampled_chains = np.array(sampled_chains)
        chains_added_to_history = np.array(chains_added_to_history)
        self.assertIs(np.array_equal(chains_added_to_history[chains_added_to_history[:,0].argsort()], sampled_chains[sampled_chains[:,0].argsort()]), True)
    
    def test_chain_sampling_multidim_model(self):
        """Test that sampling from DREAM history for multi-dimensional model when the history is known matches with expected possible samples."""
        self.params, self.like = multidmodel()
        model = Model(likelihood=self.like, sampled_parameters=self.params)
        dream = Dream(model=model)
        history_arr = mp.Array('d', [0]*2*dream.total_var_dimension)
        n = mp.Value('i', 0)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        chains_added_to_history = []
        for i in range(2):
            start = i*dream.total_var_dimension
            end = start+dream.total_var_dimension
            chain = dream.draw_from_prior(model.sampled_parameters)
            pydream.Dream_shared_vars.history[start:end] = chain
            chains_added_to_history.append(chain)       
        sampled_chains = dream.sample_from_history(nseedchains=2, DEpairs=1, ndimensions=dream.total_var_dimension)
        sampled_chains = np.array(sampled_chains)
        chains_added_to_history = np.array(chains_added_to_history)
        self.assertIs(np.array_equal(chains_added_to_history[chains_added_to_history[:,0].argsort()], sampled_chains[sampled_chains[:,0].argsort()]), True)
    
    def test_proposal_generation_nosnooker_CR1(self):
        """Test proposal generation without a snooker update with a single or multiple proposed points and a crossover value of 1 gives all dimensions changed on average as expected."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        history_arr = mp.Array('d', list(range(120)))
        n = mp.Value('i', 0)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5])
        dims_kept = 0
        for iteration in range(10000):
            proposed_pt = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=1, DEpairs=1, gamma_level=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pt), 1)
            dims_change_vec = np.squeeze(q0 == proposed_pt)
            for dim in dims_change_vec:
                if dim:
                    dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*10000.0)
        self.assertAlmostEqual(frac_kept, 0, places=1)
        dims_kept = 0
        for iteration in range(1000):
            proposed_pts = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=1, DEpairs=1, gamma_level=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pts), 5)
            for pt in proposed_pts:
                dims_change_vec = (q0 == pt)
                for dim in dims_change_vec:
                    if dim:
                        dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*1000.0*5)
        self.assertAlmostEqual(frac_kept, 0, places=1)
    
    def test_proposal_generation_nosnooker_CR33(self):
        """Test proposal generation without a snooker update with a single or multiple proposed points and a crossover value of .33 gives 1/3 of all dimensions changed on average as expected."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        history_arr = mp.Array('d', list(range(120)))
        n = mp.Value('i', 0)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5])
        dims_kept = 0
        for iteration in range(100000):
            proposed_pt = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=.33, DEpairs=1, gamma_level=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pt), 1)
            dims_change_vec = np.squeeze(q0 == proposed_pt)
            for dim in dims_change_vec:
                if dim:
                    dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*100000.0)
        self.assertAlmostEqual(frac_kept, 1-.33, places=1)
        dims_kept = 0
        for iteration in range(10000): 
            proposed_pts = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=.33, DEpairs=1, gamma_level=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pts), 5)
            for pt in proposed_pts:
                dims_change_vec = (q0 == pt)
                for dim in dims_change_vec:
                    if dim:
                        dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*10000.0*5)
        self.assertAlmostEqual(frac_kept, 1-.33, places=1)
    
    def test_proposal_generation_nosnooker_CR66(self):
        """Test proposal generation without a snooker update with a single or multiple proposed points and a crossover value of 2/3 gives 2/3 of all dimensions changed on average as expected."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        history_arr = mp.Array('d', list(range(120)))
        n = mp.Value('i', 0)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5])
        dims_kept = 0
        for iteration in range(100000):
            proposed_pt = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=.66, DEpairs=1, gamma_level=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pt), 1)
            dims_change_vec = np.squeeze(q0 == proposed_pt)
            for dim in dims_change_vec:
                if dim:
                    dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*100000.0)
        self.assertAlmostEqual(frac_kept, 1-.66, places=1)
        dims_kept = 0
        for iteration in range(10000): 
            proposed_pts = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=.66, DEpairs=1, gamma_level=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pts), 5)
            for pt in proposed_pts:
                dims_change_vec = (q0 == pt)
                for dim in dims_change_vec:
                    if dim:
                        dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*10000.0*5)
        self.assertAlmostEqual(frac_kept, 1-.66, places=1)
    
    def test_proposal_generation_snooker(self):
        """Test that proposal generation with a snooker update returns values of the expected shape."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        history_arr = mp.Array('d', list(range(120)))
        n = mp.Value('i', 0)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5])
        proposed_pt, snooker_logp, z = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=1, DEpairs=1, gamma_level=1, snooker=True)
        self.assertEqual(len(proposed_pt), step.total_var_dimension)
        proposed_pts, snooker_logp, z = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=1, DEpairs=1, gamma_level=1, snooker=True)
        self.assertEqual(len(proposed_pts), 5)
    
    def test_multitry_logp_eval(self):
        """Test that evaluation of multiple trials either in parallel or not matches with known logp values."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        logp = step.logp
        proposed_pts = [[1, 2, 3, 4], [7, 8, 9, 10], [13, 14, 15, 16]]
        logpriors, loglikes = step.mt_evaluate_logps(parallel=False, multitry=3, proposed_pts=proposed_pts, pfunc=logp)
        correct_loglikes = []
        correct_logpriors = []
        for pt in proposed_pts:
            prior, like = logp(np.array(pt))
            correct_logpriors.append(prior)
            correct_loglikes.append(like)
        self.assertEqual(np.array_equal(logpriors, np.array(correct_logpriors)), True)
        self.assertEqual(np.array_equal(loglikes, np.array(correct_loglikes)), True)
        
        logpriors, loglikes = step.mt_evaluate_logps(parallel=True, multitry=3, proposed_pts=proposed_pts, pfunc=logp)

        self.assertEqual(np.array_equal(logpriors, np.array(correct_logpriors)), True)
        self.assertEqual(np.array_equal(loglikes, np.array(correct_loglikes)), True)
    
    def test_multitry_proposal_selection(self):
        """Test that multiple trial proposal selection matches expectated choice with test logp data."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        logpriors = np.array([0, 0])
        loglikes = np.array([1000, 500])
        proposed_pts = [[1, 2, 3, 4], [7, 8, 9, 10]]
        for iteration in range(3):
            q_proposal, q_logp, noT_logp, noT_loglike, q_prior= step.mt_choose_proposal_pt(logpriors, loglikes, proposed_pts, T=1)
            self.assertEqual(np.array_equal(q_proposal, proposed_pts[0]), True)
    
    def test_crossover_prob_estimation(self):
        """Test that crossover probabilities are updated as expected when changing or not changing parameter locations and giving points that give a greater jump distance."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        dream = Dream(model=model, save_history=False)
        starting_crossover = dream.CR_probabilities
        crossover_probabilities = mp.Array('d', starting_crossover)
        n = mp.Value('i', 0)
        nCR = dream.nCR
        CR_vals = dream.CR_values
        ncrossover_updates = mp.Array('d', [0]*nCR)
        current_position_arr = mp.Array('d', [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        dream.nchains = 5
        delta_m = mp.Array('d', [0]*nCR)
        dream.chain_n = 0
        pydream.Dream_shared_vars.cross_probs = crossover_probabilities
        pydream.Dream_shared_vars.count = n
        pydream.Dream_shared_vars.ncr_updates = ncrossover_updates
        pydream.Dream_shared_vars.current_positions = current_position_arr
        pydream.Dream_shared_vars.delta_m = delta_m
        q0 = np.array([1, 2, 3, 4])
        q_new = np.array([1, 2, 3, 4])
        new_cr_probs = dream.estimate_crossover_probabilities(dream.total_var_dimension, q0, q_new, CR_vals[0])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), True)
        q_new = np.array([1.2, 2.2, 3.3, 4.4])
        new_cr_probs = dream.estimate_crossover_probabilities(dream.total_var_dimension, q0, q_new, CR_vals[0])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), True)
        q_new = np.array([2, 3, 4, 5])
        new_cr_probs = dream.estimate_crossover_probabilities(dream.total_var_dimension, q0, q_new, CR_vals[1])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), True)
        q_new = np.array([11, -15, 20, 9])
        new_cr_probs = dream.estimate_crossover_probabilities(dream.total_var_dimension, q0, q_new, CR_vals[2])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), False)
        self.assertGreater(new_cr_probs[2], starting_crossover[2])
        self.assertAlmostEqual(np.sum(new_cr_probs), 1.0, places=1)
        old_cr_probs = new_cr_probs
        for i, q_new in zip(list(range(5)), [np.array([15]), np.array([17]), np.array([19]), np.array([21]), np.array([23])]):
            new_cr_probs = dream.estimate_crossover_probabilities(dream.total_var_dimension, q0, q_new, CR_vals[1])
        self.assertEqual(np.array_equal(new_cr_probs, old_cr_probs), False)
        
    def test_history_recording_simple_model(self):
        """Test that history in memory matches with that recorded for test one-dimensional model."""
        self.param, self.like = onedmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model, model_name='test_history_recording')
        history_arr = mp.Array('d', [0]*4*step.total_var_dimension)
        n = mp.Value('i', 0)
        nchains = mp.Value('i', 3)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        pydream.Dream_shared_vars.nchains = nchains
        test_history = np.array([[1], [3], [5], [7]])
        for chainpoint in test_history:
            for point in chainpoint:
                step.record_history(nseedchains=0, ndimensions=step.total_var_dimension, q_new=point, len_history=len(history_arr))
        history_arr_np = np.frombuffer(pydream.Dream_shared_vars.history.get_obj())
        history_arr_np_reshaped = history_arr_np.reshape(np.shape(test_history))
        self.assertIs(np.array_equal(history_arr_np_reshaped, test_history), True)
        remove('test_history_recording_DREAM_chain_history.npy')
        remove('test_history_recording_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_history_recording_DREAM_chain_adapted_gammalevelprob.npy')
        
    def test_history_recording_multidim_model(self):
        """Test that history in memory matches with that recorded for test multi-dimensional model."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        dream = Dream(model=model, model_name='test_history_recording')
        history_arr = mp.Array('d', [0]*4*dream.total_var_dimension*3)
        n = mp.Value('i', 0)
        nchains = mp.Value('i', 3)
        pydream.Dream_shared_vars.history = history_arr
        pydream.Dream_shared_vars.count = n
        pydream.Dream_shared_vars.nchains = nchains
        test_history = np.array([[[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]], [[7, 8, 9, 10], [9, 12, 18, 20], [11, 14, 18, 8]], [[13, 14, 18, 4], [15, 17, 11, 8], [17, 28, 50, 4]], [[19, 21, 1, 18], [21, 19, 19, 11], [23, 4, 3, 2]]])
        for chainpoint in test_history:
            for point in chainpoint:
                dream.record_history(nseedchains=0, ndimensions=dream.total_var_dimension, q_new=point, len_history=len(history_arr))
        history_arr_np = np.frombuffer(pydream.Dream_shared_vars.history.get_obj())
        history_arr_np_reshaped = history_arr_np.reshape(np.shape(test_history))
        self.assertIs(np.array_equal(history_arr_np_reshaped, test_history), True)
        remove('test_history_recording_DREAM_chain_history.npy')
        remove('test_history_recording_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_history_recording_DREAM_chain_adapted_gammalevelprob.npy')
     
    def test_history_saving_to_disc_sanitycheck(self):
        """Test that history when saved to disc and reloaded matches."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        history = np.array([[5, 8, 10, 12], [13, 18, 20, 21], [1, .5, 9, 1e9]])
        step.save_history_to_disc(history, 'testing_history_save_')
        history_saved = np.load('testing_history_save_DREAM_chain_history.npy')
        self.assertIs(np.array_equal(history, history_saved), True)
        remove('testing_history_save_DREAM_chain_history.npy')
        remove('testing_history_save_DREAM_chain_adapted_crossoverprob.npy')
        remove('testing_history_save_DREAM_chain_adapted_gammalevelprob.npy')
    
    def test_history_file_loading(self):
        """Test that when a history file is provided it is loaded and appended to the new history."""
        self.param, self.like = onedmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model)
        old_history = np.array([1, 3, 5, 7, 9, 11])
        step.save_history_to_disc(old_history, 'testing_history_load_')
        sampled_params, logps = run_dream(self.param, self.like, niterations=3, nchains=3, history_thin=1, history_file='testing_history_load_DREAM_chain_history.npy', save_history=True, model_name='test_history_loading', verbose=False)
        new_history = np.load('test_history_loading_DREAM_chain_history.npy')
        self.assertEqual(len(new_history), (len(old_history.flatten())+(3*step.total_var_dimension*3)))
        new_history_seed = new_history[:len(old_history.flatten())]
        new_history_seed_reshaped = new_history_seed.reshape(old_history.shape)
        self.assertIs(np.array_equal(old_history, new_history_seed_reshaped), True)
        
        added_history = new_history[len(old_history.flatten())::]
        sorted_history = np.sort(added_history)
        sorted_sampled_params = np.sort(np.array(sampled_params).flatten())

        for sampled_param, history_param in zip(sorted_history, sorted_sampled_params):
            self.assertEqual(sampled_param, history_param)
        remove('testing_history_load_DREAM_chain_history.npy')
        remove('testing_history_load_DREAM_chain_adapted_crossoverprob.npy')
        remove('testing_history_load_DREAM_chain_adapted_gammalevelprob.npy')
        remove('test_history_loading_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_history_loading_DREAM_chain_adapted_gammalevelprob.npy')
        remove('test_history_loading_DREAM_chain_history.npy')
        
    def test_crossover_file_loading(self):
        """Test that when a crossover file is loaded the crossover values are set to the file values and not adapted."""
        self.param, self.like = multidmodel()
        old_crossovervals = np.array([.45, .20, .35])
        np.save('testing_crossoverval_load_DREAM.npy', old_crossovervals)
        model = Model(self.like, self.param)
        dream = Dream(model=model, crossover_file='testing_crossoverval_load_DREAM.npy', save_history=True, model_name='testing_crossover_load')
        self.assertTrue(np.array_equal(dream.CR_probabilities, old_crossovervals))
        
        sampled_vals, logps = run_dream(self.param, self.like, niterations=100, nchains=3, crossover_file='testing_crossoverval_load_DREAM.npy', model_name='testing_crossover_load', save_history=True, verbose=False)
        
        crossover_vals_after_sampling = np.load('testing_crossover_load_DREAM_chain_adapted_crossoverprob.npy')
        self.assertIs(np.array_equal(crossover_vals_after_sampling, old_crossovervals), True)
        remove('testing_crossover_load_DREAM_chain_adapted_crossoverprob.npy')
        remove('testing_crossover_load_DREAM_chain_adapted_gammalevelprob.npy')
        remove('testing_crossoverval_load_DREAM.npy')
        remove('testing_crossover_load_DREAM_chain_history.npy')

    def test_astep_onedmodel(self):
        """Test that a single step with a one-dimensional model returns values of the expected type and a move that is expected or not given the test logp."""
        """Test a single step with a one-dimensional model with a normal parameter."""
        self.param, self.like = onedmodel()
        model = Model(self.like, self.param)
        dream = Dream(model=model, save_history=False, verbose=False)
        #Even though we're calling the steps separately we need to call these functions
        # to initialize the shared memory arrays that are called in the step fxn
        pool = _setup_mp_dream_pool(nchains=3, niterations=10, step_instance=dream)
        pool._initializer(*pool._initargs)

        #Test initial step (when last logp and prior values aren't set)
        q_new, last_prior, last_like = dream.astep(q0=np.array([-5]))

        self.assertTrue(isinstance(q_new, np.ndarray))
        self.assertTrue(isinstance(last_prior, numbers.Number))
        self.assertTrue(isinstance(last_like, numbers.Number))

        #Test later iteration after last logp and last prior have been set
        q_new, last_prior, last_like = dream.astep(q0=np.array(8),last_logprior=-300, last_loglike=-500)

        self.assertTrue(isinstance(q_new, np.ndarray))
        self.assertTrue(isinstance(last_prior, numbers.Number))
        self.assertTrue(isinstance(last_like, numbers.Number))

        if np.any(q_new != np.array(8)):
            self.assertTrue(last_prior+last_like >= -800)

        else:
            self.assertTrue(last_prior == -300)
            self.assertTrue(last_like == -500)

    def test_astep_multidmodel_uniform(self):
        """Test a single step of DREAM with a multi-dimensional model and uniform prior."""

        self.param, self.like = multidmodel_uniform()
        model = Model(self.like, self.param)
        dream = Dream(model=model, save_history=False, verbose=False)

        # Even though we're calling the steps separately we need to call these functions
        # to initialize the shared memory arrays that are called in the step fxn
        pool = _setup_mp_dream_pool(nchains=3, niterations=10, step_instance=dream)
        pool._initializer(*pool._initargs)

        # Test initial step (when last logp and prior values aren't set)
        q_new, last_prior, last_like = dream.astep(q0=np.array([-5, 4, -3, 0]))

        self.assertTrue(isinstance(q_new, np.ndarray))
        self.assertTrue(isinstance(last_prior, numbers.Number))
        self.assertTrue(isinstance(last_like, numbers.Number))

        # Test later iteration after last logp and last prior have been set
        q_new, last_prior, last_like = dream.astep(q0=np.array([8, 4, -2, 9]), last_logprior=100, last_loglike=-600)

        self.assertTrue(isinstance(q_new, np.ndarray))
        self.assertTrue(isinstance(last_prior, numbers.Number))
        self.assertTrue(isinstance(last_like, numbers.Number))

        if np.any(q_new != np.array(8)):
            self.assertTrue(last_prior + last_like >= -500)

        else:
            self.assertTrue(last_prior == 100)
            self.assertTrue(last_like == -600)

    def test_mp_sampledreamfxn(self):
        """Test the multiprocessing DREAM sample function returns data of the correct shape independently of the run_dream wrapper."""
        self.params, self.like = multidmodel()
        model = Model(self.like, self.params)
        dream = Dream(model=model, verbose=False, save_history=False)

        #Even though the pool won't be used, we need to initialize the shared variables.
        pool = _setup_mp_dream_pool(nchains=3, niterations=10, step_instance=dream)
        pool._initializer(*pool._initargs)

        iterations = 10
        start = np.array([-7, 8, 1.2, 0])
        verbose = False
        nverbose = 10
        args = [dream, iterations, start, verbose, nverbose]
        sampled_params, logps = _sample_dream(args)

        self.assertEqual(len(sampled_params), 10)
        self.assertEqual(len(sampled_params[0]), 4)
        self.assertEqual(len(logps), 10)
        self.assertEqual(len(logps[0]), 1)

    def test_paralleltempering_sampledreamfxn(self):
        """test that the parallel tempering DREAM sampling function returns values of the expected shape."""
        self.params, self.like = multidmodel()
        model = Model(self.like, self.params)
        dream = Dream(model=model, verbose=False, save_history=False)

        #The pool will be used within the fxn and, also, we need to initialize the shared variables.
        pool = _setup_mp_dream_pool(nchains=3, niterations=10, step_instance=dream)
        pool._initializer(*pool._initargs)

        start = np.array([-100, 5, 8, .001])

        sampled_params, logps = _sample_dream_pt(nchains=3, niterations=10, step_instance=dream, start=start, pool=pool, verbose=False)

        self.assertEqual(len(sampled_params), 3)
        self.assertEqual(len(sampled_params[0]), 20)
        self.assertEqual(len(sampled_params[0][0]), 4)
        self.assertEqual(len(logps), 3)
        self.assertEqual(len(logps[0]), 20)
        self.assertEqual(len(logps[0][0]), 1)

    def test_mp_paralleltempering_sampledreamfxn(self):
        """Test individual chain sampling function for parallel tempering returns an object of the correct type and with a better logp."""
        self.params, self.like = multidmodel()
        model = Model(self.like, self.params)
        dream = Dream(model=model, verbose=False, save_history=False)

        # Even though the pool won't be used, we need to initialize the shared variables.
        pool = _setup_mp_dream_pool(nchains=3, niterations=10, step_instance=dream)
        pool._initializer(*pool._initargs)

        start = np.array([-.33, 10, 0, 99])
        T = .33
        last_loglike = -300
        last_logprior = -100
        args = [dream, start, T, last_loglike, last_logprior]
        qnew, logprior_new, loglike_new, dream_instance = _sample_dream_pt_chain(args)

        self.assertTrue(isinstance(qnew, np.ndarray))
        self.assertTrue((logprior_new + loglike_new) >= -400)
    
class Test_Dream_Full_Algorithm(unittest.TestCase):

    def test_history_correct_after_sampling_simple_model(self):
        """Test that the history saved matches with the returned sampled parameter values for a one-dimensional test model."""
        self.param, self.like = onedmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model, save_history=True, history_thin=1, model_name='test_history_correct', adapt_crossover=False)
        sampled_params, logps = run_dream(self.param, self.like, niterations=10, nchains=5, save_history=True, history_thin=1, model_name='test_history_correct', adapt_crossover=False, verbose=False)
        history = np.load('test_history_correct_DREAM_chain_history.npy')
        self.assertEqual(len(history), step.total_var_dimension*((10*5/step.history_thin)+step.nseedchains))
        history_no_seedchains = history[(step.total_var_dimension*step.nseedchains)::]
        sorted_history = np.sort(history_no_seedchains)
        sorted_sampled_params = np.sort(np.array(sampled_params).flatten())

        for sampled_param, history_param in zip(sorted_history, sorted_sampled_params):
            self.assertEqual(sampled_param, history_param)

        remove('test_history_correct_DREAM_chain_history.npy')
        remove('test_history_correct_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_history_correct_DREAM_chain_adapted_gammalevelprob.npy')
            
        
    def test_history_correct_after_sampling_multidim_model(self):
        """Test that the history saved matches with the returned sampled parameter values for a multi-dimensional test model."""
        self.param, self.like = multidmodel()
        model = Model(self.like, self.param)
        step = Dream(model=model, save_history=True, history_thin=1, model_name='test_history_correct', adapt_crossover=False)
        sampled_params, logps = run_dream(self.param, self.like, niterations=10, nchains=5, save_history=True, history_thin=1, model_name='test_history_correct', adapt_crossover=False, verbose=False)
        history = np.load('test_history_correct_DREAM_chain_history.npy')

        self.assertEqual(len(history), step.total_var_dimension*((10*5/step.history_thin)+step.nseedchains))
        history_no_seedchains = history[(step.total_var_dimension*step.nseedchains)::]

        sorted_history = np.sort(history_no_seedchains)
        sorted_sampled_params = np.sort(np.array(sampled_params).flatten())

        for sampled_param, history_param in zip(sorted_history, sorted_sampled_params):
            self.assertEqual(sampled_param, history_param)

        remove('test_history_correct_DREAM_chain_history.npy')
        remove('test_history_correct_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_history_correct_DREAM_chain_adapted_gammalevelprob.npy')

    def test_boundaries_obeyed_aftersampling(self):
        """Test that boundaries are respected if included."""

        self.param, self.like = multidmodel_uniform()
        model = Model(self.like, self.param)
        step = Dream(model=model, save_history=True, history_thin=1, model_name='test_boundaries',
                     adapt_crossover=False, hardboundaries=True, nverbose=10)
        sampled_params, logps = run_dream(self.param, self.like, niterations=1000, nchains=5, save_history=True,
                                          history_thin=1, model_name='test_boundaries', adapt_crossover=False,
                                          verbose=True, hardboundaries=True, nverbose=10)
        history = np.load('test_boundaries_DREAM_chain_history.npy')
        variables = model.sampled_parameters
        dim = 0
        for var in variables:
            interval = var.interval()
            dim += var.dsize

        lowerbound = interval[0]
        upperbound = interval[1]

        npoints = int(len(history)/float(dim))
        reshaped_history = np.reshape(history, (npoints, dim))

        print('reshaped history: ',reshaped_history)
        print('upper ',upperbound,' and lower ',lowerbound)
        print('lower bounds: ',(reshaped_history<lowerbound).any())
        print('upper bounds: ',(reshaped_history>upperbound).any())
        print('disobeyed lower: ',reshaped_history[reshaped_history<lowerbound])
        print('disobeyed upper: ', reshaped_history[reshaped_history>upperbound])

        self.assertFalse((reshaped_history<lowerbound).any())
        self.assertFalse((reshaped_history>upperbound).any())

        remove('test_boundaries_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_boundaries_DREAM_chain_adapted_gammalevelprob.npy')
        remove('test_boundaries_DREAM_chain_history.npy')


class Test_DREAM_examples(unittest.TestCase):

    def test_CORM_example(self):
        """Test that the CORM example runs and returns values of the expected shape."""
        nchains = corm_kwargs['nchains']
        corm_kwargs['niterations'] = 100
        corm_kwargs['verbose'] = False
        #Check likelihood fxn works
        logp = corm_like([-5, -3, .1, 10, 8, 4, .33, -.58, 99, 1, 0, 11])

        #Check entire algorithm will run and give results of the expected shape
        sampled_params, logps = run_dream(**corm_kwargs)
        self.assertEqual(len(sampled_params), nchains)
        self.assertEqual(len(sampled_params[0]), 100)
        self.assertEqual(len(sampled_params[0][0]), 12)
        self.assertEqual(len(logps), nchains)
        self.assertEqual(len(logps[0]), 100)
        self.assertEqual(len(logps[0][0]), 1)
        remove('corm_dreamzs_5chain_DREAM_chain_adapted_crossoverprob.npy')
        remove('corm_dreamzs_5chain_DREAM_chain_adapted_gammalevelprob.npy')
        remove('corm_dreamzs_5chain_DREAM_chain_history.npy')

    def test_mixturemodel_example(self):
        """Test that the mixture model example runs and returns values of the expected shape."""
        nchains = mix_kwargs['nchains']
        mix_kwargs['niterations'] = 100
        mix_kwargs['verbose'] = False
        mix_kwargs['save_history'] = False

        #Check likelihood fxn works
        logp = mix_like(np.array([1, -9, 3, .04, 2, -8, 11, .001, -1, 10]))

        #Check that sampling runs and gives output of expected shape
        sampled_params, logps = run_dream(**mix_kwargs)
        self.assertEqual(len(sampled_params), nchains)
        self.assertEqual(len(sampled_params[0]), 100)
        self.assertEqual(len(sampled_params[0][0]), 10)
        self.assertEqual(len(logps), nchains)
        self.assertEqual(len(logps[0]), 100)
        self.assertEqual(len(logps[0][0]), 1)
        remove('mixturemodel_seed.npy')

    def test_ndimgaussian_example(self):
        """Test that the n-dimensional gaussian example runs and returns values of the expected shape."""
        nchains = ndimgauss_kwargs['nchains']
        ndimgauss_kwargs['niterations'] = 100
        ndimgauss_kwargs['verbose'] = False
        ndimgauss_kwargs['save_history'] = False

        #Check likelihood fxn runs
        logp = ndimgauss_like(np.random.random_sample((200,))*10)

        #Check sampling runs and gives output of expected shape
        sampled_params, logps = run_dream(**ndimgauss_kwargs)
        self.assertEqual(len(sampled_params), nchains)
        self.assertEqual(len(sampled_params[0]), 100)
        self.assertEqual(len(sampled_params[0][0]), 200)
        self.assertEqual(len(logps), nchains)
        self.assertEqual(len(logps[0]), 100)
        self.assertEqual(len(logps[0][0]), 1)
        remove('ndim_gaussian_seed.npy')

    def test_robertson_example(self):
        """Test that the Robertson example runs and returns values of the expected shape."""
        nchains = robertson_kwargs['nchains']
        robertson_kwargs['niterations'] = 100
        robertson_kwargs['verbose'] = False
        robertson_kwargs['save_history'] = False

        #Check likelihood fxn runs
        logp = robertson_like([3, 8, .11])

        #Check sampling runs and gives output of expected shape
        sampled_params, logps = run_dream(**robertson_kwargs)
        self.assertEqual(len(sampled_params), nchains)
        self.assertEqual(len(sampled_params[0]), 100)
        self.assertEqual(len(sampled_params[0][0]), 3)
        self.assertEqual(len(logps), nchains)
        self.assertEqual(len(logps[0]), 100)
        self.assertEqual(len(logps[0][0]), 1)

    def test_robertson_nopysb_example(self):
        """Test that the Robertson example without PySB runs and returns values of the expected shape."""

        nchains = rob_nop_kwargs['nchains']
        rob_nop_kwargs['niterations'] = 100
        rob_nop_kwargs['verbose'] = False
        rob_nop_kwargs['save_history'] = False

        #Check likelihood fxn runs
        logp = rob_nop_like([3, 8, .11])

        #Check sampling runs and gives output of expected shape
        sampled_params, logps = run_dream(**rob_nop_kwargs)
        self.assertEqual(len(sampled_params), nchains)
        self.assertEqual(len(sampled_params[0]), 100)
        self.assertEqual(len(sampled_params[0][0]), 3)
        self.assertEqual(len(logps[0]), 100)
        self.assertEqual(len(logps[0][0]), 1)

if __name__ == '__main__':
    unittest.main()
    
