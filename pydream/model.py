# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:40:32 2016

@author: Erin
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import TimeoutError


class Model(object):
    
    def __init__(self, likelihood, sampled_parameters):
        self.likelihood = likelihood
        if type(sampled_parameters) is list:
            self.sampled_parameters = sampled_parameters
        else:
            self.sampled_parameters = [sampled_parameters]
        
    def total_logp(self, q0):
        prior_logp = 0
        var_start = 0
        for param in self.sampled_parameters:
            var_end = param.dsize + var_start
            try:
                prior_logp += param.prior(q0[var_start:var_end])
            except IndexError:
                # raised if q0 is a single scalar
                prior_logp += param.prior(q0)
            var_start += param.dsize

        # Evaluate logp(s)
        with ProcessPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(self.likelihood, q0)
                loglike = future.result(timeout=5)
            except TimeoutError:
                loglike = -np.inf
        return prior_logp, loglike
