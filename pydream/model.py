# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:40:32 2016

@author: Erin
"""
import numpy as np
from pebble import ProcessPool
from concurrent.futures import TimeoutError


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
        with ProcessPool(max_workers=1) as pool:
            future = pool.schedule(self.likelihood, [q0], timeout=5)
            try:
                loglike = future.result()
            except TimeoutError:
                future.cancel()  # Be explicit
                loglike = -np.inf
                print('TimeoutError')
            finally:
                pool.close()
                pool.join()

        # 🔍 Memory check after likelihood evaluation
        import psutil, os
        print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")

        return prior_logp, loglike
