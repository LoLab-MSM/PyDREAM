# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:34 2016

@author: Erin
"""

import numpy as np

class SampledParam():
    def __init__(self, scipy_distribution, *args, **kwargs):
        self.dist = scipy_distribution(*args, **kwargs)
        self.dsize = self.random().size

    def interval(self, alpha=1):
        return self.dist.interval(alpha)

    def random(self):
        return self.dist.rvs()

    def prior(self, q0):
        logp = np.sum(self.dist.logpdf(q0))

        return logp
    
class FlatParam(SampledParam):
    def __init__(self, test_value):
        self.dsize = test_value.size
        
    def prior(self, q0):
        return 0
        