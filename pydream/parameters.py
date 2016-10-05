# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:34 2016

@author: Erin
"""

import numpy as np

class SampledParam():
    """A SciPy-based parameter prior class.
    Initialization arguments:
        scipy_distribution: A SciPy statistical distribution (i.e. scipy.stats.norm)
        *args: Arguments for the SciPy distribution
        **kwargs: keyword arguments for the SciPy distribution

    Attributes:
        dist: The SciPy distribution underlying the parameter
        dsize: The dimension of the parameter
        """
    def __init__(self, scipy_distribution, *args, **kwargs):
        self.dist = scipy_distribution(*args, **kwargs)
        self.dsize = self.random().size

    def interval(self, alpha=1):
        """Return the interval for a given alpha value."""
        return self.dist.interval(alpha)

    def random(self):
        """Return a random value drawn from this prior."""
        return self.dist.rvs()

    def prior(self, q0):
        """Return the prior log probability given a point.
        Args:
            q0: A location in parameter space.
            """
        logp = np.sum(self.dist.logpdf(q0))

        return logp
    
class FlatParam(SampledParam):
    """A Flat parameter class (returns 0 at all locations).
    Initialization arguments:
    test_value: A representative value for the parameter.  Used the infer parameter dimension, which is needed in the DREAM algorithm.
    Attributes:
        dsize: The dimension of the parameter.
        """
    def __init__(self, test_value):
        self.dsize = test_value.size
        
    def prior(self, q0):
        return 0
        