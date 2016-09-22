# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:34 2016

@author: Erin
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform


class SampledParam():
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.dsize = np.array(value).size
    
class NormalParam(SampledParam):
    def __init__(self, name, value, mu, sd):
        SampledParam.__init__(self, name, value)

        self.mu = mu
        self.sd = sd
        self.dis = None
    
    def random(self):
        return np.random.normal(self.mu, self.sd)
    
    def prior(self, q0):
        if self.dis == None:
            self.dis = norm(loc=self.mu, scale=self.sd)
        logp = np.sum(self.dis.logpdf(q0))
        return logp

class UniformParam(SampledParam):
    def __init__(self, name, value, lower, upper):
        SampledParam.__init__(self, name, value)

        self.lower = lower
        self.upper = upper
        self.range = self.upper - self.lower
        self.dis = None
        
    def random(self):
        return np.random.uniform(self.lower, self.upper)
    
    def prior(self, q0):
        if self.dis == None:
            self.dis = uniform(loc=self.lower, scale=self.range)
        logp = np.sum(self.dis.logpdf(q0))
        return logp
    
class FlatParam(SampledParam):
    def __init__(self, name, value):
        SampledParam.__init__(self, name, value)
        
    def prior(self, q0):
        return 0
        