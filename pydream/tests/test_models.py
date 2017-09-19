# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:58:09 2016

@author: Erin

#Parameters defined for simple example statistical models for testing DREAM
"""

from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import numpy as np

def onedmodel():
    """One dimensional model with normal prior."""
    
    mu = -2
    sd = 3
    x = SampledParam(norm, loc=mu, scale=sd)
    like = simple_likelihood   
    
    return [x], like

def multidmodel():
    """Multidimensional model with normal prior."""
    
    mu = np.array([-6.6, 3, 1.0, -.12])
    sd = np.array([.13, 5, .9, 1.0])
    
    x = SampledParam(norm, loc=mu, scale=sd)
    like = simple_likelihood
    
    return [x], like

def multidmodel_uniform():
    """Multidimensional model with uniform priors."""

    lower = np.array([-5, -9, 5, 3])
    upper = np.array([10, 2, 7, 8])
    range = upper-lower

    x = SampledParam(uniform, loc=lower, scale=range)
    like =simple_likelihood

    return [x], like

def simple_likelihood(param):
    """Flat likelihood."""
    
    return np.sum(param + 3)
