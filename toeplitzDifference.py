#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:09:10 2019

@author: xies
"""
import numpy as np
from scipy.linalg import toeplitz

def forward_difference(length):
    """
    
    Toeplitz matrix for forward difference
    
    """
    r = np.zeros(length)
    c = np.zeros(length)
    r[0] = -1
    r[length-1] = 1
    # c[1] = 1
    return toeplitz(r,c)

def backward_difference(length):
    """ returns a toeplitz matrix
      for backward differences
    """
    r = np.zeros(length)
    c = np.zeros(length)
    r[0] = 1
    # r[length-1] = -1
    c[1] = -1
    return toeplitz(r,c).T

def central_difference(length):
    """ returns a toeplitz matrix
      for central differences (kernelsize = 1)
    """
    r = np.zeros(length)
    c = np.zeros(length)
    r[1] = .5
    # r[length-1] = -.5
    c[1] = -.5
    # c[length-1] = .5
    M = toeplitz(r,c).T
    # Boundary conditions: fixed
    M[-1,-1] = 1
    M[-1,-2] = -1
    return M
