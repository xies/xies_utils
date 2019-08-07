#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:09:10 2019

@author: xies
"""
from scipy.linalg import toeplitz

def forward_difference(size):
 """ returns a toeplitz matrix
   for forward differences
 """
 r = np.zeros(size)
 c = np.zeros(size)
 r[0] = -1
 r[size-1] = 1
 c[1] = 1
 return toeplitz(r,c)

def backward_difference(size):
 """ returns a toeplitz matrix
   for backward differences
 """
 r = np.zeros(size)
 c = np.zeros(size)
 r[0] = 1
 r[size-1] = -1
 c[1] = -1
 return toeplitz(r,c).T

def central_difference(size):
 """ returns a toeplitz matrix
   for central differences (kernelsize = 1)
 """
 r = np.zeros(size)
 c = np.zeros(size)
 r[1] = .5
 r[size-1] = -.5
 c[1] = -.5
 c[size-1] = .5
 return toeplitz(r,c).T
