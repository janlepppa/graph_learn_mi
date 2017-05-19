#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:41:05 2017

@author: janleppa
"""

from slTests import doTests
import numpy as np

if __name__ == "__main__":

    ns = [125,250,500,1000,2000] #sample size    
    ntests = 25 #number of tests
    k = 3 # k for the knnEstimator
    folder = "k3UPinf"        
    useTrans = False
    upperThres = np.inf
        
    doTests("randomGMLarge15", ntests = ntests, ns = ns,k = k ,folderName = folder,upperThres = upperThres,useTransformation=useTrans)

