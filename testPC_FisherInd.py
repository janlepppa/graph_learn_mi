#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:11:18 2017

@author: janleppa
"""

from fisherEstimator import fisherEstimator, parCorr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def createData(n, corr = 0.1):
    cov = np.array([[1, corr,], [corr, 1]])
    mean = [0,0]
    
    
    X = np.random.multivariate_normal(mean, cov, n)
    return(X)
    #print(np.corrcoef(X.T))
    
def createData2(n, C):
    
    X = np.random.multivariate_normal([0,0,0,0], C, n)
    
    #print(np.abs(parCorr(X[:,[0]],X[:,[3]],X[:,[1,2]])-(-IC[0,3])))
    
    return(X)
    
# Z-transformed sample correlations should (approximately) follow a normal distribution with mean given by the transformed population
# correlation and sd standard deviation 1/sqrt(N - 3). Plots the histogram and the pdf of normal distribution 
def correlationTest(n,bins = 100, ntests = 10000, corr = 0):
    
    
    
    meanZ = np.arctanh(corr)
    sdZ = 1/np.sqrt((n - 3))
    
    X = []

    for ii in range(0,ntests):
        D = createData(n,corr)
        sampleCorr = np.corrcoef(D.T)[0,1]
        X.append(np.arctanh(sampleCorr))
        
        
    
    minX = np.min(X) - np.std(X)
    maxX = np.max(X) + np.std(X)
    
    grid = np.linspace(minX,maxX, 1000)
    
    
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    
    histt = ax.hist(X,bins = bins, normed = True)
    h = ax.plot(grid, norm.pdf(grid, loc = meanZ, scale = sdZ), lw=2)
    
    plt.show()

# same for partial correlation    
def pcTest(n,bins = 100, ntests = 10000, x = 0, y = 1):
    
    IC = np.array([[1, 0, 0, 0.9], [0,1,0.2,0],[0,0.2,1,0.1],[0.9,0,0.1,1]])
    C = np.linalg.inv(IC)
    
    z = set(range(0,C.shape[1]))
    z.remove(x)
    z.remove(y)
    
    print("True partial correlation: ", -IC[x,y])
    meanZ = np.arctanh(-IC[x,y])
    sdZ = 1/np.sqrt((n - 3 - 2))
    
    X = []

    for ii in range(0,ntests):
        D = createData2(n,C)
        samplePC = parCorr(D[:,[x]],D[:,[y]],D[:,list(z)])
        
        X.append(np.arctanh(samplePC))
        
        
    
    minX = np.min(X) - np.std(X)
    maxX = np.max(X) + np.std(X)
    
    grid = np.linspace(minX,maxX, 1000)
    
    
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    
    histt = ax.hist(X,bins = bins, normed = True)
    h = ax.plot(grid, norm.pdf(grid, loc = meanZ, scale = sdZ), lw=2)
    
    plt.show()

# Check that the p-values used in fisherEstimator make sense
def correlationTest2(n, corr = 0, samples = 10000):
    
    D = createData(n,corr)
    sampleCorr = np.corrcoef(D.T)[0,1]
        
    ff = fisherEstimator()
    
    z = ff._fisherZ(sampleCorr)
    indep, pValue = ff._fisherTest(D[:,0],D[:,1])
    
    
    count = 0
    # sample data from the population distirbution and compute and count correaltions whose absolute value exceeds the observed one
    for ii in range(0,samples):
        D = createData(n,corr)
        sampleCorri = np.corrcoef(D.T)[0,1]
        zi = ff._fisherZ(sampleCorri)
        if(np.abs(zi) >= np.abs(z)):
            count += 1
            
    print("True correlation:                                                ", corr)
    print("Observed:                                                        ", sampleCorr)        
    print("P-value:                                                         ", pValue)
    print("proportion of transformed corrs. exceeding the observed:         ", count/samples)
    
# same for partial correlation. x and y refer to variables (the columns of data matrix, from 0 to 3)
def pcTest2(n, x = 0, y = 1, samples = 10000):
    
    # inverse of covariance
    IC = np.array([[1, 0, 0, 0.9], [0,1,0.2,0],[0,0.2,1,0.1],[0.9,0,0.1,1]])
    
    # covariance matrix
    C = np.linalg.inv(IC)
    
    z = set(range(0,C.shape[1]))
    z.remove(x)
    z.remove(y)
    
    D = createData2(n,C)
    
    samplePC = parCorr(D[:,[x]],D[:,[y]],D[:,list(z)])
        
    ff = fisherEstimator()
    
    Z = ff._fisherZ(samplePC)
    indep, pValue = ff._fisherTest(D[:,[x]],D[:,[y]],D[:,list(z)])
    
    
    count = 0
    
    # sample data from the population distirbution and compute and count transformed pcs whose absolute value exceeds the observed one
    for ii in range(0,samples):
        D = createData2(n,C)
        samplePCi = parCorr(D[:,[x]],D[:,[y]],D[:,list(z)])
        Zi = ff._fisherZ(samplePCi)
        if(np.abs(Zi) >= np.abs(Z)):
            count += 1
    
    print("True pc:                                                   ", -IC[x,y])
    print("Observed:                                                  ", samplePC)
    print("P-value:                                                   ", pValue)
    print("proportion of transformed pc exceeding the observed:       ", count/samples)
    