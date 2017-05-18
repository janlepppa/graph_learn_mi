#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:48:51 2016

@author: janleppa
"""
from knnEstimator import knnEstimator
from fisherEstimator import fisherEstimator, parCorr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


def plotErrors(results,ks,samples,title):
    nks, nsamples = results.shape
    
    plt.figure()
    plt.xlim(0.9,len(samples) + 0.1)
    x = list(range(1,len(samples) + 1,1))
    for ii in range(0,nks):
        lab = "k = " + str(ks[ii])
        plt.plot(x,results[ii,:],label = lab,marker = 'o')
        plt.xticks(x, samples)
        
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best')
    
    plt.title(title)
    plt.savefig(title + ".pdf")
    plt.show()
    
def compErrors(data,samples, ks, trueMI,p,eps):
    nsamples = len(samples)
    nks = len(ks)
    res1 = np.zeros((nks,nsamples))
    ni = 0
    
    for n in samples:
        ki = 0
        for k in ks:
            X = data[:n,:]
            aa = knnEstimator(k = k, p = p, eps = eps, noiseA=0)
            res1[ki,ni] = trueMI - aa._entropy(X)   
            ki += + 1
    
        ni += 1
        
    return(res1)
    
def compErrors2(x,y,samples, ks, trueMI,p,eps,z=None):  
    nsamples = len(samples)
    nks = len(ks)
    res1 = np.zeros((nks,nsamples))
    ni = 0
    
    for n in samples:
        ki = 0
        for k in ks:
            aa = knnEstimator(k = k, p = p, eps = eps, noiseA=0)
            
            if(z is not None):
                res1[ki,ni] = trueMI -aa._cmi1(x[:n,:],y[:n,:],z[:n,:]) 
                
            else:
                res1[ki,ni] = trueMI -aa._cmi1(x[:n,:],y[:n,:],z)
                
            ki += + 1
        ni += 1
        
    return(res1)
    
# test entropy estimators, plot absolute error as a function of sample size  
def testEntropy(seed, p = float('inf'), eps = 0.0):
    np.random.seed(seed)
    samples = [50,100,1000,5000,10000, 25000, 50000]
    ks = [1,3,5,7,9]

    # normally distributed data
    mu = 2.5
    sigma = 3
    data1 = np.random.normal(loc = mu,scale = sigma,size = (max(samples),1))
    trueE1 = np.log(sigma*np.sqrt(2*np.pi*np.e))   #in nats  
    res1 = compErrors(data1,samples,ks,trueE1,p,eps)   
    plotErrors(res1,ks,samples,'Gaussian RV')
    
    # logistic
    data2 = np.random.logistic(loc = 0.0,scale = 1.0,size = (max(samples),1))
    trueE2 = 2   #in nats
    res2 = compErrors(data2,samples,ks,trueE2,p,eps)
    plotErrors(res2,ks,samples,'Logistic RV')
    
    # uniform
    a = -1
    b = 3
    data3 = np.random.uniform(low = a,high = b,size = (max(samples),1))
    trueE3 = np.log(b-a)
    res3 = compErrors(data3,samples,ks,trueE3,p,eps)
    plotErrors(res3,ks,samples,'Uniform RV')
    
    # multivariate normal
    mean = [0,0,0]
    cov = [[1,0.2,0],[0.2,1,0.5],[0,0.5,1]] 
    data4 = np.random.multivariate_normal(mean,cov,max(samples))
    trueE4 = 0.5*(len(mean)*np.log(2*np.pi*np.e) + np.linalg.slogdet(cov)[1])
    res4 = compErrors(data4,samples,ks,trueE4,p,eps)
    plotErrors(res4,ks,samples,'Mv normal RV')
 
# test MI estimator version 1, plot absolute error as a function of sample size    
def testMI(seed, c = 0.1, p = float('inf'), eps = 0.0):
    
    # multivarial normal
    np.random.seed(seed)
    samples = [50,100,4000,10000,25000, 50000]
    ks = [1,3, 5, 10,40]

    n = max(samples)
    cov_m = [[1.0, c], [c, 1.0]]
    data = np.random.multivariate_normal([0, 0], cov_m, n)
    #trueMI = -1/2*np.log(1-np.corrcoef(data, rowvar=0)[0, 1]**2)
    trueMI = -1/2*np.log(1-c**2)
    res1 = compErrors2(data[:,[0]],data[:,[1]],samples, ks, trueMI,p,eps,z=None)
    plotErrors(res1,ks,samples,'MI: Bivariate normal RV, estimator 1')
    
    #res2 = compErrors2(data[:,[0]],data[:,[1]],samples, ks, trueMI,p,eps,z=None)
    #plotErrors(res2,ks,samples,'MI: Bivariate normal RV, estimator 2')
    
def testCMI(seed, tests = 10, p = float('inf'), eps = 0.0):
    
    # multivarial normal
    np.random.seed(seed)
    samples = [50,100,1000,10000,25000]
    ks = [1,3,5,10,15,25,40]

    n = max(samples)
    
    ic = np.array([[1.0, -0.2, 0],[-0.2, 1.0, 0.6],[0, 0.6, 1.0]])
    c = np.linalg.inv(ic)
    
    
    nsamples = len(samples)
    nks = len(ks)
    res1 = np.zeros((nks,nsamples))
    res2 = np.zeros((nks,nsamples))
    res3 = np.zeros((nks,nsamples))
      
    for tt in range(0,tests):
        data = np.random.multivariate_normal([0, 0, 0], c, n)
        
        x,y,z = 0,2,1
        cmixy_zT = mvnCMI(c,[x],[y],[z])
        res1 = res1 + compErrors2(data[:,[x]],data[:,[y]],samples, ks, cmixy_zT,p,eps,z = data[:,[z]])/tests
        pc1= parCorr(data[:,[x]],data[:,[y]],data[:,[z]])
        
        x,y,z = 1,2,0
        cmixy_zT = mvnCMI(c,[x],[y],[z])
        res2 = res2 +  compErrors2(data[:,[x]],data[:,[y]],samples, ks, cmixy_zT,p,eps,z = data[:,[z]])/tests
        pc2= parCorr(data[:,[x]],data[:,[y]],data[:,[z]])
      
                                   
        x,y,z = 0,1,2
        cmixy_zT = mvnCMI(c,[x],[y],[z])
        res3 = res3 + compErrors2(data[:,[x]],data[:,[y]],samples, ks, cmixy_zT,p,eps,z = data[:,[z]])/tests
        pc3= parCorr(data[:,[x]],data[:,[y]],data[:,[z]])
      
    plotErrors(res1,ks,samples,"pc=" + str(ic[0,2]))
    plotErrors(res2,ks,samples,"pc=" + str(ic[1,2]))
    plotErrors(res3,ks,samples,"pc=" + str(ic[0,1]))

 
# entropy for Gaussian variables, input: (scalar) variance or covariance matrix    
def mvnEntropy(cov_mat):
    if(np.isscalar(cov_mat)):
        return(0.5*(np.log(2*np.pi*np.e) +  np.log(cov_mat)))
    else:
        return(0.5*(cov_mat.shape[0]*np.log(2*np.pi*np.e) + np.linalg.slogdet(cov_mat)[1]))

def extSubmat(mat,variables):
    return(mat[variables,:][:,variables])
    
    

    
# MUTUAL INFORMATION for Gaussian variables, cov_mat should be a numpy array!
#If z argument is not given, compute mutual information: I(x,y) = H(x) + H(y)- H(x,y)
#I(x,y | z) = H(x,z) + H(y,z) - H(x,y,z) - H(z) , x,y,z are lists containing indices of variables referring to cov_mat   
def mvnCMI(cov_mat,x,y,z = None):
    if(z is None):
            xy = x + y
            Hxy = mvnEntropy(extSubmat(cov_mat,xy))
            Hy = mvnEntropy(extSubmat(cov_mat,y))
            Hx = mvnEntropy(extSubmat(cov_mat,x))
    
            return(Hx+Hy-Hxy)           
    else:     
        xz = x + z
        yz = y + z
        xyz = x + y + z
    
        Hxz = mvnEntropy(extSubmat(cov_mat,xz))
        Hyz = mvnEntropy(extSubmat(cov_mat,yz))
        Hxyz = mvnEntropy(extSubmat(cov_mat,xyz))
        Hz = mvnEntropy(extSubmat(cov_mat,z))
    
        return(Hxz + Hyz - Hxyz - Hz)
