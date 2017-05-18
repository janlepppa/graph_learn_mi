#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:18:44 2017

@author: janleppa
"""
import numpy as np
from huge import hugeGenerateData, hugeLearnGraph
from graphUtils import HD
#from scipy.stats import wishart
from scipy.linalg import block_diag
from structLearn import structLearn
from knnEstimator import knnEstimator
from fisherEstimator import fisherEstimator

class dataGenerator:
    def __init__(self, testName = (True,"gaussian"), rng = None):
        self.testName = testName
        if rng is None:
            self.rng = np.random.RandomState()             
        else:
            self.rng = rng
        
    def createData(self,n):
        if type(self.testName) == tuple:     
            linear, noise = self.testName
            return(self.genDataSmallUG(n,linear,noise))
        elif self.testName == "largeUG":
            return(self.genDataLargeUG(n,21,linear = False,noise = "t"))
        elif self.testName == "largeUG_g":
            return(self.genDataLargeUG(n,21,linear = False,noise = "gaussian"))
        elif self.testName == "largeUG_u":
            return(self.genDataLargeUG(n,21,linear = False,noise = "uniform"))
        elif self.testName == "randomUG":
            return(self.randomGaussUG(n,d = 10))
        elif self.testName == "randomUGnonPara":
            return(self.randomNonParaUG(n,d = 10))
        elif self.testName == "randomUGLarge":
            return(self.randomGaussUG(n,d = 20))
        elif self.testName == "randomUGnonParaLarge":
            return(self.randomNonParaUG(n,d = 20))
        else:    
            print("oops")
                    

    # generates data for the small UG experiment                
    def genDataSmallUG(self,n, linear = True, noise = "gaussian"):
 
        if linear == True:
            X1 = self.sampleNoise(n,noiseType = noise)
            X2 = 0.20*X1 + self.sampleNoise(n,noiseType = noise)
            X3 = 0.50*X2 + self.sampleNoise(n,noiseType = noise)
            X4 = 0.25*X3 + self.sampleNoise(n,noiseType = noise)
            X5 = 0.35*X2 + 0.55*X3  + self.sampleNoise(n,noiseType = noise)
            X6 = 0.65*X5 + self.sampleNoise(n,noiseType = noise)
            X7 = 0.90*X3 + 0.25*X5 + self.sampleNoise(n,noiseType = noise)
            
        if linear == False:
            X1 = self.sampleNoise(n,noiseType = noise)
            X2 = 2*np.cos(X1) + self.sampleNoise(n,noiseType = noise)
            X3 = 2*np.sin(np.pi*X2) + self.sampleNoise(n,noiseType = noise)
            X4 = 3*np.cos(X3) + self.sampleNoise(n,noiseType = noise)
            X5 = 0.75*X2*X3 + self.sampleNoise(n,noiseType = noise)
            X6 = 2.5*X5 + self.sampleNoise(n,noiseType = noise)
            X7 = 3*np.cos(0.2*X3) + np.log(np.abs(X5)) + self.sampleNoise(n,noiseType = noise)
            
        X = np.hstack((X1,X2,X3,X4,X5,X6,X7))
        
        G = np.array([[0,1,0,0,0,0,0],
                      [0,0,1,0,1,0,0],
                      [0,0,0,1,1,0,1],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,1,1],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]])
        
        G = 1*(G.T + G > 0)
        
        return(X,G)
        
    def genDataLargeUG(self,n,d, linear = True, noise = "gaussian"):
        assert d % 7 == 0
            
        multip = np.int(d/7)
        
        datas = []
        G = np.array([])
        
        for ii in range(0,multip):
            x,g = self.genDataSmallUG(n,linear = linear,noise = noise)
            datas.append(x) 
            G = block_diag(G,g)
            
        X = np.hstack(tuple(datas))    
                
        return(X,G)
              
    # returns noise from given (gaussian,uniform,t) distribution    
    def sampleNoise(self,n, noiseType = "gaussian"):
        if noiseType == "gaussian":
            return(self.rng.normal(0, 1, (n,1)))
        if noiseType == "uniform":
            return(self.rng.uniform(-1,1,(n,1)))
        if noiseType == "t":
            return(self.rng.standard_t(2,(n,1)))
            
    # creates Gaussian data with a random underlying undirected graph        
    def randomGaussUG(self,n,d, prob = None):
        
        seed = self.rng.randint(1,1e9)
        X,G = hugeGenerateData(n,d, graph = "random", prob = prob,seed = seed)
        
        return(X,G)
    
    # similar to above one, except the data is now transformed X -> X**3     
    def randomNonParaUG(self,n,d, prob = None):
        
        seed = self.rng.randint(1,1e9)
        X,G = hugeGenerateData(n,d, graph = "random", prob = prob,seed = seed)
        X = X**3
        return(X,G)
        
        
    # create adjacency matrix of a random graph with "d" nodes and approximately "expectEdges" number of edges    
    def randUG(self,d,expectEdges = 10):
     
        G = np.zeros((d,d),dtype = np.int)
        possibleEdges = d*(d-1)/2
        prob = expectEdges/possibleEdges
    
        for ii in range(0,d):
            for jj in range(ii +1,d):
                u = self.rng.uniform(0,1,1)
                if u < prob:
                    G[ii,jj] = 1
                    G[jj,ii] = 1
        
        return(G)
         
