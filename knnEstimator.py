#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:05:07 2017

@author: janleppa
"""
# Constructor parameters:
#k              number of nearest neighbours used
#p              norm used in distance computations, either "float("inf")" (max/sup-norm) or 2 (Euclidean)
#               HOWEVER MI estimates with 2-norm don't seem to converge as they should so this should be used with caution 
#noiseA         noise to be added in order to deal with ties when searching knns   
#permutations   number of permutations used in mutual information or conditional mi based independence tests
#parallel       compute permutation test in parallel
#sig            parameter used when deciding on indepedence
#rng            instance of numpy.random.RandomState(), this will be reseeded in each permutation test made in parallel
#               in order to get reproducable results this should be given AND the global  numpy.random seed should be set                
  
import numpy as np
from scipy.special import psi, gammaln
import scipy.spatial as scsp
import multiprocessing as mp
from fisherEstimator import fisherEstimator

class knnEstimator:
    
    def __init__(self, 
                 k = 3, 
                 p = float("inf"),
                 eps = 0.0,
                 noiseA = 0.0,
                 permutations = 200,
                 permuteZ = False,
                 parallel = 4,
                 sig = 0.05,
                 rng = None,
                 LOWERthres = 0.001,
                 UPPERthres = np.inf,
                 corrCheck = True):
        self.k = k
        self.p = p
        self.eps = eps
        self.noiseA = noiseA                
        self.permutations = permutations
        self.parallel = parallel
        self.sig = sig
        self.permuteZ = permuteZ
        self.LOWERthres = LOWERthres
        self.UPPERthres = UPPERthres
        self.corrCheck = False
        
        if corrCheck == True:
            self.corrCheck = True
            self.corrTest = fisherEstimator(sig = sig)
        
        if rng is None:
            self.rng = np.random.RandomState()      # random number generator for permutation tests, this will reseed
        else:
            self.rng = rng
            
    def independent(self,x,y,z = None):
        indep, estMI, varMI, prcn, MIs = self._permutationTest(x,y,z)
        
        return(indep, estMI)
            
    def dependence(self,x,y,z = None):
        
        estMI = self._cmi1(x,y,z)
        
        return(estMI)
        
    def _log_cd(self, d):
        hd = 0.5*d
        return((hd)*np.log(np.pi) - gammaln(1 + hd) - d*np.log(2))
        
    # Entropy estimator by Kraskov et al. The inåut data x should be Nxd numpy array (N obs ervations of d-dimensional variable)   
    def _entropy(self,x):
        N,d = x.shape 
        
        if(np.isinf(self.p)):
            log_c_d = 0
        elif(self.p == 2):
            log_c_d = self._log_cd(d)
        else:
            print("Check that p is either 2 or infinity")
            
        # distance to k:th neighbour averaged over all the points
        xNoise = x + self.noiseA*self.rng.normal(loc = 0.0, scale = 1.0, size = (N,d)) 
        tree = scsp.cKDTree(xNoise)
        avg_dist = np.mean(np.log(2*tree.query(xNoise,k = self.k + 1, eps = self.eps, p = self.p)[0][:,self.k]))
            
        # Kraskov Eq. (20)
        ent = psi(N) - psi(self.k) + log_c_d + d*avg_dist
    
        return(ent)

    # Mutual information estimator by Kraskov et al. The inåut datas x and y should be Nxd1 and Nxd2 numpy arrays. (version 1, Eq (8))
    def _mi1(self,x,y):
        assert(x.shape[0] == y.shape[0])
        
        if(np.array_equal(x,y)):
            return(self._entropy(x))
        
        N,d1 = x.shape
        d2 = y.shape[1]

        # add small noise and construct KD-trees
        if(self.noiseA <= 0.0):
            addNoise = False
        else:
            addNoise = True
        
        
        if(addNoise == True):
            xNoise = x + self.noiseA*self.rng.normal(loc = 0.0, scale = 1.0, size = (N,d1))
            yNoise = y + self.noiseA*self.rng.normal(loc = 0.0, scale = 1.0, size = (N,d2))
    
        else:
            xNoise = x
            yNoise = y
    
        xyNoise = np.hstack((xNoise,yNoise)) 
        
        treeXY = scsp.cKDTree(xyNoise)
        treeX = scsp.cKDTree(xNoise)
        treeY = scsp.cKDTree(yNoise)
        
        # find the distance to the k:th neighbour in for every point in (x,y) space
        Kdists = treeXY.query(xyNoise,k = self.k + 1, eps = self.eps, p = float("inf"))[0][:,self.k] # k + 1 since 1st neighbour is always the point itself 
        
         
        avg_psi_n_x = 0
        avg_psi_n_y = 0
        
        for ii in range(0,N):
            #print(len(treeX.query_ball_point(xNoise[ii,:],Kdists[ii]-epss,eps = eps, p = p )))
            epss = 1e-10
            
            if epss > Kdists[ii]:   # avioids situations where query distance would be 0 or less
                epss = 9*Kdists[ii]/10
         
            avg_psi_n_x += psi(len(treeX.query_ball_point(xNoise[ii,:],Kdists[ii]-epss,eps = self.eps, p = self.p )))/N # no +1 since it is implicitly included (query ball contains always the point itself)
            avg_psi_n_y += psi(len(treeY.query_ball_point(yNoise[ii,:],Kdists[ii]-epss,eps = self.eps, p = self.p )))/N                                       
        
        # Eq. (8) in Kraskov et al.
        #print("avgspi: ",(avg_psi_n_x + avg_psi_n_y))
        
        mi = psi(self.k) + psi(N) - (avg_psi_n_x + avg_psi_n_y)
        
        if(self.p == 2):
            norm_term = self._log_cd(d1) - self._log_cd(d2)
            return(mi + norm_term)
        
        if(self.p == float("inf")):
            return(mi)
        
        
        return(mi)
        
              
    # coditional mutual information estimator based on Kraskov estimator version 1.    
    def _cmi1(self,x,y,z = None):
        if(z is None):
            return(self._mi1(x,y))
        
        assert(x.shape[0] == y.shape[0] & z.shape[0] == x.shape[0])
        N,d1 = x.shape
        d2 = y.shape[1]
        d3 = z.shape[1]
    
        # add small noise and construct KD-trees
        if(self.noiseA <= 0.0):
            addNoise = False
        else:
            addNoise = True
        
        
        if(addNoise == True):
            xNoise = x + self.noiseA*self.rng.normal(loc = 0.0, scale = 1.0, size = (N,d1))
            yNoise = y + self.noiseA*self.rng.normal(loc = 0.0, scale = 1.0, size = (N,d2))
            zNoise = z + self.noiseA*self.rng.normal(loc = 0.0, scale = 1.0, size = (N,d3))
        else:
            xNoise = x
            yNoise = y
            zNoise = z
        
        xyzNoise = np.hstack((xNoise,yNoise,zNoise)) 
        xzNoise = np.hstack((xNoise,zNoise))
        yzNoise = np.hstack((yNoise,zNoise)) 
        
        treeXYZ = scsp.cKDTree(xyzNoise)
        treeXZ = scsp.cKDTree(xzNoise)
        treeYZ = scsp.cKDTree(yzNoise)
        treeZ = scsp.cKDTree(zNoise)
        
        # find the distance to the k:th neighbour in for every point in (x,y) space using max-norm
        Kdists = treeXYZ.query(xyzNoise,k = self.k + 1, eps = self.eps, p = self.p)[0][:,self.k] # k + 1 since 1st neighbour is always the point itself 
        
        
        avg_psi_n_xz = 0
        avg_psi_n_yz = 0
        avg_psi_n_z = 0
        
        for ii in range(0,N):
            #print(len(treeX.query_ball_point(xNoise[ii,:],Kdists[ii]-epss,eps = eps, p = p )))
            epss = 1e-10
            if epss > Kdists[ii]:   # avioids situations where query distance would be 0 or less
                epss = 9*Kdists[ii]/10
            
            avg_psi_n_xz += psi(len(treeXZ.query_ball_point(xzNoise[ii,:],Kdists[ii]-epss,eps = self.eps, p = self.p )))/N # no +1 since query ball contains always the point itself
            avg_psi_n_yz += psi(len(treeYZ.query_ball_point(yzNoise[ii,:],Kdists[ii]-epss,eps = self.eps, p = self.p )))/N
            avg_psi_n_z += psi(len(treeZ.query_ball_point(zNoise[ii,:],Kdists[ii]-epss,eps = self.eps, p = self.p )))/N                                       
        
        # Eq. can be found in Vejmelka and Palus section II.B.1
        cmi = psi(self.k) - (avg_psi_n_xz + avg_psi_n_yz - avg_psi_n_z)
        return(cmi)
        
    # pwermutation test for (conditional) independence  
    def _permutationTest(self,x,y,z = None):
        estMI = self._cmi1(x,y,z)
        
        # return just the estimated Mutual information if number of permutations = 1
        if(self.permutations == 1):
             
             return(None, estMI, None, None, None)
        
        
        # skip permutation tests if correlation is detected oOR if MI is low (under specified threshold) and correlation test implies independence   
        if(self.corrCheck):
            corr, negPval = self.corrTest.independent(x,y,z) # result of a partial correlation based inpendence test
                
            if(corr == False and estMI > self.UPPERthres): # linear dependence detected and estimated MI is over the threshold
                return(False, estMI, None, None, None)              
            
            if(estMI < self.LOWERthres and corr == True): # low MI and low correlation
                return(True, estMI, None, None, None)
                
        # permutation tests    
        rr = np.random.randint(0,2**32-2-self.permutations) # THIS relies on global seed 
        args = self._argIterator(x,y,z,rr)
   
        if(self.parallel == 1):            
            MIs = list(map(self._permMI,args))                                                                
        else:
  
            pool = mp.Pool(self.parallel)
            
            MIs = pool.map(self._permMI,args)
            
            pool.close()
            pool.join()
                
            
        varMI = np.var(MIs)
        extremes = (np.array(MIs) >= estMI).sum() 
        estPVal = (extremes + 1)/(self.permutations + 1) 
        
        indep = True # null-hypothesis

        if(estPVal < self.sig):
            indep = False
            
        if(len(np.unique(MIs)) != self.permutations):
            print("some of the permuted MIs are exactly equal!!!")
        
        return(indep, estMI, varMI, estPVal, MIs)
        
    # one argument function to call permutation test
    def _permMI(self,args):
        x,y,z,seed = args
        
        if seed is not None:
            self.rng.seed(seed)
                  
        permutedY = self.rng.permutation(y)    
  
        if z is None:                     
            return(self._cmi1(x,permutedY))
        else:
            if self.permuteZ:
                permutedZ = self.rng.permutation(z)
                return((self._cmi1(x,permutedY, permutedZ)))
            else:
                return(self._cmi1(x,permutedY, z))
    
    # creates an iterator over arguments for the permutation tests            
    def _argIterator(self,x,y,z,startSeed):
        for ii in range(0,self.permutations):
            yield (x,y,z,startSeed+ii)
            