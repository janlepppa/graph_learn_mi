#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:11:36 2017

@author: janleppa
"""
from knnEstimator import knnEstimator
import numpy as np

# Functions and stuff shared by the all MB algorithms
class MBAlgorithms:
    
    def __init__(self, X, algorithm, estimator = None ):
        self.cache = dict()
        self.nTests = 0
        self.X = X
        
        if estimator is None:
            self.estimator = knnEstimator()
        else:    
            self.estimator = estimator
        self.algorithm = algorithm
        
    def clearCache(self):
        self.cache = dict()
        
    def resetTestCounter(self):
        self.nTests = 0
        
    def _dependence(self, var_inx ,y ,MB):
        yi = self.X[:,[y]]
        x = self.X[:,[var_inx]]
    
        if(len(MB) == 0):
            z = None
        else:
            z = self.X[:,list(MB)]       #conditioning set
                 
        inCache, key = self._isCached(var_inx, y, MB)
        
        if(inCache):
            estMI = self.cache[key][1]
        else:
            estMI = self.estimator.dependence(x,yi,z)     
        return estMI
        
            
    def _doIndepTest(self, var_inx ,y ,MB):
        if(MB is None):
            z = None
        elif(len(MB) == 0):
            z = None
        else:
            z = self.X[:,list(MB)]       #conditioning set
            
        assert(np.isscalar(var_inx) & np.isscalar(y))

        yi = self.X[:,[y]]
        x = self.X[:,[var_inx]]
    
        inCache, key = self._isCached(var_inx, y, MB)
        if(inCache):
            indep = self.cache[key][0]
        else:
            indep, estMI = self.estimator.independent(x,yi,z)
            #print("Testing", var_inx, y, "given", MB)
            #print("         ", indep, estMI)
            self._putInCache(key, (indep,estMI))
            self.nTests += 1
            
        return indep
        
        
    def _isCached(self, x,y,z):
            key = self._returnKey(x,y,z)
            
            if(key in self.cache):
                #print("cache used", x, y ,z)
                return(True, key)
            else:
                return(False,key)
                
    # key is tuple containing tuples (x,y) (sorted) and (z) (sorted)            
    def _returnKey(self, x, y, z):
        
        xy = [x, y]
        xy.sort()
        
        zz = list(z)
        zz.sort()
        
        return((tuple(xy),tuple(zz)))
        
    def _putInCache(self, key, value):
        self.cache[key] = value     
        
# Grow-Shrink algorithm, TODO: add UG mode (some independence tests might be redundant if faithfulness to UG is assumes)
class GS(MBAlgorithms):
    
    def __init__(self, X, estimator = None):
        if estimator is None:
            estimator = knnEstimator()
        MBAlgorithms.__init__(self,X,"Grow-Shrink", estimator)    
        
    # Find Markov blanket for variable whose (column index) is var_inx 
    def findMB(self, var_inx):
        d = self.X.shape[1]

        cands = set(range(0,d))
        cands.remove(var_inx)               # initial candidate variables for entering the Markov blanket
        
        newCands = set(cands)
        MB = set()                          # initilize Markov blanket to an empty set
        
        # grow phase
        grow = True
        
        while(grow):
    
            grow = False
                  
            for y in cands:
            
                indep = self._doIndepTest(var_inx, y, MB)
                    
                if(indep == False):         # x (conditionally) dependent on y
                    MB.add(y)               # add y to blankets
                    newCands.remove(y)      # remove from candidate set
                    grow = True             # Markov blanket changed, try to add more (if there are candidates left)
            
                    
            cands = set(newCands)        
            if(len(cands) == 0):
                grow = False
                
                           
        # shrink phase
        if(len(MB) <= 1):
            shrink = False                  # no need to shrink if MB size is 0 or 1 (with 1 we would be repeating one test)
        else:
            shrink = True
            
        while(shrink):
            
            #print(MB)
            shrink = False
            newMB = set(MB)
            
            for y in MB:
                newMB.remove(y)
                
                indep = self._doIndepTest(var_inx, y, newMB)
                
                if(indep == False):
                    newMB.add(y)        # conditional independence does not hold, put variable back 
                    
                
            if(MB != newMB):            # blanket changed, update MB and continue shrinking
                shrink = True
                MB = set(newMB)
            
               
            #print(MB)
        return(MB)
 
 # Incremental Association Markov Blanket algorithm       
class IAMB(MBAlgorithms):
    def __init__(self, X, estimator = None, mode = 'DAG'):
        if estimator is None:
            estimator = knnEstimator()
        MBAlgorithms.__init__(self,X,"IAMB", estimator)
        self.mode = mode
        
    def findMB(self,var_inx):
        d = self.X.shape[1]

        cands = set(range(0,d))
        cands.remove(var_inx)               # initial candidate variables for entering the Markov blanket
        
        MB = set()                          # initilize Markov blanket to an empty set
        
        # add variables to the blanket
        addVariables = True
        
        while(addVariables):
            
            addVariables = False
            
            highestMI = -np.inf
            highestMIindex = np.nan
         
            for y in cands:
                
                miXY = self._dependence(var_inx,y,MB)
                              
                if(miXY > highestMI):
                    highestMIindex = y
                    highestMI = miXY
                    
            
            # check (conditional independence)
            indep = self._doIndepTest(var_inx, highestMIindex , MB)
            
            if(indep == False):
                
                MB.add(highestMIindex)        
                cands.remove(highestMIindex)
                
                if(len(cands) > 0):
                    addVariables = True # MB changed, continue adding
            else:
                if(self.mode == "UG"): # Faithfulness to Markov network assumed
                    cands.remove(highestMIindex) # we can remove the independent variable from the subsequent tests
                    
                    if(len(cands) > 0):
                        addVariables = True # MB changed, continue adding
        
        ########### removal phase
        if(len(MB) <= 1):
            removeVariables = False
        else:
             removeVariables = True
             
        while(removeVariables):
            
            newMB = set(MB)
            removeVariables = False
            
            for y in MB:
                newMB.remove(y)
                
                
                indep = self._doIndepTest(var_inx, y, newMB)
               
                if(indep == False):
                    newMB.add(y)        # conditional independence does not hold, put variable back 
                    
                
            if(MB != newMB):            # blanket changed, update MB and continue shrinking
                removeVariables = True
                MB = set(newMB)
                
                
        return(MB)
        
# interleaved IAMB, each succesfull addition step is follow by a deletion step            
class interIAMB(MBAlgorithms):
        def __init__(self, X, estimator = None, mode = 'DAG'):
            if estimator is None:
                estimator = knnEstimator()
            
            MBAlgorithms.__init__(self,X,"interIAMB", estimator)
            self.mode = mode
        
        def findMB(self,var_inx):
            d = self.X.shape[1]
    
            cands = set(range(0,d))
            cands.remove(var_inx)               # initial candidate variables for entering the Markov blanket
            
            MB = set()                          # initilize Markov blanket to an empty set
            
            # add variables to the blanket
            addVariables = True
            
            #print("Adding variables: ")
            while(addVariables):
                
                addVariables = False
                
                highestMI = -np.inf
                highestMIindex = np.nan
             
                for y in cands:
                    
                    miXY = self._dependence(var_inx,y,MB)
                    
                    #print("var", y, ",   MI = ", miXY)
                    
                    if(miXY > highestMI):
                        highestMIindex = y
                        highestMI = miXY
                        
                
                # check (conditional independence)
                indep = self._doIndepTest(var_inx, highestMIindex , MB)
                
                if(indep == False):        
                    MB.add(highestMIindex)        
                    cands.remove(highestMIindex)
                 
                    # possible removal phase
                    if(len(MB) <= 1):
                        removeVariables = False
                    else:
                        removeVariables = True
                 
                    while(removeVariables):
                
                        newMB = set(MB)
                        removeVariables = False
                
                        for y in MB:
                            newMB.remove(y)
                    
                            indep = self._doIndepTest(var_inx, y, newMB)
                    
                            if(indep == False):
                                newMB.add(y)        # conditional independence does not hold, put variable back 
                        
                    
                        if(MB != newMB):            # blanket changed, update MB and continue shrinking
                            removeVariables = True
                            MB = set(newMB)
                    
                    
                    if(len(cands) > 0):
                        addVariables = True # MB changed, continue adding
                        
                else:
                    if(self.mode == "UG"): # Faithfulness to Markov network assumed
                        cands.remove(highestMIindex) # we can remove the independent variable from the subsequent tests
                    
                        if(len(cands) > 0):
                            addVariables = True # MB changed, continue adding  
         
            return(MB)
            
#class KIAMB(MBAlgorithms):
#    def __init__(self, X, K = 0.8, estimator = knnEstimator(), mode = 'DAG'):
#        MBAlgorithms.__init__(self,X,"IAMB", estimator)
#        self.mode = mode
#        self.K = K
#        
#    def findMB(self,var_inx):
#        d = self.X.shape[1]
#
#        cands = set(range(0,d))
#        cands.remove(var_inx)               # initial candidate variables for entering the Markov blanket
#        
#        MB = set()                          # initilize Markov blanket to an empty set
#        
#        # add variables to the blanket
#        addVariables = True
#        
#        while(addVariables):
#            
#            addVariables = False
#            
#            highestMI = -np.inf
#            highestMIindex = np.nan
#         
#            for y in cands:
#                
#                miXY = self._dependence(var_inx,y,MB)
#                              
#                if(miXY > highestMI):
#                    highestMIindex = y
#                    highestMI = miXY
#                    
#            
#            # check (conditional independence)
#            indep = self._doIndepTest(var_inx, highestMIindex , MB)
#            
#            if(indep == False):
#                
#                MB.add(highestMIindex)        
#                cands.remove(highestMIindex)
#                
#                if(len(cands) > 0):
#                    addVariables = True # MB changed, continue adding
#            else:
#                if(self.mode == "UG"): # Faithfulness to Markov network assumed
#                    cands.remove(highestMIindex) # we can remove the independent variable from the subsequent tests
#                    
#                    if(len(cands) > 0):
#                        addVariables = True # MB changed, continue adding
#        
#        ########### removal phase
#        if(len(MB) <= 1):
#            removeVariables = False
#        else:
#             removeVariables = True
#             
#        while(removeVariables):
#            
#            newMB = set(MB)
#            removeVariables = False
#            
#            for y in MB:
#                newMB.remove(y)
#                
#                
#                indep = self._doIndepTest(var_inx, y, newMB)
#               
#                if(indep == False):
#                    newMB.add(y)        # conditional independence does not hold, put variable back 
#                    
#                
#            if(MB != newMB):            # blanket changed, update MB and continue shrinking
#                removeVariables = True
#                MB = set(newMB)
#                
#                
#        return(MB)
#    
#            
#            
#class IAMBscore(MBAlgorithms):
#    # Similar as the IAMB but no independence tests are made. Nodes are added/removed if the conditional mutual information is over/below the given threshold
#        def __init__(self, X, threshold = 0.005, estimator = knnEstimator()):
#            MBAlgorithms.__init__(self,X,"IAMBscore", estimator)         
#            self.threshold = threshold
#            
#        def findMB(self,var_inx):
#            d = self.X.shape[1]
#    
#            cands = set(range(0,d))
#            cands.remove(var_inx)               # initial candidate variables for entering the Markov blanket
#            
#            MB = set()                          # initilize Markov blanket to an empty set
#            
#            # add variables to the blanket
#            addVariables = True
#            
#            #globalScore = 0
#            
#            
#            while(addVariables):
#                
#                addVariables = False
#       
#                highestMIindex = np.nan
#                highestMI = -np.inf
#             
#                for y in cands:
#                    
#                    miXY = self._dependence(var_inx,y,MB)
#                                  
#                    if(miXY > highestMI):
#                        highestMIindex = y
#                        highestMI = miXY
#                        
#                
#                if(not np.isnan(highestMIindex)):        # adding some variable increased the score
#                    if(highestMI > self.threshold):
#                        MB.add(highestMIindex)        
#                        cands.remove(highestMIindex)
#                    
#                        if(len(cands) > 0):
#                            addVariables = True # MB changed, continue adding
#                      
#            if(len(MB) <= 1):
#                removeVariables = False
#            else:
#                 removeVariables = True
#                 
#            while(removeVariables):
#                
#                newMB = set(MB)
#                removeVariables = False
#                               
#                for y in MB:
#                    newMB.remove(y)
#                    
#                    miXY = self._dependence(var_inx,y,newMB)
#                    
#                                        
#                    if(miXY < self.threshold):   # removing the variable keeps the mutual information between variable and new Blanket almost the same
#                        continue
#                    else:
#                        newMB.add(y)       #  removing did not increase the score, put it back
#           
#                if(MB != newMB):            # blanket changed, update MB and continue shrinking
#                    removeVariables = True
#                    MB = set(newMB)
#                    
#                    
#            return(MB)        
            
        
        
                

        
          

    