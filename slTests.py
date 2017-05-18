#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:47:46 2017

@author: janleppa
"""
import numpy as np
from knnEstimator import knnEstimator
from structLearn import structLearn
from fisherEstimator import fisherEstimator
from graphUtils import HD
from dataGenerator import dataGenerator
from huge import hugeLearnGraph,transform
import multiprocessing as mp
import copy
import os
import pickle
from sklearn.preprocessing import scale

def doTests(testName, 
            folderName = None, 
            seed = 123456, 
            ntests = 25, 
            ns = None, 
            k = 3,  
            methods = None,
            upperThres = -np.inf,
            lambdaRatio = 0.01,
            useTransformation = False,
            SAVE = True):
    
    if methods is None:
        methods = ["knnMI_AND","knnMI_OR","fisherZ_AND","fisherZ_OR","mb_RIC","glasso_RIC","mb_STARS","glasso_STARS","mb_auto"]

    if ns is None:
        ns = [125,250,500,1000,2000]
    
    
    # different random number generators for data generation and for the knnMI method (permutation tests) 
    rrData = np.random.RandomState(seed)
    rr1 = np.random.RandomState(seed + 1) 
    
    # global rng is also used..
    np.random.seed(seed)
 
    
    cores = mp.cpu_count()
    knnEst1 = knnEstimator(k = k,rng = rr1, parallel= cores, UPPERthres= upperThres)
    
    fEst = fisherEstimator()
       
    # initilize dictionaries for results
    res = {"HD" : [], "UG" : [], "sparsity" : []} # measured quantities are keys
    Nres = {n : copy.deepcopy(res) for n in ns}
    allRes = {method : copy.deepcopy(Nres) for method in methods}
              
    # used parameters          
    parameters = {"seed": seed, 
                  "ntests": 0,
                  "ns": ns,
                  "testName" : testName,
                  "methods" : methods,
                  "k" : k,
                  "lambdaRatio" : lambdaRatio,
                  "upperThres" : upperThres,
                  "trueUGs" : []}    
                
    # create folder where to save the resuls              
    if folderName is None:
        directory = "tests"
    else:
        directory = "tests/" + folderName
    
    if not os.path.exists(directory):
        os.makedirs(directory)            
        
    if type(testName) == tuple: # convert test name to string
 
        if(testName[0] == True):
            testNameString = "Linear"
        else:
            testNameString = "Nonlinear"

        filee = testNameString + testName[1]    
    else:
        filee = testName

    filename = directory + "/" + filee + ".p"
    
     
    # create object for generating data and run the tests
    dd = dataGenerator(testName,rng = rrData)
    
    nonPara = True # use non-paranormal transformation for glasso and mb
    
    #DEBUG
    errorCount = 0     
    
    for tt in range(0,ntests):
        
        print("test ",tt + 1,"/", ntests, sep="")
        Xall,G = dd.createData(np.max(ns))
        
        for n in ns:          
            
            X = Xall[:n,:]
            X = scale(X) # zero mean, sd one for all the features
            
            if useTransformation:
                X = transform(X) # non-paranormal transformation for every method
                print("Transformation used.")
                nonPara = False # no need to perform the transformation twice when glasso/mb is called

            print("............sample size: ",n)
            
            
            # find Markov blankets for knnMI method
            if "knnMI_AND" in methods or "knnMI_OR" in methods:
  
                knnSl = structLearn(X, ci_estimator= knnEst1)    
                knnSl.findMoralGraph()
                
            # same for fisherZ based method    
            if "fisherZ_AND" in methods or "fisherZ" in methods:
                fishSl = structLearn(X, ci_estimator= fEst)
                fishSl.findMoralGraph()
    
            for method in methods:
                
                sp = np.nan # record sparsities of estimated graphs for glasso/mb, for other methods use just nan (graphs are saved so sparsity is easy to compute)
                
                # DEBUG
                seeeds = np.random.RandomState
                
                if method == "knnMI_AND":
                    estUG = knnSl.getMoralGraph("AND")
                elif method == "knnMI_OR":
                    estUG = knnSl.getMoralGraph("OR")                
                elif method == "fisherZ_AND":
                    estUG = fishSl.getMoralGraph("AND")
                elif method == "fisherZ_OR":
                    estUG = fishSl.getMoralGraph("OR")                    
                elif method == "glasso_RIC":
                    estUG, sp = hugeLearnGraph(X,method = "glasso", modelSelectCrit= "ric", nonPara=nonPara, lambdaRatio= lambdaRatio) 
                elif method == "glasso_BIC":
                    estUG, sp = hugeLearnGraph(X,method = "glasso", modelSelectCrit= "ebic", nonPara=nonPara, ebicTuning= 0.0,lambdaRatio= lambdaRatio)
                elif method == "glasso_EBIC":
                    estUG, sp = hugeLearnGraph(X,method = "glasso", modelSelectCrit= "ebic", nonPara=nonPara, ebicTuning= 0.5,lambdaRatio= lambdaRatio)
                elif method == "mb_RIC":
                    estUG,sp = hugeLearnGraph(X,method = "mb", modelSelectCrit= "ric", nonPara=nonPara,lambdaRatio= lambdaRatio) 
                elif method == "mb_auto":
                    estUG,sp = hugeLearnGraph(X,method = "mb", modelSelectCrit= "mbDefault", nonPara=nonPara)     
                elif method == "mb_STARS":
                    estUG,sp = hugeLearnGraph(X,method = "mb", modelSelectCrit= "stars", nonPara=nonPara,lambdaRatio= lambdaRatio)    
                elif method == "glasso_STARS":
                    estUG,sp = hugeLearnGraph(X,method = "glasso", modelSelectCrit= "stars", nonPara=nonPara,lambdaRatio= lambdaRatio)
                else:
                    print("unspecified method!!")
                    hd = np.nan
              
                # DEBUG    
                if (estUG == estUG.T).all() == False:
                    errors = {"testName" : testName, "data": X, "method" : method, "currentSeed" : seeeds, "estUG": estUG, "trueUG": G, "testNumber" : tt +1 }
                    errorCount += 1
                    
                    if type(testName) == str:
                        testNameString = testName
                    else:
                        testNameString = filee
                        
                    path = directory + "/errors" + testNameString + "_" + str(errorCount) + ".p"
                    saveResults(errors,path)
      
                    ## force symmetry on UG
                    estUG = 1*(estUG + estUG.T == 2)
                                        
                # compute Hamming distance
                hd = HD(G,estUG)    
                    
                # save stuff
                print(method,hd)
                allRes[method][n]["HD"].append(hd)
                allRes[method][n]["UG"].append(estUG)
                allRes[method][n]["sparsity"].append(sp)
                
        # save the true UG (this is differs between the tests only in random graph cases)
        parameters["trueUGs"].append(G)
               
        # save results after every 5 tests    
        if (tt + 1) % 5 == 0 and SAVE:
            parameters["ntests"] = tt + 1
            res = (allRes,parameters) 
            saveResults(res,filename)
    
    # final results
    parameters["ntests"] = tt + 1
    res = (allRes,parameters) 
    if SAVE:
        saveResults(res,filename)
    
    return(res)
    
def saveResults(res,pathToSave):       
        pickle.dump(res,open(pathToSave,"wb")) 
        
    


	       
    
    