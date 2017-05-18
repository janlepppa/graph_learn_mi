#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:35:54 2017

@author: janleppa
"""
from slTests import doTests
import argparse
import numpy as np

def runSmallNetWorkTests(ntests,ns,k,folder,upperThres,useTrans):
    doTests((True,"gaussian"), ntests = ntests, ns = ns,k = k, folderName = folder, upperThres = upperThres, useTransformation=useTrans)
    doTests((True,"t"), ntests = ntests, ns = ns,k = k ,folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    doTests((True,"uniform"), ntests = ntests, ns = ns,k = k,folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    doTests((False,"gaussian"), ntests = ntests, ns = ns,k = k, folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    doTests((False,"t"), ntests = ntests, ns = ns,k = k ,folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    doTests((False,"uniform"), ntests = ntests, ns = ns,k = k,folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    
def runLargeNetworkTest(ntests,ns,k,folder,upperThres,useTrans):
    doTests("largeUG", ntests = ntests, ns = ns,k = k,folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    
def runLargeNetworkTest_Gaussian(ntests,ns,k,folder,upperThres,useTrans):
    doTests("largeUG_g", ntests = ntests, ns = ns,k = k,folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    
def runLargeNetworkTest_uniform(ntests,ns,k,folder,upperThres,useTrans):
    doTests("largeUG_u", ntests = ntests, ns = ns,k = k,folderName = folder,upperThres = upperThres,useTransformation=useTrans)
    
def randomUG(ntests,ns,k,folder,upperThres,useTrans):
    doTests("randomUG", ntests = ntests, ns = ns,k = k, folderName = folder, upperThres = upperThres,useTransformation=useTrans)
    doTests("randomUGLarge", ntests = ntests, ns = ns,k = k, folderName = folder, upperThres = upperThres,useTransformation=useTrans)
    
def randomUGnonPara(ntests,ns,k,folder,upperThres,useTrans):
    doTests("randomUGnonPara", ntests = ntests, ns = ns,k = k, folderName = folder, upperThres = upperThres,useTransformation=useTrans)
    doTests("randomUGnonParaLarge", ntests = ntests, ns = ns,k = k, folderName = folder, upperThres = upperThres,useTransformation=useTrans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testname",help = "name of the test")
    parser.add_argument("usetrans",help = "use the paranormal transformation True/False")
    parser.add_argument("pcskip",help = "skip tests if partial correlation detected True/False")
    args = parser.parse_args()
    
    testName = args.testname
    
    if args.usetrans == "True":
        useTrans = True
    elif args.usetrans == "False":
        useTrans = False
    else:
        print("Using transformation")
        useTrans = True
    
    if args.pcskip == "True":
        upperThres = -np.inf
    elif args.pcskip == "False":
        upperThres = np.inf
    else:
        print("Permutation tests are skipped if non-zero partial correlation is detected")
        upperThres = -np.inf
        
            
    ns = [125,250,500,1000,2000] #sample size    
    ntests = 25 #number of tests
    k = 3 # k for the knnEstimator
    
    # pick save folder name according to the parameters
    if useTrans == True and upperThres == -np.inf:
        folder = "k3noUPtrans"
    elif useTrans == True and upperThres == np.inf:
        folder = "k3UPinftrans"
    elif useTrans == False and upperThres == -np.inf:
        folder = "k3noUP"
    elif useTrans == False and upperThres == np.inf:
        folder = "k3UPinf"
    else:
        folder = "shouldNotHappen"
        
        
    if testName == "smallNetwork":
        runSmallNetWorkTests(ntests,ns,k,folder,upperThres,useTrans)
    elif testName == "largeNetwork":
        runLargeNetworkTest(ntests,ns,k,folder,upperThres,useTrans)
    elif testName == "largeNetwork_g":
        runLargeNetworkTest_Gaussian(ntests,ns,k,folder,upperThres,useTrans)
    elif testName == "largeNetwork_u":
        runLargeNetworkTest_uniform(ntests,ns,k,folder,upperThres,useTrans)
    elif testName == "randomUG":
        randomUG(ntests,ns,k,folder,upperThres,useTrans)
    elif testName == "randomUGNonPara":
        randomUGnonPara(ntests,ns,k,folder,upperThres,useTrans)
    else:
        print("That test does not exist..")
        
    
   
        
        
    
        

    
     
