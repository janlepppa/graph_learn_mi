#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:57:17 2017

@author: janleppa
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product

def loadRes(location):
    return(pickle.load(open(location,"rb")))
    
# check that tuning parameter ranges are sensible in all tests (smallest one yields a graph that is denser than the generating model and
# the largest is known to yield an empty graph)    
def checkAllSparsities(folderName, testNames = None, methods = None):
    if testNames is None:
        
        linear = [True,False]
        noises = ["gaussian", "t", "uniform"]
        testNames = list(product(linear, noises)) + ["randomUG", "randomUGnonPara","randomUGLarge", "randomUGnonParaLarge","largeUG","largeUG_u","largeUG_g"] 
        
        
    for test in testNames:
        checkSparsities(test,folderName=folderName,methods=methods)
        
def plotAll(folderName,leg = False):
    plotLargeNetworks(folder= folderName, leg=leg) 
    plotSmallNetworks(folder = folderName,leg=leg)
        

# check that all the tuning parameter values were chosen in a sensible manner (one test)    
def checkSparsities(testName,folderName, methods = None):
    if type(testName) == tuple: # convert test name to string
     
            if(testName[0] == True):
                testNameString = "Linear"
            else:
                testNameString = "Nonlinear"
    
            testName = testNameString + testName[1]


    filee = "tests/" + folderName + "/" +testName + ".p"
        
    res,parameters = loadRes(filee)
    
    ns = parameters["ns"]
    ntests = parameters["ntests"]
    
    if methods is None:
        methods = ["mb_STARS","glasso_STARS","mb_RIC","glasso_RIC"]

    success = True

    for method in methods:
        
        for tt in range(0,ntests):
            trueGraph = parameters["trueUGs"][tt]
            d = trueGraph.shape[0]
            trueSparsity = (np.sum(trueGraph)/2)/(d*(d-1)/2)
            
            for n in ns:
                maxSparsity =  np.max(res[method][n]["sparsity"][tt]) # actually the density of the graph....
                if trueSparsity > maxSparsity: # true network is denser than the densest on the lasso path
                    print("FAIL: ",testName, method, trueSparsity,maxSparsity)
                    success = False
    if success:
        print(testName, ": tuning parameter range is sensible in all",ntests ,"tests")
    
def plotLargeRandomNetworks(font = 15, folder = None, methods = None, leg = False):
    
    if methods is None:
       methods = ["knnMI_AND","fisherZ_AND","mb_STARS","glasso_STARS","mb_auto"]
    
    tests = ["randomGM2", "randomGM3"]
             
    titles = ["Random GM, infinite", "Random GM, finite"]

    #allTestNames = list(zip(tests,titles))
    #nTests = len(allTestNames)
    #ii = 1
    
    for (test, title) in zip(tests,titles):
        #if ii == nTests:
        #    leg = True
        
        plotRes(testName = test, folderName= folder, methods= methods, font= font, title = title, leg = leg)
        #ii += 1
    
    
def plotLargeNetworks(font = 15, folder = None, methods = None, leg = False):
    
    if methods is None:
       methods = ["knnMI_AND","fisherZ_AND","mb_STARS","glasso_STARS","mb_auto"]
    
    tests = ["largeUG","largeUG_u","largeUG_g","randomUG", "randomUGnonPara","randomUGLarge", "randomUGnonParaLarge"]
             
    titles = ["Large network + t","Large network + uniform","Large network + Gaussian","Random Gaussian", "Random Non-paranormal","Random Gaussian Large", "Random Non-paranormal Large"]

    #allTestNames = list(zip(tests,titles))
    #nTests = len(allTestNames)
    #ii = 1
    
    for (test, title) in zip(tests,titles):
        #if ii == nTests:
        #    leg = True
        
        plotRes(testName = test, folderName= folder, methods= methods, font= font, title = title, leg = leg)
        #ii += 1
        
def plotSmallNetworks(font = 15, folder = None, methods = None, leg = False):
    
    if methods is None:
       methods = ["knnMI_AND","fisherZ_AND","mb_STARS","glasso_STARS","mb_auto"]
    
    linear = [True,False]
    noises = ["Gaussian", "t", "Uniform"]

    tests = list(product(linear, noises))
    #nTests = len(tests)
    #ii = 1
    
    
    for test in tests:
        
        #if ii == nTests:
        #    leg = True
        
        
        if test[0] == True:
            title = "Linear + "
        else:
            title = "Non-linear + "
        
        title = title + test[1]
        
        plotRes(testName = (test[0],test[1].lower()), folderName= folder, methods= methods, font= font, title = title, leg = leg)

        #ii += 1
        
    
def plotRes(testName, folderName = None, font = 12, methods = None, leg= False, title = None):
    # load the right file
    if type(testName) == tuple: # convert test name to string
     
            if(testName[0] == True):
                testNameString = "Linear"
            else:
                testNameString = "Nonlinear"
    
            testName = testNameString + testName[1]

    if folderName is None:
        filee = "tests/" + testName + ".p"
    else:
        filee = "tests/" + folderName + "/" +testName + ".p"
        
    res,parameters = loadRes(filee)
    
    ns = parameters["ns"]

    if methods is None:
       methods = ["knnMI_AND","fisherZ_AND","mb_STARS","glasso_STARS","mb_auto"]

    HDs = np.zeros((len(methods),len(ns)))
    SDs = np.zeros((len(methods),len(ns)))

    ii = 0
    for method in methods:
        
        jj = 0
        
        for n in ns:
            hds = res[method][n]["HD"]
            
            HDs[ii,jj] = np.mean(hds)
            SDs[ii,jj] = np.std(hds)/np.sqrt(len(hds)) # standard error of the mean
           

            jj += 1
            
        ii += 1
    
    print("ntests = " ,len(hds))    
    ############## plot    
    x = range(0,len(ns)) 
    
    plt.rcParams.update({'font.size': font})
        
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_ylim(-0.1, np.ceil(np.amax(HDs) + np.mean(SDs.flatten())))
    epsss = 0.05
    ax.set_xlim(x[0]-epsss, x[len(x)-1]+ epsss)
    for i in range(0,len(methods)):
        lab = methods[i]
        ax.errorbar(x,HDs[i,:],SDs[i,:], label = lab, marker = 'o', linewidth = 2)
   
    ax.set_xticks(range(0,len(ns)))    
    ax.set_xticklabels(ns)    
    ax.set_ylabel("Hamming distance")
    ax.set_xlabel("Sample size")    
    ax.grid()      

    if title is not None:      
        ax.set_title(title)
               
    if leg is True:    
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':12})
        

    folder = "tests/" + folderName + "/figures"
    
    if not os.path.exists(folder):
            os.makedirs(folder)
          
    filename = folder + "/" + testName + ".pdf"
    
    fig.savefig(filename)
    plt.show()    
    
def plotOnlyLegend(folderName = "k3UPinf",testName ="largeUG", font = 15, methods = None):
    # load the right file
    if type(testName) == tuple: # convert test name to string
     
            if(testName[0] == True):
                testNameString = "Linear"
            else:
                testNameString = "Nonlinear"
    
            testName = testNameString + testName[1]

    if folderName is None:
        filee = "tests/" + testName + ".p"
    else:
        filee = "tests/" + folderName + "/" +testName + ".p"
        
    res,parameters = loadRes(filee)
    
    #ns = parameters["ns"]

    if methods is None:
       methods = ["knnMI_AND","fisherZ_AND","mb_STARS","glasso_STARS","mb_auto"]

    plt.rcParams.update({'font.size': font})
        
    fig = plt.figure(figsize=(5,0.02))
    ax = plt.subplot(111)
    for i in range(0,len(methods)):
        lab = methods[i]
        ax.plot([],label = lab,linewidth = 3)
        
        
    #ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=len(methods), mode="expand", borderaxespad=0.)
    
    ax.legend(bbox_to_anchor=(0., 0, 0, 0), loc=3,
          ncol=len(methods))
    
    ax.set_axis_off()    
        
    folder = "tests/" + folderName + "/figures"
    
    if not os.path.exists(folder):
            os.makedirs(folder)
          
    filename = folder + "/legend.png"
    
    fig.savefig(filename,bbox_inches='tight',pad_inches = 0)
    plt.show()
    


    
    