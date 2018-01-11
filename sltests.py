import numpy as np
from knnestimator import KnnEstimator
from structlearn import StructLearn
from fishertest import FisherCI
from rcit import KCIT, RCIT
from graphutils import HD
from datagenerator import DataGenerator
from huge import hugeLearnGraph,transform
import multiprocessing as mp
import copy
import os
import pickle
from sklearn.preprocessing import scale
from itertools import product

def doTests(testName, 
            folderName = None, 
            seed = 123456, 
            ntests = 25, 
            ns = None, 
            k = 3,  
            k_perm = None,
            methods = None,
            lambdaRatio = 0.01,
            useTransformation = False,
            SAVE = True):
    
    if methods is None:
        
        methods = ["knnMI_AND",
                   "knnMI_OR",
                   "fisherZ_AND",
                   "fisherZ_OR",
                   "mb_RIC",
                   "glasso_RIC",
                   "mb_STARS",
                   "glasso_STARS",
                   "mb_auto"]

    if ns is None:
        ns = [125,250,500,1000,2000]
    
    
    # different random number generators for data generation and for the knnMI method (permutation tests) 
    rrData = np.random.RandomState(seed)
    rr1 = np.random.RandomState(seed + 1) 
    
    # global rng is also used..
    np.random.seed(seed)
     
    cores = mp.cpu_count()
    
    # conditional independence tests
    knnEst1 = KnnEstimator(k = k,rng = rr1, parallel= cores,k_perm = k_perm)
    fEst = FisherCI()
    
    if "KCIT_OR" in methods or "KCIT_AND" in methods: 
        k_cit = KCIT(seed = seed + 2)
    if "RCIT_OR" in methods or "RCIT_AND" in methods:
        r_cit = RCIT(seed = seed + 3)
        
       
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
                  "trueUGs" : []}    
                
    # create folder where to save the resuls              
    if folderName is None:
        directory = "tests"
    else:
        directory = "tests/" + folderName
    
    if not os.path.exists(directory):
        os.makedirs(directory)            
        
    test_str = __test2str(testName)
    filename = directory + "/" + test_str + ".p"
     
    # create object for generating data and run the tests
    dd = DataGenerator(testName,rng = rrData)
    
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
                  
            # kernel methods
            if "KCIT_OR" in methods or "KCIT_AND" in methods:
                kcitSl = StructLearn(X,ci_estimator = k_cit)
                kcitSl.findMoralGraph()
                
            if "RCIT_OR" in methods or "RCIT_AND" in methods: 
                rcitSl = StructLearn(X,ci_estimator = r_cit)
                rcitSl.findMoralGraph()
                
            # find Markov blankets for knnMI method
            if "knnMI_AND" in methods or "knnMI_OR" in methods:
                if k < 1:
                    knnEst1.k = max(3,int(np.ceil(k*n)))
                                        
                knnSl = StructLearn(X, ci_estimator= knnEst1)    
                knnSl.findMoralGraph()
                
            # same for fisherZ based method    
            if "fisherZ_AND" in methods or "fisherZ" in methods:
                fishSl = StructLearn(X, ci_estimator= fEst)
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
                elif method == "KCIT_AND":
                    estUG = kcitSl.getMoralGraph("AND")
                elif method == "KCIT_OR":
                    estUG = kcitSl.getMoralGraph("OR")
                elif method == "RCIT_AND":
                    estUG = rcitSl.getMoralGraph("AND")
                elif method == "RCIT_OR":
                    estUG = rcitSl.getMoralGraph("OR")
                else:
                    print("unspecified method!!")
                    hd = np.nan
              
                # DEBUG    
                if (estUG == estUG.T).all() == False:
                    errors = {"testName" : testName, "data": X, "method" : method, "currentSeed" : seeeds, "estUG": estUG, "trueUG": G, "testNumber" : tt +1 }
                    errorCount += 1
                
                    path = directory + "/errors_" + test_str + "_" + str(errorCount) + ".p"
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
    
    
def compareKnns(testName,
                  seed = 123432,
                  folderName = "knn_est_test",
                  ntests = 25,
                  ns = None,
                  SAVE = True):
    
    if ns is None:
        ns = [125,250,500,1000, 2000]
    
    # different random number generators for data generation and for the knnMI method (permutation tests) 
    rrData = np.random.RandomState(seed)
    rr1 = np.random.RandomState(seed + 1) 
    
    # global rng is also used..
    np.random.seed(seed)
 
    cores = mp.cpu_count()
    
    k_values = [0.01,0.1,0.2,3,5]
    local_perm = [True,False]
    graph_rules = ["AND","OR"]

    methods = list(product(k_values,local_perm))
    method_names = [__method2str(method) for method in product(k_values,local_perm,graph_rules)]
     
    # initilize dictionaries for results
    res = {"HD" : [], "UG" : []} # measured quantities are keys
    Nres = {n : copy.deepcopy(res) for n in ns}
    allRes = {method : copy.deepcopy(Nres) for method in method_names}
              
    # used parameters          
    parameters = {"seed": seed, 
                  "ntests": 0,
                  "ns": ns,
                  "testName" : testName,
                  "methods" : method_names,
                  "trueUGs" : []}    
                
    # create folder where to save the resuls              
    if folderName is None:
        directory = "tests"
    else:
        directory = "tests/" + folderName
    
    if not os.path.exists(directory):
        os.makedirs(directory)            
        
    test_str = __test2str(testName)
    filename = directory + "/" + test_str + ".p"
    
    # create object for generating data and run the tests
    dd = DataGenerator(testName,rng = rrData)
    
    for tt in range(0,ntests):
        
        print("test ",tt + 1,"/", ntests, sep="")
        Xall,G = dd.createData(np.max(ns))
        
        for n in ns:          
            
            X = Xall[:n,:]
            X = scale(X) # zero mean, sd one for all the features
            
            print("............sample size: ",n)
            
            for method in methods:
                
                kk,local = method
                
                if kk < 1:
                    k = max(3,int(np.ceil(kk*n)))
                else:
                    k = kk
                
                if local:
                    k_perm = 5
                else:
                    k_perm = None
                    
                knnest = KnnEstimator(k = k,k_perm = k_perm, rng = rr1, parallel = cores)
                    
                knn_sl = StructLearn(X, ci_estimator= knnest)    
                knn_sl.findMoralGraph()
                          
                for graph_rule in graph_rules:
                    
                    est_ug = knn_sl.getMoralGraph(graph_rule)
                    method_name = __method2str( (method[0],method[1],graph_rule) )
                
                    # compute Hamming distance                
                    hd = HD(G,est_ug)    
                        
                    # save stuff
                    print(method_name,hd)
                    allRes[method_name][n]["HD"].append(hd)
                    allRes[method_name][n]["UG"].append(est_ug)
               
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
     
def __method2str(comb):
    k,local,graph_rule = comb
      
    s = "k_" + str(k)
    
    if k < 1:
        s = s + "n"
    
    if local:
        s = s + "_local"
        
    s = s + "_" + graph_rule
        
    return(s)
    
def __test2str(testname):
    if type(testname) == tuple: # convert test name to string
 
        if(testname[0] == True):
            linearity = "Linear"
        else:
            linearity = "Nonlinear"

        test_str = linearity + testname[1]    
    elif type(testname) == str:
        test_str = testname
    else:
        raise TypeError("unkown test")
        
    return(test_str)
        
def saveResults(res,pathToSave):       
        pickle.dump(res,open(pathToSave,"wb")) 
        
    


	       
    
    
