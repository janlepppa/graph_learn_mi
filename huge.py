import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri # passing numpy objects to R functions
#from sklearn.preprocessing import scale
rpy2.robjects.numpy2ri.activate()
hugeR = importr('huge') 
from scipy.stats import norm

def hugeLearnGraph(X, nonPara = True, method = "mb", nTunings = 20, modelSelectCrit = "ric", ebicTuning = 0.5, lambdaRatio = None, verbose = False):
    
    n,d = X.shape
       
    if nonPara:
        #X = hugeR.huge_npn(X, verbose = verbose) # transform the data
        X = transform(X, returnNumpyArray= False, verbose = verbose)
  
    asR = robjects.r['as']
    
    if method == "mb" and modelSelectCrit == "mbDefault":
        
        # single value for the regularization parameter
        alpha = 0.05
        lambbda = 2/np.sqrt(n)*norm.ppf(1 - alpha/(2*d**2))
         
        # cannot pass "lambda = x" argument to hugeR.huge function directly because "lambda" is illegal variable name..
        lambda_r = robjects.FloatVector([lambbda])
        robjects.rinterface.globalenv['lambda_r'] = lambda_r # define variable in R environment
        lambdaArg = robjects.reval('lambda = lambda_r') # this can be passed 

        est= hugeR.huge(X, lambdaArg, method = method, verbose = verbose, sym = "and")
        
        #print(robjects.r['get']('method',est))
        #print(robjects.r['get']('sym',est))
        path = robjects.r['get']('path',est)[0]
        estG = np.array(asR(path,"matrix"),dtype = np.int)
             
    else:
        if lambdaRatio is not None:            
            est= hugeR.huge(X,lambda_min_ratio = lambdaRatio, method = method, nlambda = nTunings, verbose = verbose)
        else:
        # estimate graphs for a range of hyperparameters
            est= hugeR.huge(X, method = method, nlambda = nTunings, verbose = verbose)
    
        # set seed to obtain reproducable results (depends on the numpy global seed) this affects only if criterion is "ric" or "stars"
        seed = np.random.randint(1,1e9)
        robjects.r['set.seed'](seed)
        
        #ebic_r = robjects.FloatVector([ebicTuning])
        #robjects.rinterface.globalenv['ebicGamma'] = ebic_r # define variable in R environment
        #ebicArg = robjects.reval('ebic.gamma = ebicGamma') # this can be passed 
    
        # do model selection
        methodRes = hugeR.huge_select(est,ebic_gamma = ebicTuning, criterion = modelSelectCrit, verbose = verbose)
        
        G_R = robjects.r['get']('refit',methodRes)
        asR = robjects.r['as']
        estG = np.array(asR(G_R,"matrix"),dtype = np.int)
    
    # sparsity/ies for the estimated graph(s)
    sparsity = np.array(robjects.r['get']('sparsity',est))
    
    return (estG,sparsity)
    
def transform(X, returnNumpyArray = True, verbose = False):
    if returnNumpyArray:
        return np.array(hugeR.huge_npn(X, verbose = verbose))
    else:
        return hugeR.huge_npn(X, verbose = verbose)
    
    
def hugeGenerateData(n,d, graph = "random", prob = None,seed = None):
    
    if seed is not None:
        robjects.r['set.seed'](seed) # the data generating DAG is same for each sample size
        
    if prob is None:
        X_R = hugeR.huge_generator(n,d, graph = graph, verbose = False)
    else:
        X_R = hugeR.huge_generator(n,d, graph = graph, prob = prob, verbose = False)
        
    # get the data matrix    
    X = np.array(robjects.r['get']('data',X_R))
    
    # and the adjacency matrix
    G_R = robjects.r['get']('theta',X_R)
    
    asR = robjects.r['as']
    G = np.array(asR(G_R,"matrix"),dtype = np.int)
    return X,G


    
        
