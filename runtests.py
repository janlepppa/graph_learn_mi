from sltests import doTests
import argparse
from itertools import product

def runSmallNetWorkTests(ntests,ns,k,folder,useTrans):
    linear = [True,False]
    noisetypes = ["gaussian","t","uniform"]

    testnames = product(linear,noisetypes)
    
    for testname in testnames:
        doTests(testname, ntests = ntests, ns = ns,k = k, folderName = folder, useTransformation=useTrans)

def runLargeNetworkTest(ntests,ns,k,folder,useTrans):
    doTests("largeUG", ntests = ntests, ns = ns,k = k,folderName = folder,useTransformation=useTrans)
    
def runLargeNetworkTest_Gaussian(ntests,ns,k,folder,useTrans):
    doTests("largeUG_g", ntests = ntests, ns = ns,k = k,folderName = folder,useTransformation=useTrans)
    
def runLargeNetworkTest_uniform(ntests,ns,k,folder,useTrans):
    doTests("largeUG_u", ntests = ntests, ns = ns,k = k,folderName = folder,useTransformation=useTrans)
    
def randomUG(ntests,ns,k,folder,useTrans):
    doTests("randomUG", ntests = ntests, ns = ns,k = k, folderName = folder,useTransformation=useTrans)
    doTests("randomUGLarge", ntests = ntests, ns = ns,k = k, folderName = folder,useTransformation=useTrans)
    
def randomUGnonPara(ntests,ns,k,folder,useTrans):
    doTests("randomUGnonPara", ntests = ntests, ns = ns,k = k, folderName = folder,useTransformation=useTrans)
    doTests("randomUGnonParaLarge", ntests = ntests, ns = ns,k = k, folderName = folder,useTransformation=useTrans)

def randomGM(ntests,ns,k,folder,useTrans):
    doTests("randomGMSmall15", ntests = ntests, ns = ns,k = k ,folderName = folder,useTransformation=useTrans)

def randomGMnew(ntests,ns,k,folder,useTrans):      
    doTests("randomGMLarge15", ntests = ntests, ns = ns,k = k ,folderName = folder,useTransformation=useTrans)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("testname",help = "name of the test")
    parser.add_argument("usetrans",help = "use the paranormal transformation True/False")
    parser.add_argument("folder",help = "folder to save the results to")
    
    args = parser.parse_args()
    
    testName = args.testname
    folder = args.folder
    
    if args.usetrans == "True":
        useTrans = True
    elif args.usetrans == "False":
        useTrans = False
    else:
        print("Using non-paranormal transformation for all methods")
        useTrans = True
       

    ns = [125,250,500,1000,2000] #sample size 
    ntests = 25 # number of tests    
    k = 5 # k for the knnEstimator
    
    if testName == "smallNetwork":
        runSmallNetWorkTests(ntests,ns,k,folder,useTrans)
    elif testName == "largeNetwork":
        runLargeNetworkTest(ntests,ns,k,folder,useTrans)
    elif testName == "largeNetwork_g":
        runLargeNetworkTest_Gaussian(ntests,ns,k,folder,useTrans)
    elif testName == "largeNetwork_u":
        runLargeNetworkTest_uniform(ntests,ns,k,folder,useTrans)
    elif testName == "randomUG":
        randomUG(ntests,ns,k,folder,useTrans)
    elif testName == "randomUGNonPara":
        randomUGnonPara(ntests,ns,k,folder,useTrans)
    elif testName == "randomGMsmall":
        randomGM(ntests,ns,k,folder,useTrans)
    elif testName == "randomGMnlarge":
        randomGMnew(ntests,ns,k,folder,useTrans)
    else:
        print("That test does not exist..")
        
    
   
        
        
    
        

    
     
