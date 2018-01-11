from sltests import doTests
from itertools import product

args = {"folderName" : "kernel_tests",
            "ntests" : 25,
            "ns" : [125,250,500,1000,2000],
            "SAVE" : True,
            "methods" : ["RCIT_OR","RCIT_AND","KCIT_OR","KCIT_AND"]
            } 

def small_networks():
    linear = [True,False]
    noisetypes = ["gaussian","t","uniform"]
    testnames = product(linear,noisetypes)

    for test_name in testnames:
        doTests(test_name,**args)      
            
def small_random_networks():    
    doTests("randomUGnonPara", **args)
    
def large_network():
    doTests("largeUG",**args)
    

def large_random_networks_2():
    doTests("randomUGnonParaLarge",**args)


#if __name__ == "__main__":
#       
#    # small networks
#    linear = [True,False]
#    noisetypes = ["gaussian","t","uniform"]
#    testnames = product(linear,noisetypes)
#
#    for test_name in testnames:
#        doTests(test_name,**args)
#        
#    
#    # large network with t-noise
#    doTests("largeUG",**args)
    
    