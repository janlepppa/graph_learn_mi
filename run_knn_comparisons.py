from sltests import compareKnns
from itertools import product

def small_networks():
    linear = [True,False]
    noisetypes = ["gaussian","t","uniform"]
    
    args = {"folderName" : "knn_est_test",
            "ntests" : 25,
            "ns" : [125,250,500,1000],
            "SAVE" : True,
            "seed" : 6666}    
    
    testnames = product(linear,noisetypes)

    for test_name in testnames:
        compareKnns(test_name,**args)        
    
        
#if __name__ == "__main__":
#    small_networks()
        
    
            

    
    
