import numpy as np
from scipy.stats import norm

# Independence testing using Fisher's z-transform
class FisherCI:
    def __init__(self, sig = 0.05):
        self.sig = sig
          
    def independent(self, x, y, z = None):
        indep, pVal = self._fisherTest(x,y,z)
        
        return (indep, -1.0*pVal) # NEGAtive p-value as a measure of dependence
         
    def dependence(self,x,y,z = None):
        indep, pVal = self._fisherTest(x,y,z)
        return -1.0*pVal
    
    # check using Fisher z-transformation test whether variables are independent, retuns tuple (True/False, pVal)    
    def _fisherTest(self, x ,y ,z = None):
        
        if z is None:
            assert x.shape[0] == y.shape[0]
            dz = 0
        elif len(z) == 0:
            assert x.shape[0] == y.shape[0]
            dz = 0
        else:
            assert x.shape[0] == y.shape[0] and y.shape[0] == z.shape[0]
            dz = z.shape[1]
        
        n = x.shape[0]
        pc = parCorr(x,y,z)
            
        arg = np.sqrt(n - dz - 3)*np.abs(pc)
        pVal = 2*( 1 - norm.cdf(arg)  ) 
        
        indep = True # null-hypothesis, (normal) variables are independent given some others
        if pVal < self.sig:
            indep = False

        return indep, pVal
        
    # fisher's z-transform for (partial) correlation    
    def _fisherZ(self, pc):
        return np.arctanh(pc) 
  
# returns sample partial correlation between variables x,y and z, if z is None, correlation is returned
# x and y should be column vectors and z (sample size times number of conditioning variables)-matrix          
def parCorr(x,y,z = None):
    
    if z is None:
        return np.corrcoef(x.T,y.T)[0,1]
    else:
 
        X = np.hstack((x,y,z))
    
        C  = np.cov(X.T)
        IC = np.linalg.inv(C)
    
        pc = -IC[0,1]/np.sqrt(IC[0,0]*IC[1,1])
    
        return pc
