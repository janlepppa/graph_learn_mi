import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
rcitR = importr('RCIT') 

class RCIT:
    '''
    Wraps RCIT and RCot independence tests found at https://github.com/ericstrobl/RCIT
    '''
    def __init__(self, 
                 sig = 0.05,
                 approx = "lpd4",
                 corr = False,
                 num_f = 25,
                 seed = None):
        
        self.sig = sig
        self.approx = approx
        self.num_f = num_f
        self.rng = None
        
        if corr:
            self.ci_test =  rcitR.RCoT
        else:
            self.ci_test = rcitR.RCIT
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
   
        
    def independent(self,x,y,z = None):
        indep,p = self._rcittest(x,y,z)
        return(indep,-p)
        
    def dependence(self,x,y,z = None):
        _,p = self._rcittest(x,y,z)
        return(-p)
        
    def _rcittest(self,x,y,z = None):
        
        if self.rng is not None:
            r_seed = self.rng.randint(0,2**31)
        else:
            r_seed = robjects.reval('NULL')
         
        if z is None:
            r_z = robjects.reval('NULL')
        else:
            r_z = z
     
            
        p_val = self.ci_test(x,y, 
                             z = r_z, 
                             approx = self.approx, 
                             num_f = self.num_f,
                             seed = r_seed)[0][0]
                           
        indep = True

        if p_val < self.sig:
            indep = False
            
        return(indep,p_val)
        
class KCIT:
    '''
    Wraps KCIT independence test found at https://github.com/ericstrobl/RCIT
    '''
    def __init__(self, 
                 sig = 0.05,
                 bootstrap = True,
                 seed = None):
        
        self.sig = sig
        self.bootstrap = bootstrap
   
        if seed is not None:
            #self.rng = np.random.RandomState(seed)
            robjects.r['set.seed'](seed)

    def independent(self,x,y,z = None):
        indep,p = self._kcittest(x,y,z)
        return(indep,-p)
        
    def dependence(self,x,y,z = None):
        _,p = self._kcittest(x,y,z)
        return(-p)
        
    def _kcittest(self,x,y,z = None):
        
        # x and y are assumed to be column vectors
        if x.ndim == 1:
            x = np.reshape(x,(len(x),1))
        if y.ndim == 1:
            y = np.reshape(y,(len(y),1))
        
            
        #if self.rng is not None:
        #    r_seed = self.rng.randint(0,2**31)
        #    robjects.r['set.seed'](r_seed)
         
        if z is None:
            r_z = robjects.reval('NULL')
        else:
            r_z = z
            
            if r_z.ndim == 1:
                r_z = np.reshape(z,(len(z),1))
                
     
        p_val = rcitR.KCIT(x,y, 
                             z = r_z, 
                             Bootstrap = self.bootstrap)[0]
                           
        indep = True

        if p_val < self.sig:
            indep = False
            
        return(indep,p_val)
        
        
