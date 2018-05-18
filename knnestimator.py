import numpy as np
from scipy.special import psi, gammaln
import scipy.spatial as scsp
import multiprocessing as mp
from fishertest import FisherCI
from sklearn.neighbors import KDTree

class KnnEstimator:
    '''
    k              number of nearest neighbours used
    p              norm used in distance computations, either "float("inf")" (max/sup-norm) or 2 (Euclidean)
                   HOWEVER MI estimates with 2-norm don't seem to converge as they should so this should be used with caution  
    permutations   number of permutations used in mutual information or conditional mi based independence tests
    parallel       integer, compute permutation test in parallel using 'parallel' number of cores
    sig            significance level, used when deciding on indepedence
    rng            instance of numpy.random.RandomState() used in permutation tests 
                   in order to get reproducable results this should be given AND the global numpy.random seed should be set   
                   (this actually now a bit redundant and should be changed...)
    corrCheck      boolean, tells whether permutation test are skipped in certain cituations based on the result of partial correaltion based test 
    LOWERthres     if estimated MI is below this and partial correlation based test accepts independence, permutation test are skipped
    UPPERthres     if estimated MI is above this and partial correlation based test rejects independence, permutation test are skipped.
                   if set to 'np.inf' this has no effect
    k_perm         hyperparameter related no local permutation scheme. 'None' implies simple permutation, values 5-10 are suggested     
                   if local permutation is used
    '''    
    def __init__(self, 
                 k = 3, 
                 p = float("inf"),
                 permutations = 200,
                 parallel = 4,
                 sig = 0.05,
                 rng = None,
                 LOWERthres = 0.001,
                 UPPERthres = np.inf,
                 k_perm = None,
                 corrCheck = True):
    
        self.k = k
        self.p = p             
        self.permutations = permutations
        self.parallel = parallel
        self.sig = sig
        self.LOWERthres = LOWERthres
        self.UPPERthres = UPPERthres
        self.corrCheck = False
        self.k_perm = k_perm
        
        if corrCheck == True:
            self.corrCheck = True
            self.corrTest = FisherCI(sig = sig)
        
        if rng is None:
            self.rng = np.random.RandomState()      # random number generator for permutation tests
        else:
            self.rng = rng
            
    def independent(self,x,y,z = None):
        
        x = self.__vec2array(x)
        y = self.__vec2array(y)
        z = self.__vec2array(z) 
        
        indep, estMI, varMI, prcn, MIs = self._permutationTest(x,y,z)
        
        return indep, estMI
            
    def dependence(self,x,y,z = None):
        
        x = self.__vec2array(x)
        y = self.__vec2array(y)
        z = self.__vec2array(z) 
        
        estMI = self._cmi1(x,y,z)
        
        return estMI
        
    def _cd2(self,d):
        hd = 0.5*d
        return np.exp((hd)*np.log(np.pi) - gammaln(1 + hd))
        
    def _log_cd(self, d):
        hd = 0.5*d
        return (hd)*np.log(np.pi) - gammaln(1 + hd) - d*np.log(2)
        
    # Entropy estimator by Kraskov et al. The inåut data x should be Nxd numpy array (N obs ervations of d-dimensional variable)   
    def _entropy(self,x):
        N,d = x.shape 
        
        if np.isinf(self.p):
            log_c_d = 0
        elif(self.p == 2):
            log_c_d = self._log_cd(d)
        else:
            print("Check that p is either 2 or infinity")
            
        # distance to k:th neighbour averaged over all the points
        #x = x + self.noiseA*self.rng.normal(loc = 0.0, scale = 1.0, size = (N,d)) 
        tree = scsp.cKDTree(x)
        avg_dist = np.mean(np.log(2*tree.query(x,k = self.k + 1, p = self.p)[0][:,self.k]))
            
        # Kraskov Eq. (20)
        ent = psi(N) - psi(self.k) + log_c_d + d*avg_dist
    
        return ent

    # Mutual information estimator by Kraskov et al. The inåut datas x and y should be Nxd1 and Nxd2 numpy arrays. (version 1, Eq (8))
    def _mi1(self,x,y):
        assert x.shape[0] == y.shape[0]
        
        if np.array_equal(x,y):
            return self._entropy(x)
        
        N,d1 = x.shape
        d2 = y.shape[1]

        xy = np.hstack((x,y)) 
        
        treeXY = scsp.cKDTree(xy)
        
        # find the distance to the k:th neighbour in for every point in (x,y) space
        Kdists = treeXY.query(xy,k = self.k + 1, p = float("inf"))[0][:,self.k] # k + 1 since 1st neighbour is always the point itself 
            
        treeX =  KDTree(x,metric = 'chebyshev')    
        treeY =  KDTree(y,metric = 'chebyshev')
        
        
        # look points whose distance to query point is strictly less than the corresponding Kdist                                          
        MULTIP = 1 - 1e-10
                 
        Kdists = MULTIP*Kdists
    
        n_x = treeX.query_radius(x,Kdists,count_only = True)
        n_y = treeY.query_radius(y,Kdists,count_only = True)

        mi = psi(self.k) + psi(N) - (np.mean(psi(n_x)) + np.mean(psi(n_y)))
        
        if self.p == 2:
            norm_term = self._log_cd(d1) - self._log_cd(d2)
            return mi + norm_term
        
        if self.p == float("inf"):
            return mi
     
        return mi
        
    def _mi2(self,x,y):
        assert x.shape[0] == y.shape[0]
        
        if np.array_equal(x,y):
            return self._entropy(x)
        
        N,d1 = x.shape
        d2 = y.shape[1]

        xy = np.hstack((x,y)) 
        
        treeXY = scsp.cKDTree(xy)
        treeX = scsp.cKDTree(x)
        treeY = scsp.cKDTree(y)
        
        # find the distance to the k:th neighbour in for every point in (x,y) space
        Kdists = treeXY.query(xy,k = self.k + 1, p = 2)[0][:,self.k] # k + 1 since 1st neighbour is always the point itself 
                 
        avg_log_n_x = 0
        avg_log_n_y = 0
        
        for ii in range(0,N):     
            avg_log_n_x += np.log(len(treeX.query_ball_point(x[ii,:],Kdists[ii], p = 2 )) -1)/N # no +1 since it is implicitly included (query ball contains always the point itself)
            avg_log_n_y += np.log(len(treeY.query_ball_point(y[ii,:],Kdists[ii], p = 2 )) - 1)/N                                       
        
        # Eq. (8) in Kraskov et al.
        mi = psi(self.k) + psi(N) - (avg_log_n_x + avg_log_n_y) + np.log(self._cd2(d1)*self._cd2(d2) / (self._cd2(d1+d2)))
        
        return mi
        
    
    # coditional mutual information estimator based on Kraskov estimator version 1.    
    def _cmi1(self,x,y,z = None):
        if z is None:
            return self._mi1(x,y)
        
        assert x.shape[0] == y.shape[0] and z.shape[0] == x.shape[0]
        
        N,d1 = x.shape

        xyz = np.hstack((x,y,z)) 
        xz = np.hstack((x,z))
        yz = np.hstack((y,z)) 
        
        treeXYZ = scsp.cKDTree(xyz)
        
        # find the distance to the k:th neighbour in for every point in (x,y) space using max-norm
        Kdists = treeXYZ.query(xyz,k = self.k + 1, p = self.p)[0][:,self.k] # k + 1 since 1st neighbour is always the point itself 
                
        # use KDTrees from sklearn since these support radius searches for multiple datapoints and radiis at the same time
        treeXZ =  KDTree(xz,metric = 'chebyshev')    
        treeYZ =  KDTree(yz,metric = 'chebyshev')
        treeZ =  KDTree(z,metric = 'chebyshev')

        # look points whose distance to query point is strictly less than the corresponding Kdist                                          
        MULTIP = 1 - 1e-10
        
        Kdists = MULTIP*Kdists
    
        n_xz = treeXZ.query_radius(xz,Kdists,count_only = True)
        n_yz = treeYZ.query_radius(yz,Kdists,count_only = True)
        n_z = treeZ.query_radius(z,Kdists,count_only = True)

        # Eq. can be found in Vejmelka and Palus section II.B.1
        cmi = psi(self.k) - (np.mean(psi(n_xz)) + np.mean(psi(n_yz)) - np.mean(psi(n_z)))

        return cmi
        
    def _cmi2(self,x,y,z = None, viaMI = True):
        if z is None:
            return self._mi2(x,y)
            
        if viaMI:
            return self._mi2(x,np.hstack((y,z))) - self._mi2(x,z)
        
        assert x.shape[0] == y.shape[0] and z.shape[0] == x.shape[0]

        N,d1 = x.shape
        d2 = y.shape[1]
        d3 = z.shape[1]
           
        xyz = np.hstack((x,y,z)) 
        xz = np.hstack((x,z))
        yz = np.hstack((y,z)) 
        
        treeXYZ = scsp.cKDTree(xyz)
        treeXZ = scsp.cKDTree(xz)
        treeYZ = scsp.cKDTree(yz)
        treeZ = scsp.cKDTree(z)
        
        # find the distance to the k:th neighbour in for every point in (x,y) space using max-norm
        Kdists = treeXYZ.query(xyz,k = self.k + 1, p = 2)[0][:,self.k] # k + 1 since 1st neighbour is always the point itself 
        
        
        avg_log_n_xz = 0
        avg_log_n_yz = 0
        avg_log_n_z = 0
        
        for ii in range(0,N):
            #print(len(treeX.query_ball_point(x[ii,:],Kdists[ii]-epss,eps = eps, p = p )))

            avg_log_n_xz += np.log(len(treeXZ.query_ball_point(xz[ii,:],Kdists[ii], p = 2 )) - 1 )/N # -1 since query ball contains always the point itself
            avg_log_n_yz += np.log(len(treeYZ.query_ball_point(yz[ii,:],Kdists[ii], p = 2 )) -1  )/N
            avg_log_n_z += np.log(len(treeZ.query_ball_point(z[ii,:],Kdists[ii], p = 2 )) - 1 )/N                                       
        
          
        normTerm = np.log(self._cd2(d1 + d3)*self._cd2(d2 + d3)) - np.log(self._cd2(d1+d2+d3)*self._cd2(d3))                                                 
        
        cmi = psi(self.k) + normTerm - (avg_log_n_xz + avg_log_n_yz - avg_log_n_z)
        return cmi
        
    # pwermutation test for (conditional) independence  
    def _permutationTest(self,x,y,z = None):
        estMI = self._cmi1(x,y,z)
        
        # return just the estimated Mutual information if number of permutations = 1
        if self.permutations == 1:
             
             return (None, estMI, None, None, None)
        
        # skip permutation tests if correlation is detected oOR if MI is low (under specified threshold) and correlation test implies independence   
        if self.corrCheck:
            corr, negPval = self.corrTest.independent(x,y,z) # result of a partial correlation based inpendence test (True = independence)
            
            if z is None: # If there are no conditioning variables and correlation is detected, infer dependence 
                if corr == False:
                    return (False, estMI, None, None, None)     
                
            if corr == False and estMI > self.UPPERthres: # linear dependence detected and estimated MI is over the threshold
                return (False, estMI, None, None, None)              
            
            if estMI < self.LOWERthres and corr == True: # low MI and low partial correlation ---> independence
                return (True, estMI, None, None, None)
               
        # permutation tests
        rr = np.random.randint(0,2**32-2-self.permutations) # THIS relies on global seed
        
        
        if self.k_perm is not None and z is not None: 
            nn_list_z = self._nnlist(z,self.k_perm)            
            args = self._argIterator_local(x,y,z,rr,nn_list_z)
            mi_func = self._permMI_local            
        else:
            args = self._argIterator(x,y,z,rr)
            mi_func = self._permMI
            
             
        # run permuation tests serially or in parallel    
        if self.parallel == 1:            
            MIs = list(map(mi_func,args))                                                                
        else:  
            pool = mp.Pool(self.parallel)  
            MIs = pool.map(mi_func,args)    
            pool.close()
            pool.join()
                
            
        # collect results    
        varMI = np.var(MIs)
        extremes = (np.array(MIs) >= estMI).sum() 
        estPVal = (extremes + 1)/(self.permutations + 1) 
        
        indep = True # null-hypothesis
            
        # if observed p-value is less than significance level conlude dependence
        if estPVal < self.sig:
            indep = False
            
        # debugging    
        if len(np.unique(MIs)) != self.permutations:
            print("some of the permuted MIs are exactly equal!!!")
        
        return (indep, estMI, varMI, estPVal, MIs)
        
    # one argument function to call permutation test
    def _permMI(self,args):
        x,y,z,seed = args
        
        if seed is not None:
            self.rng.seed(seed)
                  
        permuted_y = self.rng.permutation(y)    
  
        return self._cmi1(x,permuted_y, z)
        
            
    def _permMI_local(self,args):
        x,y,z,seed,nn_list = args
        
        if seed is not None:
            self.rng.seed(seed)
                  
        permuted_y = self._localPerm(y,nn_list)
        
        return self._cmi1(x,permuted_y, z)
      
    # creates an iterator over arguments for the permutation tests            
    def _argIterator(self,x,y,z,startSeed):
        for ii in range(0,self.permutations):
            yield (x,y,z,startSeed+ii)
            
    def _argIterator_local(self,x,y,z,startSeed,nn_list):
        for ii in range(0,self.permutations):
            yield (x,y,z,startSeed+ii,nn_list)
            
            
    def _nnlist(self,z,k):      
        treeZ = scsp.cKDTree(z)

        nn_list = treeZ.query(z, k = k, p = float('inf'))[1]
    
        return nn_list
            
    def _localPerm(self,y,nn_list_z): 
        n = y.shape[0]
        permuted_y = np.zeros(y.shape)
        shuffled_indices = list(range(0,n))
        
        self.rng.shuffle(shuffled_indices)
        
        kperm = len(nn_list_z[0])
        used_indices = []

        for ii in shuffled_indices:
            
            nn_list = nn_list_z[ii]
            self.rng.shuffle(nn_list)
            
            j = nn_list[0]
            m = 0
            
            while j in used_indices and m < kperm - 1:
                m += 1
                j = nn_list[m]
                    
            permuted_y[ii,0] = y[j,0]
    
            used_indices.append(j)
            
        return permuted_y 

        
    def __vec2array(self,x):
        '''
        Reshapes a vector of n elements to an (n x 1) - numpy-array
        '''
        if x is not None:
            if x.ndim == 1:
                x = np.reshape(x,(len(x),1))                
            return x
        else:
            return None
        
            

        
    
            
