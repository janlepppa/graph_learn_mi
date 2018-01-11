import numpy as np
from huge import hugeGenerateData
from scipy.linalg import block_diag
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri # passing numpy objects to R functions

rpy2.robjects.numpy2ri.activate()
bdGarph = importr('BDgraph') 

class DataGenerator:
    def __init__(self, testName = (True,"gaussian"), rng = None):
        self.testName = testName
        if rng is None:
            self.rng = np.random.RandomState()             
        else:
            self.rng = rng
        
    def createData(self,n):
        if type(self.testName) == tuple:     
            linear, noise = self.testName
            return(self.genDataSmallUG(n,linear,noise))
        elif self.testName == "largeUG":
            return(self.genDataLargeUG(n,21,linear = False,noise = "t"))
        elif self.testName == "largeUG_g":
            return(self.genDataLargeUG(n,21,linear = False,noise = "gaussian"))
        elif self.testName == "largeUG_u":
            return(self.genDataLargeUG(n,21,linear = False,noise = "uniform"))
        elif self.testName == "randomUG":
            return(self.randomGaussUG(n,d = 10))
        elif self.testName == "randomUGnonPara":
            return(self.randomNonParaUG(n,d = 10))
        elif self.testName == "randomUGLarge":
            return(self.randomGaussUG(n,d = 20))
        elif self.testName == "randomUGnonParaLarge":
            return(self.randomNonParaUG(n,d = 20))
        elif self.testName == "randomGMSmall15":
            d = 8
            c = 0.15
            scaleMat = c*np.ones((d,d)) + (1-c)*np.eye(d)
            return(self.gmUG2(n,d = d, edges = 6, scaleMat=scaleMat))
        elif self.testName == "randomGMLarge15":
            d = 16
            c = 0.15
            scaleMat = c*np.ones((d,d)) + (1-c)*np.eye(d)
            return(self.gmUG2(n,d = d, edges = 10, scaleMat=scaleMat))
        else:
            print("Invalid test name.")
                    

    # generates data for the small UG experiment                
    def genDataSmallUG(self,n, linear = True, noise = "gaussian"):
 
        if linear == True:
            X1 = self.sampleNoise(n,noiseType = noise)
            X2 = 0.20*X1 + self.sampleNoise(n,noiseType = noise)
            X3 = 0.50*X2 + self.sampleNoise(n,noiseType = noise)
            X4 = 0.25*X3 + self.sampleNoise(n,noiseType = noise)
            X5 = 0.35*X2 + 0.55*X3  + self.sampleNoise(n,noiseType = noise)
            X6 = 0.65*X5 + self.sampleNoise(n,noiseType = noise)
            X7 = 0.90*X3 + 0.25*X5 + self.sampleNoise(n,noiseType = noise)
            
        if linear == False:
            X1 = self.sampleNoise(n,noiseType = noise)
            X2 = 2*np.cos(X1) + self.sampleNoise(n,noiseType = noise)
            X3 = 2*np.sin(np.pi*X2) + self.sampleNoise(n,noiseType = noise)
            X4 = 3*np.cos(X3) + self.sampleNoise(n,noiseType = noise)
            X5 = 0.75*X2*X3 + self.sampleNoise(n,noiseType = noise)
            X6 = 2.5*X5 + self.sampleNoise(n,noiseType = noise)
            X7 = 3*np.cos(0.2*X3) + np.log(np.abs(X5)) + self.sampleNoise(n,noiseType = noise)
            
        X = np.hstack((X1,X2,X3,X4,X5,X6,X7))
        
        G = np.array([[0,1,0,0,0,0,0],
                      [0,0,1,0,1,0,0],
                      [0,0,0,1,1,0,1],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,1,1],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]])
        
        G = 1*(G.T + G > 0)
        
        return(X,G)
        
    def genDataLargeUG(self,n,d, linear = True, noise = "gaussian"):
        assert d % 7 == 0
            
        multip = np.int(d/7)
        
        datas = []
        Gs = []
        
        for ii in range(0,multip):
            x,g = self.genDataSmallUG(n,linear = linear,noise = noise)
            datas.append(x) 
            Gs.append(g)
            
        X = np.hstack(tuple(datas))    
        G = block_diag(*Gs)        
        
        return(X,G)
           
    def gmUG2(self,n, d, edges, scaleMat = None, df = None, nMIX = None, absPC = None):
        G = self.randUG(d,edges) # random UG

        finiteMIX = True
        
        if nMIX is None:
            nMIX = n
            finiteMIX = False
            
        if scaleMat is None:
            scaleMat = np.eye(d)
            
        if df is None:
            df = 3
        
        covMats = np.zeros((d,d,nMIX))    
            
        for jj in range(0,nMIX):
            # sample inverse of covariance from G-Wishart distribution
            IC = self.sampleGWshrt(G = G,df =df,scaleMat=scaleMat,absPC=absPC)
            cov = np.linalg.inv(IC)
            covMats[:,:,jj] = cov
            
            
        X = np.zeros((n,d))
        
        for tt in range(0,n):
            
            if finiteMIX:
                cov = covMats[:,:,self.rng.randint(0,nMIX)]
            else:
                cov = covMats[:,:,tt]

            X[tt,:] = self.rng.multivariate_normal(np.zeros(d),cov)
                
        return(X,G)
            
    # returns true if matrix is positive definite, false otherwise
    def __isPosDef(self,A):
        try: 
            np.linalg.cholesky(A) # this will raise LinAlgError if A is not pos def
            return(True)
        except np.linalg.LinAlgError:
            return(False)
        
    # returns noise from given (gaussian,uniform,t) distribution    
    def sampleNoise(self,n, noiseType = "gaussian"):
        if noiseType == "gaussian":
            return(self.rng.normal(0, 1, (n,1)))
        if noiseType == "uniform":
            return(self.rng.uniform(-1,1,(n,1)))
        if noiseType == "t":
            return(self.rng.standard_t(2,(n,1)))
            
    # creates Gaussian data with a random underlying undirected graph        
    def randomGaussUG(self,n,d, prob = None):
        
        seed = self.rng.randint(1,1e9)
        X,G = hugeGenerateData(n,d, graph = "random", prob = prob,seed = seed)
        
        return(X,G)
    
    # similar to above one, except the data is now transformed X -> X**3     
    def randomNonParaUG(self,n,d, prob = None):
        
        seed = self.rng.randint(1,1e9)
        X,G = hugeGenerateData(n,d, graph = "random", prob = prob,seed = seed)
        X = X**3
        return(X,G)
        
        
    # create adjacency matrix of a random graph with "d" nodes and approximately "expectEdges" number of edges    
    def randUG(self,d,expectEdges = 10):
     
        G = np.zeros((d,d),dtype = np.int)
        possibleEdges = d*(d-1)/2
        prob = expectEdges/possibleEdges
    
        for ii in range(0,d):
            for jj in range(ii +1,d):
                u = self.rng.uniform(0,1,1)
                if u < prob:
                    G[ii,jj] = 1
                    G[jj,ii] = 1
        
        return(G)
        
    def sampleGWshrt(self,G,df,scaleMat, absPC = None):
   
        if absPC is None:
            IC = np.array(bdGarph.rgwish(n = 1,adj_g = G,b = df, D = scaleMat))[:,:,0]
        else:
            repeat = True
            IC = np.array(bdGarph.rgwish(n = 1,adj_g = G,b = df, D = scaleMat))[:,:,0]

            while(repeat):
               pc = self.covToCorr(IC)
               nonZerosOffDiagonals = np.abs(pc.ravel()[np.triu(G).ravel() > 0])
               minNonzero = np.min(nonZerosOffDiagonals)
               
               if minNonzero > absPC:
                   break;
                
        return(IC)
              
    def covToCorr(self,cov):
        D = np.diag(np.reciprocal(np.sqrt(np.diag(cov))))
        return(np.dot(np.dot(D,cov),D))