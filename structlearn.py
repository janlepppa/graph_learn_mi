from knnestimator import KnnEstimator
from mbalgs import GS, IAMB, interIAMB
import numpy as np
import itertools
from graphutils import drawGraph
import networkx as nx


class StructLearn():
    
    def __init__(self,X,
                 MBalgorithm = "IAMB",
                 ci_estimator = None,
                 symmetryRule = "AND",
                 MBresolve = "colliders",
                 mode = "UG"):
        
        self.X = X
        self.n, self.d = X.shape
        self.aMat = np.zeros((self.d,self.d), dtype = np.int)
        self.MBs = dict()
        self.symmetryRule = symmetryRule
        self.MBresolve = MBresolve
        self.mode = mode
        
        if(ci_estimator is None):           
            self.estimator = KnnEstimator()
        else:
            self.estimator = ci_estimator
            
        if(MBalgorithm == "IAMB"):
            self.MBalgorithm = IAMB(self.X, estimator = self.estimator, mode= self.mode)
        elif(MBalgorithm == "GS"):
            self.MBalgorithm = GS(self.X, estimator = self.estimator)   
        elif(MBalgorithm == "interIAMB"):
            self.MBalgorithm = interIAMB(self.X, estimator = self.estimator)
        else:
            print("Warning: MBalgorithm  \'",MBalgorithm,"\' is not defined. Using the IAMB-algorithm instead.", sep="")
            self.MBalgorithm = IAMB(self.X, self.estimator)
                     
    def __symmetrize(self, G, rule):
        if(isinstance(G,dict)): #input is dictionary containing Markov blankets
            newG = np.zeros((self.d,self.d))
            
            for ii in range(0,self.d):
                MB = G[ii]
                newG[list(MB),ii] = 1

            return(self.__symmetrize(newG,rule))        
        
        elif(isinstance(G, np.ndarray)): # inpput is adjacency matrix
            if(rule == "OR"):
                OR = 1*((G + G.T) > 0)
                OR = OR.astype(np.int)
                return(OR)
            if(rule == "AND"):
                AND = 1*((G + G.T) == 2)  
                AND = AND.astype(np.int)
                return(AND)
                
        else:
            print("Oops")
            
    def __findTriangles(self):
        triangles = dict()
        for i in range(self.d):
            a_x = np.flatnonzero(np.array(self.aMat[:, i]))
            for j in a_x:
                a_y = np.flatnonzero(np.array(self.aMat[:, j]))
                tris = set(a_y).intersection(a_x)
                if len(tris) > 0:
                    if (j, i) not in triangles:
                        triangles[(i, j)] = tris
        return (triangles)

    # returns nodes that reachable from nodes in set "A"  in G_XY (UG where all directed links involving x or y are removed)   
    def __reachableFrom(self,A,x,y):
        
        aMatCopy = np.copy(self.aMat)
        
        # remove all direct links containing x or y
        aMatCopy[x,:] = 0
        aMatCopy[y,:] = 0
        aMatCopy[:,y] = 0
        aMatCopy[:,x] = 0

        G_XY = nx.Graph(aMatCopy)  

        reachable = set()
        
        for W in A:
            reachable.update(set(nx.node_connected_component(G_XY,W)))    

        return(reachable)
        
        
    # main algorithm for directing v-structures after initial MB-search    
    def _resolveMBcoll(self):        
        # find all edges that are part of triangles
        triangles = self.__findTriangles()
        
        remove = []
        orient = []
        
        # loop over edges in triangles
        #######################################################################
        for edge, triXY in triangles.items():          
            S_xy = None
            X,Y = edge
            
            BdX = self.__findBoundary(X)
            BdY = self.__findBoundary(Y)    
            
            
            
            A = BdX.difference(triXY.union([Y]))
            B = BdY.difference(triXY.union([X]))
            
            if(len(A) < len(B)):
                B = A
                       
            for S in self.__allStrictSubsets(triXY):
                S = set(S) # S is tuple
                Z = B.union(S)
                
                if(self.MBalgorithm._doIndepTest(X,Y,list(Z))):
                     S_xy = Z
                     
                     break # collider set found, break inner for loop
            
                allW = triXY.difference(S)      
                D = B.intersection(self.__reachableFrom(allW,X,Y)) ## implement reachable
                Bprime = B.difference(D)
                               
                breakOuter = False

                for Sprime in self.__allStrictSubsets(D):
                    Z = Bprime.union(Sprime, S)
                    
                    if(self.MBalgorithm._doIndepTest(X,Y,list(Z))):
                     S_xy = Z
                     breakOuter = True
                     break # collider set found, break inner for loop
                    
                if breakOuter:
                    break
                

            ## save orientation directive
            if S_xy is not None:
                
                remove.append((X,Y)) # remove X --- Y
                
                for Z in triXY.difference(S_xy):
                    orient.append((X,Z,Y)) # orient X --> Z <-- Y
         
        ####################################################################### 
        # apply the found orientation directives
        #
        # remove spouses
        for edge in remove:
            e1,e2 = edge
            self.aMat[e1,e2] = 0
            self.aMat[e2,e1] = 0

        # orient edges:
        for c in orient:
            x,z,y = c
            
            # check that both edges exist
            if(self.aMat[x,z] > 0 and self.aMat[y,z] > 0): 
                self.aMat[z,x] = 0 #orient x --> z 
                self.aMat[z,y] = 0 #orient y --> z
            
        return(self.aMat)       
            
    # returns an iterator over all the strict subsets (as tuples) of set A
    # if len(A) == 0 return list containing empty tuple                    
    def __allStrictSubsets(self,A):
        maxSize = len(A)
        if maxSize == 0:
            return([()])
        else:    
            setIterator = (itertools.combinations(A,ssize) for ssize in range(0,maxSize) )
            return(itertools.chain.from_iterable(setIterator))
     
    def __findBoundary(self, varIndex):
        connected = self.aMat[:,varIndex]
        return(set(np.flatnonzero(connected)))       

    
    def _resolveMarkovBlanketsGS(self):
        # Orients the V-structures, see https://www.cs.cmu.edu/~dmarg/Papers/PhD-Thesis-Margaritis.pdf (phases 2 and 3)
         
        MBs = np.copy(self.aMat) # copy the original adjacency matrix containing Markov blanket information

        # compute the graph structure
        for x in range(0,self.d):
            
            MBx = set(np.nonzero(MBs[:,[x]])[0])

            
            for y in MBx:
                Bx = set(np.nonzero(MBs[:,[x]])[0]).difference([y])
                By = set(np.nonzero(MBs[:,[y]])[0]).difference([x])
                
                if(len(Bx) > len(By)):
                    B = By
                else:
                    B = Bx
                    
                cardB = len(B)
                
                if(cardB == 0):
                    break
                
                setSize = 0
                dependent = True
                
                while(dependent):
                    
                    if(setSize > cardB):
                        break;
                        
                    for S in self._subsets2(B,setSize):
                        indep = self.MBalgorithm._doIndepTest(x,y,list(S))
                        
                        if(indep):
                            self.aMat[x,y] = 0
                            self.aMat[y,x] = 0
                            dependent = False
                            break
                        else:
                            setSize += 1
                            
        # orient edges
        for x in range(0,self.d):
            
            MBx = self.__findBoundary(x)
     
            for y in MBx:
                
                C = MBx.difference(self.__findBoundary(y).union([y]))
                
                
                for z in C:
                    orient = True
                    self.aMat[y,x] = 1    # orient Y -> X
                    self.aMat[x,y] = 0

                    #Bz = self._findBoundary(z).difference([y]) 
                    #By = self.findBoundaty(y).difference([z])
                    
                    Bz = set(np.nonzero(MBs[:,[z]])[0]).difference([y,x])
                    By = set(np.nonzero(MBs[:,[y]])[0]).difference([z,x])
                
                    if(len(Bz) > len(By)):
                        B = By
                    else:
                        B = Bz
                    
                    cardB = len(B)    
                    setSize = 0
                    
                    
                    while(orient):
                        if(setSize > cardB):
                            break
                        
                        for S in self._subsets2(B,setSize):
                            SuX = list(S)
                            SuX.append(x)
                            
                            #print(SuX)
                            
                            indep = self.MBalgorithm._doIndepTest(y,z,SuX)
                            
                            if(indep == True):
                                orient = False
                                self.aMat[x,y] = 1  # tests yielded independence, remove orientation
                                break
                            
                        setSize += 1
                        
                    if(orient == True): # found z so that condition in phase 3 on page 35 of Margaritis thesis holds, exit the loop
                        break
                             
    def _subsets2(self,A,setsize):
        # returns an iterator over all the subsets of "A" of size "setsize"
        # if "setsize == 0", returns a list containing empty set
        if(setsize == 0):
            return([set()])
        else:
            return(itertools.combinations(A,setsize))
   
    def findMoralGraph(self):
        print("Searching Markov Blankets...")
        
        # find the Markov blanket for each node
        for ii in range(0,self.d):
            
            print("..........node ", ii + 1,"/",self.d,sep="")
            MB = self.MBalgorithm.findMB(ii)
            
            self.MBs[ii] = MB              # store the found MBs
            self.aMat[list(MB),ii] = 1         # store the found MB into adjacency matrix 
        
        # harmonize the found Markov blankets
        self.aMat = self.__symmetrize(self.aMat, self.symmetryRule)
        return(self.aMat)
                    
    def findDAG(self):
        # Run the structure learning algorithm using first the specified method for Markov blanket discovery.
        # The found Markov blankets are then resolved by orienting the V-structures 
        
        if(not self.MBs):
            print("Moral graph not found. Estimating it...")
            self.findMoralGraph()
        
        if(self.MBresolve == "GS"):
            print("Using GS")
            self._resolveMarkovBlanketsGS()
            
        elif(self.MBresolve == "colliders"):
            print("Using collider sets")
            self._resolveMBcoll()
        else:
            print("MB resolve algoritmh not found. Using collider sets...") 
            self._resolveMBcoll()
            
               
        return(self.aMat)
      
    # plot the current adjacency matric (UG or PDAG)    
    def showGraph(self):
        return(drawGraph(self.aMat))
        
    # return undirected moral graph cinstructed from the Markov blanket information (AND or OR rule for harmonizing the blankets)    
    def getMoralGraph(self,rule):
        if(bool(self.MBs) == False):
            print("Markov blanket information not found. Call 'findMoralGraph'.")
            return(None)
        else:
            return(self.__symmetrize(self.MBs,rule))
            
