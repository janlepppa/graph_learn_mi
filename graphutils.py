import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def getMaximalCliques(G):
    d = G.shape[0]
    G_nx = nx.Graph(G)
    
    # maximal cliques
    cliqs = list(nx.find_cliques(G_nx))
    
    # key = variable index, value = list containing cliques it belongs to
    cliqInxByVar = {i : [] for i in range(0,d)}

    for i in range(0,d):
        for c in cliqs:
            if i in c:
                cliqInxByVar[i].append(c)
                    
    return(cliqs,cliqInxByVar)
    
    
def drawGraph(G, nodeLabels = None):
    # Input: G              adjacency matrix (as numpy array)
    #        nodeLabels     list of strings specifying node names  
    assert(G.shape[0] == G.shape[1])
    d = G.shape[0]

    graph = nx.DiGraph()
    
    di_edges = []
    ud_edges = []
    
    for ii in range(0,d):
        graph.add_node(ii)

    for ii in range(0,d):
        for jj in range(ii+1,d):
            
            u = G[ii,jj]
            v = G[jj,ii]

            if(u > 0 and v > 0 ):
                ud_edges.append((ii,jj))
                ud_edges.append((jj,ii))
            elif(u == 0 and v > 0):
                di_edges.append((jj,ii))
            elif(u > 0 and v == 0):
                di_edges.append((ii,jj))
            else:
                continue
            
    graph.add_edges_from(di_edges)
    graph.add_edges_from(ud_edges)
    
    pos = nx.fruchterman_reingold_layout(graph)
    
    if nodeLabels is None:
        names= {ii:str(ii) for ii in range(0,d)}
    else:
        names = {ii:nodeLabels[ii] for ii in range(0,d)}
    
    #print(ud_edges)
    #print(di_edges)        
            
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph,pos, labels = names)
    nx.draw_networkx_edges(graph, pos, edgelist=di_edges, edge_color='k', arrows=True)
    nx.draw_networkx_edges(graph, pos, edgelist=ud_edges, edge_color='b', arrows=False)
    
    plt.show()
    return(graph)
                  
# Given a DAG 'G', return the Markov blanket of node 'varInx'          
def findMB_DAG(G,varInx):
    MB = set()

    #parents         
    parents = set(np.nonzero(G[:,varInx])[0])
    MB.update(parents)
    
    # children
    children = set(np.nonzero(G[varInx,:])[0])
    MB.update(children)
    
    #spouses
    if(len(children) > 0):
        for ii in children:
            parents_i = set(np.nonzero(G[:,ii])[0])
            MB.update(parents_i)
    
    # remove the variable itself from its Markov blanket
    MB.discard(varInx)

    return(MB)
  
# moralizes DAG or PDAG (where only V-structures are assmued to be directed)    
def moralize(G):
    assert(G.shape[0] == G.shape[1])
     
    # undirected version of the directed graph. bi-directed edges are intepreted as undirected ones
    newG = np.array(1*((G + G.T) > 0), dtype = np.int)
    pDAG = np.copy(G) 
    
    # find edge indices
    nonZeros = np.nonzero(G)
    edges = list(zip(nonZeros[0],nonZeros[1]))
    
    Vstructures = set()
    
    for e in edges:
        ii = e[0]   # row index
        jj = e[1]   # column index
        
        if(G[jj,ii] == 0): #edge is directed ii -> jj
            notDirectedInPDAG = True       

            jjParents = set(np.nonzero(G[:,jj])[0])
            jjParents.discard(ii)   #other parents of jj, ie. parent -> jj (might be also the other way around)
            
            # marry the parents 
            for parent in jjParents:
                if(G[jj,parent] == 0): #check that parent has also _directed_ edge to jj
                
                   # check that parents are not already connected
                   if(G[ii,parent] == 1 or G[parent,ii] == 1):
                       continue
                   else:
                       newG[ii,parent] = 1 
                       newG[parent,ii] = 1 
                       Vstructures.add(Vstructure(jj,ii,parent))
                       notDirectedInPDAG = False
                       
            if notDirectedInPDAG:
                pDAG[jj,ii] = 1
                
            
    return(newG,pDAG,Vstructures)

# returns Hamming distance between two undirected graphs    
def HD(trueUG,estUG):
    assert ((trueUG == trueUG.transpose()).all())
    assert ((estUG == estUG.transpose()).all())
    
    assert trueUG.dtype == np.int and estUG.dtype == np.int 

    return(np.sum(1*(np.abs(trueUG-estUG) > 0))/2)
    
# returns true positive and false positive rates
def tp_fp(trueUG,estUG):
    assert ((trueUG == trueUG.transpose()).all())
    assert ((estUG == estUG.transpose()).all())
    
    assert trueUG.dtype == np.int and estUG.dtype == np.int 

    d = trueUG.shape[0]

    max_edges = int(d*(d-1)/2)
    nedges = int(np.sum(trueUG)/2)
    non_edges = max_edges - nedges
    
    tps = np.sum((trueUG + estUG == 2)/2)
    fps = np.sum((trueUG - estUG == -1)/2)

    return(tps/nedges,fps/non_edges)

# simple class for representing V-structures    
class Vstructure:
    def __init__(self,child,parent1,parent2):
        self.child = child
        self.parents = [parent1,parent2]
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.child == other.child and set(self.parents) == set(other.parents)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(str(self.child) + str(sorted(self.parents)))
        
    def __str__(self):
        s = "" + str(self.parents[0]) + " --> " + str(self.child) + " <-- " + str(self.parents[1])
        return(s)
        
    