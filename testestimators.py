from knnestimator import KnnEstimator
from fishertest import FisherCI, parCorr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plotErrors(results,ks,samples,title):
    nks, nsamples = results.shape
    
    plt.figure()
    plt.xlim(0.9,len(samples) + 0.1)
    x = list(range(1,len(samples) + 1,1))
    for ii in range(0,nks):
        lab = "k = " + str(ks[ii])
        plt.plot(x,results[ii,:],label = lab,marker = 'o')
        plt.xticks(x, samples)
        
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best')
    
    plt.title(title)
    plt.savefig(title + ".pdf")
    plt.show()
    
def compErrors(data,samples, ks, trueMI,p):
    nsamples = len(samples)
    nks = len(ks)
    res1 = np.zeros((nks,nsamples))
    ni = 0
    
    for n in samples:
        ki = 0
        for k in ks:
            X = data[:n,:]
            aa = KnnEstimator(k = k, p = p)
            res1[ki,ni] = trueMI - aa._entropy(X)   
            ki += + 1
    
        ni += 1
        
    return(res1)
    
def compErrors2(x,y,samples, ks, trueMI,p,z=None):  
    nsamples = len(samples)
    nks = len(ks)
    res1 = np.zeros((nks,nsamples))
    ni = 0
    
    for n in samples:
        ki = 0
        for k in ks:
            aa = KnnEstimator(k = k, p = p)
            
            if(z is not None):
                res1[ki,ni] = trueMI -aa._cmi1(x[:n,:],y[:n,:],z[:n,:]) 
                
            else:
                res1[ki,ni] = trueMI -aa._cmi1(x[:n,:],y[:n,:],z)
                
            ki += + 1
        ni += 1
        
    return(res1)
    
# test entropy estimators, plot absolute error as a function of sample size  
def testEntropy(seed, p = float('inf')):
    np.random.seed(seed)
    samples = [50,100,1000,5000,10000, 25000, 50000]
    ks = [1,3,5,7,9,25]

    # normally distributed data
    mu = 2.5
    sigma = 3
    data1 = np.random.normal(loc = mu,scale = sigma,size = (max(samples),1))
    trueE1 = np.log(sigma*np.sqrt(2*np.pi*np.e))   #in nats  
    res1 = compErrors(data1,samples,ks,trueE1,p)   
    plotErrors(res1,ks,samples,'Gaussian RV')
    
    # logistic
    data2 = np.random.logistic(loc = 0.0,scale = 1.0,size = (max(samples),1))
    trueE2 = 2   #in nats
    res2 = compErrors(data2,samples,ks,trueE2,p)
    plotErrors(res2,ks,samples,'Logistic RV')
    
    # uniform
    a = -1
    b = 3
    data3 = np.random.uniform(low = a,high = b,size = (max(samples),1))
    trueE3 = np.log(b-a)
    res3 = compErrors(data3,samples,ks,trueE3,p)
    plotErrors(res3,ks,samples,'Uniform RV')
    
    # multivariate normal
    mean = [0,0,0]
    cov = [[1,0.2,0],[0.2,1,0.5],[0,0.5,1]] 
    data4 = np.random.multivariate_normal(mean,cov,max(samples))
    trueE4 = 0.5*(len(mean)*np.log(2*np.pi*np.e) + np.linalg.slogdet(cov)[1])
    res4 = compErrors(data4,samples,ks,trueE4,p)
    plotErrors(res4,ks,samples,'Mv normal RV')
 
# test MI estimator version 1, plot absolute error as a function of sample size    
def testMI(seed, c = 0.1, p = float('inf')):
    
    # multivarial normal
    np.random.seed(seed)
    samples = [50,100,4000,10000,25000, 50000]
    ks = [1,3, 5, 10,40]

    n = max(samples)
    cov_m = [[1.0, c], [c, 1.0]]
    data = np.random.multivariate_normal([0, 0], cov_m, n)
    #trueMI = -1/2*np.log(1-np.corrcoef(data, rowvar=0)[0, 1]**2)
    trueMI = -1/2*np.log(1-c**2)
    res1 = compErrors2(data[:,[0]],data[:,[1]],samples, ks, trueMI,p,z=None)
    plotErrors(res1,ks,samples,'MI: Bivariate normal RV, estimator 1')
    
    #res2 = compErrors2(data[:,[0]],data[:,[1]],samples, ks, trueMI,p,eps,z=None)
    #plotErrors(res2,ks,samples,'MI: Bivariate normal RV, estimator 2')
    
    
def testMI2(seed, k =3, c = 0.1, ntests = 10, p = float('inf')):
    
    np.random.seed(seed)
    samples = [100,200,400,800,1600,3200]
    
    knn= KnnEstimator(k = k)
    trueMI = -1/2*np.log(1-c**2)
    
    errors = np.zeros((2,len(samples)))
    
    
    for jj in range(0,ntests):
    
        ii = 0
        for n in samples:
            cov_m = [[1.0, c], [c, 1.0]]
            data = np.random.multivariate_normal([0, 0], cov_m, n)
            errors[0,ii] += (trueMI - knn._mi1(data[:,[0]],data[:,[1]]))/ntests
            errors[1,ii] += (trueMI - knn._mi2(data[:,[0]],data[:,[1]]))/ntests
            ii +=1
        
    plt.figure()
    plt.xlim(0.9,len(samples) + 0.1)
    x = list(range(1,len(samples) + 1,1))
    for ii in range(0,2):
        lab = "MI_" + str(ii)
        plt.plot(x,errors[ii,:],label = lab,marker = 'o')
        plt.xticks(x, samples)
        
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best')  
    plt.show()
    
    return(errors)    
    
def testCMI(seed, tests = 3, p = float('inf')):
    
    # multivarial normal
    np.random.seed(seed)
    samples = [50,100,1000,10000,25000]
    ks = [1,3,5,10,15,25,40]

    n = max(samples)
    
    ic = np.array([[1.0, -0.2, 0],[-0.2, 1.0, 0.6],[0, 0.6, 1.0]])
    c = np.linalg.inv(ic)
    
    
    
    nsamples = len(samples)
    nks = len(ks)
    res1 = np.zeros((nks,nsamples))
    res2 = np.zeros((nks,nsamples))
    res3 = np.zeros((nks,nsamples))
      
    for tt in range(0,tests):
        data = np.random.multivariate_normal([0, 0, 0], c, n)
        
        x,y,z = 0,2,1
        cmixy_zT = mvnCMI(c,[x],[y],[z])
        res1 = res1 + compErrors2(data[:,[x]],data[:,[y]],samples, ks, cmixy_zT,p,z = data[:,[z]])/tests
        pc1= parCorr(data[:,[x]],data[:,[y]],data[:,[z]])
        
        x,y,z = 1,2,0
        cmixy_zT = mvnCMI(c,[x],[y],[z])
        res2 = res2 +  compErrors2(data[:,[x]],data[:,[y]],samples, ks, cmixy_zT,p,z = data[:,[z]])/tests
        pc2= parCorr(data[:,[x]],data[:,[y]],data[:,[z]])
      
                                   
        x,y,z = 0,1,2
        cmixy_zT = mvnCMI(c,[x],[y],[z])
        res3 = res3 + compErrors2(data[:,[x]],data[:,[y]],samples, ks, cmixy_zT,p,z = data[:,[z]])/tests
        pc3= parCorr(data[:,[x]],data[:,[y]],data[:,[z]])
      
    plotErrors(res1,ks,samples,"pc=" + str(pc1))
    plotErrors(res2,ks,samples,"pc=" + str(pc2))
    plotErrors(res3,ks,samples,"pc=" + str(pc3))

def testCMI2(seed,k = 3, tests = 10):
    np.random.seed(seed)
    samples = [100,25000]


    
    ic = np.array([[1.0, -0.2, 0],[-0.2, 1.0, 0.6],[0, 0.6, 1.0]])
    c = np.linalg.inv(ic)
    c[0,0] = 10*c[0,0] 
    
    n = np.max(samples)
    nsamples = len(samples)
    res1 = np.zeros((2,nsamples))
    res2 = np.zeros((2,nsamples))
    res3 = np.zeros((2,nsamples))
      
    
    knn = KnnEstimator(k = k) 
    for tt in range(0,tests):
        
        data = np.random.multivariate_normal([0, 0, 0], c, n)
        jj = 0
        for ss in samples:
            x,y,z = 0,2,1
            cmixy_zT = mvnCMI(c,[x],[y],[z])
            res1[0,jj] += (cmixy_zT - knn._cmi1(data[:ss, [x]],data[:ss, [y]],data[:ss, [z]]) ) /tests 
            res1[1,jj] += (cmixy_zT - knn._cmi2(data[:ss, [x]],data[:ss, [y]],data[:ss, [z]]) ) /tests 
    
            
            x,y,z = 1,2,0
            cmixy_zT = mvnCMI(c,[x],[y],[z])
            res2[0,jj] += (cmixy_zT - knn._cmi1(data[:ss, [x]],data[:ss, [y]],data[:ss, [z]]) ) /tests 
            res2[1,jj] += (cmixy_zT - knn._cmi2(data[:ss, [x]],data[:ss, [y]],data[:ss, [z]]) ) /tests 
          
                                       
            x,y,z = 0,1,2
            cmixy_zT = mvnCMI(c,[x],[y],[z])
            res3[0,jj] += (cmixy_zT - knn._cmi1(data[:ss, [x]],data[:ss, [y]],data[:ss, [z]]) ) /tests 
            res3[1,jj] += (cmixy_zT - knn._cmi2(data[:ss, [x]],data[:ss, [y]],data[:ss, [z]]) ) /tests 
            jj += 1
            
            
    plt.figure()
    plt.xlim(0.9,len(samples) + 0.1)
    x = list(range(1,len(samples) + 1,1))
    for ii in range(0,2):
        lab = "MI_" + str(ii)
        plt.plot(x,res1[ii,:],label = lab,marker = 'o')
        plt.xticks(x, samples)
        
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best')  
    plt.show()
    
    plt.figure()
    plt.xlim(0.9,len(samples) + 0.1)
    x = list(range(1,len(samples) + 1,1))
    for ii in range(0,2):
        lab = "MI_" + str(ii)
        plt.plot(x,res2[ii,:],label = lab,marker = 'o')
        plt.xticks(x, samples)
        
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best')  
    plt.show()
    
    
    plt.figure()
    plt.xlim(0.9,len(samples) + 0.1)
    x = list(range(1,len(samples) + 1,1))
    for ii in range(0,2):
        lab = "MI_" + str(ii)
        plt.plot(x,res3[ii,:],label = lab,marker = 'o')
        plt.xticks(x, samples)
        
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best')  
    plt.show()
    
            
            
        
 
# entropy for Gaussian variables, input: (scalar) variance or covariance matrix    
def mvnEntropy(cov_mat):
    if(np.isscalar(cov_mat)):
        return(0.5*(np.log(2*np.pi*np.e) +  np.log(cov_mat)))
    else:
        return(0.5*(cov_mat.shape[0]*np.log(2*np.pi*np.e) + np.linalg.slogdet(cov_mat)[1]))

def extSubmat(mat,variables):
    return(mat[variables,:][:,variables])
    
# MUTUAL INFORMATION for Gaussian variables, cov_mat should be a numpy array.
#If z argument is not given, compute mutual information: I(x,y) = H(x) + H(y)- H(x,y)
#I(x,y | z) = H(x,z) + H(y,z) - H(x,y,z) - H(z) , x,y,z are lists containing indices of variables referring to cov_mat   
def mvnCMI(cov_mat,x,y,z = None):
    if(z is None):
            xy = x + y
            Hxy = mvnEntropy(extSubmat(cov_mat,xy))
            Hy = mvnEntropy(extSubmat(cov_mat,y))
            Hx = mvnEntropy(extSubmat(cov_mat,x))
    
            return(Hx+Hy-Hxy)           
    else:     
        xz = x + z
        yz = y + z
        xyz = x + y + z
    
        Hxz = mvnEntropy(extSubmat(cov_mat,xz))
        Hyz = mvnEntropy(extSubmat(cov_mat,yz))
        Hxyz = mvnEntropy(extSubmat(cov_mat,xyz))
        Hz = mvnEntropy(extSubmat(cov_mat,z))
    
        return(Hxz + Hyz - Hxyz - Hz)
    
        
def mvnormal_cmi_null(samples = 100, t = 10000,k = 3, sig = 0.05,permutations = 200,k_perm = None):
    
    icmat = np.array([[1,0,0.2],[0,1,0.8],[0.2,0.8,1]])        
    c_mat = np.linalg.inv(icmat)
    
    meann = c_mat.shape[0]*[0]

    true_cmi_dep = mvnCMI(c_mat,[1],[2],[0])
    true_cmi_indep = mvnCMI(c_mat,[0],[1],[2])
    
    
    knn = KnnEstimator(k = k, sig= sig, permutations = permutations, corrCheck= False, k_perm = k_perm)
 
    cmi_dep = []
    cmi_indep = []

    for ii in range(0,t):
        X = np.random.multivariate_normal(meann,c_mat,samples)
        c_mat_est = np.cov(X,rowvar = False)
        
        cmi_dep.append(mvnCMI(c_mat_est,[0],[2],[1]))
        cmi_indep.append(mvnCMI(c_mat_est,[0],[1],[2]))
        
        
    sns.distplot(cmi_dep,hist=False, label = "null_dep")
    sns.distplot(cmi_indep,hist=False,label = "null_indep")
    
    
    indep_dep, estMI_dep, _, estPVal_dep, MIs_dep = knn._permutationTest(X[:,[0]],X[:,[2]],X[:,[1]])
    
    MIdep_2 = []
    
    for mi in MIs_dep:
        if mi < 0:
            MIdep_2.append(0)
        else:
            MIdep_2.append(mi)
 
    indep_indep, estMI_indep, _, estPVal_indep, MIs_indep = knn._permutationTest(X[:,[0]],X[:,[1]],X[:,[2]])
    
    MIindep_2 = []
    
    for mi in MIs_indep:
        if mi < 0:
            MIindep_2.append(0)
        else:
            MIindep_2.append(mi)
    

    #print("P-val from permutation test (dependent case): ",estPVal_dep)
    #p_null_dep = np.sum(np.array(cmi_dep) >= estMI_dep )/len(cmi_dep)
    p_null_indep = np.sum(np.array(cmi_indep) >= estMI_indep )/len(cmi_indep)
    #print("P-val from the null-distribution: ", p_null_dep)
    #print("------------------")
    print("P-val from permutation test (independent case): ",estPVal_indep)
    print("P-val from the null-distribution: ", p_null_indep)
    
    print(MIindep_2)
    
    sns.distplot(MIdep_2,hist=False,label= "permutation_dep")
    sns.distplot(MIindep_2,hist=False, label = "permutation_indep")
    
    plt.legend()
    
    
    return(cmi_dep,cmi_indep,true_cmi_dep)
   
def createMeanderData(n):
    Z = (3/4)*np.random.normal(loc = 0, scale = 1/5, size= (n,1) ) + (1/4)*np.random.normal(loc = 1, scale = 1/3, size= (n,1))
    X = Z/10 + 0.5*np.sin(2*np.pi*Z) + 0.1*np.random.normal(loc = 0, scale = 1,size= (n,1) )
    Y = Z/5 + 0.5*np.sin(2*np.pi*Z + 0.35) + 0.1*np.random.normal(loc = 0, scale = 1, size= (n,1))
    
    #fig = plt.figure()
  
    return(X,Y,Z)
    
def creteMeanderDataVstructure(n):
    X =  (3/4)*np.random.normal(loc = 0, scale = 1/5, size= (n,1) ) + (1/4)*np.random.normal(loc = 1, scale = 1/3, size= (n,1))
    Y =  (3/4)*np.random.normal(loc = 0, scale = 1/5, size= (n,1) ) + (1/4)*np.random.normal(loc = 1, scale = 1/3, size= (n,1))
    Z =  X/10 + 0.5*np.sin(2*np.pi*X) + Y/5 + 0.5*np.sin(2*np.pi*Y) + 0.1*np.random.normal(loc = 0, scale = 1, size= (n,1))
    
    return(X,Y,Z)
    
    

    
def visualizeMeanderCI(samples, k = 5, permutations = 200, seed = 123, sig = 0.05, k_perm = 5,corrCheck = False, data = 1):
    
    knn = KnnEstimator(k = k, permutations = permutations, sig = sig, corrCheck= corrCheck,k_perm = None)
    knn_local = KnnEstimator(k = k, permutations = permutations, sig = sig, corrCheck= corrCheck,k_perm = k_perm)
    
    
    if data == 1:
        X,Y,Z = createMeanderData(samples)
    elif data == 2:
        X,Y,Z = creteMeanderDataVstructure(samples)
    
    indep, estMI, _ , _ , MIs = knn._permutationTest(X,Y,Z)
    indep_l, estMI_l, _ , _ , MIs_l = knn_local._permutationTest(X,Y,Z)
     
    
    plt.scatter(X,Y)
    
    plt.figure(2)
    
    sns.kdeplot(np.array(MIs),label="knn", shade=True)
    ax = sns.kdeplot(np.array(MIs_l),label="knn_local",shade = True)
    ax.axvline(x=estMI, ymin=0, ymax=1, c = "red", label = "estimated MI", linestyle = "--")
    ax.legend()
    print("knn naive permutation: ", indep, "\n","knn local permutation: ",indep_l)
    
    return(indep,indep_l,estMI,estMI_l,ax,MIs,MIs_l)
    
    
    
def testMeander(samples, k = 5, permutations = 200, seed = 123, tests = 100, sig = 0.05, k_perm = None,corrCheck = False):
    np.random.seed(seed)
    
    ff = FisherCI()
    knn = KnnEstimator(k = k, permutations = permutations, sig = sig, corrCheck= corrCheck,k_perm = k_perm)
    #knn = KnnEstimator(k = k, permutations = permutations, sig = sig)
    #maxS = np.max(samples)
    
    XIIYf = 0 
    XIIYZf = 0
    
    XIIYknn = 0 
    XIIYZknn = 0
    
    
    for ii in range(tests):
        
        X,Y,Z = createMeanderData(samples)
        #X= scale(X)
        #Y = scale(Y)
        #Z = scale(Z)
            
        indepf, depf = ff.independent(X,Y)
        indepk, depk = knn.independent(X,Y)
        
        if(indepf == False):
            XIIYf+=1    
        if(indepk == False):
            XIIYknn+=1
            
        indepf, depf = ff.independent(X,Y,Z)
        indepk, depk = knn.independent(X,Y,Z)
        
            
        if(indepf == True):
            XIIYZf+=1    
        if(indepk == True):
            XIIYZknn+=1
    print("Sample size:", samples)
    print("        Reject X || Y      Accept X || Y | Z")
    print("Fisher    ",XIIYf/tests, "           ", XIIYZf/tests )
    print("kNN       ",XIIYknn/tests, "           ", XIIYZknn/tests )
    
#if __name__ == "__main__":
    #seed = 1223234
    #testEntropy(seed)
    #testCMI2(seed,tests = 3)
    #testMI2(seed,ntests= 3)
    #testCMI(seed)
    #testMI(seed)
