import pickle
import numpy as np
import matplotlib.pyplot as plt
#import os
import seaborn as sns
from itertools import product
from graphutils import tp_fp
import pandas as pd


used_methods = ["knnMI_AND",
                "fisherZ_AND",
                "mb_STARS",
                "glasso_STARS",
                "mb_auto",
                "RCIT_AND",
                "KCIT_AND"]
                
used_methods_legend = ["knnMI",
                "fisherZ",
                "NPN_mb",
                "NPN_glasso",
                "NPN_mb_auto",
                "RCIT",
                "KCIT"]
                
knn_comp_methods = ['k_0.01n_local_AND',
                    'k_0.1n_local_AND',
                    'k_3_local_AND',
                    'k_5_local_AND',
                    'k_0.01n_AND',
                    'k_0.1n_AND',
                    'k_3_AND',
                    'k_5_AND']
                    
knn_comp_methods_legend = ['k_0.01n_local',
                    'k_0.1n_local',
                    'k_3_local',
                    'k_5_local',
                    'k_0.01n',
                    'k_0.1n',
                    'k_3',
                    'k_5']

def loadRes(location):
    return pickle.load(open(location,"rb"))
    
# check that tuning parameter ranges are sensible in all tests (smallest one yields a graph that is denser than the generating model and
# the largest is known to yield an empty graph)    
def checkAllSparsities(folderName, testNames = None, methods = None):
    if testNames is None:
        
        linear = [True,False]
        noises = ["gaussian", "t", "uniform"]
        testNames = list(product(linear, noises)) + ["randomUGnonPara","randomUGnonParaLarge","largeUG"] 
        
        
    for test in testNames:
        
        try: 
            checkSparsities(test,folderName=folderName,methods=methods)
            
        except FileNotFoundError:
            print("File corresponding to test",test, "does not exist")
            
# check that all the tuning parameter values were chosen in a sensible manner (one test)    
def checkSparsities(testName,folderName, methods = None):
    if type(testName) == tuple: # convert test name to string
     
            if testName[0] == True:
                testNameString = "Linear"
            else:
                testNameString = "Nonlinear"
    
            testName = testNameString + testName[1]


    filee = "tests/" + folderName + "/" +testName + ".p"
        
    res,parameters = loadRes(filee)
    
    ns = parameters["ns"]
    ntests = parameters["ntests"]
    
    if methods is None:
        methods = ["mb_STARS","glasso_STARS","mb_RIC","glasso_RIC"]

    success = True

    for method in methods:
        
        for tt in range(0,ntests):
            trueGraph = parameters["trueUGs"][tt]
            d = trueGraph.shape[0]
            trueSparsity = (np.sum(trueGraph)/2)/(d*(d-1)/2)
            
            for n in ns:
                maxSparsity =  np.max(res[method][n]["sparsity"][tt]) # actually the density of the graph....
                if trueSparsity > maxSparsity: # true network is denser than the densest on the lasso path
                    print("FAIL: ",testName, method, trueSparsity,maxSparsity)
                    success = False
    if success:
        print(testName, ": tuning parameter range is sensible in all",ntests ,"tests")
    
        
def gen_smallnetwork_subplot(fontscale = 2, 
                             folder = "k5", 
                             methods = used_methods,
                             leg = False,
                             lw = 4,
                             savepath = "tex/ISIT/small_networks.pdf"):
    
    linear = [True,False]
    noises = ["Gaussian", "t", "Uniform"]

    tests = list(product(linear, noises))
    
    #fig,axs = plt.subplots(nrows = 2, ncols = 3,sharex = True, sharey = True, figsize = (10,11) )
    fig,axs = plt.subplots(nrows = 2, ncols = 3,sharex = True, figsize = (12,6.5))
    axs = axs.flatten()
    
    ax_ii = 0

    for test in tests:
        
        ylab = None
        xlab = None
        
        if ax_ii == 0 or ax_ii == 3:
            ylab = "Hamming distance"
            
        if ax_ii >= 3:
            xlab = "Sample size"
               
        if test[0] == True:
            title = "Linear + "
        else:
            title = "Non-linear + "
        
        title = title + test[1]
        ax = axs[ax_ii]

        # one subplot
        load_and_plot_res(test,title,ax,
                          fontscale=fontscale,
                          folderName=folder,
                          methods=methods,
                          leg = leg,
                          linewidth = lw,
                          ylab = ylab,
                          xlab = xlab) 

        ax_ii += 1
             
    #plt.ylabel("Hamming distance")
    #plt.xlabel("Sample Size")    
        
    fig.savefig(savepath)
    plt.show()  

def gen_large_networks_subplot(fontscale = 2, 
                             folder = "k5", 
                             methods = used_methods, 
                             leg = False,
                             lw = 4,
                             savepath = "tex/ISIT/large_networks.pdf"):
    
    fig,axs = plt.subplots(nrows = 1, ncols = 3,sharex = True, figsize = (12.5,4))
    axs = axs.flatten()
    
    ax_ii = 0
    
    tests = ["randomUGnonPara","randomUGnonParaLarge","largeUG"]
             
    titles = ["Non-paranormal, small", "Non-paranormal, large","Large network + t"]
    title_ii = 0
    
    xlab = "Sample Size"
    ylab = "Hamming distance"
    
    for test in tests:
               
        title = titles[title_ii]
        ax = axs[ax_ii]

        if ax_ii > 0:
            ylab = None

        # one subplot
        load_and_plot_res(test,title,ax,
                          fontscale=fontscale,
                          folderName=folder,
                          methods=methods,
                          leg = leg,
                          linewidth = lw,
                          ylab = ylab,
                          xlab = xlab) 

        ax_ii += 1
        title_ii += 1
             
    plt.ylabel("Hamming distance")
    plt.xlabel("Sample Size")    
        
    fig.savefig(savepath)
    plt.show()  
    
    
def gen_knn_comparison_subplot(fontscale = 2, 
                             folder = "knn_est_test", 
                             methods = knn_comp_methods,
                             legend_names = knn_comp_methods_legend,
                             leg = False,
                             lw = 4,
                             savepath = "tex/ISIT/knn_comp.pdf"):
    
    linear = [True,False]
    noises = ["Gaussian", "t", "Uniform"]

    tests = list(product(linear, noises))
    
    #fig,axs = plt.subplots(nrows = 2, ncols = 3,sharex = True, sharey = True, figsize = (10,11) )
    fig,axs = plt.subplots(nrows = 2, ncols = 3,sharex = True, figsize = (12,6.5))
    axs = axs.flatten()
    
    ax_ii = 0

    for test in tests:
        
        ylab = None
        xlab = None
        
        if ax_ii == 0 or ax_ii == 3:
            ylab = "Hamming distance"
            
        if ax_ii >= 3:
            xlab = "Sample size"
               
        if test[0] == True:
            title = "Linear + "
        else:
            title = "Non-linear + "
        
        title = title + test[1]
        ax = axs[ax_ii]

        # one subplot
        load_and_plot_res(test,title,ax,
                          fontscale=fontscale,
                          folderName=folder,
                          methods=methods,
                          leg = leg,
                          linewidth = lw,
                          xlab = xlab,
                          ylab= ylab) 

        ax_ii += 1
                     
    fig.savefig(savepath)
    plt.show()  
    
    plotOnlyLegend(folderName = folder,
                   testName = test, 
                   font = fontscale, 
                   methods = methods,
                   methods_names_in_legend= legend_names,
                   saveloc = "tex/ISIT/legend2.pdf")
    
def load_and_plot_res(testName, title, ax, 
                      fontscale = 1.0, 
                      folderName = None, 
                      methods = None,
                      leg = False,
                      linewidth = 2,
                      ylab = None,
                      xlab = None):
    
    # load the right file
    if type(testName) == tuple: # convert test name to string
     
            if testName[0] == True:
                testNameString = "Linear"
            else:
                testNameString = "Nonlinear"
    
            testName = testNameString + testName[1].lower()

    if folderName is None:
        filee = "tests/" + testName + ".p"
    else:
        filee = "tests/" + folderName + "/" +testName + ".p"
        
    res,parameters = loadRes(filee)

    ns = parameters["ns"]
    print("k = ", parameters.get('k'))

    if "KCIT_AND" or "RCIT_AND" in methods: # add results for the kernel mehthods from different folder
        kernel_res,kernel_par = loadRes("tests/kernel_tests/" + testName + ".p")
        print("ntests (kernel) = ",len(kernel_res['RCIT_AND'][2000]['HD']))
        res.update(kernel_res)

    HDs = np.zeros((len(methods),len(ns)))
    SDs = np.zeros((len(methods),len(ns)))

    ii = 0
    
    for method in methods:
        
        jj = 0
        
        for n in ns:
            hds = res[method][n]["HD"]
            
            HDs[ii,jj] = np.mean(hds)
            SDs[ii,jj] = np.std(hds)/np.sqrt(len(hds)) # standard error of the mean
           
            jj += 1            
        
        ii += 1
    
    print("ntests = " ,len(hds))    
    
    x = range(0,len(ns)) 
    
    if methods == knn_comp_methods:
        linestyles = 4*["-"] + 4*["--"] 
        sns.set(style = 'ticks', palette = sns.color_palette("Set2", 4), font_scale= fontscale)

    else:
        linestyles = ['-', '--']
        linestyles = (int(len(methods)/len(linestyles)) +1 )*linestyles
        #palette = sns.color_palette("husl", 7)
        palette = sns.color_palette("Set2", 10)
        sns.set(style = 'ticks', palette = palette, font_scale= fontscale)
    sns.set_context(font_scale=fontscale)
    
    for i in range(0,len(methods)):
        lab = methods[i]
        ax.errorbar(x,HDs[i,:],SDs[i,:],ls = linestyles[i] ,label = lab, marker = 'o', linewidth = linewidth)
   
    ax.set_xticks(range(0,len(ns)))    
    ax.set_xticklabels(ns)    
    ax.grid()
    
    if xlab is not None:
        ax.set_xlabel(xlab)
        
    if ylab is not None:
        ax.set_ylabel(ylab)
    
    if title is not None:      
        ax.set_title(title)
               
    if leg is True:    
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':12})
        
    plt.tight_layout()
    
def plotOnlyLegend(folderName = "k5",
                   testName ="largeUG", 
                   font = 4,
                   lw = 12,
                   methods = None,
                   methods_names_in_legend = None,
                   saveloc = "tex/ISIT/legend.pdf"):
    # load the right file
    if type(testName) == tuple: # convert test name to string
     
            if testName[0] == True:
                testNameString = "Linear"
            else:
                testNameString = "Nonlinear"
    
            testName = testNameString + testName[1].lower()

    if folderName is None:
        filee = "tests/" + testName + ".p"
    else:
        filee = "tests/" + folderName + "/" +testName + ".p"
        
    res,parameters = loadRes(filee)
    
    if methods == knn_comp_methods:
        linestyles = 4*["-"] + 4*["--"] 
        sns.set(style = 'ticks', palette = sns.color_palette("Set2", 4), font_scale= font)
    else:
        linestyles = ['-', '--']
        linestyles = (int(len(methods)/len(linestyles)) +1 )*linestyles
        sns.set(style = 'ticks', palette = sns.color_palette("Set2", 10), font_scale= font)

        
    fig = plt.figure(figsize=(5,0.02))
    ax = plt.subplot(111)
    
    for i in range(0,len(methods)):
        #lab = methods[i]
        lab = methods_names_in_legend[i]
        ax.plot([],label = lab,linewidth = lw,ls = linestyles[i])
        
    
    ax.legend(bbox_to_anchor=(0., 0, 0, 0), loc=1,
              ncol = 4) 
    
    ax.set_axis_off()    
           
    fig.savefig(saveloc,bbox_inches='tight',pad_inches = 0, borderpad = 0.0)
    plt.show()
    
def knn_tp_fp(folder = "tests/knn_est_test", methods = None, printt = False):
    
    if methods is None:
        methods = []
        pars = loadRes(folder + "/Lineargaussian.p")
        all_methods = pars[1]["methods"]
        

        for m in all_methods:
            if m[-1] == "D": #take only the results corresponding AND rule for contructing the graphs
                methods.append(m)
        
                
                
    linear = [True,False]
    noises = ["Gaussian", "t", "Uniform"]

    tests = list(product(linear, noises))
    #nTests = len(tests)
    #ii = 1
    
    res_dict = {}

    for test in tests:
        
        if test[0] == True:
            title = "Linear + " + test[1]
            filename = "Linear" + test[1].lower()
        else:
            title = "Non-linear + " + test[1]
            filename = "Nonlinear" + test[1].lower()
            
        res,parameters = loadRes(folder + "/" + filename  + ".p")

        
        sample_sizes = parameters['ns']
         
        tp_all = np.zeros((len(sample_sizes),len(methods)))
        fp_all = np.zeros((len(sample_sizes),len(methods)))
        
        
        ntests = parameters['ntests']

        for tt in range(ntests):
            
            realUG = parameters['trueUGs'][tt]
            iii = 0
            
            for n in sample_sizes:
                jjj= 0
                
                for method in methods:
                    est_ug = res[method][n]['UG'][tt]
                    
                    tp,fp = tp_fp(realUG,est_ug)      
                    
                    tp_all[iii,jjj] += tp/ntests
                    fp_all[iii,jjj] += fp/ntests
                    
                    jjj += 1
                    
                iii += 1
                
              
                
        tp_df = pd.DataFrame(tp_all, columns = methods, index = sample_sizes)
        fp_df = pd.DataFrame(fp_all, columns = methods, index = sample_sizes)
        res_dict[title] = (tp_df,fp_df)  
        
        if printt:
            print(title)
            print(tp_df)
            print(fp_df)
        
    return res_dict