# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:40:11 2020

@author: lutobi
"""
import numpy as np
from scipy import linalg
from scipy.special import softmax
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fmin_bfgs
from math import pi
from sklearn import mixture
from datetime import datetime

from icaML_bic import icaML_bic
from icaML_traintest import icaML_traintest
from plot_learning_curves import plot_learning_curves

def select_model(X, k_int = np.arange(1,6), nRuns = 10, nFolds = 100, pct_bic = 10, lcurve_len = 5, prerun_SVD = 0, do_plots = 1):
# SELECT_MODEL    : Selects between a dimensional and a cluster model.
# function [mod_probs, negll] = select_model(X,varargin) performs model selection on a dataset by training a Maximum-Likelihood
#                   ICA model and a Gaussian Mixture model and comparing
#                   their cross-validation lI just spog-likelihoods. The number
#                   of components or dimensions is determined separately
#                   for each model by the Bayesian Information Criterion
#                   (BIC).
#
#                 INPUTS
#                   X       : Matrix containing observations as columns and
#                   dimensions as rows.
#
#                 Optional inputs:
#                   k_int   :  interval of k (number of dimensions/components)
#                              to be tested (default: 1:5)
#                   nRuns   :  number of runs to determine k (default: 10)
#                   nFolds  : number of cross-validation folds (default: 100)
#                   pct_bic : per cent of data used for computing BIC for
#                   component optimization (default: 10)
#                   lcurve_len: number of points in learning curve (default: 5)
#                   prerun_SVD: set to 1 if ICA should be calculated on a
#                   prerun SVD. Preferred option for speed, but under-estimates
#                               CV error at lower training set sizes. (default: 0)
#                   do_plots: 0: do not plot training output. 1: (default)
#                                plot training output
#
#                OUTPUTS
#                   mod_probs   : proportion of cross-validation folds where each
#                                 model was selected for the biggest training set size.
#                   info        :    struct containing number of components, negative
#                   log-likelihood for each cross-validation fold, training
#                   set sizes and number of parameters
#
# original (matlab) version by Alma Lindborg 2019 - DTU Compute, Technical University of Denmark
# translated to python by Lukasz Bienias 2020  - DTU Compute, Technical University of Denmark
    print("Model selection started")
    # normalize X
    X = X - np.mean(X)
    X = X/np.std(X)

    # make sure X has dimensions as rows and observations as columns
    M, N = np.shape(X)
    if M > N:
        print('More dimensions than data points. Is data input in the right format?')
        print('Transposing data matrix...')
        X = np.transpose(X)
        M, N = np.shape(X)
        
    models = ['ICA', 'GMM'] # models that will be used

    # optimization options for GMM
    MaxIter = 1000

    # pre-compute SVD if desired
    if prerun_SVD:
        print("Doing prerun SVD")
        Xpca = {}
        _, _, U = np.linalg.svd(np.transpose(X)) # should be u, s, vh = np.linalg.svd(np.transpose(X)), according to definition
        U = np.transpose(U)
        Xpca['U'] = U
        Xpca['SV'] = np.matmul(np.transpose(U), X)   

    ### DETERMINE NUMBER OF DIMENSIONS
    print('***COMPONENT OPTIMIZATION*** \n')
    mpks = {}
    components = {}

    for iM in range(len(models)):
        md = models[iM]

        # train on pct_bic percent of the data
        nTr = round(N*pct_bic/100)

        mpks[md] = np.zeros(nRuns,)
        print('Optimizing %s on %i data points in %i runs \n' %(md, nTr, nRuns))
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Started %s optimisation " %(md))
        print( current_time)
        
        for jR in range(nRuns):
            print('Started %i run \n' %(jR ));

            # randomly select Ntr training samples from X
            idcs = np.random.permutation(N)     

            if(md == 'ICA'): # train ICA
                if(prerun_SVD):
                    Xtr = Xpca
                    Xtr['SV'] = Xtr['SV'][:,idcs[:nTr]]
                else:
                    Xtr = X[:,idcs[:nTr]]

                P, _ = icaML_bic(Xtr, k_int, draw = 0)
                idx = np.argmax(P)
                mpk = k_int[idx]

            else: # train GMM
                Xtr = X[:,idcs[:nTr]]
                bics = np.zeros(len(k_int),)
                for iK in range(len(k_int)):
                    misses = 0
                    while 1:
                        try:
                            # Fit a Gaussian mixture with EM 
                            GMModel = mixture.GaussianMixture(n_components = k_int[iK], max_iter = MaxIter).fit(np.transpose(Xtr))
                            bics[iK] = GMModel.bic(np.transpose(Xtr))
                            break
                        except:
                            misses = misses + 1
                            if(misses > 1000): # we're stuck - get out of infinite loop
                                print('Cannot fit Gaussian mixture model due to ill-conditioned covariance matrix. Try to increase pct_bic or run PCA prior to training.')

                idx = np.argmin(bics)
                mpk = k_int[idx]

            mpks[md][jR] = mpk

        print('%s is done \n' %(md))
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Finished %s optimisation " %(md))
        print( current_time)
        
        
        # use the number of components that was chosen most often
        mpks[md] = mpks[md].astype(int)
        counts = np.bincount(mpks[md])
        components[md] = np.argmax(counts)

    print('#GMM components: %i \n' %(components['GMM']))
    print('#ICA components: %i \n' %(components['ICA']))            

    # Calculate number of parameters for each model
    npar = {}
    npar['GMM'] = int(components['GMM'] + components['GMM'] * M * (M + 1) / 2 + components['GMM'] - 1)
    npar['ICA'] = int(components['ICA'] * (2 * M - components['ICA'] + 1) / 2 + 1 + np.square(components['ICA']))

    logn = np.log10(N)
    lognpar = np.log10(max(npar['GMM'], npar['ICA']))
    if lognpar > logn:
        print('More parameters than training examples!')

    #  make an array of logarithmically spaced training set sizes
    tr_sizes = np.round(np.logspace(lognpar,logn,lcurve_len+1))
    #  exclude end points (dim < nTr < n)
    tr_sizes = tr_sizes[:-1]

    ### LEARNING CURVE
    # initialize
    tr_error = {}
    tst_error = {}

    tr_error['ICA'] = np.zeros((nFolds,len(tr_sizes)))
    tr_error['GMM'] = np.zeros((nFolds,len(tr_sizes)))
    tst_error['ICA'] = np.zeros((nFolds,len(tr_sizes)))
    tst_error['GMM'] = np.zeros((nFolds,len(tr_sizes)))

    print('***LEARNING CURVES*** \n')
    print('Estimating on %d CV folds. \n' %(nFolds))
    
    for iN in range(len(tr_sizes)):
        nTr = int(tr_sizes[iN])
        print('Training set size: %i \n' %nTr)
        print('current fold: ', end = '')

        for jF in range(nFolds):
            if not (jF % 10):
                print('%i ' %jF, end = ' ')

            # randomly split into training and validation set
            prm = np.random.permutation(N)

            # Train ICA
            if prerun_SVD:
                print("This needs to be coded afterwards!")
                Xtr = Xpca
                Xtr['SV'] = Xtr['SV'][:,prm[:nTr]]
            else:
                Xtr = X[:,prm[:nTr]]

            Xtst = X[:,prm[nTr:]]
            _, _, _, ll, ll_tst = icaML_traintest(Xtr, Xtst, components['ICA'])

            nTst = np.size(Xtst,1) 
            tr_error['ICA'][jF,iN] = -ll / nTr
            tst_error['ICA'][jF,iN] = -ll_tst / nTst

            # Train GMM
            Xtr = X[:,prm[:nTr]]
            Xtst = X[:,prm[nTr:]]
            nTst = np.size(Xtst,1) 

            misses = 0 # counter for missed optimization attempts
            while 1:
                try:
                    GMModel = mixture.GaussianMixture(n_components = components['GMM'], max_iter = MaxIter).fit(np.transpose(Xtr))
                    negll_tr = -GMModel.score(np.transpose(Xtr))
                    negll_tst = -GMModel.score(np.transpose(Xtst))
                    break
                except:
                    misses = misses + 1
                    if(misses > 1000): # we're stuck - get out of infinite loop
                        print('Cannot fit Gaussian mixture model due to ill-conditioned covariance matrix. You may have to use PCA to decrease the data dimension.')

            tr_error['GMM'][jF,iN] = negll_tr
            tst_error['GMM'][jF,iN] = negll_tst
        print('\n')
    print('...Done. \nTraining finished. \n')

    ### MODEL SELECTION
    mod_probs = {}

    prop_ica = np.sum(tst_error['ICA'] < tst_error['GMM'], axis = 0)/np.shape(tst_error['ICA'])[0]
    mod_probs['ICA'] = prop_ica[-1]
    mod_probs['GMM'] = 1-prop_ica[-1]

    info = {}
    info['cv_negll'] = tst_error
    info['ncomponents'] = components
    info['prop_ica'] = prop_ica
    info['train_sizes'] = tr_sizes
    info['nFolds'] = nFolds
    info['npar'] = npar    

    ### PLOT RESULTS
    if do_plots:
        # Fig1: Histogram of k selection
        def bins_labels(bins, **kwargs): # taken from: https://stackoverflow.com/questions/23246125/how-to-center-labels-in-histogram-plot
            bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
            plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
            plt.xlim(bins[0], bins[-1])
            
        plt.figure(figsize=(14,6))
        for i in range(len(models)):       
            plt.subplot(1,len(models),i+1)
            
            bins = range(0, max(mpks[models[i]]) + 2)
            plt.hist(mpks[models[i]], bins=bins)
            bins_labels(bins, fontsize=15)
            
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value',fontsize=15)
            plt.ylabel('Frequency',fontsize=15)
            plt.yticks(fontsize=15)
            plt.title('# components for %s' %(models[i]),fontsize=15)
        
        plt.show()    

        # Fig2: Learning curves
        plot_learning_curves(info)
    
    return(mod_probs, info)