# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:33:34 2020

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

def generate_gmix(K, M, snr, nTr, nTst, seed):
#  GENERATE GMIX: Generate a training and a test set from a Gaussian Mixture
#  model. Components have zero mean and one main direction of variance.
#    INPUTS:
#        K: Number of Gaussian components
#        M: Number of dimensions in data
#        nTr: Number of samples in training set
#        nTst: Number of samples in test set
#        snr: Signal-to-noise ratio
#        seed: random seed
# 
#    OUTPUTS:
#        Xtr: Training set (Dimensions in rows, observations in columns)
#        Xtst: Test set
#        mus: mean of each Gaussian component
#        Sigmas: covariance matrix of each Gaussian component
#        probs: mixing probability of each component
#
# original (matlab) version by Alma Lindborg 2019 - DTU Compute, Technical University of Denmark
# translated to python by Lukasz Bienias 2020  - DTU Compute, Technical University of Denmark
    tr_smpls = np.zeros((K, nTr, M))
    tst_smpls = np.zeros((K,nTst,M))
        
    np.random.seed(seed) # fix random number generator seed for replicability
    
    mus = []
    Sigmas = []
    # generate and sample from each component
    for i in range(K):
        # generate a random covariance matrix with a main direction of variance
        eigs = np.ones(M,)
        eigs[0] = 100 * eigs[0]

        L = np.diag(eigs)
        
        A = np.random.randn(M, M)
        Q, R = np.linalg.qr(A)

        # Solve systems of linear equations xA = B for x, in matlab it is enough to use '/'
        S = np.dot(np.dot(Q, L), np.linalg.pinv(Q))  # https://stackoverflow.com/questions/1007442/mrdivide-function-in-matlab-what-is-it-doing-and-how-can-i-do-it-in-python
        S = (S + np.transpose(S)) / 2             # compensate for potential round-off errors making S non-symmetric

        mu = np.zeros(M,)                         # fixed mean = 0

        # draw samples
        tr_smpls[i, :, :] = np.random.multivariate_normal(mu, S, nTr)
        tst_smpls[i, :, :] = np.random.multivariate_normal(mu, S, nTst);

        mus.append(mu)
        Sigmas.append(S)

    # Generate mixing probabilities
    if K == 1:
        p1 = np.random.beta(2, 2) # draw mixing proportions from beta distribution
        probs = np.array([p1, 1-p1])
    else: # use softmax
        probs = np.random.rand(K, 1)
#         probs = np.random.uniform(low=0.0, high=1.0, size = (K, 1))
        probs = np.transpose(softmax(probs))

    pcum = np.cumsum(probs) # cumulative probability
    
    # sample coefficients for training set match to corresponding cluster
    coeffs = np.random.rand(nTr, 1)
    bin_matrix_1 = (np.repeat(coeffs, K, axis = 1) > np.array([np.concatenate((np.array([0]), pcum[:-1])),]*nTr)  ).astype(int) 
    bin_matrix_2 = (np.repeat(coeffs, K, axis = 1) < np.array([pcum,]*nTr)    ).astype(int) 
    
    trnCluster = np.multiply(bin_matrix_1, bin_matrix_2)   
    
    # ----------------------------------- do the same for the test set --------------------------
    coeffs_tst = np.random.rand(nTst, 1)
    bin_matrix_tst_1 = (np.repeat(coeffs_tst, K, axis = 1) > np.array([np.concatenate((np.array([0]), pcum[:-1])),]*nTst)  ).astype(int) 
    bin_matrix_tst_2 = (np.repeat(coeffs_tst, K, axis = 1) < np.array([pcum,]*nTst)  ).astype(int) 

    tstCluster = np.multiply(bin_matrix_tst_1, bin_matrix_tst_2)    
    # -------------------------------------------------------------------------------------------
    
    Xtr = np.zeros((nTr,M))
    Xtst = np.zeros((nTst,M))

    # Pick samples from either component with the generated probabilities
    for i in range(K):
        Xtr = Xtr + np.multiply(np.transpose(np.array([trnCluster[:,i]])), np.squeeze(tr_smpls[i,:,:]))

        # test set
        Xtst = Xtst + (np.tile(np.multiply(np.transpose(np.array([tstCluster[:,i]])), np.squeeze(tst_smpls[i,:,:])), (M,1)))

    # transpose data matrices if there's only one training example (they get
    # flipped by squeeze)
    if nTr == 1:
        Xtr = np.transpose(Xtr)

    if nTst == 1:
        Xtst = np.transpose(Xtst)
    
    # transpose both data sets s.t. observations are in columns
    Xtr = np.transpose(Xtr)
    # Xtst = np.transpose(Xtst) # seems like we do not need it here    
    
    # variance of training and test set (combined)
    var_x = np.var(np.concatenate((Xtr, Xtst),axis=1)) 
    
    # add Gaussian noise to data
    # (scale by sqrt(var_x/snr) to get noise variance = var_x/snr)
    Xtr = Xtr + np.sqrt(var_x/snr) * np.random.randn(np.shape(Xtr)[0], np.shape(Xtr)[1])
    Xtst = Xtst + np.sqrt(var_x/snr) * np.random.randn(np.shape(Xtst)[0], np.shape(Xtst)[1])    
    
    # scale s.t. sdev = 1
    Xtr = Xtr / np.std(np.concatenate((Xtr, Xtst),axis=1))
    Xtst = Xtst / np.std(np.concatenate((Xtr, Xtst),axis=1))  


    return(Xtr, Xtst, mus, Sigmas, probs)