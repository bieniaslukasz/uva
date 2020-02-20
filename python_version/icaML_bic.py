# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:35:58 2020

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

from icaML import icaML

def icaML_bic(X, K = np.arange(2,11), draw = 0):
# icaML_bic : ICA by ML with square mixing matrix and no noise.
#
# function [P,logP]=icaML_bic(X,[K],[draw])     Use Bayes Information Criterion (BIC) [1] to
#                                               estimate number of components with the maximum
#                                               likelihood ICA (icaML) algorithm. SVD is used for
#                                               dimension reduction.
#
#                                               X     : Mixed signal matrix [M x N] or a struct
#                                                       holding the solution to the [U,S,V]=SVD(X),
#                                                       thus X.U=U, X.S=S and X.V=V.
#                                               K     : Vector or scalar holding the number of
#                                                       IC components to investigate. Values
#                                                       must be greater than 1. Default K=[2:10].
#                                               draw  : Output run-time information if draw=1.
#                                                       Default draw=0.
#
#                                               P     : Normalized model probability.
#                                               logP  : Model negative log probability.
# - Version 1.3 (Revised 5/12-2019)
# - version 1.2a
# - by Thomas Kolenda 2002 - IMM, Technical University of Denmark
# Revised: 5/12-2019, Alma Lindborg, allin@dtu.dk
# translated to python by Lukasz Bienias 2020  - DTU Compute, Technical University of Denmark

#   * k = 0 now a possible choice
#   * compatible with updated icaML.m (version 1.6)

# Bibtex references:
# [1]
#  @article{Hansen2001BIC,
#       author =       "Hansen, L. K. and Larsen, J. and Kolenda, T.",
#       title =        "Blind Detection of Independent Dynamic Components",
#       journal =      "In proc. IEEE ICASSP'2001",
#       volume =       "5",
#       pages =        "3197--3200",
#       year =         "2001",
#       url =          "http://www.imm.dtu.dk/pubdb/views/edoc_download.php/827/pdf/imm827.pdf",
#  }

    if type(X) is not dict:
        # Reduce dimension with PCA
        M, N = np.shape(X)
        if N>M: # Transpose if matrix is flat to speed up the svd and later transpose back again
            if draw == 1:
                print('Transpose SVD')
            _, _, U = np.linalg.svd(np.transpose(X))  
            U = np.transpose(U)
        else:
            if draw == 1:
                print('SVD')
            U, _, _ = np.linalg.svd(X) 

        X = np.matmul(np.transpose(U), X)
    else: # use pre-computed SVD decomposition
        U = X['U']
        X = X['SV']

    M, N = np.shape(X)

    if draw == 1:
        print('M=%d N=%d' %(M,N)) 
        
    index = -1
    logP = np.zeros(len(K))

    for k in K:
        index = index + 1
        if draw == 1:
            print('dim=%d' %(k), end = ' ')
        
        # estimate ICA
        if k == 0:
            _, _, _,ll = icaML(X,k)
        else: # avoid doing SVD for each k - treat as quadratic
            _, _, _,ll = icaML(X[:k,:])
            if k < M: # noise space log-likelihood has to be added
                X_noise = X[k:,:]
                tr_cov = np.trace(np.dot(X_noise, np.transpose(X_noise)))
                ll = ll - 0.5*(M-k)*N*(np.log(tr_cov)+1+np.log(2*pi)-np.log(N*(M-k)))
        
        # add bic term to log-likelihood
        dim = k * (2 * M - k + 1)/2 + 1 + k * k # number of parameters to estimate
        ll = ll - 0.5 * dim * np.log(N)
        logP[index] = ll
    
    # Normalize
    P = np.exp((logP - np.max(logP)) / N )
    P = P / np.sum(P)
    
#     # display a bar plot with probabilities of having k number of sources:
#     plt.figure(figsize=(10,10))
#     plt.bar(K,P)
#     plt.ylabel('P(K)')
#     plt.xlabel('K')
#     plt.show()
            
    return (P,logP)