# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:36:51 2020

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

def icaML(X, K = None, par = [1,  1e-4,  1e-8,  1000], debug_draw = 0):
# icaML     : ICA by ML (Infomax) with square mixing matrix and no noise.
#
# function [S,A,U,ll,info]=icaML(X,[K],[par]) Independent component analysis (ICA) using
#                                       maximum likelihood, square mixing matrix and
#                                       no noise [1] (Infomax). Source prior is assumed
#                                       to be p(s)=1/pi*exp(-ln(cosh(s))). For optimization
#                                       the BFGS algorithm is used [2]. See code for
#                                       references.
#
#                                       X  : Mixed signals
#                                       K  : Number of source components.
#                                           For K=0 (default) number of sources are equal to number of
#                                           observations.
#                                           For K < number of observations, SVD is used to reduce the
#                                           dimension.
#                                       par:  Vector with 4 elements:
#                                           (1)  :  Expected length of initial step
#                                           Stopping criteria:
#                                           (2)  :  Gradient  ||g||_inf <= par(2)
#                                           (3)  :  Parameter changes ||dW||_2  <= par(3)*(par(3) + ||W||_2)
#                                           (4)  :  Maximum number of iterations
#                                           Any illegal element in  opts  is replaced by its
#                                           default value,  [1  1e-4*||g(x0)||_inf  1e-8  100]
#                                       debug_draw : Draw debug information
#
#                                       S  : Estimated source signals with variance
#                                            scaled to one.
#                                       A  : Estimated mixing matrix
#                                       U  : Principal directions of preprocessing PCA.
#                                            If K (the number of sources) is equal to the number
#                                            of observations then no PCA is performed and U=eye(K).
#                                       ll : Log likelihood for estimated sources
#                                       info :  Performance information, vector with 6 elements:
#                                           (1:3)  : final values of [ll  ||g||_inf  ||dx||_2]
#                                           (4:5)  : no. of iteration steps and evaluations of (ll,g)
#                                           (6)    : 1 means stopped by small gradient
#                                                    2 means stopped by small x-step
#                                                    3 means stopped by max number of iterations.
#                                                    4 means stopped by zero step.
#
#
#                                       Ex. Separate with number of sources equal number of
#                                       observations.
#
#                                               [S,A] = icaML(X);
#
#                                       Ex. Separate with number of sources equal to k, using SVD
#                                       as pre-processing.
#
#                                               [S,A,U] = icaML(X,k);
#
# - version 1.6 (Revised 5/12-2019)
# - version 1.5 (Revised 9/9-2003)
# - IMM, Technical University of Denmark
# - version 1.4
# - by Thomas Kolenda 2002 - IMM, Technical University of Denmark
# Revised: 5/12-2019, Alma Lindborg, allin@dtu.dk
#   * Log-likelihood of noise included in output variable ll for the
#     non-quadratic case
#   * code revised for improved readability
# Revised: 9/9-2003, Mads, mad@imm.dtu.dk
#   * Fixed help message to inform the user about U.
#   * Removed the automatic use of PCA in the quadratic case.
# Bibtex references:
# [1]
#   @article{Bell95,
#       author =       "A. Bell and T.J. Sejnowski",
#       title =        "An Information-Maximization Approach to Blind Separation and Blind Deconvolution",
#       journal =      "Neural Computation",
#       year =         "1995",
#       volume =       "7",
#       pages =        "1129-1159",
#   }
#
# [2]
#   @techreport{Nielsen01:unopt,
#       author =        "H.B. Nielsen",
#       title =         "UCMINF - an Algorithm for Unconstrained, Nonlinear Optimization ",
#       institution =   "IMM, Technical University of Denmark",
#       number =        "IMM-TEC-0019",
#       year =          "2001",
#       url =           "http://www.imm.dtu.dk/pubdb/views/edoc_download.php/642/ps/imm642.ps",
#   }
# Translated to python by Lukasz Bienias 2020  - DTU Compute, Technical University of Denmark    
    
    # Objectve function to be minimized:
    def objfun(theta):
        # returns the negative log likelihood and its gradient w.r.t. W
        X = X_obj 
        M = M_obj 
        N = N_obj 

        W = np.reshape(theta, (M,M))
        S = np.dot(W, X)

        f = -( N * np.log(np.abs(np.linalg.det(W))) - np.sum(np.sum(np.log(np.cosh(S)), axis =0), axis =0) - N*M*np.log(pi) )
        return(f)

    # gradient of the objective function
    def objfun_der(theta):
        X = X_obj 
        M = M_obj 
        N = N_obj

        W = np.reshape(theta, (M,M))

        S = np.dot(W, X)

        dW = -(N * np.linalg.inv(np.transpose(W)) - np.dot(np.tanh(S), np.transpose(X)))
        dW = np.ndarray.flatten(dW)
        return(dW)


    MaxNrIte = par[3]
    debug = debug_draw
    
    if debug == 1:
        print('\n** Start icaML ***************************************\n')
        
    # Scale X to avoid numerical problems
    Xorig = X
    scaleX = np.max(np.abs(X[:]))

    X = X/scaleX
    M, N = np.shape(X)

    # set number of source parameters
    if K == None:
        K = M
    
    # Reduce dimension with PCA
    if N>M: # Transpose if matrix is flat to speed up the svd and later transpose back again; REMARK: I have converted it, so case with transpose should be checked
        if debug == 1:
            print('Transpose SVD')
        _, _, U = np.linalg.svd(np.transpose(X)) 
        U = np.transpose(U)
    else:
        if draw == 1:
            print('SVD')
        U, _, _ = np.linalg.svd(X) 

    # project into PCA space
    X = np.matmul(np.transpose(U), X) 
    
    if K > 0: # reduce number of dimensions
        X = X[:K,:]
    
    # initialize optimize parameters
    ucminf_opt = par
    
    # initialize variables
    D, N = np.shape(X)

    W = np.eye(D)

    if debug == 1:
        print('Number of samples %d - Number of sources %d\n' %(N,D))
        
    # optimize
    if debug == 1:
        print('Optimize ICA ... ')

#     par = {}
#     par['X'] = X
#     par['M'] = D
#     par['N'] = N
    
    X_obj = X
    M_obj = D
    N_obj = N

    # Initial value of W
    theta0 = W

    # BFGS method for unconstrained nonlinear optimization
    W = fmin_bfgs(objfun, theta0, fprime=objfun_der, disp = 0)
    
    W = np.reshape(W, (D,D))
    
    # estimates
    A = np.linalg.pinv(W)
    S = np.dot(W, X)
    
    if debug == 1:
        print('done optimizing ICA!')
        
    #  sort components according to energy
    Avar = np.diag(np.dot(np.transpose(A),A))/D
    Svar = np.diag(np.dot(S, np.transpose(S)))/N
    sig = np.multiply(Avar, Svar)

    indx = np.argsort(sig)
    S = S[np.flip(indx, 0),:]
    A = A[:,np.flip(indx, 0)]
    
    # scale back
    A = np.multiply(A, scaleX)
    
    # log likelihood
    ll = 0
    if K > 0:
        ll = ll - N * np.log(np.abs(np.linalg.det(A))) - np.sum(np.sum(np.log(np.cosh(S)), axis =0), axis =0) - N*K*np.log(pi)

    # log-likelihood of non-included PCA space
    if K < M:
        X_noise = np.dot(np.transpose(U), Xorig)
        X_noise = X_noise[K:,:]
        tr_cov = np.trace(np.dot(X_noise, np.transpose(X_noise)))
        ll = ll - 0.5 * (M - K) * N * (np.log(tr_cov) + 1 + np.log(2 * pi) - np.log(N * (M-K)))
        
    if debug == 1:
        print('** End of icaML ************************\n\n')

    return (S, A, U, ll)