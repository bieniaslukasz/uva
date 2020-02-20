# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:37:55 2020

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

def icaML_traintest(X, Xtest, K, par = [1,  1e-4,  1e-8,  1000], debug_draw = 0):
# ICAML_TRAINTEST     : ICA by ML (Infomax) with square mixing matrix and no noise.
# function [S,St,A,ll,ll_test,info]=icaML_traintest(X,[K],[par]) Independent component analysis (ICA) using
#                                       maximum likelihood, square mixing matrix and
#                                       no noise [1] (Infomax). Source prior is assumed
#                                       to be p(s)=1/pi*exp(-ln(cosh(s))). For optimization
#                                       the BFGS algorithm is used [2]. See code for
#                                       references. Predictions and
#                                       log-likelihood for a test set are
#                                       also calculated.
#
#                                       X  : Mixed signals
#                                       K  : Number of source components.
#                                           For K=M (default) number of sources are equal to number of
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
#                                       St : Estimated sorce signals on
#                                       test set
#                                       A  : Estimated mixing matrix
#                                       U  : Principal directions of preprocessing PCA.
#                                            If K (the number of sources) is equal to the number
#                                            of observations then no PCA is performed and U=eye(K).
#                                       ll : Log likelihood for estimated sources
#                                       ll_test: Log likelihood for test
#                                       set estimations
#                                       info :  Performance information, vector with 6 elements:
#                                           (1:3)  : final values of [ll  ||g||_inf  ||dx||_2]
#                                           (4:5)  : no. of iteration steps and evaluations of (ll,g)
#                                           (6)    : 1 means stopped by small gradient
#                                                    2 means stopped by small x-step
#                                                    3 means stopped by max number of iterations.
#                                                    4 means stopped by zero step.
#
# original (matlab) version by Alma Lindborg 2019 - DTU Compute, Technical University of Denmark
# based on icaML.m function by Thomas Kolenda
# translated to python by Lukasz Bienias 2020  - DTU Compute, Technical University of Denmark

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

    # set number of source parameters        
    if type(X) is dict: # if SVD is already performed
        U = X['U']
        X = X['SV']
        M, N = np.shape(X)
        if K == None:
            K = M     
    else:        
        M, N = np.shape(X)
        if K == None:
            K = M         
        if K > 0 and K < M:
            if N > M: # Transpose if matrix is flat to speed up the svd and later transpose back again
                _, _, U = np.linalg.svd(np.transpose(X))  
                U = np.transpose(U)
            else:
                U, _, _ = np.linalg.svd(X)    
        else:
            U = np.eye(np.shape(X)[0]) # Don't rotate
            print("Ten nieciekawy moment")
            if debug == 1:
                print('Don''t use SVD')
                
        # project into PCA space
        X = np.matmul(np.transpose(U), X)

    # Scale X to avoid numerical problems
    Xpca = X
    scaleX = np.max(np.abs(X[:]))
    X = X/scaleX
    
    if K > 0:
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
        X_noise = Xpca[K:,:]
        tr_cov = np.trace(np.dot(X_noise, np.transpose(X_noise)))
        ll = ll + 0.5 * (M - K) * N * (np.log(N * (M - K)) - np.log(2 * pi) - np.log(tr_cov) - 1)

        
    ### TEST SET CALCULATIONS
    Xt_pca = np.matmul(np.transpose(U), Xtest)

    # test error
    Ntst = np.shape(Xtest)[1]
    ll_test = 0
        
    St = []

    # predictions on test set & log-likelihood of signal space
    if K > 0:
        St = np.dot(W, (Xt_pca[:K,:] / scaleX)) # scale in same way as training set
        ll_test = ll_test - Ntst * np.log(np.abs(np.linalg.det(A))) - np.sum(np.sum(np.log(np.cosh(St)))) - Ntst * K * np.log(pi)
    
    # log-likelihood of non-included PCA space
    if K < M:
        Xt_noise = Xt_pca[K:,:]
        tst_cov = np.trace(np.dot(Xt_noise, np.transpose(Xt_noise)))
        ll_test = ll_test + 0.5 * (M - K) * (Ntst * (np.log(N * (M-K)) - np.log(2 * pi) - np.log(tr_cov)) - N * tst_cov / tr_cov)

    if K > 0:
        A = np.dot(U[:,:K], A) # project A back to data space

    if debug == 1:
        print('** End of icaML ************************\n\n')                    
    
    return(S,St,A,ll,ll_test)