# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:43:37 2020

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


def generate_dim(K, M, snr, nTr, nTst, seed):
    #GENERATE_DIMENSIONAL Generate dimensional Independent Component data from Laplace distribution
    #Inputs:
    #   K: number of source components
    #   M: number of dimensions in data
    #   snr: signal-to-noise ratio (var(signal)/var(noise))
    #   Ntr: number of samples in training set
    #   Ntst: number of samples in test set
    #   seed: random seed number
    #Outputs:
    #   Xtrain: Training set
    #   Xtest: Test set
    #   Atrue: Mixing matrix
    #   Strue: Source signals
    #
    # original (matlab) version by Alma Lindborg 2019 - DTU Compute, Technical University of Denmark
    # translated to python by Lukasz Bienias 2020  - DTU Compute, Technical University of Denmark
    
    # ----------------- VERSION CREATED IN PURPOSE OF HAVING SAME NUMBERS AS MATLAB HAS -----------------
    # # Generate mixed signals from artificial super Gaussian sources and mixing matrix
    # Atrue = np.random.randn(M, K)

    # Generate mixed signals from artificial super Gaussian sources and mixing matrix
    Atrue = np.random.randn(M, K)

    # draw Strue from Laplace distribution
    Strue = np.multiply(np.sign(np.random.rand(K, nTr) - 0.5), np.random.exponential(1, (K, nTr)))
    Xtrain = np.dot(Atrue, Strue)
    
    # test set
    Stest = np.multiply(np.sign(np.random.rand(K, nTst) - 0.5), np.random.exponential(1, (K,nTst)))
    Xtest = np.dot(Atrue, Stest)

    # variance of training and test set (combined)
    var_x = np.var(np.concatenate((Xtrain, Xtest),axis=1))

    # add Gaussian noise to data
    # (scale by sqrt(var_x/snr) to get noise variance = var_x/snr)
    Xtrain = Xtrain + np.sqrt(var_x/snr) * np.random.randn(np.shape(Xtrain)[0], np.shape(Xtrain)[1])
    Xtest = Xtest + np.sqrt(var_x/snr) * np.random.randn(np.shape(Xtest)[0], np.shape(Xtest)[1])

    # scale s.t. sdev = 1
    Xtrain = Xtrain / np.std(np.concatenate((Xtrain, Xtest),axis=1))
    Xtest = Xtest / np.std(np.concatenate((Xtrain, Xtest),axis=1))

    return(Xtrain, Xtest, Atrue, Strue)
