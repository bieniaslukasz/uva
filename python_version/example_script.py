# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:39:50 2020

@author: lutobi
"""
from generate_dim import generate_dim
from generate_gmix import generate_gmix
from select_model import select_model

import numpy as np
from scipy import linalg
from scipy.special import softmax
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fmin_bfgs
from math import pi
from sklearn import mixture
import sys

def main(gen, K, M, snr, N, nTst, seed):
    print("Example script is running.")
    
    if gen == "dim":
        X, _, _, _ = generate_dim(K, M, snr, N, nTst, seed)
    elif gen == "gmix":
        X, _, _, _, _ = generate_gmix(K, M, snr, N, nTst, seed)
    else:
        print("Data cannot be generated, specified type of data is not recognised.")
        
    mod_probs, info = select_model(X)

if __name__ == '__main__':
    if sys.argv[1:]:
        gen = str(sys.argv[1]) # type of data generated
        K = int(sys.argv[2]) # number of underlying sources/components
        M = int(sys.argv[3]) # dimension of data
        snr = int(sys.argv[4]) # signal-to-noise ratio
        N = int(sys.argv[5]) # number of training samples, N = 1000
        nTst = int(sys.argv[6]) # Number of samples in test set
        seed = int(sys.argv[7]) # random seed
    else:
        print("Default arguments used.")
        gen = "dim"
        K = 2      # number of underlying sources/components
        M = 10     # dimension of data
        snr = 2    # signal-to-noise ratio
        N = 1000   # number of training samples, N = 1000
        nTst = 1   # Number of samples in test set
        seed = 1   # random seed
    print("Type of data generated: ", gen)
    print("Number of underlying sources/components: ", K)
    print("Dimension of data: ", M)
    print("Signal-to-noise ratio: ", snr)
    print("Number of training samples: ", N)
    print("Number of samples in test set: ", nTst)
    print("Random seed: ", seed)

    main(gen, K, M, snr, N, nTst, seed)