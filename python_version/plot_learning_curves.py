# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:39:32 2020

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

def plot_learning_curves(*argv):
# PLOT_LEARNING_CURVES: Plot learning curves for two competing models.
#
# function plot_learning_curves(info) 
#
#                             info:   struct from function output of 
#                             select_model.m
#
# function plot_learning_curves(cv_negll, tr_sizes)
#                             cv_negll:   struct with 2 fields (one per 
#                             model), each storing a 
#                             nFolds x length(tr_sizes) matrix of cross-
#                             validation errors.
#                             
#                             tr_sizes:   vector containing the training
#                             set size corresponding to each column of
#                             errors in cv_negll.
#
# original (matlab) version by Alma Lindborg 2019 - DTU Compute, Technical University of Denmark
# translated to python by Lukasz Bienias 2020  - DTU Compute, Technical University of Denmark
    if len(argv) == 1:
        info = argv[0]
        cv_negll = info['cv_negll']
        models = list(cv_negll.keys())
        tr_sizes = info['train_sizes']
        prop_ica = info['prop_ica']
    else:
        cv_negll = argv[0]
        tr_sizes = argv[1]
        models = list(cv_negll.keys())
        prop_ica = np.sum(cv_negll[models[0]] < cv_negll[models[1]]) / np.shape(cv_negll[models[0]])[0]
        
    plt.figure(figsize=(14,10))
    
    # Subplot 1: Learning curve 
    plt.subplot(len(models), 1, 1)
    colors = ['C0', 'C1']
    for iM in range(len(models)):
        md = models[iM]
        plt.semilogx(tr_sizes, np.mean(cv_negll[md], axis = 0))
        plt.errorbar(tr_sizes, np.mean(cv_negll[md], axis = 0), np.std(cv_negll[md], axis = 0) / np.sqrt(np.shape(cv_negll[md])[0]),  label = md, color = colors[iM], ecolor = 'red', barsabove = True, capsize = 8.0)
    
    plt.title('Learning curve')
    plt.xlim(tr_sizes[0] * 0.9, tr_sizes[-1] * 1.1)
    plt.xlabel('Training set size')
    plt.ylabel('cross-validation error')
    plt.legend(loc="upper right")
    
    # Subplot 2: Model selections
    plt.subplot(len(models), 1, 2)
    plt.semilogx(tr_sizes, prop_ica, label = models[0])
    plt.semilogx(tr_sizes, 1-prop_ica, label = models[0])

    plt.title('Model selection')
    plt.xlim(tr_sizes[0] * 0.9, tr_sizes[-1] * 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Training set size')
    plt.ylabel('Proportion selected')
    plt.legend(models)  
    
    plt.show()