#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Performing feature selection on a given dataset

@author: Henrique de Almeida Machado da Silveira

October 29th, 2017
"""

from skfeature.function.similarity_based import reliefF
from skfeature.function.information_theoretical_based import FCBF

def feature_select(X_train, y_train, X_test, number_features = 10 , method = "ReliefF", debug = False):

    idx = None
    
    # Displays the current dimension of X
    if debug:
        print("Data dimension before feature selection: " + str(X_train.shape))
    
    # Performs feature selection with the specified method, also getting
    # the list idx of the selected feature indices
    if method == "ReliefF":
        score = reliefF.reliefF(X_train, y_train)
        idx = reliefF.feature_ranking(score)
        X_train_new = X_train[:, idx[0:number_features]]
        X_test_new = X_test[:, idx[0:number_features]]
    elif method == "FCBF":
        idx, _ = FCBF.fcbf(X_train, y_train, n_selected_features=number_features)
        X_train_new = X_train[:, idx[0:number_features]]
        X_test_new = X_test[:, idx[0:number_features]]
    else:
        raise ValueError('\'method\' should be either \'ReliefF\' or \'FCBF\'')
    
    # Displays the dimension of X, after feature selection
    if debug:
        print("Data dimension after feature selection: " + str(X_train_new.shape))
    
    # Returns feature selected data and the indices
    return (X_train_new, X_test_new, idx)