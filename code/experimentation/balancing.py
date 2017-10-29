#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Balancing imbalanced gene expression datasets

@author: Henrique de Almeida Machado da Silveira

October 29th, 2017
"""

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss 
from collections import Counter

# Balances a given dataset using a given balancing method
def balance_data(X, y, method = "SMOTE", debug = False):    
    if method == "SMOTE":        
        balancing_method = SMOTE(random_state = 42)
    elif method == "SMOTETomek":
        balancing_method = SMOTETomek(random_state = 42)
    elif method == "NearMiss":
        balancing_method = NearMiss(version = 2, random_state = 42)
    else:
        raise ValueError('\'method\' should be either \'SMOTE\', \'SMOTETomek\' or \'NearMiss\'')
        
    # Count the number of examples of each class
    if debug:
        print(sorted(Counter(y).items()))
        
    # Balances data
    X_resampled, y_resampled = balancing_method.fit_sample(X, y)
    
    # Check if oversampling worked
    if debug:
        print(sorted(Counter(y_resampled).items()))
        
    # Returns the balanced data
    return (X_resampled, y_resampled)