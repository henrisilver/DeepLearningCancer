#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline used to perform experiments with gene expression datasets

@author: Henrique de Almeida Machado da Silveira

October 29th, 2017
"""

import csv
import pandas as pd
from glob import glob
from algorithms import cross_validate, cross_validate_SDAE
from utils import compare_selected_features

# Gets all the CSV files in the working directory
filenames = glob("*.csv")

# Raise error if no datasets are available in the working directory
if len(filenames) == 0:
    raise FileNotFoundError("No data files found in the target directory!")

# Lists of different options to be considered in each experiment
feature_selection_methods = ["ReliefF", "SDAE", "FCBF"]
number_of_features = [10, 50, 100]
balancing_methods = ["SMOTE", "SMOTETomek", "NearMiss"]
algorithms = ["CNN", "MLP", "DecisionTree"]

# Creates output file, for storing the results
with open("results/results.csv", "w") as results_file:
    wr = csv.writer(results_file,
                    delimiter = ",",
                    quoting=csv.QUOTE_NONNUMERIC)
    
    wr.writerow(["Dataset",
                 "Algoritmo de seleção de atributos",
                 "Número de atributos selecionados",
                 "Método de Balanceamento",
                 "Classificador",
                 "G-Mean"])

    # For each dataset
    for filename in filenames:

        # Importing the dataset
        dataset = pd.read_csv(filename)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        # Performs each experiment as a combination of the feature selection
        # method, the number of features to be selected,
        # the data balancing method and the classification algorithm
        for balancing_method in balancing_methods:
            for alg in algorithms:
                for num in number_of_features:
                    selected_columns_relieff = None
                    selected_columns_fcbf = None
                    for feature_selection_method in feature_selection_methods:
                        if feature_selection_method == "SDAE":
                            # Gets the score for the current experiment using SDAE
                            score = cross_validate_SDAE(X = X, y = y, filename = filename, balancing_method = balancing_method, number_features=num, noise_factor=0.5, kfold_splits=5, model=alg)
                        elif feature_selection_method == "ReliefF":
                            # Gets the score for the current experiment and
                            # the names of the selected columns if 10 features
                            # are being selected
                            if num == 10:
                                score, selected_columns_relieff = cross_validate(model = alg, X = X, y = y, kfold_splits = 5, filename = filename, column_names = dataset.columns, balancing_method = balancing_method, ft_selection_method = feature_selection_method, number_features = num)
                            else:
                                score = cross_validate(model = alg, X = X, y = y, kfold_splits = 5, filename = filename, column_names = dataset.columns, balancing_method = balancing_method, ft_selection_method = feature_selection_method, number_features = num)
                        else: #FCBF
                            # Gets the score for the current experiment and
                            # the names of the selected columns if 10 features
                            # are being selected
                            if num == 10:
                                score, selected_columns_fcbf = cross_validate(model = alg, X = X, y = y, kfold_splits = 5, filename = filename, column_names = dataset.columns, balancing_method = balancing_method, ft_selection_method = feature_selection_method, number_features = num)
                            else:
                                score = cross_validate(model = alg, X = X, y = y, kfold_splits = 5, filename = filename, column_names = dataset.columns, balancing_method = balancing_method, ft_selection_method = feature_selection_method, number_features = num)

                        wr.writerow([filename, feature_selection_method, num, balancing_method, alg, score])
                    
                    # If both FCBF and ReliefF feature selection methods were
                    # performed for the current dataset and the current
                    # balancing method and classification algorithm, compares
                    # the selected features from each method for all the folds
                    # from cross validation
                    if num == 10 and selected_columns_relieff is not None and selected_columns_fcbf is not None:
                        for i in range(5):
                            compare_selected_features(selected_columns_relieff[i], selected_columns_fcbf[i], filename, balancing_method, alg, i)