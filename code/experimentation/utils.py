#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities file, gathering different functions used throughout the code

@author: Henrique de Almeida Machado da Silveira

October 29th, 2017
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Computes dimensions of the intermediate hidden layers of a SDAE, given the
# input dimension (begin_interval) and the output dimension (end_interval).
# The ratio between consecutive dimensions is constant
def getEncodingValues(begin_interval, end_interval):
    
    rate = (end_interval / begin_interval) ** (1/3.0)
    
    values = [end_interval//rate, end_interval//(rate**2)]
    
    return list(map(int, values))

# Reshapes X to be used by a CNN, which is required by the Conv1D layer
def reshape_for_CNN(X):
    X_new_shape = X.reshape(X.shape[0], X.shape[1], 1)
    return X_new_shape

# Gets the name of the output file used to generate DecisionTree models' visualizations
def get_out_file(filename, balancing_method, ft_selection_method, number_features, i):
    return "results/imgs/dot/{}_{}_{}_{}_{}.dot".format(filename,
                                                        balancing_method,
                                                        ft_selection_method,
                                                        number_features,
                                                        i)

# Creates a confusion matrix and plots it using Matplotlib
# Adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def create_confusion_matrix(y_true,
                            y_pred,
                            filename,
                            balancing_method,
                            ft_selection_method,
                            number_features,
                            model,
                            normalize = False):
    
    classes = ["TP", "NT"]
    cnf_matrix = confusion_matrix(y_true, y_pred)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão para\n{} - {}\n{} - {} - {} atributos".format(filename, model, balancing_method, ft_selection_method, number_features))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.savefig('results/confmat/{}_{}_{}_{}_{}.png'.format(filename, model, balancing_method, ft_selection_method, number_features))
    plt.close()

# Generates a plot comparing the selected features from both FCBF and ReliefF. Uses Matplotlib
# Adapted from: https://matplotlib.org/examples/pylab_examples/barchart_demo.html
def compare_selected_features(selected_columns_relieff,
                              selected_columns_fcbf,
                              filename,
                              balancing_method,
                              alg,
                              k):
    n_groups = 3
    intersection = [x for x in selected_columns_relieff if x in selected_columns_fcbf]
    only_relieff = [x for x in selected_columns_relieff if x not in intersection]
    only_fcbf = [x for x in selected_columns_fcbf if x not in intersection]
    
    intersection.sort()
    only_relieff.sort()
    only_fcbf.sort()
    
    results = [int(len(only_relieff)), int(len(only_fcbf)), int(len(intersection))]
    
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4

    plt.figure(figsize=(5,5))
    plt.bar(index, results, bar_width, alpha=opacity, color='b')
    plt.xlabel('Método de seleção de atributos')
    plt.ylabel('Número de atributos')
    plt.title('Análise dos atributos selecionados\n{} - {}\n{} - k = {}'.format(filename, alg, balancing_method, k))
    plt.xticks(index, ('Apenas ReliefF', 'Apenas FCBF', 'Ambos'))
    plt.yticks(np.arange(max(results)+1), tuple(range(max(results)+1)))
    
    relieff_strings = [', '.join(only_relieff[3 * i: 3 * i + 3]) for i in range(0, int(np.ceil(len(only_relieff) / 3)))]
    relieff_strings = "Apenas ReliefF:\n" + '\n'.join(relieff_strings)
    
    fcbf_strings = [', '.join(only_fcbf[3 * i: 3 * i + 3]) for i in range(0, int(np.ceil(len(only_fcbf) / 3)))]
    fcbf_strings = "Apenas FCBF:\n" + '\n'.join(fcbf_strings)
    
    intersection_strings = [', '.join(intersection[3 * i: 3 * i + 3]) for i in range(0, int(np.ceil(len(intersection) / 3)))]
    intersection_strings = "Ambos:\n" + '\n'.join(intersection_strings)
    
    y_offset_title = 0.22
    
    full_string = "Atributos selecionados:\n" + relieff_strings + "\n" + fcbf_strings + "\n" + intersection_strings
    
    plt.gcf().text(0.02, y_offset_title, full_string, fontsize=10, verticalalignment="center")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.6)
    plt.subplots_adjust(right=0.6)
    plt.savefig('results/feature_comp/{}_{}_{}_{}.png'.format(filename, alg, balancing_method, k))
    plt.close()