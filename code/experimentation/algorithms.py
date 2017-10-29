#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Algorithms for classification and dimensionality reduction.

@author: Henrique de Almeida Machado da Silveira

October 29th, 2017
"""

from balancing import balance_data
from ftselection import feature_select
from imblearn.metrics import geometric_mean_score
from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, Input
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from utils import create_confusion_matrix, getEncodingValues, get_out_file, reshape_for_CNN

# Executes an experiment on the current dataset. Given the dataset, split
# into X(independent variables) and y (target variable), perform the selected
# balancing method, the specified feature selection technique and use the
# classification model specified to perform classification. Cross validation is
# used and the mean of the results is used.
def cross_validate(model,
                   X,
                   y,
                   kfold_splits,
                   filename,
                   column_names,
                   balancing_method,
                   ft_selection_method,
                   number_features):
    scores = []
    y_trues = []
    y_preds = []
    columns_selected = []

    # Instantiate the cross validator
    skf = StratifiedKFold(n_splits = kfold_splits,
                          shuffle = True,
                          random_state = 42)
    
    # For each fold
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        
        # Separates training set from test set
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Balance training dataset according to the given balancing method
        X_train, y_train = balance_data(X = X_train,
                                        y = y_train,
                                        method = balancing_method,
                                        debug = False)
        
        # Performs feature selection based on the feature selection method specified
        X_train, X_test, idx = feature_select(X_train = X_train,
                                              y_train = y_train,
                                              X_test=X_test, 
                                              number_features = number_features,
                                              method = ft_selection_method,
                                              debug = False)

        # Gets the column names associated with the selected columns
        columns = column_names[idx[0:number_features]]
        input_size = X_train.shape[1]

        # Instantiates the selected model and performs training, using
        # the trained model to predict labels for the test set
        mdl = None
        if model == "CNN":
            X_train = reshape_for_CNN(X_train)
            X_test = reshape_for_CNN(X_test)
            mdl = cnn(input_size)
            mdl.fit(X_train, y_train, epochs = 200, batch_size = 32)
            y_pred = np.around(np.asarray(mdl.predict(X_test, batch_size = 32)))
        elif model == "MLP":
            mdl = mlp(input_size)
            mdl.fit(X_train, y_train, epochs = 200, batch_size = 32)
            y_pred = np.around(np.asarray(mdl.predict(X_test, batch_size = 32)))
        else: # DecisionTree
            mdl = DecisionTreeClassifier()
            mdl = mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)         
            out_file = get_out_file(filename, 
                                    balancing_method,
                                    ft_selection_method,
                                    number_features,
                                    i)
            export_graphviz(mdl,
                            out_file = out_file,
                            feature_names = columns,
                            class_names = ["TP", "NT"],
                            filled = True,
                            rounded = True,
                            special_characters = True)
        
        # Appends the predicted labels and the true labels of this fold
        # to the respective list, so that the predicted and true labels of
        # the median fold are saved to generate the confusiion matrix later
        y_preds.append(y_pred)
        y_trues.append(y_test)
        
        # Calculates the score of this fold and appends it to the score list
        score = geometric_mean_score(y_test, y_pred)
        scores.append(score)
        
        # If the specified number_features is 10, we are going to compare
        # the columns selected by different feature selection methods. We
        # append the columns which were selected in this fold to the list
        # which will later be returned
        if number_features == 10:
            columns_selected.append(columns)

    # Finds the fold which has the median results
    median_index = scores.index(np.median(scores))
    
    # Generates the confusion matrix using the predicted and true labels
    # associated with the median results
    create_confusion_matrix(y_trues[median_index],
                            y_preds[median_index],
                            filename, balancing_method,
                            ft_selection_method,
                            number_features, model)

    # In case the classification model is a Decision Tree, the plot of that
    # tree is generated for visualization. The tree corresponding to the
    # median result is used fot that
    if model == "DecisionTree":
        out_file = get_out_file(filename,
                                balancing_method,
                                ft_selection_method,
                                number_features,
                                median_index)
        os.system("dot -Tpng {} -o results/imgs/{}_{}_{}_{}_tree.png".format(out_file,
                                                                             filename,
                                                                             balancing_method,
                                                                             ft_selection_method,
                                                                             number_features))
    
    # If the columns selected by different feature seletion methods are going
    # to be compared, they are returned with the mean of the scores. Otherwise,
    # only the scores are returned
    if number_features == 10:
        return (np.mean(scores), columns_selected)
    
    return np.mean(scores)

# Model of the MLP used for classification. It contains 3 hidden layers with
# the same number of nodes as the input layer. The dropout technique is also
# used, with a dropout rate of 10%.
def mlp(input_size):
    model = Sequential()

    model.add(Dense(input_size, 
                    input_dim = input_size,
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "glorot_uniform",
                    activation = "tanh"))

    model.add(Dropout(0.1))

    model.add(Dense(input_size,
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "glorot_uniform",
                    activation = "tanh"))

    model.add(Dropout(0.1))

    model.add(Dense(input_size,
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "glorot_uniform",
                    activation = "tanh"))

    model.add(Dropout(0.1))

    # Output layer consisting of just one node for classification
    model.add(Dense(1,
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "glorot_uniform",
                    activation = "sigmoid",
                    name = "output"))

    model.compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = ["accuracy"])

    return model

# Model of the CNN used for classification. A convolution layer is used with
# 5 filters and kernel size equal to half of the input dimension. A pooling
# layer comes next, and after that a fully-connected layer is used with the
# 10% Dropout technique. The output layer has only one node and is used
    # for classification.
def cnn(input_size):
    model = Sequential()
    
    model.add(Conv1D(filters = 5,
                     kernel_size = input_size//2,
                     activation = "relu",
                     input_shape = (input_size, 1)))
    
    model.add(GlobalMaxPooling1D())
    
    model.add(Dense(input_size,
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "glorot_uniform",
                    activation = "tanh"))
    
    model.add(Dropout(0.1))
    
        # Output layer consisting of just one node for classification
    model.add(Dense(1,
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "glorot_uniform",
                    activation = "sigmoid",
                    name = "output"))
    
    model.compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = ["accuracy"])

    return model

# Executes an experiment on the current dataset. Given the dataset, split
# into X(independent variables) and y (target variable), perform the selected
# balancing method, dimesnionality reduction using a SDAE and use the
# classification model specified to perform classification. Cross validation is
# used and the mean of the results is used.
def cross_validate_SDAE(X, y, filename, balancing_method, number_features, noise_factor, kfold_splits, model):
    
    input_size = X.shape[1]

    # Getting dimensions of the intermediate hidden layers
    encoding_intermediate_values = getEncodingValues(number_features, input_size)

    scores = []
    y_trues = []
    y_preds = []

    # Instantiate the cross validator
    skf = StratifiedKFold(n_splits = kfold_splits,
                          shuffle = True,
                          random_state = 42)
    
    # For each fold
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):

        # Separates training set from test set
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Balance training dataset according to the given balancing method
        X_train, y_train = balance_data(X = X_train,
                                        y = y_train,
                                        method = balancing_method,
                                        debug = False)
    
        # Trains each hidden layer separately (layerwise training
        # of the stacked autoencoder)
        hidden_layer_1, encoded_output = train_layerwise_SDAE(X_train = X_train,
                                                                input_size = input_size, 
                                                                hidden_layer_size = encoding_intermediate_values[0], 
                                                                noise_factor = noise_factor)
        
        hidden_layer_2, encoded_output = train_layerwise_SDAE(X_train = encoded_output,
                                                                input_size = encoded_output.shape[1], 
                                                                hidden_layer_size = encoding_intermediate_values[1], 
                                                                noise_factor = noise_factor)
        
        hidden_layer_3, encoded_output = train_layerwise_SDAE(X_train = encoded_output,
                                                                input_size = encoded_output.shape[1], 
                                                                hidden_layer_size = number_features, 
                                                                noise_factor = noise_factor)
        
        # Creates a deep network consisting of the hidden layer and a classification
        # output layer to perform fine tuning (supervised learning)
        input_layer = Input(shape = (input_size, ))
        layer1 = hidden_layer_1(input_layer)
        layer2 = hidden_layer_2(layer1)
        layer3 = hidden_layer_3(layer2)
        ouput_layer = Dense(1, activation = "sigmoid", name = "output_layer")(layer3)
        
        autoencoder = Model(inputs = input_layer, outputs = ouput_layer)
        autoencoder.compile(optimizer = "adadelta", loss = "binary_crossentropy")
        
        # Fine tuning
        autoencoder.fit(X_train, y_train, epochs = 10, batch_size = 32)
        
        # Extracts hidden layers to create model for dimensionality reduction
        sdae = Model(inputs = input_layer, outputs = layer3)

        # Performs classification and retrieves the score and the predicted labels
        score, y_pred = classify(X_train, X_test, y_train, y_test, sdae, model)

        # Adds this fold's score to the score list
        scores.append(score)
        
        # Appends the predicted labels and the true labels of this fold
        # to the respective list, so that the predicted and true labels of
        # the median fold are saved to generate the confusiion matrix later
        y_preds.append(y_pred)
        y_trues.append(y_test)
        
    # Finds the fold which has the median results
    median_index = scores.index(np.median(scores))
    
    # Generates the confusion matrix using the predicted and true labels
    # associated with the median results
    create_confusion_matrix(y_trues[median_index],
                            y_preds[median_index],
                            filename,
                            balancing_method,
                            "SDAE",
                            number_features, model)

    # Returns the mean of the scores resulting from cross validation
    return np.mean(scores)

# This fucntion is used to train each layer of the SDAE
def train_layerwise_SDAE(X_train, input_size, hidden_layer_size, noise_factor):
    
    # Adding Gaussian noise to input data (for the denoising autoencoder)
    X_train_noisy = X_train + noise_factor * np.random.normal(loc = 0.0,
                                                              scale = 1.0,
                                                              size = X_train.shape)
    X_train_noisy = preprocessing.scale(X_train_noisy)
    
    # Training one layer of the SDAE: creating architecture
    input_layer = Input(shape = (input_size, ))
    hidden_layer = Dense(hidden_layer_size, activation = "relu")(input_layer)
    output_layer = Dense(input_size, activation = "sigmoid")(hidden_layer)

    # Training one layer of the SDAE: fitting model    
    autoencoder = Model(inputs = input_layer, outputs = output_layer)
    autoencoder.compile(optimizer = "adadelta", loss = "mean_squared_error")
    autoencoder.fit(X_train_noisy, X_train, epochs = 3, batch_size = 32)
    
    # Obtaining higher level representation of input by encoding it
    encoder = Model(inputs = input_layer, outputs = hidden_layer)
    encoded_output = encoder.predict(X_train, batch_size = 32)
    encoded_output = preprocessing.StandardScaler().fit_transform(encoded_output)
    
    # Returns the trained hidden layer and the encoded output produced
    # to apply in the next layer of the stack
    return (autoencoder.layers[1], encoded_output)
    
# Function used to perform classification in experiments using SDAEs
def classify(X_train, X_test, y_train, y_test, sdae, model):
    
    # Encodes X_train, reducing its dimensionality
    X_train = sdae.predict(X_train, batch_size = 32)
    X_train = preprocessing.StandardScaler().fit_transform(X_train)

    # Encodes X_test, reducing its dimensionality
    X_test = sdae.predict(X_test, batch_size = 32)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)

    # Trains classifier according to the specified model. Then, performs
    # prediction and uses G-MEAN as metric to evaluate performance
    mdl = None
    if model == "CNN":
        X_train = reshape_for_CNN(X_train)
        X_test = reshape_for_CNN(X_test)
        input_size = X_train.shape[1]
        mdl = cnn(input_size)
        mdl.fit(X_train, y_train, epochs = 200, batch_size = 32)
        y_pred = np.around(np.asarray(mdl.predict(X_test, batch_size = 32)))
    elif model == "MLP":
        input_size = X_train.shape[1]
        mdl = mlp(input_size)
        mdl.fit(X_train, y_train, epochs = 200, batch_size = 32)
        y_pred = np.around(np.asarray(mdl.predict(X_test, batch_size = 32)))
    else: # Model is decision tree
        mdl = DecisionTreeClassifier()
        mdl = mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
    
    # Returns classification score
    return geometric_mean_score(y_test, y_pred), y_pred
