# Starter code for CS 165B MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }


def FindHyperParams(mlp, params):
    clf = GridSearchCV(mlp, params)
    return clf

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    training_label = training_data['target']
    training_data.drop('target', axis=1, inplace=True)

    le = preprocessing.LabelEncoder()
    sc = preprocessing.StandardScaler()

    training_data['QUANTIZED_INC'] = le.fit_transform(training_data['QUANTIZED_INC'])
    training_data['QUANTIZED_AGE'] = le.fit_transform(training_data['QUANTIZED_AGE'])
    training_data['QUANTIZED_WORK_YEAR'] = le.fit_transform(training_data['QUANTIZED_WORK_YEAR'])
    training_data['NAME_INCOME_TYPE'] = le.fit_transform(training_data['NAME_INCOME_TYPE'])
    training_data['NAME_EDUCATION_TYPE'] = le.fit_transform(training_data['NAME_EDUCATION_TYPE'])
    training_data['NAME_FAMILY_STATUS'] = le.fit_transform(training_data['NAME_FAMILY_STATUS'])
    training_data['NAME_HOUSING_TYPE'] = le.fit_transform(training_data['NAME_HOUSING_TYPE'])

    training_data.drop('OCCUPATION_TYPE', axis=1, inplace=True)
    #training_data.drop('QUANTIZED_INC', axis=1, inplace=True)
    #training_data.drop('QUANTIZED_AGE', axis=1, inplace=True)
    #training_data.drop('QUANTIZED_WORK_YEAR', axis=1, inplace=True)

    scale_training = sc.fit_transform(training_data)
    training_data.loc[:,:] = scale_training

    # Highest F1 so far: hidden = (128,64), max_iter = 1000, lr_init = 0.0003, batch = 16 (drop OCCUPATION_TYPE)

    params = {
        'hidden_layer_sizes': [(128,),(64,128),(128,64),(128,32,32),(128,128),(64,64,32,32)],
        'max_iter': [1000],
        'learning_rate_init': [0.1,0.01,0.001,0.3,0.03,0.003],
        'batch_size': [8,16,32,64,128],
    }

    mlp = MLPClassifier()
    clf = FindHyperParams(mlp, params)
    clf.fit(scale_training, training_label)
    
    testing_data['QUANTIZED_INC'] = le.fit_transform(testing_data['QUANTIZED_INC'])
    testing_data['QUANTIZED_AGE'] = le.fit_transform(testing_data['QUANTIZED_AGE'])
    testing_data['QUANTIZED_WORK_YEAR'] = le.fit_transform(testing_data['QUANTIZED_WORK_YEAR'])
    testing_data['NAME_INCOME_TYPE'] = le.fit_transform(testing_data['NAME_INCOME_TYPE'])
    testing_data['NAME_EDUCATION_TYPE'] = le.fit_transform(testing_data['NAME_EDUCATION_TYPE'])
    testing_data['NAME_FAMILY_STATUS'] = le.fit_transform(testing_data['NAME_FAMILY_STATUS'])
    testing_data['NAME_HOUSING_TYPE'] = le.fit_transform(testing_data['NAME_HOUSING_TYPE'])

    testing_data.drop('OCCUPATION_TYPE', axis=1, inplace=True)
    #testing_data.drop('QUANTIZED_INC', axis=1, inplace=True)
    #testing_data.drop('QUANTIZED_AGE', axis=1, inplace=True)
    #testing_data.drop('QUANTIZED_WORK_YEAR', axis=1, inplace=True)

    scale_testing = sc.fit_transform(testing_data)
    testing_data.loc[:,:] = scale_testing

    predict = mlp.predict(scale_testing)
    
    return predict


if __name__ == '__main__':

    training = pd.read_csv('data/train.csv')
    development = pd.read_csv('data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)

    


    


