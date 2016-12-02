'''
Script mainly from =>
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''

import read_data
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, f_regression
from sklearn.cross_validation import StratifiedKFold

def list_to_percentiles(numbers):
    pairs = zip(numbers, range(len(numbers)))
    pairs.sort(key=lambda p: p[0])
    result = [0 for i in range(len(numbers))]
    for rank in xrange(len(numbers)):
        original_index = pairs[rank][1]
        result[original_index] = int( rank * 100.0 / (len(numbers)-1))
    return result

def trainModel(model, varselect = True, datasetRead = "base", modelname= "", nbags = 5,
               params = {}, randp = False, shift = False, data = []):


    if datasetRead == "base":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetBase()
    elif datasetRead == "cox":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetcox()
    elif datasetRead == "coxpca":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetcoxpca()
    elif datasetRead == "impact":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetImpact()
    elif datasetRead == "indata":
        xtrain, xtest, id_train, id_test, y = data[0], data[1], data[2], data[3], data[4]

    if varselect:
        selector = SelectPercentile(f_regression, percentile=70)

        selector.fit(xtrain, y)

        xtrain = selector.transform(xtrain)
        xtest = selector.transform(xtest)
    shift_val = 200
    early_stopping = 20
    ## cv-folds
    nfolds = 5
    lossl = list_to_percentiles(np.log(y+shift_val).ravel())
    folds = StratifiedKFold(lossl, n_folds=5, shuffle = True, random_state = 20)
    #folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

    ## train models
    i = 0
    
    nepochs = 55
    pred_oob = np.zeros(xtrain.shape[0])
    pred_test = np.zeros(xtest.shape[0])

    for (inTr, inTe) in folds:
        xtr = xtrain[inTr]
        xte = xtrain[inTe]
        if shift:
            ytr = np.log(y[inTr]+shift_val).ravel()
            yte = np.log(y[inTe]+shift_val).ravel()
        else:
            ytr = np.log(y[inTr]).ravel()
            yte = np.log(y[inTe]).ravel()

        pred = np.zeros(xte.shape[0])
        for j in range(nbags):
            if randp:
                params['random_state'] = 1337+j
            model.set_params(**params)
            model.fit(xtr, ytr)
            if shift:
                pred += np.exp(model.predict(xte) )-shift_val
                pred_test += np.exp(model.predict(xtest))-shift_val
            else:
                pred += np.exp(model.predict(xte) )
                pred_test += np.exp(model.predict(xtest))
        pred /= nbags
        pred_oob[inTe] = pred
        i += 1
        #print('Fold ', i, '- MAE:', score)
    pred_test /= (nfolds*nbags)

    #print('Total - MAE:', mean_absolute_error(y, pred_oob))
    loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
    ## train predictions
    df = pd.DataFrame({'id': id_train, 'loss': pred_oob})


    perf = mean_absolute_error(y, pred_oob)

    df.to_csv(loc+'opt_sk/preds_oob_'+modelname+str(perf)+'.csv', index = False)

    ## test predictions



    df = pd.DataFrame({'id': id_test, 'loss': pred_test})
    df.to_csv(loc+'opt_sk/submission_'+modelname+str(perf)+'.csv', index = False)

    return perf