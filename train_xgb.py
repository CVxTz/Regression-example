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

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0.5290
params['min_child_weight'] = 4.2922
params['colsample_bytree'] = 0.3085
params['subsample'] = 0.9930
params['max_depth'] = 7
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 1337

xtrain, xtest, id_train, id_test, y = read_data.readDataSetBase()

early_stopping = 10
## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 1
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])
d_test = xgb.DMatrix(xtest)
for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    d_train = xgb.DMatrix(xtr, label=ytr)
    d_valid = xgb.DMatrix(xte, label=yte)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping)
        pred += clf.predict(d_valid)
        pred_test += clf.predict(d_test)
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(y, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('C:/Users/jenazad\PycharmProjects/Regression-example/submissions/preds_oob_xgb.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('C:/Users/jenazad\PycharmProjects/Regression-example/submissions/submission_xgb.csv', index = False)