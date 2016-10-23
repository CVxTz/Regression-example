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

varselect = True

varselect = True
base = False

if base:
    xtrain, xtest, id_train, id_test, y = read_data.readDataSetBase()
else:
    xtrain, xtest, id_train, id_test, y = read_data.readDataSetcox()

params = {}
params['booster'] = 'gbtree' #gbtree
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.075
#params['gamma'] = 0.5290
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.7
params['subsample'] = 0.7
params['max_depth'] = 7
#params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 1337


if varselect:
    selector = SelectPercentile(f_regression, percentile=70)

    selector.fit(xtrain, y)

    xtrain = selector.transform(xtrain)
    xtest = selector.transform(xtest)

early_stopping = 20
## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))
## train models
i = 0
nbags = 5
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])
d_test = xgb.DMatrix(xtest)
for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = np.log(y[inTr]).ravel()
    xte = xtrain[inTe]
    yte = np.log(y[inTe]).ravel()
    d_train = xgb.DMatrix(xtr, label=ytr)
    d_valid = xgb.DMatrix(xte, label=yte)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        params['random_state'] = 1337+j
        clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping, feval=xg_eval_mae, maximize=False)
        pred += np.exp(clf.predict(d_valid) )
        pred_test += np.exp(clf.predict(d_test))
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte), pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(y, pred_oob))
loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv(loc+'submissions/preds_oob_xgb_fs_log.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(loc+'submissions/submission_xgb_fs_log.csv', index = False)