


'''
Script mainly from =>
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''

import numpy as np
np.random.seed(1337)
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import skew, boxcox
from sklearn.decomposition import PCA

loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

def readData1(nrows = None):
    ## read data
    list_names = ["XGBRegressor1", "XGBRegressor2", "XGBRegressor3", "XGBRegressor4", "XGBRegressor5", "XGBRegressor6", "SGDRegressor1",
                  "Ridge1", "PassiveAggressiveRegressor1", "LinearRegression1", "Lasso1", "HuberRegressor1", "ffnn2", "ffnn1", "ffnn3"]


    list_train = []
    list_test = []
    for modelname in list_names:
        train_f = pd.read_csv(loc+'stacking_preds/preds_oob_xgb_'+modelname+'.csv', nrows = None)
        test_f = pd.read_csv(loc+'stacking_preds/submission_xgb_'+modelname+'.csv', nrows = None)
        train_f = train_f.rename(index=str, columns={"id": "id", "loss": "loss"+modelname})
        train_f.index = train_f['id']
        test_f = test_f.rename(index=str, columns={"id": "id", "loss": "loss"+modelname})
        test_f.index = test_f['id']
        list_train.append(train_f)
        list_test.append(test_f)

    train_f = pd.concat(list_train, axis = 1, join="inner").T.groupby(level=0).first().T
    test_f = pd.concat(list_test, axis = 1, join="inner").T.groupby(level=0).first().T


    train = pd.read_csv(loc+'data/train.csv', nrows = None)
    test = pd.read_csv(loc+'data/test.csv', nrows = None)

    test['loss'] = np.nan

    train = train[["loss", "id"]]
    test = test[["loss", "id"]]

    train = train.merge(train_f, on="id", how="inner")
    test = test.merge(test_f, on="id", how="inner")
    ## response and IDs
    y = train['loss'].values
    id_train = train['id'].values
    id_test = test['id'].values

    ## stack train test
    ntrain = train.shape[0]




    listcols = list(train.columns.values)


    train = train[listcols]
    test = test[listcols]

    tr_te = pd.concat((train, test), axis = 0)

    listcols.remove("loss")
    listcols.remove("id")

    f_num = listcols
    print f_num
    print tr_te
    scaler = StandardScaler()
    tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))


    ## sparse train and test data
    xtr_te = tmp
    xtrain = xtr_te[:ntrain, :]
    xtest = xtr_te[ntrain:, :]

    print('Dim train', xtrain.shape)
    print('Dim test', xtest.shape)

    return xtrain, xtest, id_train, id_test, y


