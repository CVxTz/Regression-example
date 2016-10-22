


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



def readDataSetBase(nrows = None):
    ## read data
    train = pd.read_csv('C:/Users/jenazad\PycharmProjects/Regression-example/data/train.csv', nrows = None)
    test = pd.read_csv('C:/Users/jenazad\PycharmProjects/Regression-example/data/test.csv', nrows = None)

    ## set test loss to NaN
    test['loss'] = np.nan

    ## response and IDs
    y = train['loss'].values
    id_train = train['id'].values
    id_test = test['id'].values

    ## stack train test
    ntrain = train.shape[0]
    tr_te = pd.concat((train, test), axis = 0)

    ## Preprocessing and transforming to sparse data
    sparse_data = []

    f_cat = [f for f in tr_te.columns if 'cat' in f]
    for f in f_cat:
        dummy = pd.get_dummies(tr_te[f].astype('category'))
        tmp = csr_matrix(dummy)
        sparse_data.append(tmp)

    f_num = [f for f in tr_te.columns if 'cont' in f]
    scaler = StandardScaler()
    tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
    sparse_data.append(tmp)

    del(tr_te, train, test)

    ## sparse train and test data
    xtr_te = hstack(sparse_data, format = 'csr')
    xtrain = xtr_te[:ntrain, :]
    xtest = xtr_te[ntrain:, :]

    print('Dim train', xtrain.shape)
    print('Dim test', xtest.shape)

    return xtrain, xtest, id_train, id_test, y