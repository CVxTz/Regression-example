


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

def readDataSetBase(nrows = None):
    ## read data
    train = pd.read_csv(loc+'data/train.csv', nrows = None)
    test = pd.read_csv(loc+'data/test.csv', nrows = None)

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

def readDataSetcox(nrows = None):
    ## read data
    train = pd.read_csv(loc+'data/train.csv', nrows = None)
    test = pd.read_csv(loc+'data/test.csv', nrows = None)

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

    for f in f_num:
        tr_te[f+"_cox"], _ = boxcox(tr_te[f].values+1)

        tr_te[f+"_log"] = np.log(tr_te[f].values+1)


    print tr_te.columns


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
	
def readDataSetcoxpca(nrows = None):
    ## read data
    train = pd.read_csv(loc+'data/train.csv', nrows = None)
    test = pd.read_csv(loc+'data/test.csv', nrows = None)

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

    for f in f_num:
        tr_te[f+"_cox"], _ = boxcox(tr_te[f].values+1)

        tr_te[f+"_log"] = np.log(tr_te[f].values+1)


    print tr_te.columns


    f_num = [f for f in tr_te.columns if 'cont' in f]

    scaler = StandardScaler()
    pca = PCA()
    tmp = csr_matrix(scaler.fit_transform( pca.fit_transform(tr_te[f_num]) ))
    sparse_data.append(tmp)

    del(tr_te, train, test)

    ## sparse train and test data
    xtr_te = hstack(sparse_data, format = 'csr')
    xtrain = xtr_te[:ntrain, :]
    xtest = xtr_te[ntrain:, :]

    print('Dim train', xtrain.shape)
    print('Dim test', xtest.shape)

    return xtrain, xtest, id_train, id_test, y

def readDataSetImpact(nrows = None):
    ## read data
    train = pd.read_csv(loc+'data/train.csv', nrows = None)
    test = pd.read_csv(loc+'data/test.csv', nrows = None)

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

    for f in f_cat:
        mapping_dict = (pd.concat([tr_te[[f]].fillna(-1), tr_te[['loss']]], axis=1).groupby(f).mean()).to_dict()
        transformations = {f: value for key, value in mapping_dict.iteritems() }
        tr_te[[f+"_impact"]] = tr_te[[f]].copy().fillna(-1).replace(to_replace=transformations).fillna(-1)

    f_impact = [f+"_impact" for f in f_cat]
    #print tr_te[f_impact]

    f_num = [f for f in tr_te.columns if 'cont' in f]+f_impact
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

def readDataSetLexical(nrows = None):
    ## read data
    train = pd.read_csv(loc+'data/train.csv', nrows = None)
    test = pd.read_csv(loc+'data/test.csv', nrows = None)

    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    cats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train, test, numeric_feats)
    for col in cats:
        train_test[col] = train_test[col].apply(encode)

    ss = StandardScaler()
    train_test[numeric_feats] = \
        ss.fit_transform(train_test[numeric_feats].values)
    train = train_test.iloc[:ntrain, :].copy()
    test = train_test.iloc[ntrain:, :].copy()
    test.drop('loss', inplace=True, axis=1)
    feats = numeric_feats+ cats

    return train[feats], test[feats], train['id'], test['id'], train["loss"]

def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    # compute skew and do Box-Cox transformation (Tilli)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain

def encode(charcode):
    r = 0
    ln = len(charcode)
    if(ln > 2):
        print("Error: Expected Maximum of Two Characters!")
        exit(0)
    for i in range(ln):
        r += (ord(charcode[i])-ord('A')+1)*26**(ln-i-1)
    return r