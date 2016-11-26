


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
    print('Dim train', train[feats].shape)
    print('Dim test', test[feats].shape)
    print train["loss"]

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

def readDataSetLexicalcomp(nrows = None):
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
    #print('Dim train', train[feats].shape)
    #print('Dim test', test[feats].shape)
    #print train["loss"]
    result = [('cat80', 'cat101', 0.53907180825224987, '-'), ('cat12', 'cat80', 0.5244145983395182, '-'), ('cat79', 'cat101', 0.50374293055346264, '+'), ('cat12', 'cat79', 0.48725474852416256, '+'), ('cat81', 'cat87', 0.41124259427488874, '-'), ('cat10', 'cat81', 0.40216609930497876, '-'), ('cat1', 'cat87', 0.39796687934844266, '-'), ('cat1', 'cat10', 0.37682725921935278, '-'), ('cat2', 'cat57', 0.33760439710228435, '+'), ('cat2', 'cat72', 0.33609668480943178, '+'), ('cat9', 'cat57', 0.32829734766307028, '+'), ('cat9', 'cat72', 0.32745999355935101, '+'), ('cat11', 'cat103', 0.31971760507461972, '+'), ('cat7', 'cat11', 0.31074403922891847, '+'), ('cat13', 'cat111', 0.3088851702393845, '+'), ('cat7', 'cat13', 0.30659326029319017, '+'), ('cat103', 'cat111', 0.2867545818618395, '+'), ('cat3', 'cat89', 0.27279044359043342, '+'), ('cat16', 'cat89', 0.2717765899443455, '+'), ('cat3', 'cat23', 0.26426274934931032, '+'), ('cat23', 'cat90', 0.26005007308112243, '+'), ('cat16', 'cat73', 0.25840263818584408, '-'), ('cat36', 'cat90', 0.25701116287606585, '+'), ('cat36', 'cat73', 0.24896439920385346, '-'), ('cat6', 'cat53', 0.22227280865202451, '-'), ('cat6', 'cat114', 0.21272517182728345, '-'), ('cat4', 'cat5', 0.21148010931581968, '+'), ('cat50', 'cat53', 0.20352936403984986, '/'), ('cat50', 'cat114', 0.19807987652587153, '/'), ('cat4', 'cat38', 0.19268232415585612, '+'), ('cat5', 'cat28', 0.19149919238692711, '+'), ('cat28', 'cat38', 0.18467524873381091, '+'), ('cat25', 'cat40', 0.17596090113808627, '+'), ('cont2', 'cat40', 0.16769798208846023, '+'), ('cont2', 'cat25', 0.15759355959907909, '+'), ('cat24', 'cat82', 0.14735626150526968, '-'), ('cat8', 'cat82', 0.1454692313049977, '-'), ('cat14', 'cat24', 0.13734357354930476, '+'), ('cat14', 'cat41', 0.13353824779963153, '+'), ('cat76', 'cat85', 0.12980695082088076, '+'), ('cat8', 'cat41', 0.12939582956814316, '+'), ('cont3', 'cat102', 0.12225565703895201, '+'), ('cat44', 'cat102', 0.12101575710907629, '+'), ('cont3', 'cat76', 0.11997431947787257, '+'), ('cat29', 'cat44', 0.11742191624417585, '+'), ('cat29', 'cat105', 0.11623900437176257, '+'), ('cat45', 'cat105', 0.11307908648407171, '+'), ('cat45', 'cat85', 0.11002306134365768, '+'), ('cont7', 'cat17', 0.10508640161523232, '+'), ('cont7', 'cat26', 0.10398589409066376, '+'), ('cat17', 'cat26', 0.10159479067982102, '+')]
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    for tup in result:
        if tup[3] == "+":
            resultvector = scaler1.fit_transform(train[tup[0]]) + scaler2.fit_transform(train[tup[1]])
            resultvector_ = scaler1.transform(test[tup[0]]) + scaler2.transform(test[tup[1]])
        if tup[3] == "-":
            resultvector = scaler1.fit_transform(train[tup[0]]) - scaler2.fit_transform(train[tup[1]] )
            resultvector_ = scaler1.transform( test[tup[0]] ) - scaler2.transform( test[tup[1]])
        if tup[3] == "*":
            resultvector = np.multiply(scaler1.fit_transform( train[tup[0]] ), scaler2.fit_transform( train[tup[1]] ) )
            resultvector_ = np.multiply(scaler1.transform(test[tup[0]]) , scaler2.transform(test[tup[1]] ) )
        if tup[3] == "/":
            resultvector = np.divide(scaler1.fit_transform(  train[tup[0]] ), np.absolute(scaler2.fit_transform(  train[tup[1]] )) +1 )
            resultvector_ = np.divide(scaler1.transform( test[tup[0]] ), np.absolute(scaler2.transform( test[tup[1]] ) ) +1 )

        train[tup[0]+tup[3]+tup[1]] = resultvector
        test[tup[0]+tup[3]+tup[1]] = resultvector_
        feats.append(tup[0]+tup[3]+tup[1])

    return train[feats], test[feats], train['id'], test['id'], train["loss"]
