
import pandas as pd
import numpy as np
import scipy.stats as stats
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, boxcox


def readDataSetLexical(nrows = None):
    ## read data
    train = pd.read_csv('data/train.csv', nrows = None)
    test = pd.read_csv('data/test.csv', nrows = None)

    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    cats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train, test, numeric_feats)
    for col in cats:
        train_test[col] = train_test[col].apply(encode)

    ss = StandardScaler()
    train_test[numeric_feats] =         ss.fit_transform(train_test[numeric_feats].values)
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


train, test, trainid, testid, trainloss = readDataSetLexical()


scaler = StandardScaler()


colnames = train.columns.values


# In[8]:

trainscaled = np.array(scaler.fit_transform(train) )


# In[9]:

print trainscaled


# In[10]:

featurenum = trainscaled.shape[1]
feature_corr_list = []
for i in range(featurenum):
    corr = stats.pearsonr(trainscaled[:,i], trainloss)
    if abs(corr[0]) > 0.01:
        feature_corr = (i, abs(corr[0]))
        feature_corr_list.append(feature_corr)


# In[11]:

feature_corr_list = sorted(feature_corr_list, key=lambda tup: -tup[1])


# In[12]:

#print feature_corr_list
feature_corr_list_ = [(colnames[t[0]], t[1]) for t in feature_corr_list]
print feature_corr_list_


# In[13]:

feature_corr_list_minus = []
for i in range(featurenum-1):
    if i%7 == 0:
        print i
    for j in range(i+1, featurenum):

        corr = stats.pearsonr(trainscaled[:,i]-trainscaled[:,j], trainloss)
        if abs(corr[0]) > 0.1:
            feature_corr = (i, j,  abs(corr[0]))
            feature_corr_list_minus.append(feature_corr)

feature_corr_list_minus = sorted(feature_corr_list_minus, key=lambda tup: -tup[2])


# In[14]:

#print feature_corr_list_minus
feature_corr_list_minus_ = [(colnames[t[0]], colnames[t[1]], t[2]) for t in feature_corr_list_minus]
print feature_corr_list_minus_


# In[32]:

feature_corr_list_plus = []
for i in range(featurenum-1):
    if i%7 == 0:
        print i
    for j in range(i+1, featurenum):

        corr = stats.pearsonr(trainscaled[:,i]+trainscaled[:,j], trainloss)
        if abs(corr[0]) > 0.1:
            feature_corr = (i, j,  abs(corr[0]))
            feature_corr_list_plus.append(feature_corr)

feature_corr_list_plus = sorted(feature_corr_list_plus, key=lambda tup: -tup[2])


# In[33]:

print feature_corr_list_plus


# In[38]:

feature_corr_list_multi = []
for i in range(featurenum-1):
    if i%7 == 0:
        print i
    for j in range(i+1, featurenum):

        corr = stats.pearsonr(np.multiply(trainscaled[:,i], trainscaled[:,j] ) , trainloss)
        if abs(corr[0]) > 0.1:
            feature_corr = (i, j,  abs(corr[0]))
            feature_corr_list_multi.append(feature_corr)

feature_corr_list_multi = sorted(feature_corr_list_multi, key=lambda tup: -tup[2])


# In[39]:

feature_corr_list_multi


# In[40]:

feature_corr_list_divide = []
for i in range(featurenum-1):
    if i%7 == 0:
        print i
    for j in range(featurenum):

        corr = stats.pearsonr(np.divide(trainscaled[:,i], np.absolute(trainscaled[:,j]) +1 ) , trainloss)
        if abs(corr[0]) > 0.1:
            feature_corr = (i, j,  abs(corr[0]))
            feature_corr_list_divide.append(feature_corr)

feature_corr_list_divide = sorted(feature_corr_list_divide, key=lambda tup: -tup[2])


# In[41]:

feature_corr_list_divide


# In[52]:

print train['cat79'].describe()


# In[19]:

dictlim = {}
lim = 2
result = []
for tup in feature_corr_list_minus_:

    if tup[0] not in dictlim:
        dictlim[tup[0]] = 0
    if tup[1] not in dictlim:
        dictlim[tup[1]] = 0

    if dictlim[tup[1]] < lim and dictlim[tup[0]] < lim:
        result.append(tup)

        dictlim[tup[0]] += 1

        dictlim[tup[1]] += 1


print result


# In[20]:

dictlim


# In[ ]: