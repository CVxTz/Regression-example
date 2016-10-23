'''
Script mainly from =>
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''


'''
Ideas to test out :  variable selection
ideas to test out : log(x+1) cox box
'''
import read_data
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, f_regression


varselect = True
base = False

if base:
    xtrain, xtest, id_train, id_test, y = read_data.readDataSetBase()
else:
    xtrain, xtest, id_train, id_test, y = read_data.readDataSetcox()

def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)



if varselect:
    selector = SelectPercentile(f_regression, percentile=90)

    selector.fit(xtrain, y)

    xtrain = selector.transform(xtrain)
    xtest = selector.transform(xtest)

early_stopping = 10
## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 5
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = np.log(y[inTr]).ravel()
    xte = xtrain[inTe]
    yte = np.log(y[inTe]).ravel()
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        print('bag', j)
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 0)
        pred += np.exp( model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0] )
        pred_test += np.exp( model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0] )

    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte), pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(y, pred_oob))
loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv(loc+'submissions/preds_oob_nn_log_cox.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(loc+'submissions/submission_nn_log_cox.csv', index = False)