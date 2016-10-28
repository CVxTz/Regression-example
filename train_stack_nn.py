'''
Script mainly from =>
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''
from keras.layers.core import Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise
from keras.regularizers import l2, l1, l1l2, activity_l1l2
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
import read_data
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import math
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, f_regression


def trainModelNn(varselect = False, datasetRead = "base", modelname= "", nbags = 5,
               params = {}, randp = False, shift = False):


    if datasetRead == "base":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetBase()
    elif datasetRead == "cox":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetcox()
    elif datasetRead == "coxpca":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetcoxpca()


    if varselect:
        selector = SelectPercentile(f_regression, percentile=70)

        selector.fit(xtrain, y)

        xtrain = selector.transform(xtrain)
        xtest = selector.transform(xtest)

    early_stopping = 20
    ## cv-folds
    nfolds = 5
    folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

    ## train models
    i = 0
    shift_val = 200
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


            model = create_model(xtr.shape[1])(**params)
            lrs = LearningRateScheduler(init_lr(params["lr"]))
            early = EarlyStopping(patience=2)
            fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                      nb_epoch = nepochs,
                                      samples_per_epoch = xtr.shape[0],
                                      callbacks=[lrs],
                                      verbose = 1)
            pred += np.exp( model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0] )
            pred_test += np.exp( model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0] )

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
    df.to_csv(loc+'stacking_preds/preds_oob_xgb_'+modelname+'.csv', index = False)

    ## test predictions

    df = pd.DataFrame({'id': id_test, 'loss': pred_test})
    df.to_csv(loc+'stacking_preds/submission_xgb_'+modelname+'.csv', index = False)

    return mean_absolute_error(y, pred_oob)


def create_model(inputsize):
    def model_fn(optimizer='adam', init='he_normal', regl1 = 0.0001, lr = 0.001, depth=2,
                 nb_neurone=50, dropout_rate=0.5, BN = True ):
        dict_opt = {'adam': Adam(lr=lr), 'sgd': SGD(lr=lr), "rmsprop": RMSprop(lr=lr), "adagrad": Adagrad(lr=lr)}
        opt = dict_opt[optimizer]

        model = Sequential()
        model.add(Dense(nb_neurone, init=init , input_shape=(inputsize,), W_regularizer=l1(l=regl1)))
        model.add(PReLU())
        if BN:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        for i in range(depth):
            model.add(Dense(nb_neurone, init=init, W_regularizer=l1(l=regl1)))
            model.add(PReLU())
            if BN:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, W_regularizer=l1(l=regl1)))
        model.compile(loss = 'mae',
                      optimizer=opt)
        return model
    return model_fn


def init_lr(lr):
    def step_decay(epoch):
        initial_lrate = lr
        drop = 0.5
        epochs_drop = 5
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    return step_decay

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