'''
Script mainly from =>
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''
from keras.layers.core import Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise
from keras.regularizers import l2, l1, l1l2, activity_l1l2
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
import read_data
import read_stack
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import math
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


def trainModelNn(varselect = False, datasetRead = "base", modelname= "", nbags = 5,
               params = {}, randp = False, model2 = False):

    sparse = True
    if datasetRead == "base":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetBase()
    elif datasetRead == "cox":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetcox()
    elif datasetRead == "coxpca":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetcoxpca()
    elif datasetRead == "stack":
        xtrain, xtest, id_train, id_test, y = read_stack.readData1()
    elif datasetRead == "impact":
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetImpact()
    elif datasetRead == "comp":
        scaler = StandardScaler()
        xtrain, xtest, id_train, id_test, y = read_data.readDataSetLexicalcomp()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)
        sparse = False

    if varselect:
        selector = SelectPercentile(f_regression, percentile=70)

        selector.fit(xtrain, y)

        xtrain = selector.transform(xtrain)
        xtest = selector.transform(xtest)

    early_stopping = 20
    ## cv-folds
    nfolds = 10
    lossl = list_to_percentiles(y)
    folds = StratifiedKFold(lossl, n_folds=5, shuffle = True, random_state = 20)
    #folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

    ## train models
    i = 0
    shift_val = 200
    nepochs = 70
    pred_oob = np.zeros(xtrain.shape[0])
    pred_test = np.zeros(xtest.shape[0])
    loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

    for (inTr, inTe) in folds:
        xtr = xtrain[inTr]
        xte = xtrain[inTe]

        ytr = y[inTr]
        yte = y[inTe]

        pred = np.zeros(xte.shape[0])
        for j in range(nbags):
            if randp:
                params['random_state'] = 1337+j

            early = EarlyStopping(patience=5)
            outmodel = loc+"modelCheckPoint/bestmodel.weights.best.hdf5"
            checkpoint = ModelCheckpoint(outmodel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            if model2:
                model = create_model2(xtr.shape[1])()
                clbcks = [early, checkpoint]
            else:
                model = create_model(xtr.shape[1])(**params)
                lrs = LearningRateScheduler(init_lr(params["lr"]))
                clbcks = [lrs, early, checkpoint]



            if sparse:
                fit = model.fit_generator(generator = batch_generator(xtr, ytr, 32, True),
                                          nb_epoch = nepochs,
                                          samples_per_epoch = xtr.shape[0],
                                          callbacks=clbcks,
                                          validation_data= batch_generator(xte, yte, 32, True),
                                          nb_val_samples=xte.shape[0],
                                          verbose = 1)
                model.load_weights(outmodel)
                model.compile(loss = 'mae', optimizer = 'adadelta')
                pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0]
                pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0]
            else:
                fit = model.fit(xtr, ytr,
                                          batch_size = 128,
                                          nb_epoch = nepochs,
                                          callbacks=clbcks,
                                          validation_data=(xte, yte),
                                          verbose = 1)
                model.load_weights(outmodel)
                model.compile(loss = 'mae', optimizer = 'adadelta')
                pred += model.predict(xte, batch_size=800)[:,0]
                pred_test += model.predict(xtest, batch_size = 800)[:,0]

        pred /= nbags
        pred_oob[inTe] = pred
        i += 1
        #print('Fold ', i, '- MAE:', score)
    pred_test /= (nfolds*nbags)

    #print('Total - MAE:', mean_absolute_error(y, pred_oob))

    ## train predictions
    df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
    df.to_csv(loc+'stacking_preds/preds_oob_xgb_'+modelname+'.csv', index = False)

    ## test predictions

    df = pd.DataFrame({'id': id_test, 'loss': pred_test})
    df.to_csv(loc+'stacking_preds/submission_xgb_'+modelname+'.csv', index = False)

    return mean_absolute_error(y, pred_oob)


def create_model(inputsize):
    def model_fn(optimizer='adam', init='he_normal', regl1 = 0.0001, regl2 = 0.0001, lr = 0.001, depth=2,
                 nb_neurone=50, dropout_rate=0.5, BN = True, prelu = True ):
        dict_opt = {'adam': Adam(lr=lr), 'sgd': SGD(lr=lr), "rmsprop": RMSprop(lr=lr), "adagrad": Adagrad(lr=lr), "adadelta": Adadelta(lr=lr)}
        opt = dict_opt[optimizer]

        model = Sequential()
        model.add(Dense(nb_neurone, init=init , input_shape=(inputsize,), W_regularizer=l1l2(l1=regl1, l2=regl2 )))
        if prelu == True:
            model.add(PReLU())
        else:
            model.add(LeakyReLU())
        if BN:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        for i in range(depth):
            model.add(Dense(nb_neurone, init=init, W_regularizer=l1l2(l1=regl1, l2=regl2 ) ) )
            if prelu == True:
                model.add(PReLU())
            else:
                model.add(LeakyReLU())
            if BN:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, W_regularizer=l1l2(l1=regl1, l2=regl2 ), init=init))
        model.compile(loss = 'mae',
                      optimizer=opt)
        return model
    return model_fn

def create_model2(inputsize):
    def model_fn():
        model = Sequential()
        model.add(Dense(400, input_dim = inputsize, init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(50, init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adadelta')
        return(model)

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

