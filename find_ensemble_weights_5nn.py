from keras.layers.core import Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise, Input
from keras.regularizers import l1l2
import os
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def writeResultModel(modelname, perf, params, other):
    loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
    f = open(loc+'opt_result.txt','a')
    f.write(modelname+ '\n')
    f.write(other+ '\n')
    f.write(str(perf)+ 'MAE \n')
    f.write(str(params)+ '\n')
    f.write('\n')
    f.close()



def list_to_percentiles(numbers):
    pairs = zip(numbers, range(len(numbers)))
    pairs.sort(key=lambda p: p[0])
    result = [0 for i in range(len(numbers))]
    for rank in xrange(len(numbers)):
        original_index = pairs[rank][1]
        result[original_index] = int( rank * 100.0 / (len(numbers)-1))
    return result


loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

files_test = \
[loc+"submissions/submission_xgb_customobj_comp.csv",
loc+"stacking_preds/submission_xgb_ffnn28.csv",
loc+"stacking_preds/submission_5fold-average-xgb_fairobj_1132.143663.csv",
loc+"stacking_preds/submission_xgb_ffnn30.csv",
loc+"stacking_preds/submission_xgb_ffnn27.csv",
loc+"stacking_preds/submission_xgb_HuberRegressor1.csv",
loc+"stacking_preds/submission_xgb_XGBRegressor7.csv",
loc+"stacking_preds/submission_xgb_XGBRegressor10.csv",
loc+"stacking_preds/submission_xgb_XGBRegressor11.csv",
loc+"stacking_preds/submission_xgb_ffnn29.csv",
loc+"stacking_preds/submission_xgb_ffnn20.csv",
loc+"stacking_preds/submission_xgb_ffnn5.csv",
loc+"submissions/submission_xgb_customobj.csv",
loc+"submissions/submission_nn.csv",
loc+"submissions/submission_nn_base.csv",
loc+"submissions/submission_xgb_fs.csv",
loc+"stacking_preds/submission_xgb_ffnn4.csv",
loc+"stacking_preds/submission_xgb_ffnn_stack_3.csv",
loc+"stacking_preds/submission_xgb_LinearRegression1.csv",
loc+"stacking_preds/submission_xgb_SGDRegressor1.csv",
loc+"stacking_preds/submission_xgb_Lasso1.csv",
loc+"stacking_preds/submission_xgb_XGBRegressor1.csv",
loc+"submissions/submission_xgb.csv",
loc+"stacking_preds/submission_xgb_ffnn_stack_1.csv",
loc+"stacking_preds/submission_xgb_PassiveAggressiveRegressor1.csv",
loc+"stacking_preds/submission_xgb_KNeighborsRegressor2.csv"] #KNeighborsRegressor2



files_train = [loc+"submissions/preds_oob_xgb_customobj_comp.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn28.csv",
loc+"stacking_preds/preds_oob_xgb__5fold-average-xgb_fairobj_.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn30.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn27.csv",
loc+"stacking_preds/preds_oob_xgb_HuberRegressor1.csv",
loc+"stacking_preds/preds_oob_xgb_XGBRegressor7.csv",
loc+"stacking_preds/preds_oob_xgb_XGBRegressor10.csv",
loc+"stacking_preds/preds_oob_xgb_XGBRegressor11.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn29.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn20.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn5.csv",
loc+"submissions/preds_oob_xgb_customobj.csv",
loc+"submissions/preds_oob_nn.csv",
loc+"submissions/preds_oob_nn_base.csv",
loc+"submissions/preds_oob_xgb_fs.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn4.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn_stack_3.csv",
loc+"stacking_preds/preds_oob_xgb_LinearRegression1.csv",
loc+"stacking_preds/preds_oob_xgb_SGDRegressor1.csv",
loc+"stacking_preds/preds_oob_xgb_Lasso1.csv",
loc+"stacking_preds/preds_oob_xgb_XGBRegressor1.csv",
loc+"submissions/preds_oob_xgb.csv",
loc+"stacking_preds/preds_oob_xgb_ffnn_stack_1.csv",
loc+"stacking_preds/preds_oob_xgb_PassiveAggressiveRegressor1.csv",
loc+"stacking_preds/preds_oob_xgb_KNeighborsRegressor2.csv"]

dict_test = {}
dict_train = {}

colnames = []
for i in range(len(files_train)):
    colnames.append("loss"+str(i+1))
    dict_test[i] = pd.read_csv(files_test[i]).rename(index=str, columns={"loss": "loss"+str(i+1)})
    dict_train[i] = pd.read_csv(files_train[i]).rename(index=str, columns={"loss": "loss"+str(i+1)})

for i in range(len(files_train)):
    if i == 0:
        data = dict_test[i]
        datat = dict_train[i]
    else:
        data = data.merge(dict_test[i], on="id")
        datat = datat.merge(dict_train[i], on="id")
new_colnames = []
print "quad feat : "
for i in range( len( colnames)):
    print i
    for j in range(i, len( colnames)):
        data[colnames[i]+"_"+colnames[j]] = 0*data[colnames[i]]*data[colnames[j]]/5000
        datat[colnames[i]+"_"+colnames[j]] = 0*datat[colnames[i]]*datat[colnames[j]]/5000
        new_colnames.append(colnames[i]+"_"+colnames[j])


train = pd.read_csv(loc+'data/train.csv', nrows = None)

train = train[["loss", "id"]]

datat = datat.merge(train, on="id")


all_colnames = new_colnames + colnames
scaler = StandardScaler()
Xt = scaler.fit_transform(datat[all_colnames])
X = scaler.transform(data[all_colnames])

pca = PCA()

Xt = pca.fit_transform(Xt)
X = pca.transform(X)

def create_model2(inputsize):
    def model_fn(depth=2, do_rate=0, l1_=0, l2_=0, nb_neurones=30, activation='relu', init='he_normal'):

        nb_neurones = int(nb_neurones)
        depth = int(depth)

        model = Sequential()
        model.add(Dense(nb_neurones, input_dim = inputsize, init=init, W_regularizer=l1l2(l1_, l2_)))
        model.add(Dropout(do_rate))
        model.add(Activation(activation))
        if depth>1:
            for i in range(depth-1):
                model.add(Dense(nb_neurones, init=init, W_regularizer=l1l2(l1_, l2_)))
                model.add(Dropout(do_rate))
                model.add(Activation(activation))
        model.add(Dense(1, input_dim = inputsize, init=init, W_regularizer=l1l2(l1_, l2_)))
        model.compile(loss = 'mae', optimizer = 'adadelta')
        return(model)

    return model_fn
def init_lr(lr):
    def step_decay(epoch):
        initial_lrate = lr
        drop = 0.9
        epochs_drop = 2
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    return step_decay

loss = np.array(train['loss']).ravel()
lossl = list_to_percentiles(loss)
n_folds = 5
folds = StratifiedKFold(lossl, n_folds=n_folds, shuffle = True, random_state = 1337)


def score(params):
    pred = 0
    perf = 0

    for i, (train_index, test_index) in enumerate(folds):
        print('\n Fold %d' % (i+1))
        X_train, X_val = Xt[train_index,:], Xt[test_index,:]
        y_train, y_val = loss[train_index], loss[test_index]
        model = create_model2(X.shape[1])(**params)
        outmodel = loc+"modelCheckPoint/bestmodel.weights.best.hdf5"
        checkpoint = ModelCheckpoint(outmodel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(patience=3)
        lrs = LearningRateScheduler(init_lr(0.1))

        hist = model.fit(X_train, y_train, nb_epoch= 15, batch_size=50, validation_data=(X_val, y_val), verbose=0, callbacks=[early, checkpoint]) #, checkpoint , callbacks=[early]

        #model.load_weights(outmodel)
        #model.compile(loss = 'mae', optimizer = 'adadelta')
        pred += model.predict(X)/n_folds
        perf += hist.history['val_loss'][-1]/n_folds

    writeResultModel("nn_stack", perf, params, "")
    print params
    print perf
    data["loss"] = pred
    data[["id", "loss"]].to_csv(loc+"opt_sub/submission_nn_"+str(perf)+".csv", index=False)
    return perf

#depth=2, do_rate=0, l1_=0, l2_=0, nb_neurones=30, activation='relu', init='he_normal'

def optimize(trials):
    space = {
        'depth' : hp.quniform('depth', 1, 3.1, 1),
        'nb_neurones' : hp.quniform('nb_neurones', 10, 1000, 1),
        'do_rate' : hp.quniform('do_rate', 0.0, 0.95, 0.01),
        'l1_' : hp.loguniform('l1_', -10, 2),
        'l2_' : hp.loguniform('l2_', -10, 2),
        #'activation' : hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
        'init' : hp.choice('init',  ['he_normal', 'he_uniform', 'glorot_uniform']),
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best


trials = Trials()

optimize(trials)