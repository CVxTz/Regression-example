from keras.layers.core import Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise
from keras.regularizers import l2, l1, l1l2, activity_l1l2
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA

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
loc+"stacking_preds/submission_xgb_XGBRegressor1.csv"]



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
loc+"stacking_preds/preds_oob_xgb_XGBRegressor1.csv"]

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
    def model_fn():
        model = Sequential()
        model.add(Dense(1, input_dim = inputsize, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adadelta')
        return(model)

    return model_fn
def init_lr(lr):
    def step_decay(epoch):
        initial_lrate = lr
        drop = 0.95
        epochs_drop = 5
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    return step_decay

model = create_model2(X.shape[1])()
outmodel = loc+"modelCheckPoint/bestmodel.weights.best.hdf5"
checkpoint = ModelCheckpoint(outmodel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(patience=5)
lrs = LearningRateScheduler(init_lr(0.1))
loss = np.array(train['loss']).ravel()
shuf = np.random.permutation(len(loss))
hist = model.fit(Xt[shuf,:], loss[shuf], nb_epoch= 1000, batch_size=50, validation_split=0.2, callbacks=[early, checkpoint])

model.load_weights(outmodel)
model.compile(loss = 'mae', optimizer = 'adadelta')
pred = model.predict(X)


data["loss"] = pred

data[["id", "loss"]].to_csv(loc+"submissions/submission_avg"+str(hist.history['val_loss'][-1])+".csv", index=False)

