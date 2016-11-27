from keras.layers.core import Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise, Input
import os
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


fair_constant = 2
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

shift = 200

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)

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
loc+"stacking_preds/submission_xgb_PassiveAggressiveRegressor1.csv"]



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
loc+"stacking_preds/preds_oob_xgb_PassiveAggressiveRegressor1.csv"]

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


loss = np.array(train['loss']).ravel()

train_y = np.log(loss + shift)

lossl = list_to_percentiles(loss)



nbag = 5
final_preds = 0
final_score = 0

d_test = xgb.DMatrix(X)


for b in range(nbag):

    params = {
    	'seed': b*100,
    	'colsample_bytree': 0.7,
    	'silent': 1,
    	'subsample': 0.95,
    	'learning_rate': 0.03,
    	'objective': 'reg:linear',
    	'max_depth': 5,
    	'min_child_weight': 100,
    	'booster': 'gbtree'}
    print str(nbag)+"................................................................................................................................"
    n_folds = 10
    folds = StratifiedKFold(lossl, n_folds=n_folds, shuffle = True, random_state = 20+100*b)

    pred = 0
    perf = 0

    #kf = KFold(Xt.shape[0], n_folds=n_folds, shuffle=True, random_state=b)
    for i, (train_index, test_index) in enumerate(folds):
        print('\n Fold %d' % (i+1))
        X_train, X_val = Xt[train_index,:], Xt[test_index,:]
        y_train, y_val = train_y[train_index], train_y[test_index]

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        clf = xgb.train(params,
                        d_train,
                        100000,
                        watchlist,
                        early_stopping_rounds=50,
                        obj=fair_obj,
                        feval=xg_eval_mae)

        scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)

        cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))

        y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit)) - shift
        #model.load_weights(outmodel)
        #model.compile(loss = 'mae', optimizer = 'adadelta')
        pred += y_pred/n_folds
        perf += cv_score/n_folds

    final_score += perf/nbag
    final_preds += pred/nbag


data["loss"] = final_preds



data[["id", "loss"]].to_csv(loc+"submissions/submission_stack_xgb"+str(final_score)+".csv", index=False)

