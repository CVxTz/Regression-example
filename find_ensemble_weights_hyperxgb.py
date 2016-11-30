
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from hyperopt import fmin, tpe, hp, Trials

import xgboost as xgb

def writeResultModel(modelname, perf, params, other):
    loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
    f = open(loc+'opt_result.txt','a')
    f.write(modelname+ '\n')
    f.write(other+ '\n')
    f.write(str(perf)+ 'MAE \n')
    f.write(str(params)+ '\n')
    f.write('\n')
    f.close()

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

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
loc+"stacking_preds/submission_xgb_KNeighborsRegressor2.csv",
loc+"stacking_preds/submission_xgb_RandomForestRegressor2.csv",
loc+"stacking_preds/submission_xgb_RandomForestRegressor1.csv",
loc+"stacking_preds/submission_xgb_KNeighborsRegressor4.csv",
loc+"stacking_preds/submission_xgb_KNeighborsRegressor3.csv"]



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
loc+"stacking_preds/preds_oob_xgb_KNeighborsRegressor2.csv",
loc+"stacking_preds/preds_oob_xgb_RandomForestRegressor2.csv",
loc+"stacking_preds/preds_oob_xgb_RandomForestRegressor1.csv",
loc+"stacking_preds/preds_oob_xgb_KNeighborsRegressor4.csv",
loc+"stacking_preds/preds_oob_xgb_KNeighborsRegressor3.csv"]


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

#pca = PCA()

#Xt = pca.fit_transform(Xt)
#X = pca.transform(X)


loss = np.array(train['loss']).ravel()
lossl = list_to_percentiles(loss)
n_folds = 10
folds = StratifiedKFold(lossl, n_folds=n_folds, shuffle = True, random_state = 1337)

d_test = xgb.DMatrix(X)

shift = 200
loss = np.log(loss+ shift).ravel()

def score(params):
    params['max_depth'] = int(params['max_depth'])
    pred = 0
    perf = 0
    pred_test = np.zeros(X.shape[0])
    early_stopping = 20
    cvscore = 0

    for i, (train_index, test_index) in enumerate(folds):
        print('\n Fold %d' % (i+1))
        X_train, X_val = Xt[train_index,:], Xt[test_index,:]
        y_train, y_val = loss[train_index], loss[test_index]

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)

        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
        clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping, feval=xg_eval_mae, maximize=False)


        pred = np.exp(clf.predict(d_valid) ) - shift

        pred_test += (np.exp(clf.predict(d_test)) - shift)/n_folds

        cvscore +=  mean_absolute_error(np.exp(y_val) - shift, pred)/n_folds

    perf = cvscore

    writeResultModel("xgb_stack", perf, params, "")
    print params
    print perf
    data["loss"] = pred_test
    data[["id", "loss"]].to_csv(loc+"xgb_sub/submission_xgb_"+str(perf)+".csv", index=False)
    return perf

#depth=2, do_rate=0, l1_=0, l2_=0, nb_neurones=30, activation='relu', init='he_normal'

def optimize(trials):
    space = {
    'eta' : hp.quniform('eta', 0.007, 0.5, 0.002),
    'max_depth' : hp.quniform('max_depth', 1, 13, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 200, 1),
    'subsample' : hp.quniform('subsample', 0.99, 1, 0.05),
    'gamma' : hp.quniform('gamma', 0.0, 10, 0.01),
    'alpha' : hp.quniform('alpha', 0.0, 10, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.99, 1, 0.01),
    'objective' : "reg:linear",
    'silent' : 1,
    'booster': 'gbtree'
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best


trials = Trials()

optimize(trials)