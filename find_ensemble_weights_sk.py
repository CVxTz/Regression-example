
import os
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


#model = KNeighborsRegressor()
#params = {"n_neighbors":5, "n_jobs":-1}
#modelname = "knn1"

model = KNeighborsRegressor()
params = {"n_neighbors":50, "n_jobs":-1}
modelname = "knn2"#

def list_to_percentiles(numbers):
    pairs = zip(numbers, range(len(numbers)))
    pairs.sort(key=lambda p: p[0])
    result = [0 for i in range(len(numbers))]
    for rank in xrange(len(numbers)):
        original_index = pairs[rank][1]
        result[original_index] = int( rank * 100.0 / (len(numbers)-1))
    return result

def evalmodel(model, params, modelname):
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
     loc+"opt_sk/submission_HuberRegressor11238.70739646.csv",
     loc+"opt_sk/submission_Lasso11237.27124071.csv",
     loc+"opt_sk/submission_PassiveAggressiveRegressor11250.09800089.csv",
     loc+"opt_sk/submission_Ridge11237.98865498.csv",
     loc+"opt_sk/submission_SGDRegressor11241.96375313.csv",
      loc+"submissions/submission_fairobj_1130.839299.csv"]



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
                   loc+"opt_sk/preds_oob_HuberRegressor11238.70739646.csv",
                   loc+"opt_sk/preds_oob_Lasso11237.27124071.csv",
                   loc+"opt_sk/preds_oob_PassiveAggressiveRegressor11250.09800089.csv",
                   loc+"opt_sk/preds_oob_Ridge11237.98865498.csv",
                   loc+"opt_sk/preds_oob_SGDRegressor11241.96375313.csv",
                                  loc+"submissions/preds_oob_xgb_fairobj_1130.83929874.csv"]

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
    #new_colnames = []
    #print "quad feat : "
    #for i in range( len( colnames)):
    #    print i
    #    for j in range(i, len( colnames)):
    #        data[colnames[i]+"_"+colnames[j]] = 0*data[colnames[i]]*data[colnames[j]]/5000
    #        datat[colnames[i]+"_"+colnames[j]] = 0*datat[colnames[i]]*datat[colnames[j]]/5000
    #        new_colnames.append(colnames[i]+"_"+colnames[j])


    train = pd.read_csv(loc+'data/train.csv', nrows = None)

    train = train[["loss", "id"]]

    datat = datat.merge(train, on="id")


    all_colnames =  colnames
    scaler = StandardScaler()
    Xt = scaler.fit_transform(datat[all_colnames])
    X = scaler.transform(data[all_colnames])

    #pca = PCA()
    #
    #Xt = pca.fit_transform(Xt)
    #X = pca.transform(X)


    loss = np.array(train['loss']).ravel()
    lossl = list_to_percentiles(loss)

    nbag = 5
    final_preds = 0
    final_score = 0
    pred_oob = np.zeros(shape = (len(loss), 1)).ravel()
    for b in range(nbag):
        print str(nbag)+"................................................................................................................................"
        n_folds = 10
        folds = StratifiedKFold(lossl, n_folds=n_folds, shuffle = True, random_state = 20+100*b)

        pred = 0
        perf = 0
        pred_test = np.zeros(shape = (len(loss), 1)).ravel()

        #kf = KFold(Xt.shape[0], n_folds=n_folds, shuffle=True, random_state=b)
        for i, (train_index, test_index) in enumerate(folds):
            print('\n Fold %d' % (i+1))
            X_train, X_val = Xt[train_index,:], Xt[test_index,:]
            y_train, y_val = loss[train_index], loss[test_index]
            model.set_params(**params)
            model.fit(X_train, y_train)

            predoo = model.predict(X_val)

            pred_test[test_index] += predoo
            pred += model.predict(X)/n_folds
            perf += mean_absolute_error(y_val, predoo)/n_folds

        final_score += perf/nbag
        final_preds += pred/nbag
        pred_oob += pred_test/nbag
    #

    data["loss"] = final_preds

    data[["id", "loss"]].to_csv(loc+"submissions/submission_sk"+modelname+"_"+str(final_score)+".csv", index=False)#

    train["loss"] = pred_oob

    train[["id", "loss"]].to_csv(loc+"submissions/preds_oob_sk"+modelname+"_"+str(final_score)+".csv", index=False)