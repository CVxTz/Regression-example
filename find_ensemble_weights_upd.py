
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

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


def mae_loss_func(weights):
    """
    scipy minimize will pass the weights as a numpy array
    """
    #print weights
    final_prediction = 0
    shift = weights[-1]
    weights = weights[0:(len(weights)-1)]
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    fun_val = mean_absolute_error(train['loss'], final_prediction+shift)
    print fun_val

    return fun_val
all_colnames = new_colnames + colnames
#scaler = StandardScaler()
#datat[all_colnames] = scaler.fit_transform(datat[all_colnames])
#data[all_colnames] = scaler.transform(data[all_colnames])
number_cols = 40
selectbest = SelectKBest(f_regression, k=number_cols)
Xt = selectbest.fit_transform(datat[new_colnames], train["loss"])
X = selectbest.transform(data[new_colnames])
newcol_select = ['loss_new_'+str(i) for i in range(1, number_cols+1)]
for i in range( number_cols):

    data[newcol_select[i]] = X[:,i]
    datat[newcol_select[i]] = Xt[:,i]
# finding the optimum weights part 1
predictions = []
predictions_t = []
selectcolnames = colnames+ newcol_select
for col in selectcolnames: # all_colnames:
    predictions.append(datat[col])
    predictions_t.append(data[col])



# the algorithms need a starting value, right not we chose 0.5 for all weights
# its better to choose many random starting points and run minimize a few times
starting_values = np.random.rand(( len(predictions)+ 1),1) # [3, -2, 3, 7, 0.1, 1, 8, 3, 20, 100]
# adding constraints and a different solver as suggested by user 16universe
starting_values[0:len( colnames)] = starting_values[0:len( colnames)] + 1.0/len( colnames)
cons = ({'type': 'eq', 'fun': lambda w: 1-sum(w)}) #

# our weights are bound between 0 and 1
bounds = [(-1, 1)] * ( len(predictions)+ 1)

res = minimize(mae_loss_func, starting_values, method='SLSQP', options={'disp': True} ) #, constraints=cons , bounds=bounds SLSQP BFGS

print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))


def mae_func(weights):
    """
    scipy minimize will pass the weights as a numpy array
    """
    final_prediction = 0
    shift = weights[-1]
    weights = weights[0:(len(weights)-1)]
    for weight, prediction in zip(weights, predictions_t):
            final_prediction += weight*prediction

    return final_prediction+shift


data["loss"] = mae_func( res['x'] )

data[["id", "loss"]].to_csv(loc+"submissions/submission_avg"+"_"+str(res['fun'])+".csv", index=False)