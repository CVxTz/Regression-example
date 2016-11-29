
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

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

train = pd.read_csv(loc+'data/train.csv', nrows = None)

train = train[["loss", "id"]]

datat = datat.merge(train, on="id")


def mae_loss_func(weights):
    """
    scipy minimize will pass the weights as a numpy array
    """
    final_prediction = 0
    shift = weights[-1]
    weights = weights[0:(len(weights)-1)]
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return mean_absolute_error(train['loss'], final_prediction+shift)


# finding the optimum weights
predictions = []
for i in range(1, (len(files_train)+1)):
    predictions.append(datat['loss'+str(i)])

# the algorithms need a starting value, right not we chose 0.5 for all weights
# its better to choose many random starting points and run minimize a few times
starting_values = np.random.rand(( len(predictions)+ 1),1) # [3, -2, 3, 7, 0.1, 1, 8, 3, 20, 100]
# adding constraints and a different solver as suggested by user 16universe
cons = ({'type': 'eq', 'fun': lambda w: 1-sum(w)}) #

# our weights are bound between 0 and 1
bounds = [(0, 10)] * ( len(predictions)) + [(-100, 100)]

res = minimize(mae_loss_func, starting_values, method='SLSQP', bounds=bounds) #, constraints=cons , bounds=bounds

print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))

predictions_t = []
for i in range(1,( len(files_train)+1)):
    predictions_t.append(data['loss'+str(i)])


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