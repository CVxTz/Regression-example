
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
loc+"submissions/submission_xgb_fs.csv"]



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
loc+"submissions/preds_oob_xgb_fs.csv"]

df1 = pd.read_csv(files_test[0]).rename(index=str, columns={"loss": "loss1"})
df2 = pd.read_csv(files_test[1]).rename(index=str, columns={"loss": "loss2"})
df3 = pd.read_csv(files_test[2]).rename(index=str, columns={"loss": "loss3"})
df4 = pd.read_csv(files_test[3]).rename(index=str, columns={"loss": "loss4"})
df5 = pd.read_csv(files_test[4]).rename(index=str, columns={"loss": "loss5"})
df6 = pd.read_csv(files_test[5]).rename(index=str, columns={"loss": "loss6"})
df7 = pd.read_csv(files_test[6]).rename(index=str, columns={"loss": "loss7"})
df8 = pd.read_csv(files_test[7]).rename(index=str, columns={"loss": "loss8"})
df9 = pd.read_csv(files_test[8]).rename(index=str, columns={"loss": "loss9"})
df10 = pd.read_csv(files_test[9]).rename(index=str, columns={"loss": "loss10"})
df11 = pd.read_csv(files_test[10]).rename(index=str, columns={"loss": "loss11"})
df12 = pd.read_csv(files_test[11]).rename(index=str, columns={"loss": "loss12"})
df13 = pd.read_csv(files_test[12]).rename(index=str, columns={"loss": "loss13"})
df14 = pd.read_csv(files_test[13]).rename(index=str, columns={"loss": "loss14"})
df15 = pd.read_csv(files_test[14]).rename(index=str, columns={"loss": "loss15"})
df16 = pd.read_csv(files_test[15]).rename(index=str, columns={"loss": "loss16"})



dt1 = pd.read_csv(files_train[0]).rename(index=str, columns={"loss": "loss1"})
dt2 = pd.read_csv(files_train[1]).rename(index=str, columns={"loss": "loss2"})
dt3 = pd.read_csv(files_train[2]).rename(index=str, columns={"loss": "loss3"})
dt4 = pd.read_csv(files_train[3]).rename(index=str, columns={"loss": "loss4"})
dt5 = pd.read_csv(files_train[4]).rename(index=str, columns={"loss": "loss5"})
dt6 = pd.read_csv(files_train[5]).rename(index=str, columns={"loss": "loss6"})
dt7 = pd.read_csv(files_train[6]).rename(index=str, columns={"loss": "loss7"})
dt8 = pd.read_csv(files_train[7]).rename(index=str, columns={"loss": "loss8"})
dt9 = pd.read_csv(files_train[8]).rename(index=str, columns={"loss": "loss9"})
dt10 = pd.read_csv(files_train[9]).rename(index=str, columns={"loss": "loss10"})
dt11 = pd.read_csv(files_train[10]).rename(index=str, columns={"loss": "loss11"})
dt12 = pd.read_csv(files_train[11]).rename(index=str, columns={"loss": "loss12"})
dt13 = pd.read_csv(files_train[12]).rename(index=str, columns={"loss": "loss13"})
dt14 = pd.read_csv(files_train[13]).rename(index=str, columns={"loss": "loss14"})
dt15 = pd.read_csv(files_train[14]).rename(index=str, columns={"loss": "loss15"})
dt16 = pd.read_csv(files_train[15]).rename(index=str, columns={"loss": "loss16"})


data = df1.merge(df2, on="id").merge(df3, on="id").merge(df4, on="id")\
    .merge(df5, on="id").merge(df6, on="id").merge(df7, on="id")\
    .merge(df8, on="id").merge(df9, on="id").merge(df10, on="id")\
    .merge(df11, on="id").merge(df12, on="id").merge(df13, on="id")\
    .merge(df14, on="id").merge(df15, on="id").merge(df16, on="id")

datat = dt1.merge(dt2, on="id").merge(dt3, on="id").merge(dt4, on="id")\
    .merge(dt5, on="id").merge(dt6, on="id").merge(dt7, on="id")\
    .merge(dt8, on="id").merge(dt9, on="id").merge(dt10, on="id")\
    .merge(dt11, on="id").merge(dt12, on="id").merge(dt13, on="id")\
    .merge(dt14, on="id").merge(dt15, on="id").merge(dt16, on="id")

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
bounds = [(-10, 10)] * ( len(predictions)+ 1)

res = minimize(mae_loss_func, starting_values, method='SLSQP') #, constraints=cons , bounds=bounds

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