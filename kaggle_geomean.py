import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import sys

loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

files_test = [loc+"stacking_preds/submission_xgb_XGBRegressor10.csv", loc+"submissions/submission_nn_base.csv"]
files_train = [loc+"stacking_preds/preds_oob_xgb_XGBRegressor10.csv", loc+"submissions/preds_oob_nn_base.csv"]

df1 = pd.read_csv(files_test[0]).rename(index=str, columns={"loss": "loss1"})
df2 = pd.read_csv(files_test[1]).rename(index=str, columns={"loss": "loss2"})

dt1 = pd.read_csv(files_train[0]).rename(index=str, columns={"loss": "loss1"})
dt2 = pd.read_csv(files_train[1]).rename(index=str, columns={"loss": "loss2"})


data = df1.merge(df2, on="id")

datat = dt1.merge(dt2, on="id")

train = pd.read_csv(loc+'data/train.csv', nrows = None)

train = train[["loss", "id"]]

datat = datat.merge(train, on="id")

p_seq = [i*0.01 for i in range(101)]
shift_seq = [(i-50) for i in range(101)]
minloss = 1000000
opt_p = 0.5
opt_shift = 0
for shift in shift_seq:
    for p in p_seq:
        #print p
        datat["losst"] = np.multiply(np.power( np.array(datat["loss1"]),p) ,np.power( np.array(datat["loss2"]),1-p)) + shift
        loss = mean_absolute_error(datat["loss"].values, datat["losst"].values)
        if minloss>loss:
            minloss = loss
            opt_p = p
            opt_shift = shift



data["loss"] = np.multiply(np.power( np.array(data["loss1"]),opt_p) ,np.power( np.array(data["loss2"]),1-opt_p)) + opt_shift

data[["id", "loss"]].to_csv(loc+"submissions/submission_geomean"+str(opt_p)+"_"+str(opt_shift)+"_"+str(minloss)+".csv", index=False)