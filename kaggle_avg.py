import pandas as pd
from sklearn.metrics import mean_absolute_error
import sys

loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

files_test = [loc+"submissions/submission_xgb_customobj_comp.csv", loc+"stacking_preds/submission_xgb_ffnn28.csv",
loc+"stacking_preds/submission_5fold-average-xgb_fairobj_1132.143663.csv", loc+"stacking_preds/submission_xgb_ffnn30.csv", loc+"stacking_preds/submission_xgb_ffnn27.csv"]
files_train = [loc+"submissions/preds_oob_xgb_customobj_comp.csv", loc+"stacking_preds/preds_oob_xgb_ffnn28.csv",
loc+"stacking_preds/preds_oob_xgb__5fold-average-xgb_fairobj_.csv", loc+"stacking_preds/preds_oob_xgb_ffnn30.csv", loc+"stacking_preds/preds_oob_xgb_ffnn27.csv"]



df1 = pd.read_csv(files_test[0]).rename(index=str, columns={"loss": "loss1"})
df2 = pd.read_csv(files_test[1]).rename(index=str, columns={"loss": "loss2"})
df3 = pd.read_csv(files_test[2]).rename(index=str, columns={"loss": "loss3"})
df4 = pd.read_csv(files_test[3]).rename(index=str, columns={"loss": "loss4"})
df5 = pd.read_csv(files_test[4]).rename(index=str, columns={"loss": "loss5"})

dt1 = pd.read_csv(files_train[0]).rename(index=str, columns={"loss": "loss1"})
dt2 = pd.read_csv(files_train[1]).rename(index=str, columns={"loss": "loss2"})
dt3 = pd.read_csv(files_train[2]).rename(index=str, columns={"loss": "loss3"})
dt4 = pd.read_csv(files_train[3]).rename(index=str, columns={"loss": "loss4"})
dt5 = pd.read_csv(files_train[4]).rename(index=str, columns={"loss": "loss5"})


data = df1.merge(df2, on="id").merge(df3, on="id").merge(df4, on="id").merge(df5, on="id")

datat = dt1.merge(dt2, on="id").merge(dt3, on="id").merge(dt4, on="id").merge(dt5, on="id")

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
        #datat["losst"] = p*(datat["loss2"]+ datat["loss4"]+ datat["loss5"])/3 + (1-p)*(datat["loss1"]+ datat["loss3"])/2 + shift
        datat["losst"] = p*(datat["loss4"]) + (1-p)*(datat["loss3"]) + shift
        datat["losst"] = p*(datat["loss4"]) + (1-p)*(datat["loss3"]) + shift
        loss = mean_absolute_error(datat["loss"].values, datat["losst"].values)
        if minloss>loss:
            minloss = loss
            opt_p = p
            opt_shift = shift



#data["loss"] = opt_p*(data["loss2"]+ data["loss4"]+ data["loss5"])/3 + (1-opt_p)*(data["loss1"]+ data["loss3"])/2 +opt_shift
data["loss"] = opt_p*(data["loss4"]) + (1-opt_p)*(data["loss3"]) +opt_shift

data[["id", "loss"]].to_csv(loc+"submissions/submission_avg"+str(opt_p)+"_"+str(opt_shift)+"_"+str(minloss)+".csv", index=False)