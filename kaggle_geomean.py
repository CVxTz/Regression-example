

import pandas as pd
import sys
import numpy as np


loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

files = [loc+"submissions/xgb_starter_v2.sub.csv", loc+"submissions/submission_nn.csv"]
loc_outfile = "average.csv"

df1 = pd.read_csv(files[0]).rename(index=str, columns={"loss": "loss1"})

df2 = pd.read_csv(files[1]).rename(index=str, columns={"loss": "loss2"})




data = df1.merge(df2, on="id")

data["loss"] =  np.multiply(np.power( np.array(data["loss1"]),0.5) ,np.power( np.array(data["loss2"]),0.5))

data[["id", "loss"]].to_csv(loc+"submissions/submission_geomean.csv", index=False)