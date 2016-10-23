import pandas as pd
import sys

loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

files = [loc+"submissions/submission_xgb_fs_log.csv", loc+"submissions/submission_nn.csv"]
loc_outfile = "average.csv"

df1 = pd.read_csv(files[0]).rename(index=str, columns={"loss": "loss1"})

df2 = pd.read_csv(files[1]).rename(index=str, columns={"loss": "loss2"})




data = df1.merge(df2, on="id")

data["loss"] = 0.75*data["loss1"] + 0.25*data["loss2"]

data[["id", "loss"]].to_csv(loc+"submissions/submission_avg.csv", index=False)