import pandas as pd
import sys

loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

files_test = [loc+"submissions/submission_avg1119.64993307.csv", loc+"submissions/submission_avg1117.77612863.csv",
              loc+"submissions/submission_avg1117.91202723.csv", loc+"submissions/submission_avg_1117.94987899.csv"]

loc_outfile = "average.csv"

df1 = pd.read_csv(files_test[0]).rename(index=str, columns={"loss": "loss1"})

df2 = pd.read_csv(files_test[1]).rename(index=str, columns={"loss": "loss2"})

df3 = pd.read_csv(files_test[2]).rename(index=str, columns={"loss": "loss3"})

df4 = pd.read_csv(files_test[3]).rename(index=str, columns={"loss": "loss4"})

data = df1.merge(df2, on="id").merge(df3, on="id").merge(df4, on="id")

data["loss"] = ( data["loss2"] + data["loss1"] + data["loss3"] + data["loss4"] )/4

data[["id", "loss"]].to_csv(loc+"submissions/submission_avg.csv", index=False)