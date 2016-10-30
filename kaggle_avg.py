import pandas as pd
import sys

loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"

files = [loc+"stacking_preds/submission_xgb_XGBRegressor9.csv", loc+"stacking_preds/submission_xgb_ffnn3.csv"]
loc_outfile = "average.csv"

df1 = pd.read_csv(files[0]).rename(index=str, columns={"loss": "loss1"})

df2 = pd.read_csv(files[1]).rename(index=str, columns={"loss": "loss2"})




data = df1.merge(df2, on="id")

data["loss"] = 0.5*data["loss1"] + 0.5*data["loss2"]

data[["id", "loss"]].to_csv(loc+"submissions/submission_avg.csv", index=False)