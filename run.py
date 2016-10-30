
from train_stack_sk import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb


def writeResultModel(modelname, perf, params, other):
    loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
    f = open(loc+'stack_result.txt','a')
    f.write(modelname+ '\n')
    f.write(other+ '\n')
    f.write(str(perf)+ 'MAE \n')
    f.write(str(params)+ '\n')
    f.write('\n')
    f.close()


#
model = xgb.XGBRegressor()
modelname = "XGBRegressor7"
params = {"n_estimators": int(2012 / 0.9), "nthread":-1,
          "colsample_bytree":0.5, "subsample":0.8, "reg_alpha":1, "gamma":1, "learning_rate":0.01, 'min_child_weight': 1, 'max_depth': 12}
perf = trainModel(model, params=params, nbags=5, modelname = modelname, randp=False, shift=True, varselect=False)
other = "nbags=5, randp=False, shift=True, varselect=False"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)


#
model = xgb.XGBRegressor()
modelname = "XGBRegressor8"
params = {"n_estimators": int(2012 / 0.9), "nthread":-1,
          "colsample_bytree":0.5, "subsample":0.8, "reg_alpha":1, "gamma":1, "learning_rate":0.01, 'min_child_weight': 1, 'max_depth': 9}
perf = trainModel(model, params=params, nbags=3, modelname = modelname, randp=False, shift=True, varselect=False)
other = "nbags=5, randp=False, shift=True, varselect=False"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)


#
model = xgb.XGBRegressor()
modelname = "XGBRegressor9"
params = {"n_estimators": int(2012 / 0.9), "nthread":-1,
          "colsample_bytree":0.5, "subsample":0.8, "reg_alpha":1, "gamma":1, "learning_rate":0.01, 'min_child_weight': 1, 'max_depth': 15}
perf = trainModel(model, params=params, nbags=3, modelname = modelname, randp=False, shift=True, varselect=False)
other = "nbags=5, randp=False, shift=True, varselect=False"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)




# #
# model = xgb.XGBRegressor()
# modelname = "XGBRegressor1"
# params = {"n_estimators": 1500, "nthread":-1, "colsample_bytree":0.9, "subsample":0.9, "reg_alpha":5, "reg_lambda":3}
# perf = trainModel(model, params=params, nbags=5, modelname = modelname, randp=False)
# other = "nbags=5, randp=True"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
# #
# model = xgb.XGBRegressor()
# params = {"n_estimators": 1200, "nthread":-1, "colsample_bytree":0.8, "subsample":0.8}
# modelname = "XGBRegressor2"
# perf = trainModel(model, params=params, nbags=5, modelname = modelname, randp=False)
# other = "nbags=5, randp=True"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
#
# #
# model = xgb.XGBRegressor()
# params = {"n_estimators": 1500, "nthread":-1, "colsample_bytree":0.9, "subsample":0.8}
# modelname = "XGBRegressor3"
# perf = trainModel(model, params=params, nbags=5, modelname = modelname, randp=False)
# other = "nbags=5, randp=True"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
#
# #
# model = xgb.XGBRegressor(reg_alpha=1, reg_lambda=3)
# params = {"n_estimators": 1400, "nthread":-1, "colsample_bytree":0.8, "subsample":0.9, "reg_alpha":1, "reg_lambda":3}
# modelname = "XGBRegressor4"
# perf = trainModel(model, params=params, nbags=5, modelname = modelname, randp=False)
# other = "nbags=5, randp=True"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)

#
# model = xgb.XGBRegressor()
# modelname = "XGBRegressor5"
# params = {"n_estimators": int(2012 / 0.9), "nthread":-1,
#           "colsample_bytree":0.5, "subsample":0.8, "reg_alpha":1, "gamma":1, "learning_rate":0.01}
# perf = trainModel(model, params=params, nbags=3, modelname = modelname, randp=False, shift=True, varselect=False)
# other = "nbags=5, randp=False, shift=True, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = xgb.XGBRegressor()
# modelname = "XGBRegressor6"
# params = {"n_estimators": int(2012 / 0.9), "nthread":-1,
#           "colsample_bytree":0.5, "subsample":0.8, "reg_alpha":5, "gamma":10, "learning_rate":0.01}
# perf = trainModel(model, params=params, nbags=3, modelname = modelname, randp=False, shift=True, varselect=False)
# other = "nbags=5, randp=False, shift=True, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
# model = Lasso()
# params = {}
# modelname = "Lasso1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = Ridge()
# params = {}
# modelname = "Ridge1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = LinearRegression()
# params = {}
# modelname = "LinearRegression1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = HuberRegressor()
# params = {}
# modelname = "HuberRegressor1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = PassiveAggressiveRegressor()
# params = {}
# modelname = "PassiveAggressiveRegressor1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = SGDRegressor()
# params = {}
# modelname = "SGDRegressor1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = KNeighborsRegressor()
# params = {"n_neighbors":5, "n_jobs":-1}
# modelname = "KNeighborsRegressor1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = KNeighborsRegressor()
# params = {"n_neighbors":10, "n_jobs":-1}
# modelname = "KNeighborsRegressor2"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = KNeighborsRegressor()
# params = {"n_neighbors":15, "n_jobs":-1}
# modelname = "KNeighborsRegressor3"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = KNeighborsRegressor()
# params = {"n_neighbors":30, "n_jobs":-1}
# modelname = "KNeighborsRegressor4"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False)
# other = "nbags=1, randp=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = RandomForestRegressor()
# params = {"criterion" : "mse", "n_estimators": 200, "n_jobs":-1}
# modelname = "RandomForestRegressor1"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=True)
# other = "nbags=5, randp=True"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
# model = RandomForestRegressor()
# params = {"criterion" : "mse", "n_estimators": 300, "n_jobs":-1}
# modelname = "RandomForestRegressor2"
# perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=True)
# other = "nbags=5, randp=True"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
# #
