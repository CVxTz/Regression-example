from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from find_ensemble_weights_sk import evalmodel

#model = KNeighborsRegressor()
#params = {"n_neighbors":5, "n_jobs":-1}
#modelname = "knn1"

#model = KNeighborsRegressor()
#params = {"n_neighbors":50, "n_jobs":-1}
#modelname = "knn2"#

model = KNeighborsRegressor()
params = {"n_neighbors":10, "n_jobs":-1}
modelname = "knn3"#

evalmodel(model, params, modelname)
model = KNeighborsRegressor()
params = {"n_neighbors":20, "n_jobs":-1}
modelname = "knn4"#

evalmodel(model, params, modelname)


model = RandomForestRegressor()
params = {"criterion" : "mae", "n_estimators": 200, "n_jobs":-1}
modelname = "RF1"#

evalmodel(model, params, modelname)

model = RandomForestRegressor()
params = {"criterion" : "mse", "n_estimators": 200, "n_jobs":-1}
modelname = "RF2"#

evalmodel(model, params, modelname)

model = ExtraTreesRegressor()
params = {"n_estimators": 200, "n_jobs":-1}
modelname = "ET1"#

evalmodel(model, params, modelname)

model = ExtraTreesRegressor()
params = {"n_estimators": 500, "n_jobs":-1}
modelname = "ET2"#

evalmodel(model, params, modelname)

