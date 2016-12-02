from hyperoptUtils import *
import read_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

xtrain, xtest, id_train, id_test, y = read_data.readDataSetBase()

data = [xtrain, xtest, id_train, id_test, y]

max_evals = 50

#model = Lasso()
#space = {
#'alpha' : hp.loguniform('alpha', -10, 0),
#         }
#modelname = "Lasso1"
#optimizehyper(space, model, modelname, max_evals, data)
#
##
#model = Ridge()
#space = {
#'alpha' : hp.loguniform('alpha', -10, 0),
# #        }
#modelname = "Ridge1"
#optimizehyper(space, model, modelname, max_evals, data)

#
model = HuberRegressor()
space = {
'alpha' : hp.loguniform('alpha', -10, 0),
'epsilon' : hp.uniform('epsilon', 1.01, 3000)
         }
modelname = "HuberRegressor1"
optimizehyper(space, model, modelname, max_evals, data)

#
model = PassiveAggressiveRegressor()
space = {
'C' : hp.loguniform('C', -10, 2),
         }
modelname = "PassiveAggressiveRegressor1"
optimizehyper(space, model, modelname, max_evals, data)
#
model = SGDRegressor()
space = {
'alpha' : hp.loguniform('alpha', -10, 0),
'l1_ratio' : hp.loguniform('l1_ratio', -1, 1)
         }
modelname = "SGDRegressor1"
optimizehyper(space, model, modelname, max_evals, data)
